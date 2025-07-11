from sqlalchemy import text
import pandas as pd
from datetime import date

from db.db import get_engine
from typing import List, Dict, Any

# ============================ DB 설정 ============================

# SQLAlchemy 엔진 객체 생성 (db/db.py에 정의된 get_engine 사용)
engine = get_engine()

# ============================ 카테고리 정의 ============================

# 카테고리 코드와 설명 매핑
CATEGORIES = {
    "employee":       "사원 정보",
    "department":     "내가 소속된 부서의 구성원 정보",
    "commute":        "나의 출퇴근 통계 정보",
    "my_vacation":    "나의 신청한 휴가 일정",
    "team_vacation":  "부서 동료의 예정된 1달 간 휴가 일정",
    "business_trip":  "나의 예정된 1달 간 출장 일정",
    "kpi":            "현재 진행중인 나의 KPI 현황",
    "holiday":        "다가오는 휴일",
    "company":        "회사 정보",
}

# ============================ 카테고리별 SQL 쿼리 ============================

CATEGORY_QUERIES: Dict[str, str] = {
    # 1. 사원 정보
    "employee": """
        SELECT e.emp_no, e.email,
               d.name AS department_name, d.contact AS department_contact,
               he.name AS department_head, p.name AS position_name,
               ROUND(e.remaining_dayoff_hours/8,2) AS remaining_dayoff_days,
               e.remaining_refresh_days
          FROM employee e
     LEFT JOIN department d   ON e.dept_id      = d.dept_id
     LEFT JOIN dept_head dh    ON d.dept_id      = dh.dept_id
     LEFT JOIN employee he     ON dh.emp_id      = he.emp_id
     LEFT JOIN position p      ON e.position_id  = p.position_id
         WHERE e.emp_id = :employee_id
    """,

    # 2. 부서 구성원 (자신 부서 + 하위 부서 포함)
    "department": """
        WITH RECURSIVE sub AS (
          SELECT dept_id FROM employee WHERE emp_id = :employee_id
          UNION ALL
          SELECT d.dept_id
            FROM department d
            JOIN sub s ON d.parent_dept_id = s.dept_id
        )
        SELECT
          e.name          AS member_name,
          p.name          AS position_name,
          e.email         AS email,
          e.contact       AS phone_number
        FROM employee e
        JOIN sub         s ON e.dept_id       = s.dept_id
        LEFT JOIN position p ON e.position_id = p.position_id
    """,

    # 3. 출퇴근 통계 (이번 달)
    "commute": """
        SELECT
            SUM(TIMESTAMPDIFF(MINUTE, start_at, end_at) - break_time) AS total_minutes,
            SUM(CASE WHEN type_id = (SELECT type_id FROM work_type WHERE type_name='OVERTIME') 
                    THEN TIMESTAMPDIFF(MINUTE, start_at, end_at) - break_time ELSE 0 END) AS overtime_minutes,
            SUM(CASE WHEN TIME(start_at) >= '22:00' OR TIME(end_at) < '06:00'
                    THEN TIMESTAMPDIFF(MINUTE, start_at, end_at) - break_time ELSE 0 END) AS night_minutes,
            SUM(CASE WHEN DATE(start_at) IN (SELECT date FROM holiday)
                    THEN TIMESTAMPDIFF(MINUTE, start_at, end_at) - break_time ELSE 0 END) AS holiday_minutes
        FROM work
        WHERE emp_id = :employee_id
        AND YEAR(start_at) = YEAR(CURDATE())
        AND MONTH(start_at) = MONTH(CURDATE())
    """,

    # 4. 휴가 일정 (같은 부서 동료 + 본인 예정 휴가)
    "my_vacation": """
        SELECT
        vt.description   AS vacation_name,
        v.start_date,
        v.end_date,
        v.reason
        FROM vacation v
        JOIN approve a         ON v.approve_id = a.approve_id
        JOIN vacation_type vt  ON v.vacation_type_id = vt.vacation_type_id
        WHERE a.emp_id = :employee_id
        AND v.start_date >= CURDATE()
    """,

    "team_vacation": """
        SELECT
        vt.description   AS vacation_name,
        v.start_date,
        v.end_date,
        v.reason,
        e.name           AS employee_name
        FROM vacation v
        JOIN approve a         ON v.approve_id = a.approve_id
        JOIN vacation_type vt  ON v.vacation_type_id = vt.vacation_type_id
        JOIN employee e        ON a.emp_id = e.emp_id
        WHERE e.dept_id = (
            SELECT dept_id FROM employee WHERE emp_id = :employee_id
        )
        AND e.emp_id != :employee_id
        AND v.start_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 1 MONTH)
    """,
    # 5. 출장 일정 (다음 1개월, 승인된 건만)
    "business_trip": """
        SELECT
          b.type      AS trip_type,
          b.place     AS trip_place,
          b.start_date,
          b.end_date,
          b.reason,
          b.cost      AS estimated_cost
        FROM business_trip b
        JOIN approve a ON b.approve_id = a.approve_id
        WHERE a.emp_id    = :employee_id
          AND a.status_id = 2
          AND b.start_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 1 MONTH)
    """,

    # 6. KPI 현황 (승인된 목표만)
    "kpi": """
        SELECT
          goal_value    AS kpi_goal,
          kpi_progress  AS progress_rate,
          deadline
        FROM kpi
        WHERE emp_id    = :employee_id
          AND status_id = 2
    """,

    # 7. 다가오는 휴일 (3개월 이내)
    "holiday": """
        SELECT holiday_name, date
        FROM holiday
        WHERE date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 3 MONTH)
        ORDER BY date
    """,

    # 8. 회사 정보 (기본 정보 1건)
    "company": """
        SELECT name AS company_name, chairman AS company_chairman,
               address AS company_address, establish_date, work_start_time
        FROM company
        LIMIT 1
    """
}

# ============================ 유틸 함수 ============================

def minutes_to_hour_minute_str(minutes: int) -> str:
    """
    분 단위를 'X시간 Y분' 형태의 문자열로 변환
    예: 90 -> '1시간 30분', 120 -> '2시간'
    """
    hours, mins = divmod(minutes, 60)
    return f"{hours}시간 {mins}분" if mins else f"{hours}시간"

# ============================ 카테고리별 요약 포맷터 ============================

def format_category_summary(category: str, data: List[Dict[str, Any]]) -> str:
    """
    카테고리별 요약을 구조적으로 명확하게 만들어주는 함수
    """
    header = f"### [{category}] {CATEGORIES.get(category, category)}"

    if not data:
        return f"{header}\n관련 정보가 없습니다."

    if category == "employee":
        e = data[0]
        lines = [
            f"사번: {e['emp_no']}",
            f"이메일: {e['email']}",
            f"소속 부서: {e['department_name']} ({e['department_contact']})",
        ]
        if e.get("department_head"):
            lines.append(f"부서장: {e['department_head']}")
        lines.append(f"직위: {e['position_name']}")
        lines.append(f"잔여 연차: {e['remaining_dayoff_days']}일")
        if e.get("remaining_refresh_days", 0) > 0:
            lines.append(f"잔여 리프레시 휴가: {e['remaining_refresh_days']}일")
        return f"{header}\n" + "\n".join(lines)

    elif category == "department":
        return f"{header}\n" + "\n".join(
            f"{r['member_name']} ({r['position_name']}, {r['email']}, {r['phone_number']})"
            for r in data
        )

    elif category == "commute":
        c = data[0]
        return (
            f"{header}\n"
            f"사원님의 이번 달 현재까지의 출퇴근 통계는 다음과 같습니다.\n"
            f"- 총 근무시간: {minutes_to_hour_minute_str(c['total_minutes'])}\n"
            f"- 연장 근무시간: {minutes_to_hour_minute_str(c['overtime_minutes'])}\n"
            f"- 야간 근무시간: {minutes_to_hour_minute_str(c['night_minutes'])}\n"
            f"- 휴일 근무시간: {minutes_to_hour_minute_str(c['holiday_minutes'])}"
        )

    elif category == "my_vacation":
        return f"{header}\n사원님의 예정 휴가 일정은 다음과 같습니다.\n" + "\n".join(
            f"{r['start_date']}~{r['end_date']}: {r['vacation_name']} ({r.get('reason', '사유 없음')})"
            for r in data
        )

    elif category == "team_vacation":
        return f"{header}\n같은 부서 동료의 이번 달 휴가 일정은 다음과 같습니다.\n" + "\n".join(
            f"{r['employee_name']} - {r['start_date']}~{r['end_date']}: {r['vacation_name']} ({r.get('reason', '사유 없음')})"
            for r in data
        )

    elif category == "business_trip":
        return f"{header}\n사원님의 다가오는 1개월 간의 출장 일정은 다음과 같습니다.\n" + "\n".join(
            f"{r['start_date']}~{r['end_date']} | {r['trip_type']} | {r['trip_place']} | "
            f"사유: {r['reason']} | 예상경비: {r['estimated_cost']}원"
            for r in data
        )


    elif category == "kpi":
        return f"{header}\n현재 사원님의 KPI 목표 현황은 다음과 같습니다.\n" + "\n".join(
            f"목표: {r['kpi_goal']} | 진척도: {r['progress_rate']}% | 마감일: {r['deadline']}"
            for r in data
        )


    elif category == "holiday":
        return f"{header}\n다가오는 3개월 간의 회사 휴일은 다음과 같습니다.\n" + "\n".join(
            f"{r['date']}: {r['holiday_name']}" for r in data
        )

    elif category == "company":
        c = data[0]
        return (
            f"{header}\n"
            f"회사명: {c['company_name']}\n"
            f"대표명: {c['company_chairman']}\n"
            f"주소: {c['company_address']}\n"
            f"설립일: {c['establish_date']}\n"
            f"출근시간: {c['work_start_time']}"
        )

    return f"{header}\n정보를 찾을 수 없습니다."

# ============================ 요약 메인 함수 ============================

def fetch_employee_summary(employee_id: int, categories: List[str]) -> str:
    """
    주어진 사원 ID와 카테고리 목록에 따라 정보를 조회하고
    자연어로 요약된 문자열을 반환합니다.
    """
    summaries = []
    today_str = date.today().isoformat()
    summaries.append(f"※ 답변 시 참고해야 할 오늘의 날짜는 {today_str} 입니다.")
    
    with engine.connect() as conn:
        for category in categories:
            sql = CATEGORY_QUERIES.get(category)
            if not sql:
                continue
            df = pd.read_sql(text(sql), conn, params={"employee_id": employee_id})
            data = df.to_dict(orient="records")
            summary = format_category_summary(category, data)
            summaries.append(summary)
    return "\n\n".join(summaries)
