from fastapi import FastAPI, APIRouter, HTTPException
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool
import pandas as pd

from db.db import get_engine
from pydantic import BaseModel
from typing import Any, Dict, List


# ===== DB 설정 =====
engine = get_engine()

# ===== FastAPI & Router 생성 =====
app = FastAPI()

# ===== 카테고리 정의 =====
CATEGORIES = {
    "employee":      "사원 정보",         # 사번, 이메일, 부서(부서명·부서연락처·부서장), 직위명, 잔여 연차(일 환산), 잔여 리프레시 휴가
    "department":    "부서 구성원",       # 같은(및 하위) 부서 소속 사원 목록(이름·직위·이메일·연락처)
    "commute":       "출퇴근 통계",       # 현재 달 총근무시간·연장·야간·휴일근무시간 합계
    "vacation":      "휴가 일정",         # 이번 달 같은 부서(하위 부서 포함) 휴가 + 나의 예정 휴가
    "business_trip": "출장 일정",         # 다음 1개월 내 나의 승인된 출장(유형·장소·기간·사유·예상경비)
    "kpi":           "KPI 현황",          # 승인된 KPI 목표(status_id=2) → 목표치·진척도·마감기한
    "holiday":       "다가오는 휴일",     # 향후 3개월 내 휴일(명칭·날짜)
    "company":       "회사 정보",         # 회사명·대표명·주소·설립일·출근시간
}

# ===== SQL 쿼리 정의 =====
CATEGORY_QUERIES = {
    # 1) 사원 정보
    "employee": """
        SELECT
          e.emp_no,
          e.email,
          d.name           AS department_name,
          d.contact        AS department_contact,
          he.name          AS department_head,
          p.name           AS position_name,
          ROUND(e.remaining_dayoff_hours / 8, 2) AS remaining_dayoff_days,
          e.remaining_refresh_days
        FROM employee e
        LEFT JOIN department d   ON e.dept_id      = d.dept_id
        LEFT JOIN dept_head dh    ON d.dept_id      = dh.dept_id
        LEFT JOIN employee he     ON dh.emp_id      = he.emp_id
        LEFT JOIN position p     ON e.position_id  = p.position_id
        WHERE e.emp_id = :employee_id
    """,

    # 2) 부서 구성원 (자신 부서 + 하위 부서)
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

    # 3) 출퇴근 통계 (현재 달)
    "commute": """
        SELECT
            SUM(TIMESTAMPDIFF(MINUTE, start_at, end_at) - break_time)                       AS total_minutes,
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

    # 4) 휴가 일정
    "vacation": """
        -- 같은 부서(및 하위) 동료 휴가
        SELECT
          vt.description   AS vacation_name,
          v.start_date,
          v.end_date,
          v.reason
        FROM vacation v
        JOIN approve a         ON v.approve_id = a.approve_id
        JOIN vacation_type vt  ON v.vacation_type_id = vt.vacation_type_id
        WHERE a.emp_id IN (
          -- 부서 재귀
          WITH RECURSIVE sub AS (
            SELECT dept_id FROM employee WHERE emp_id = :employee_id
            UNION ALL
            SELECT d.dept_id FROM department d JOIN sub s ON d.parent_dept_id = s.dept_id
          )
          SELECT e.emp_id FROM employee e JOIN sub ON e.dept_id = sub.dept_id
        )
          AND YEAR(v.start_date) = YEAR(CURDATE())
          AND MONTH(v.start_date) = MONTH(CURDATE())

        UNION ALL

        -- 나의 예정 휴가
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

    # 5) 출장 일정 (다음 1개월, status_id=2)
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

    # 6) KPI 현황 (status_id=2)
    "kpi": """
        SELECT
          goal_value    AS kpi_goal,
          kpi_progress  AS progress_rate,
          deadline
        FROM kpi
        WHERE emp_id    = :employee_id
          AND status_id = 2
    """,

    # 7) 다가오는 휴일 (향후 3개월)
    "holiday": """
        SELECT
          holiday_name,
          date
        FROM holiday
        WHERE date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 3 MONTH)
        ORDER BY date
    """,

    # 8) 회사 정보
    "company": """
        SELECT
          name             AS company_name,
          chairman         AS company_chairman,
          address          AS company_address,
          establish_date,
          work_start_time
        FROM company
        LIMIT 1
    """,
}

# ====== 유틸 함수 ======
def minutes_to_hour_minute_str(minutes: int) -> str:
    """
    분 단위를 'X시간 Y분' 형식의 문자열로 변환합니다.
    예: 90 -> '1시간 30분', 120 -> '2시간'
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}시간 {mins}분" if mins else f"{hours}시간"

# ===== 자연어 요약 함수 =====
def format_category_summary(category: str, data: List[Dict[str, Any]]) -> str:
    if not data:
        return f"{CATEGORIES[category]} 관련 정보가 없습니다."

    # 1) 사원 정보
    if category == "employee":
        e = data[0]
        s = [
            f"사번: {e['emp_no']}",
            f"이메일: {e['email']}",
            f"소속 부서: {e['department_name']} ({e['department_contact']})",
        ]
        if e.get("department_head"):
            s.append(f"부서장: {e['department_head']}")
        s.append(f"직위: {e['position_name']}")
        s.append(f"잔여 연차: {e['remaining_dayoff_days']}일")
        if e.get("remaining_refresh_days", 0) > 0:
            s.append(f"잔여 리프레시 휴가: {e['remaining_refresh_days']}일")
        return "사원 정보:\n" + "\n".join(s)

    # 2) 부서 구성원
    elif category == "department":
        lines = [
            f"{row['member_name']} ({row['position_name']}, {row['email']}, {row['phone_number']})"
            for row in data
        ]
        return "부서 구성원 목록:\n" + "\n".join(lines)

    # 3) 출퇴근 통계
    elif category == "commute":
        c = data[0]
        return (
            f"이번 달 출퇴근 통계:\n"
            f"- 총 근무시간: {minutes_to_hour_minute_str(c['total_minutes'])}\n"
            f"- 연장 근로시간: {minutes_to_hour_minute_str(c['overtime_minutes'])}\n"
            f"- 야간 근로시간: {minutes_to_hour_minute_str(c['night_minutes'])}\n"
            f"- 휴일 근로시간: {minutes_to_hour_minute_str(c['holiday_minutes'])}"
        )

    # 4) 휴가 일정
    elif category == "vacation":
        lines = [
            f"{row['start_date']}~{row['end_date']}: {row['vacation_name']} ({row.get('reason','무사유')})"
            for row in data
        ]
        return "이번 달 휴가 일정:\n" + "\n".join(lines)

    # 5) 출장 일정
    elif category == "business_trip":
        lines = [
            f"{row['start_date']}~{row['end_date']} | {row['trip_type']} | {row['trip_place']} | 사유: {row['reason']} | 예상경비: {row['estimated_cost']}원"
            for row in data
        ]
        return "다가오는 출장 일정:\n" + "\n".join(lines)

    # 6) KPI 현황
    elif category == "kpi":
        lines = [
            f"목표: {row['kpi_goal']} | 진척도: {row['progress_rate']}% | 마감일: {row['deadline']}"
            for row in data
        ]
        return "KPI 현황:\n" + "\n".join(lines)

    # 7) 다가오는 휴일
    elif category == "holiday":
        lines = [
            f"{row['date']}: {row['holiday_name']}"
            for row in data
        ]
        return "다가오는 휴일 목록:\n" + "\n".join(lines)

    # 8) 회사 정보
    elif category == "company":
        c = data[0]
        return (
            f"회사명: {c['company_name']}\n"
            f"대표명: {c['company_chairman']}\n"
            f"주소: {c['company_address']}\n"
            f"설립일: {c['establish_date']}\n"
            f"출근시간: {c['work_start_time']}"
        )

    # fallback
    return f"{CATEGORIES.get(category, category)} 정보를 찾을 수 없습니다."

# ===== 요청 모델 =====
class CategoryQueryRequest(BaseModel):
    employee_id: int
    categories: List[str]  # 예: ["vacation", "commute"]

# ===== 카테고리 데이터 요약 API =====
@app.post("/employee-context")
async def get_employee_context(request: CategoryQueryRequest):
    result: Dict[str, str] = {}
    with engine.connect() as conn:
        for category in request.categories:
            if category not in CATEGORY_QUERIES:
                continue

            # 블로킹 I/O를 스레드 풀로 분리
            df = await run_in_threadpool(
                lambda: pd.read_sql(
                    text(CATEGORY_QUERIES[category]),
                    conn,
                    params={"employee_id": request.employee_id},
                )
            )

            data = df.to_dict(orient="records")
            result[category] = format_category_summary(category, data)

    return result