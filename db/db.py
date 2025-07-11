from sqlalchemy import create_engine
from config import DB_HOST, DB_PORT, DB_NAME, DB_USERNAME, DB_PASSWORD

def get_engine():
    url = (
        f"mariadb+mariadbconnector://{DB_USERNAME}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(url)