from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, Field, create_engine, Session, select, JSON
from sqlalchemy import Column, JSON, Text

DB_URL = "sqlite:///chat_state.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

# ---------- ORM models ----------
class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    role: str                           # "user" | "assistant"
    content: str
    ts: datetime = Field(default_factory=datetime.utcnow)

class UserSettings(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    settings: dict = Field(
        default_factory=dict,
        sa_column=Column(
            JSON().with_variant(Text, "sqlite")  # Text on SQLite, JSON elsewhere
        )
    )


SQLModel.metadata.create_all(engine)

# ---------- thin helpers ----------
def save_turn(user_id: str, role: str, content: str):
    with Session(engine) as s:
        s.add(Message(user_id=user_id, role=role, content=content))
        s.commit()

def get_recent_history(user_id: str, limit: int = 20) -> List[dict]:
    with Session(engine) as s:
        stmt = (
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.ts.desc())
            .limit(limit)
        )
        rows = reversed(s.exec(stmt).all())       # chronological order
    return [{"role": m.role, "content": m.content} for m in rows]

def get_settings(user_id: str) -> dict:
    with Session(engine) as s:
        row = s.get(UserSettings, user_id)
        return row.json if row else {}

def upsert_settings(user_id: str, new_json: dict):
    with Session(engine) as s:
        row = s.get(UserSettings, user_id)
        if row:
            row.json.update(new_json)
        else:
            row = UserSettings(user_id=user_id, json=new_json)
            s.add(row)
        s.commit()
