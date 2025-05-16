from datetime import datetime, timedelta
from typing import List, Optional
from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy import Column, JSON, Text, delete
import json
import numpy as np

from common import chat_completion, get_embedding

DB_URL = "sqlite:///chat_state.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

# ---------- ORM models ----------
class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: str
    started: datetime = Field(default_factory=datetime.utcnow)
    last_ts: datetime = Field(default_factory=datetime.utcnow)
    ended: Optional[datetime] = None
    summary: Optional[str] = None
    summary_embedding: Optional[str] = Field(default=None, sa_column=Column(Text))  # JSON string of List[float]
    last_accessed_ts: Optional[datetime] = Field(default=None) # When this summary was last used for context
    access_count: int = Field(default=0)

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conv_id: int = Field(foreign_key="conversation.id")
    role: str  # "user" | "assistant"
    content: str
    ts: datetime = Field(default_factory=datetime.utcnow)

class UserSettings(SQLModel, table=True):
    channel_id: str = Field(primary_key=True)
    settings: dict = Field(
        default_factory=dict,
        sa_column=Column(
            JSON().with_variant(Text, "sqlite")  # Text on SQLite, JSON elsewhere
        )
    )


# create tables if they don't exist
SQLModel.metadata.create_all(engine)


def _get_active_conv(channel_id: str, s: Session) -> Conversation | None:
    stmt = select(Conversation).where(
        Conversation.channel_id == channel_id,
        Conversation.ended.is_(None)
    )
    return s.exec(stmt).first()

def start_new_conversation(channel_id: str, s: Session) -> Conversation:
    now = datetime.utcnow()
    conv = Conversation(channel_id=channel_id, started=now, last_ts=now)
    s.add(conv)
    s.commit()
    s.refresh(conv)
    return conv

def append_turn(channel_id: str, role: str, content: str):
    now = datetime.utcnow()
    with Session(engine) as s:
        conv = _get_active_conv(channel_id, s)

        if not conv:
            conv = start_new_conversation(channel_id, s)

        s.add(Message(conv_id=conv.id, role=role, content=content, ts=now))

        # update last_ts on user activity
        if role == "user":
            conv.last_ts = now

        s.commit()

def summarise_and_close(conv: Conversation) -> Optional[str]:
    """Ask the LLM for a compressed abstract, store it, wipe raw turns."""
    with Session(engine) as s:
        s.add(conv)  # make sure the conversation is in the session

        # # get all messages in the conversation
        # raw_msgs = s.exec(select(Message).where(Message.conv_id == conv.id).order_by(Message.ts)).all()
        #
        # # if no messages, just close
        # if not raw_msgs:
        #     conv.ended = datetime.utcnow()
        #     s.commit()
        #     return

        # get all messages in the conversation
        raw_msgs = load_context(conv.channel_id)

        # -------- LLM summary --------
        plain_dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in raw_msgs)
        summary_text = chat_completion(
            messages=[
                {"role": "system", "content":
                 "You are a memory encoding algorithm, designed to summarize conversations to below a third its original length. "
                 "You will be provided a conversation between a user and an assistant, and you will provide a chronological narrative summary from the point-of-view of the assistant. "
                 "Some past memories of the assistant relevant to the current conversation are provided for context; avoid repeating information already encoded elsewhere in memory. "
                 "Highlight any significant decisions, preferences, facts, or quotes that could be relevant in later discussions. "},
                {"role": "user", "content": plain_dialogue}
            ],
            end_user="summariser"  # tag the request with a special user-id
        )

        # store summary and embedding
        conv.summary = summary_text
        embedding = get_embedding(summary_text)
        conv.summary_embedding = json.dumps(embedding)

        # initialize access statistics
        conv.ended = conv.last_ts
        conv.access_count = 0
        conv.last_accessed_ts = datetime.utcnow()

        # clean up old messages
        s.exec(delete(Message).where(Message.conv_id == conv.id))
        s.commit()

    return summary_text


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if not isinstance(vec1, np.ndarray): vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray): vec2 = np.array(vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # should not happen, but just in case

    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)


def load_context(channel_id: str, max_summaries: int = 10, max_active_turns: int = 100) -> list[dict]:
    """Returns [memory-summaries] + [live messages]"""
    with Session(engine) as s:

        # get all past conversations
        all_past_conversations_stmt = (
            select(Conversation)
            .where(
                Conversation.channel_id == channel_id,
                Conversation.summary.is_not(None),
                Conversation.summary_embedding.is_not(None)
            )
            .order_by(Conversation.ended.desc())
        )
        past_conversations = s.exec(all_past_conversations_stmt).all()

        # get active messages
        conv = _get_active_conv(channel_id, s)
        live = []

        summaries_to_use = []
        if conv:  # if there's an active conversation, use embeddings to find relevant summaries

            # get active messages
            stmt = (
                select(Message)
                .where(Message.conv_id == conv.id)
                .order_by(Message.ts.desc())
                .limit(max_active_turns)
            )
            live = [{
                "role": m.role,
                "content": m.content
            } for m in reversed(s.exec(stmt).all())]

            # get embedding for the active messages
            plain_dialogue = '\n'.join(f"{m['role']}: {m['content']}" for m in live)
            query_embedding = np.array(get_embedding(plain_dialogue))

            # calculate similarity with past summaries
            candidate_summaries = []
            for conv in past_conversations:
                summary_emb_vector = np.array(json.loads(conv.summary_embedding))
                similarity = _cosine_similarity(query_embedding, summary_emb_vector)
                candidate_summaries.append({"conv": conv, "similarity": similarity})

            # Sort by similarity (highest first)
            candidate_summaries.sort(key=lambda x: x["similarity"], reverse=True)

            # Select top N relevant summaries
            selected_candidates = candidate_summaries[:max_summaries]

            # update access statistics
            now = datetime.utcnow()
            for item in selected_candidates:
                conv_to_update = item["conv"]
                # s.add(conv_to_update) # Re-attach if necessary, or get by ID
                db_conv = s.get(Conversation, conv_to_update.id) # Get fresh instance for update
                if db_conv:
                    db_conv.access_count += 1
                    db_conv.last_accessed_ts = now
                    s.add(db_conv) # Add to session for commit
                summaries_to_use.append(conv_to_update) # Use original conv object for summary text

            # Sort selected relevant summaries by their original end date (chronological for context)
            summaries_to_use.sort(key=lambda c: c.ended)

            if selected_candidates: # Only commit if there were updates
                s.commit()

        else:   # if there's no active conversation, don't worry about embeddings
            summaries_to_use = past_conversations[::-1][:max_summaries]   # get `max_summaries` most recent summaries

        mem_blocks = [{
            "role": "developer",
            "content": f"CONVERSATION MEMORY [{c.ended.strftime('%Y-%m-%d %H:%M')}]: {c.summary}"
        } for c in summaries_to_use]

    return mem_blocks + live

def find_inactive_conversations(max_age: timedelta = timedelta(minutes=60)) -> List[Conversation]:
    """Find conversations that are older than max_age and have not been closed."""
    with Session(engine) as s:
        cutoff_time = datetime.utcnow() - max_age
        stmt = select(Conversation).where(
            Conversation.ended.is_(None),
            Conversation.last_ts < cutoff_time
        )
        return s.exec(stmt).all()

def get_settings(channel_id: str) -> dict:
    with Session(engine) as s:
        row = s.get(UserSettings, channel_id)
        return row.settings if row else {}

def upsert_settings(channel_id: str, new_json: dict):
    with Session(engine) as s:
        row = s.get(UserSettings, channel_id)
        if row:
            # Ensure 'settings' is a dict before updating if it could be None initially
            if row.settings is None:
                row.settings = {}
            row.settings.update(new_json)
            s.add(row) # Mark as changed
        else:
            row = UserSettings(channel_id=channel_id, settings=new_json)
            s.add(row)
        s.commit()
