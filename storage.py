import os
from datetime import datetime, timedelta
from typing import List, Optional, cast
from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy import Column, Text, delete
import json
import numpy as np

from common import chat_completion, get_embedding

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

DB_URL = "sqlite:///chat_state.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
MAX_SUMMARIES_FOR_SUMMARIZATION = int(os.getenv("MAX_SUMMARIES_FOR_SUMMARIZATION", 5))  # Number of memory blocks to use for summarization

# ---------- ORM models ----------
class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    channel_id: str
    started: datetime = Field(default_factory=datetime.utcnow)
    last_user_ts: datetime = Field(default_factory=datetime.utcnow)
    started_by_user: bool = Field(default=True)  # True if the conversation was started by the user
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
    allow_proactive_checkin: bool = Field(default=False)
    asked_about_checkins: bool = Field(default=False)
    active_hours_utc_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    days_idle: int = Field(default=1)

    @property
    def active_hours_utc(self) -> List[int]:
        if self.active_hours_utc_json:
            try:
                hours = json.loads(self.active_hours_utc_json)
                return cast(List[int], hours)  # Ensure type checker knows it's List[int]
            except json.JSONDecodeError:
                return []
        return []

    @active_hours_utc.setter
    def active_hours_utc(self, hours: List[int]):
        # Ensure unique hours and sort them
        self.active_hours_utc_json = json.dumps(sorted(list(set(hours))))


# create tables if they don't exist
SQLModel.metadata.create_all(engine)


def _get_active_conv(channel_id: str, s: Session) -> Optional[Conversation]:
    stmt = select(Conversation).where(
        Conversation.channel_id == channel_id,
        Conversation.summary.is_(None)
    )
    return s.exec(stmt).first()

def get_or_create_user_settings(s: Session, channel_id: str) -> UserSettings:
    settings = s.get(UserSettings, channel_id)
    if not settings:
        settings = UserSettings(channel_id=channel_id)
        s.add(settings)
        # No commit here, caller should handle commit.
    return settings

def update_user_setting_fields(channel_id: str, **kwargs):
    with Session(engine) as s:
        user_settings = get_or_create_user_settings(s, channel_id)
        for key, value in kwargs.items():
            if hasattr(user_settings, key):
                setattr(user_settings, key, value)
            else:
                log.warning(f"Attempted to set unknown attribute {key} on UserSettings for {channel_id}")
        s.add(user_settings) # Mark as changed
        s.commit()

def start_new_conversation(channel_id: str, s: Session, started_by_user: bool) -> Conversation:
    now = datetime.utcnow()
    conv = Conversation(channel_id=channel_id, started=now, last_user_ts=now, started_by_user=started_by_user)
    s.add(conv)
    s.commit()
    s.refresh(conv)
    return conv

def append_turn(channel_id: str, role: str, content: str):
    now = datetime.utcnow()
    with Session(engine) as s:
        conv = _get_active_conv(channel_id, s)

        if not conv:
            conv = start_new_conversation(channel_id, s, started_by_user=(role == "user"))

        s.add(Message(conv_id=conv.id, role=role, content=content, ts=now))

        # update last_user_ts on user activity
        if role == "user":
            conv.last_user_ts = now
            s.add(conv)

            # update active hours for the user
            settings = get_or_create_user_settings(s, channel_id)
            current_active_hours = settings.active_hours_utc
            current_hour = now.hour
            if current_hour not in current_active_hours:
                current_active_hours.append(current_hour)
                settings.active_hours_utc = current_active_hours
            s.add(settings)

        s.commit()

def summarise_and_close(conv: Conversation) -> Optional[str]:
    """Ask the LLM for a compressed abstract, store it, wipe raw turns."""
    with Session(engine) as s:
        s.add(conv)  # make sure the conversation is in the session

        # get all messages in the conversation
        raw_msgs = load_context(conv.channel_id, max_summaries=MAX_SUMMARIES_FOR_SUMMARIZATION)

        # split memory from live messages
        mem_blocks = []
        live_messages = []
        for turn in raw_msgs:
            if turn["role"] == "developer":
                mem_blocks.append(turn)
            else:
                live_messages.append(turn)

        # -------- LLM summary --------
        plain_dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in live_messages)
        summary_text = chat_completion(
            messages=[
                {"role": "system", "content":
                 "You are a memory encoding algorithm, designed to summarize conversations to below a third its original length. "
                 "You will be provided a conversation between a user and an assistant, and you will provide a chronological narrative summary from the point-of-view of the assistant (mirroring the assistant's tone). "
                 "Some past memories of the assistant relevant to the current conversation are provided for context; avoid repeating information already encoded elsewhere in memory. "
                 "Highlight any significant decisions, preferences, facts, or quotes that could be relevant in later discussions. "},
                *mem_blocks,
                {"role": "user", "content": plain_dialogue}
            ]
        ).content

        # store summary and embedding
        conv.summary = summary_text
        embedding = get_embedding(summary_text)
        conv.summary_embedding = json.dumps(embedding)

        # initialize access statistics
        conv.access_count = 0
        conv.last_accessed_ts = datetime.utcnow()
        s.add(conv)

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


def load_context(channel_id: str, *, max_active_turns: Optional[int] = None, max_summaries: int = 10) -> list[dict]:
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
            .order_by(Conversation.last_user_ts.desc())
        )
        past_conversations = s.exec(all_past_conversations_stmt).all()

        # get the active conversation for a channel
        conv = _get_active_conv(channel_id, s)
        live_messages = []

        summaries_to_use = []
        if conv:  # if there's an active conversation, use embeddings to find relevant summaries

            # there are only live messages when there's an active conversation
            stmt = (
                select(Message)
                .where(Message.conv_id == conv.id)
                .order_by(Message.ts.desc())
            )
            if max_active_turns is not None:
                stmt = stmt.limit(max_active_turns)
            live_messages = [{
                "role": m.role,
                "content": m.content
            } for m in reversed(s.exec(stmt).all())]

            # get embedding for the active messages
            plain_dialogue = '\n'.join(f"{m['role']}: {m['content']}" for m in live_messages)
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
            summaries_to_use.sort(key=lambda c: c.last_user_ts)

            if selected_candidates: # Only commit if there were updates
                s.commit()

        else:   # if there's no active conversation, don't worry about embeddings
            summaries_to_use = past_conversations[::-1][:max_summaries]   # get `max_summaries` most recent summaries

        mem_blocks = [{
            "role": "developer",
            "content": f"CONVERSATION MEMORY [{c.last_user_ts.strftime('%Y-%m-%d %H:%M')}]: {c.summary}"
        } for c in summaries_to_use]

    return mem_blocks + live_messages

def find_inactive_conversations(max_age: timedelta = timedelta(minutes=60)) -> List[Conversation]:
    """Find conversations that are older than max_age and have not been closed."""
    with Session(engine) as s:
        cutoff_time = datetime.utcnow() - max_age
        stmt = select(Conversation).where(
            Conversation.summary.is_(None),
            Conversation.last_user_ts < cutoff_time
        )
        return s.exec(stmt).all()


def get_channels_for_proactive_checkin(s: Session) -> List[tuple[datetime, UserSettings]]:
    """
    Finds users who opted in for check-ins and whose end of last conv is beyond the threshold.
    """
    candidate_settings: List[tuple[datetime, UserSettings]] = []

    # Get all users who allow proactive check-ins or who haven't been asked yet
    eligible_users_stmt = (
        select(UserSettings)
        .where(
            UserSettings.allow_proactive_checkin.is_(True) |
            UserSettings.asked_about_checkins.is_(False)
        )
    )
    eligible_user_settings_list = s.exec(eligible_users_stmt).all()

    for user_setting in eligible_user_settings_list:
        channel_id = user_setting.channel_id
        last_user_interaction_time: Optional[datetime] = None

        # Use the user's specific setting for days_idle
        user_specific_idle_delta = timedelta(days=user_setting.days_idle)
        # user_specific_idle_delta = timedelta(minutes=30)  # For testing purposes, set to 30 minutes
        cutoff_time = datetime.utcnow() - user_specific_idle_delta

        # Check the most recent message of the most recent conversation.
        last_summarized_conv_stmt = (
            select(Conversation.last_user_ts)
            .where(Conversation.channel_id == channel_id)
            .order_by(Conversation.last_user_ts.desc())
            .limit(1)
        )
        last_summarized_conv_latest_time = s.exec(last_summarized_conv_stmt).one_or_none()
        if last_summarized_conv_latest_time:
            last_user_interaction_time = last_summarized_conv_latest_time
            # If no conv, they are not eligible

        if last_user_interaction_time and last_user_interaction_time < cutoff_time:
            candidate_settings.append((last_user_interaction_time, user_setting))

    return candidate_settings
