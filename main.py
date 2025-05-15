import os
import threading
import time
from collections import defaultdict, deque
from llm_tool import tool
import openai

import logging
logging.basicConfig(
    level=logging.INFO,
)

from AutoDiscord import DiscordAccount, PacketType
from storage import save_turn, get_recent_history, get_settings


# ---------- basic config ----------
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
MODEL_NAME      = os.getenv("MODEL", "gpt-4.1-mini")
MAX_HISTORY_TURNS = int(os.getenv("HISTORY_MSGS", 50))  # per speaker
SYSTEM_PROMPT   = os.getenv(
    "SYSTEM_PROMPT",
    "You are a friendly, concise Discord assistant."
)

if not DISCORD_TOKEN:
    raise ValueError("Set DISCORD_TOKEN (or TOKEN) in your environment")
if not LITELLM_API_KEY:
    raise ValueError("Set LITELLM_API_KEY in your environment")


llm = openai.OpenAI(
    base_url=os.getenv("LITELLM_PROXY_URL", "http://localhost:4000"),
    api_key=LITELLM_API_KEY,
)

def chat_completion(messages: list[dict], *, end_user: str):
    """
    Send the request through the proxy, tagging it with Discord user-id
    so spend is broken down per account in the dashboard.
    """
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        user=end_user
    )

    return resp.choices[0].message.content.strip()


# ---------- Discord wiring ----------
client = DiscordAccount(DISCORD_TOKEN, do_reconnect=True)
BOT_USER_ID = client.info["user"]["id"]  # so we can ignore ourselves

def generate_and_send(user_id: str, channel_id: str, reply_to_id: str, msgs: list[dict]):
    """Runs inside a worker thread – sends typing, gets LLM reply, posts back."""
    # Show typing indicator while the model thinks
    def typing_loop():
        while not stop_flag.is_set():
            client.start_typing(channel_id)
            time.sleep(8)  # Discord’s typing indicator lasts 10s, and is extended every 8s

    stop_flag = threading.Event()
    threading.Thread(target=typing_loop, daemon=True).start()

    try:

        # Get the assistant's reply
        assistant_reply = chat_completion(msgs, end_user=user_id)

        # Update memory with the assistant's reply
        save_turn(user_id, "assistant", assistant_reply)

        # Post the reply, threading it to the original user message
        client.send_message(
            channel_id,
            assistant_reply,
            # reference_message_id=reply_to_id
        )
    finally:
        stop_flag.set()   # stop typing loop

@client.on_packet([PacketType.MESSAGE_CREATE])
def handle_message(pkt: dict):
    """Main entry-point for each incoming Discord message."""
    # Ignore messages from ourselves or other bots
    author_id   = pkt["author"]["id"]
    channel_id  = pkt["channel_id"]
    message_id  = pkt["id"]
    content     = pkt["content"]

    # get chat history fresh from disk
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += get_recent_history(author_id, MAX_HISTORY_TURNS)

    if author_id == BOT_USER_ID or pkt["author"].get("bot"):
        logging.debug(f"Ignoring message from self or bot: {content}")
        return

    # save the message to the database
    save_turn(author_id, "user", content)

    # ----- spawn a worker thread so we don't block the websocket heartbeat -----
    threading.Thread(
        target=generate_and_send,
        args=(author_id, channel_id, message_id, msgs),
        daemon=True
    ).start()


# ---------- run ----------
if __name__ == "__main__":
    try:
        client.start_listeners()  # blocking
    except KeyboardInterrupt:
        print("Shutting down…")
        client.close_session()
