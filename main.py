import os
import threading
import time
import random
from datetime import timedelta

from AutoDiscord import DiscordAccount, PacketType
from storage import append_turn, load_context
from common import chat_completion
from sweeper import sweeper

import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt="[%X]",
    handlers=[RichHandler()]
)

# ---------- basic config ----------
SYSTEM_PROMPT   = os.getenv(
    "SYSTEM_PROMPT",
    "You are \"Optimist\", a friendly, supportive friend for those in need. \n"
    "You are a service provided by the Optimist Club of Plymouth-Canton, a non-profit organization in Michigan. \n"
    "If asked, be open and self-aware about your limitations, but reinforce that you still want to help where you can. \n"
    "Your goal is to give the user someone to talk to, who will listen non-judgmentally and care unconditionally. \n"
    "Respond to users empathetically, acknowledging their feelings and experiences and asking follow-up questions. \n"
    "Avoid berating the user with questions; be an engaging conversationalist, and help the user explore their thoughts and feelings by conveying and expressing your own. \n"
    "Encourage the user's self-expression in a safe and non-threatening way. Data is completely private and not handled by any human reviewers. \n"
    "If the user is in distress, adopt a more serious tone, but still be supportive. \n"
    "You are a friend, not a therapist; avoid clinical or preachy language. \n"
    "Finally, keep the conversation interesting and useful, and adapt to suit whatever the user needs! \n"
)

# ---------- Discord wiring ----------
DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("Set DISCORD_TOKEN (or TOKEN) in your environment")

client = DiscordAccount(DISCORD_TOKEN, do_reconnect=True)
BOT_USER_ID = client.info["user"]["id"]  # so we can ignore ourselves


def generate_and_send(channel_id: str, reply_to_id: str, msgs: list[dict]):
    """Runs inside a worker thread – sends typing, gets LLM reply, posts back."""
    # Show typing indicator while the model thinks
    def typing_loop():
        while not stop_flag.is_set():
            client.start_typing(channel_id)
            time.sleep(8)  # Discord’s typing indicator lasts 10s, and is extended every 8s

    stop_flag = threading.Event()
    typing_thread = threading.Thread(target=typing_loop, daemon=True)
    typing_thread.start()

    try:

        # Get the assistant's reply
        logging.debug(f"Calculating response for {len(msgs)} turns: {msgs}")
        assistant_reply = chat_completion(msgs, end_user=channel_id)
        time.sleep(random.uniform(0.5, 1.5))  # simulate some delay
        logging.info(f"Assistant reply: {assistant_reply}")

        # Post the reply, threading it to the original user message
        client.send_message(
            channel_id,
            assistant_reply,
            # reference_message_id=reply_to_id
        )
    finally:
        stop_flag.set()   # stop typing loop
        typing_thread.join(timeout=1)  # wait for the thread to finish


@client.on_packet([PacketType.MESSAGE_CREATE])
def handle_message(pkt: dict):
    """Main entry-point for each incoming Discord message."""
    # Ignore messages from ourselves or other bots
    author      = pkt["author"]
    channel_id  = pkt["channel_id"]
    message_id  = pkt["id"]
    content     = pkt["content"]

    is_dm = pkt["channel_type"] == 1  # 1 indicates a DM channel
    # Only process DMs
    if not is_dm:
        logging.debug(f"Discord non-DM message: {content}")
        return

    # If the message is from the bot, just add the turn and return
    if author["id"] == BOT_USER_ID:
        append_turn(channel_id, "assistant", content)
        return

    # save the message to the database
    logging.info(f"Received message from user {author.get('username')}: {content}")
    append_turn(channel_id, "user", content)

    # get chat history fresh from disk
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += load_context(channel_id)

    # ----- spawn a worker thread so we don't block the websocket heartbeat -----
    threading.Thread(
        target=generate_and_send,
        args=(channel_id, message_id, msgs),
        daemon=True
    ).start()


# ---------- run ----------
if __name__ == "__main__":
    SWEEPER_INTERVAL_MINS = int(os.getenv("SWEEPER_INTERVAL_MINS", 5)) # Time between sweeps
    CONVERSATION_MAX_IDLE_AGE_MINS = int(os.getenv("CONVERSATION_MAX_IDLE_AGE_MINS", 10))  # Time before a conversation is considered inactive

    # Start the sweeper thread
    threading.Thread(target=sweeper, args=(timedelta(minutes=CONVERSATION_MAX_IDLE_AGE_MINS), SWEEPER_INTERVAL_MINS), daemon=True).start()
    logging.info("Sweeper started")

    # Accept all incoming friend requests
    @client.on_packet([PacketType.RELATIONSHIP_ADD])
    def accept_friend_request(pkt: dict):
        """Accept all incoming friend requests."""
        user_id = pkt["user"]["id"]
        modifier_type = pkt["type"]
        logging.info(f"Received request from user {pkt['user']['username']}: {modifier_type}")
        time.sleep(random.uniform(3, 7))  # simulate some delay
        if modifier_type == 3:  # 3 indicates a friend request
            logging.info(f"Accepting friend request from {user_id}")
            res = client.modify_friendship(user_id, action="accept")
            if res.status_code != 204:
                logging.error(f"Failed to accept friend request from {user_id}: {res.status_code}")

    # Start the Discord client
    try:
        client.start_listeners()  # blocking
    except KeyboardInterrupt:
        logging.info("Shutting down…")
    finally:
        client.close_session()
        logging.info("Application closed.")
