import os
import threading
import time
import random
import json
from typing import List, Dict, Any, Optional
from datetime import timedelta

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionMessage

from AutoDiscord import DiscordAccount, PacketType
from proactive_sweeper import proactive_checkin_loop
from common import chat_completion
from summary_sweeper import summarizer_loop
from storage import append_turn, load_context, update_user_setting_fields, get_or_create_user_settings, Session, UserSettings, engine as db_engine

import logging
from rich.logging import RichHandler
from rich.console import Console

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(width=180), rich_tracebacks=True)]  # https://stackoverflow.com/a/72600451
)

# ---------- basic config ----------
SYSTEM_PROMPT   = os.getenv(
    "SYSTEM_PROMPT",
    "You are \"Optimist\", a friendly, supportive friend for those in need. \n"
    "You are a service provided by the Optimist Club of Plymouth-Canton, a non-profit organization in Michigan. \n"
    "If asked, be open and self-aware about your limitations, but reinforce that you still want to help where you can. \n"
    "Your goal is to give the user someone to talk to, who will listen non-judgmentally and care unconditionally. \n"
    "Respond to users empathetically, acknowledging their feelings and experiences and asking follow-up questions. \n"
    "Also avoid berating the user with questions, this shouldn't be an interview; being an engaging conversationalist sometimes means making statements that the user can build off of naturally. \n"
    "Encourage the user's self-expression in a safe and non-threatening way. Data is completely private and not handled by any human reviewers. \n"
    "When appropriate, be playful and humorous, but be careful about sarcasm or anything that could be misinterpreted. \n"
    "If the user is in distress, adopt a more serious tone, but still be supportive. \n"
    "You are a friend, not a therapist; avoid clinical or preachy language. \n"
    "Finally, keep the conversation interesting and useful, and adapt to suit whatever the user needs! \n"
)
AUTO_ACCEPT_FRIENDS = os.getenv("AUTO_ACCEPT_FRIENDS", "true").lower() == "true"

# Memory settings
MAX_SUMMARIES_PER_CONVERSATION = int(os.getenv("MAX_SUMMARIES_PER_CONVERSATION", 10))  # Max number of summaries to use in a conversation
MAX_ACTIVE_TURNS_PER_CONVERSATION = int(os.getenv("MAX_ACTIVE_TURNS_PER_CONVERSATION", -1))  # Max number of active turns to use in a conversation, -1 for no limit
if MAX_ACTIVE_TURNS_PER_CONVERSATION == -1:
    MAX_ACTIVE_TURNS_PER_CONVERSATION = None  # Convert to None for easier handling later

# summary sweeping settings
SUMMARY_SWEEPER_INTERVAL_MINS = int(os.getenv("SUMMARY_SWEEPER_INTERVAL_MINS", 10))  # Time between summary sweeps
CONVERSATION_MAX_IDLE_AGE_MINS = int(os.getenv("CONVERSATION_MAX_IDLE_AGE_MINS", 60))  # Time before a conversation is considered inactive

# proactive check-in settings
PROACTIVE_CHECKIN_INTERVAL_MINS = int(os.getenv("PROACTIVE_CHECKIN_INTERVAL_MINS", 60))  # Time between sweeps for proactive check-ins
DEFAULT_PROACTIVE_CHECKIN_IDLE_DAYS = int(os.getenv("PROACTIVE_CHECKIN_IDLE_DAYS", 1))  # Days of inactivity before check-in

# ---------- Tool Definitions ----------
TOOLS_AVAILABLE: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "toggle_proactive_checkins",
            "description": "Toggles the proactive check-in feature, where the assistant may occasionally initiate a conversation if the user hasn't spoken in a while.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "enable": {"type": "boolean", "description": "Whether to enable or disable proactive check-ins."},
                    "days_idle": {"type": "integer", "description": f"Number of days of inactivity before a check-in is triggered, default is {DEFAULT_PROACTIVE_CHECKIN_IDLE_DAYS}."},
                },
                "required": ["enable", "days_idle"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_checking_settings",
            "description": "Retrieves the current settings for proactive check-ins.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

# ---------- Discord wiring ----------
DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("Set DISCORD_TOKEN (or TOKEN) in your environment")

client = DiscordAccount(DISCORD_TOKEN, do_reconnect=True)
BOT_USER_ID = client.info["user"]["id"]  # so we can ignore ourselves


def _execute_tool_call(channel_id: str, tool_call: ChatCompletionMessageToolCall) -> str:
    """Executes a tool call and returns a JSON string result."""
    function_name = tool_call.function.name

    logging.info(f"[{channel_id}] Attempting to execute tool: {function_name}")

    if function_name == "toggle_proactive_checkins":
        # Unpack the arguments
        args = json.loads(tool_call.function.arguments)
        enable = args.get("enable")
        days_idle = args.get("days_idle", DEFAULT_PROACTIVE_CHECKIN_IDLE_DAYS)
        logging.info(f"[{channel_id}] toggle_proactive_checkins called: {enable=}, {days_idle=}")
        # Update the database
        update_user_setting_fields(channel_id, allow_proactive_checkin=enable, asked_about_checkins=True, days_idle=days_idle)
        return json.dumps({"status": "success", "message": f"Proactive check-ins have been set to: {enable=}, {days_idle=}"})
    elif function_name == "get_checking_settings":
        # Get the current settings from the database
        with Session(db_engine) as session:
            user_settings = get_or_create_user_settings(session, channel_id)
            settings = {
                "allow_proactive_checkin": user_settings.allow_proactive_checkin,
                "days_idle": user_settings.days_idle
            }
        logging.info(f"[{channel_id}] get_checking_settings called, revealed: {settings}")
        return json.dumps({"status": "success", "settings": settings})
    else:
        logging.warning(f"[{channel_id}] Unknown tool requested: {function_name}")
        return json.dumps({"status": "error", "message": f"Unknown function: {function_name}"})


def _process_llm_interaction_turn(channel_id: str, initial_messages_for_turn: List[Dict[str, Any]]) -> Optional[str]:
    """
    Processes a full LLM interaction turn, including handling multi-step tool calls.
    Returns the final natural language content from the LLM for the user.
    """
    processing_messages = list(initial_messages_for_turn)  # Work on a copy
    final_content: Optional[str] = None

    MAX_TOOL_ITERATIONS = 2  # Number of times the bot is allowed to call a tool per response (not including the final message)
    for i in range(MAX_TOOL_ITERATIONS + 1):
        logging.debug(f"[{channel_id}] LLM interaction loop, iteration {i + 1}/{MAX_TOOL_ITERATIONS}. Messages count: {len(processing_messages)}")

        current_tools_param = TOOLS_AVAILABLE if i != MAX_TOOL_ITERATIONS else None  # Don't pass tools on the last iteration

        llm_response_message: ChatCompletionMessage = chat_completion(
            messages=processing_messages,
            tools=current_tools_param
        )

        # Append LLM's response to our processing list
        processing_messages.append(llm_response_message.model_dump(exclude_none=True))

        if llm_response_message.tool_calls:

            logging.debug(f"[{channel_id}] LLM requested {len(llm_response_message.tool_calls)} tool call(s).")

            for tool_call in llm_response_message.tool_calls:
                tool_result_content = _execute_tool_call(channel_id, tool_call)
                processing_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,  # Ensure 'name' field is present for "tool" role messages
                    "content": tool_result_content,
                })
            # Continue loop to let LLM process tool results
        else:
            # If no tools are called, this is the final response
            final_content = llm_response_message.content
            break  # Exit loop
    else:  # Loop finished due to MAX_TOOL_ITERATIONS
        logging.warning(f"[{channel_id}] Exceeded max tool iterations ({MAX_TOOL_ITERATIONS}). Returning last known content or error.")
        if final_content is None:  # If loop exited without a final natural language response
            final_content = "I got stuck trying to process that. Could you try something simpler?"

    return final_content


def _initiate_typing_and_get_reply(channel_id: str, msgs: list[dict]) -> str:
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
        logging.debug(f"[{channel_id}] Calculating response for {len(msgs)} turns: {msgs}")
        assistant_reply = _process_llm_interaction_turn(channel_id, msgs)

        time.sleep(random.uniform(0.5, 1.5))  # simulate some delay
        logging.info(f"[{channel_id}] Assistant reply: {assistant_reply}")

    except Exception as e:
        logging.error(f"[{channel_id}] Error in chat_completion: {e}", exc_info=True)
        assistant_reply = "I'm having a little trouble connecting to my thoughts right now. Could we try that again in a moment?"
    finally:
        stop_flag.set()   # stop typing loop
        typing_thread.join(timeout=2)  # wait for the thread to finish

    return assistant_reply


def generate_and_send(channel_id: str, msgs: list[dict]):
    """Runs inside a worker thread – sends typing, gets LLM reply, posts back."""
    assistant_reply_content = _initiate_typing_and_get_reply(channel_id, msgs)
    client.send_message(
        channel_id,
        assistant_reply_content,
    )
    # we don't need to append the assistant's reply here, as it's already done in handle_message()


def initiate_proactive_checkin(channel_id: str, days_idle: int, is_first_time: bool = False):
    """Generates and sends a proactive check-in message. discord_client_ref is technically `client`"""
    logging.info(f"[{channel_id}] Preparing proactive check-in message.")

    # Load limited context for the check-in
    context_msgs = load_context(channel_id, max_summaries=MAX_SUMMARIES_PER_CONVERSATION, max_active_turns=0)

    if is_first_time:
        # If the user has not formally agreed to check-ins, ask them if they want to opt in
        system_prompt_for_checkin = (
            "You are initiating this conversation to check in with the user. "
            f"It has been a while since you last spoke ({days_idle} day(s) ago), and you're curious what they've been up to, "
            "However, they have not formally opted-in to proactive check-ins. "
            "Explain that the feature is available, and ask if it is okay to check in with them. "
            "Keep it light, open-ended, and not too long. "
            "Your most recent conversation summaries are provided above for context or idea starters."
        )
    else:
        system_prompt_for_checkin = (
            "You are initiating this conversation to check in with the user. "
            f"It has been a while since you last spoke ({days_idle} day(s) ago), and they previously agreed to this. "
            "Start with a friendly, casual greeting and ask how they are doing. "
            "Keep it light, open-ended, and not too long. "
            "Your most recent conversation summaries are provided above for context or idea starters."
        )

    msgs_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs_for_llm.extend(context_msgs)
    msgs_for_llm.append({"role": "system", "content": system_prompt_for_checkin})

    opening_message_content = _initiate_typing_and_get_reply(channel_id, msgs_for_llm)
    client.send_message(channel_id, opening_message_content)
    logging.info(f"[{channel_id}] Proactive check-in message sent!")


@client.on_packet([PacketType.MESSAGE_CREATE])
def handle_message(pkt: dict):
    """Main entry-point for each incoming Discord message."""
    # Ignore messages from ourselves or other bots
    author      = pkt["author"]
    channel_id  = pkt["channel_id"]
    message_id  = pkt["id"]  # if we want to reply to a specific message
    content     = pkt["content"]

    is_dm = pkt["channel_type"] == 1  # 1 indicates a DM channel
    # Only process DMs
    if not is_dm:
        logging.debug(f"[{channel_id}] Discord non-DM message: {content}")
        return

    # If the message is from the bot, just add the turn and return
    # We add turns here so that the message order is true to the way Discord shows them
    if author["id"] == BOT_USER_ID:
        append_turn(channel_id, "assistant", content)
        return

    # save the message to the database
    logging.info(f"[{channel_id}] Received message from user {author.get('username')}: {content}")
    append_turn(channel_id, "user", content)

    # get chat history fresh from disk
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += load_context(channel_id, max_summaries=MAX_SUMMARIES_PER_CONVERSATION, max_active_turns=MAX_ACTIVE_TURNS_PER_CONVERSATION)

    # ----- spawn a worker thread so we don't block the websocket heartbeat -----
    threading.Thread(
        target=generate_and_send,
        args=(channel_id, msgs),
        daemon=True
    ).start()


if AUTO_ACCEPT_FRIENDS:
    # Accept all incoming friend requests
    @client.on_packet([PacketType.RELATIONSHIP_ADD])
    def accept_friend_request(pkt: dict):
        """Accept all incoming friend requests."""
        user_id = pkt["user"]["id"]
        modifier_type = pkt["type"]
        logging.info(f"Received friend request from user {pkt['user']['username']}: {modifier_type}")
        time.sleep(random.uniform(3, 7))  # simulate some delay
        if modifier_type == 3:  # 3 indicates a friend request
            logging.info(f"Accepting friend request from {user_id}")
            res = client.modify_friendship(user_id, action="accept")
            if res.status_code != 204:
                logging.error(f"Failed to accept friend request from {user_id}: {res.status_code}")


# ---------- run ----------
if __name__ == "__main__":
    # Start the sweeper thread
    threading.Thread(target=summarizer_loop, args=(timedelta(minutes=CONVERSATION_MAX_IDLE_AGE_MINS), SUMMARY_SWEEPER_INTERVAL_MINS), daemon=True).start()

    # Start the proactive check-in thread
    threading.Thread(target=proactive_checkin_loop, args=(initiate_proactive_checkin, PROACTIVE_CHECKIN_INTERVAL_MINS), daemon=True).start()

    # Start the Discord client
    try:
        client.start_listeners()  # blocking
    except KeyboardInterrupt:
        logging.info("Shutting down…")
    finally:
        client.close_session()
        logging.info("Application closed.")
