from datetime import timedelta
import time, logging

from storage import summarise_and_close, find_inactive_conversations

log = logging.getLogger("summarizer")
log.setLevel(logging.DEBUG)

def sweeper(max_age: timedelta = timedelta(minutes=60), wait_time_mins: int = 5):
    """Background loop: every 5 min (default) close idle conversations."""
    while True:
        try:
            inactive_conversations = find_inactive_conversations(max_age)
            log.info(f"Found {len(inactive_conversations)} inactive conversations")
            for conv in inactive_conversations:
                log.info(f"Auto-summarizing convo {conv.id} (channel {conv.channel_id})")
                summary = summarise_and_close(conv)
                log.info(f"Summarized convo: {summary}")
        except Exception as e:
            log.exception("Sweeper error: %s", e)
        time.sleep(wait_time_mins * 60)  # wait for the next cycle
