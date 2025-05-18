from datetime import timedelta
import time, logging

from storage import summarise_and_close, find_inactive_conversations

log = logging.getLogger("summarizer")
log.setLevel(logging.DEBUG)

def summarizer_loop(max_age: timedelta = timedelta(minutes=60), wait_time_mins: int = 5):
    """Background loop: every `wait_time_mins` minutes close idle conversations."""
    log.info(f"Summary sweeper started. Max idle age: {max_age.seconds//60} mins, Check interval: {wait_time_mins} mins.")

    while True:
        try:
            inactive_conversations = find_inactive_conversations(max_age)
            log.debug(f"Found {len(inactive_conversations)} inactive conversations")
            for conv in inactive_conversations:
                log.info(f"Auto-summarizing convo {conv.id} (channel {conv.channel_id})")
                summary = summarise_and_close(conv)
                log.info(f"Summarized convo: {summary}")
        except Exception as e:
            log.exception(f"Sweeper error: {e}")

        time.sleep(wait_time_mins * 60)  # wait for the next cycle
