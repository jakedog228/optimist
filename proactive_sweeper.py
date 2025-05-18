import time
import logging
import random
from datetime import datetime, timedelta
from typing import Callable

from sqlmodel import Session
from storage import engine as db_engine, get_channels_for_proactive_checkin

log = logging.getLogger("proactive_service")
log.setLevel(logging.DEBUG)


def proactive_checkin_loop(
        proactive_message_callback: Callable[[str, int, bool], None],
        check_interval_mins: int,
):
    log.info(
        f"Proactive checker service started. Check interval: {check_interval_mins} mins"
    )

    while True:
        try:
            now_utc = datetime.utcnow()
            current_utc_hour = now_utc.hour

            with Session(db_engine) as session:
                eligible_user_settings = get_channels_for_proactive_checkin(session)

            log.debug(f"Found {len(eligible_user_settings)} users eligible for proactive check-in.")

            for last_user_interaction_time, user_setting in eligible_user_settings:
                user_active_hours = user_setting.active_hours_utc
                channel_id = user_setting.channel_id
                first_time = user_setting.asked_about_checkins == False

                # We also need to check if the current time is within the user's active hours
                if not user_active_hours or current_utc_hour in user_active_hours:

                    # if this is the first time, we want to disable this check-in exception for future checks
                    if first_time:
                        user_setting.asked_about_checkins = True
                        with Session(db_engine) as session:
                            session.add(user_setting)
                            session.commit()

                    # Figure out the number of days since the last message
                    time_since_last_message = now_utc - last_user_interaction_time
                    days_since_last_message = time_since_last_message.days

                    # Send the proactive message
                    log.info(f"Initiating proactive check-in for channel {channel_id} after {days_since_last_message} silent days. Current UTC hour: {current_utc_hour}, User active hours: {user_active_hours or 'Any'}.")
                    proactive_message_callback(channel_id, days_since_last_message, first_time)

                    # Stagger messages, 10-30 seconds
                    time.sleep(random.uniform(10, 30))
                else:
                    log.debug(f"Skipping check-in for channel {channel_id}. Current UTC hour {current_utc_hour} not in user's active hours {user_active_hours}.")

        except Exception as e:
            log.exception(f"Error in proactive_checker_loop: {e}")

        time.sleep(check_interval_mins * 60)  # wait for the next cycle
