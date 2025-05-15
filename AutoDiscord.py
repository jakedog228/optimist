import ssl

import requests
import websocket  # pip install websocket-client
import threading
import asyncio
import enum
import random

from typing import Callable, Any
from json import loads, dumps
import os

# Discord API base URL
API_BASE = "https://discord.com/api/v9"


# enumerate the different types of packets that can be received
class PacketType(enum.Enum):
    # Full list can be found here: https://discord.com/developers/docs/events/gateway
    SESSIONS_REPLACE = "SESSIONS_REPLACE"
    TYPING_START = "TYPING_START"  # {'user_id': '601805146598670367', 'timestamp': 1745432483, 'channel_id': '1361090208556122193'}
    MESSAGE_CREATE = "MESSAGE_CREATE"  # {'type': 0, 'tts': False, 'timestamp': '2025-04-23T18:21:21.989000+00:00', 'pinned': False, 'nonce': '1364667480840077312', 'mentions': [], 'mention_roles': [], 'mention_everyone': False, 'id': '1364667481935052832', 'flags': 0, 'embeds': [], 'edited_timestamp': None, 'content': 'hello', 'components': [], 'channel_type': 1, 'channel_id': '1361090208556122193', 'author': {'username': 'shuep', 'public_flags': 64, 'primary_guild': None, 'id': '601805146598670367', 'global_name': 'Sheep', 'discriminator': '0', 'collectibles': None, 'clan': None, 'avatar_decoration_data': None, 'avatar': '5c9bd5ce5ec806ba7d006a02da014be7'}, 'attachments': []}
    MESSAGE_UPDATE = "MESSAGE_UPDATE"
    MESSAGE_DELETE = "MESSAGE_DELETE"
    MESSAGE_REACTION_ADD = "MESSAGE_REACTION_ADD"
    PRESENCE_UPDATE = "PRESENCE_UPDATE"
    CHANNEL_UNREAD_UPDATE = "CHANNEL_UNREAD_UPDATE"
    VOICE_STATE_UPDATE = "VOICE_STATE_UPDATE"
    CONVERSATION_SUMMARY_UPDATE = "CONVERSATION_SUMMARY_UPDATE"
    NONE = None


class DiscordAccount:
    def __init__(self, token: str, do_reconnect: bool = True):
        self.token = token
        self.do_reconnect = do_reconnect
        self.headers = self.get_headers(token)
        self.websocket, self.info = self.create_session()
        self.listeners: list[Callable[[dict, Any], None], PacketType] = []
        self.print_info()
        print()

    def print_info(self):
        print(f'{len(self.info["sessions"])} total sessions running, current Session ID: {self.info["session_id"]}')
        print(f'{len(self.info["guilds"])} servers connected')
        print(f'{len(self.info["private_channels"])} DMs/GCs connected')

    def get_headers(self, token: str):
        return {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Authorization": token
        }

    def get_guild(self, guild_id: str or int) -> dict or None:
        """Get information about a guild by ID."""
        for guild in self.info["guilds"]:
            if guild["id"] == str(guild_id):
                return guild
        return None

    def get_channel(self, channel_id: str or int, guild_id: str or int = None) -> dict or None:
        """
        Get information about a channel by ID.
        If a guild id is provided, search within the guild. Otherwise, search in private channels.
        """
        if guild_id is not None:  # get guild channels
            guild = self.get_guild(guild_id)
            if guild is None:
                return None
            channels = guild["channels"]
        else:  # get private channels
            channels = self.info["private_channels"]

        # find the channel with the given ID
        for channel in channels:
            if channel["id"] == str(channel_id):
                return channel
        return None

    def get_user(self, user_id: str or int) -> dict or None:
        """Get information about a user by ID."""
        for user in self.info["users"]:
            if user["id"] == str(user_id):
                return user
        return None

    def delete_message(self, channel_id, message_id):
        requests.delete(f'https://discord.com/api/v9/channels/{channel_id}/messages/{message_id}', headers=self.headers)

    def send_message(self, channel_id: str or int, message: str, reference_message_id: str = None, filepaths: list[str] = None) -> dict:
        """Sends a message to the specified channel"""
        if filepaths is None:
            filepaths = []
        payload = {'content': message}

        # if a message ID is provided, set it as a reference
        if reference_message_id:
            payload['message_reference'] = {'channel_id': str(channel_id), 'message_id': str(reference_message_id)}

        # if a file path is provided, upload the file
        if filepaths:
            uploaded_files = self.upload_files(channel_id, filepaths)
            payload['attachments'] = []
            for i, (filename, uploaded_filename) in enumerate(uploaded_files):
                payload['attachments'].append({
                    "id": str(i),
                    "uploaded_filename": uploaded_filename,
                    "filename": filename,
                })

        res = requests.post(f'{API_BASE}/channels/{channel_id}/messages', headers=self.headers, json=payload).json()

        return res  # response includes message ID, useful for editing/deleting/referencing later

    def edit_message(self, channel_id: str or int, message_id: str or int, message: str):
        payload = {'content': message}
        requests.patch(f'{API_BASE}/channels/{channel_id}/messages/{message_id}', headers=self.headers, json=payload)

    def start_typing(self, channel_id: str or int):
        """Starts typing in the specified channel."""
        requests.post(f'{API_BASE}/channels/{channel_id}/typing', headers=self.headers)

    def create_dm_channel(self, recipient_ids: list[str or int]) -> str:
        """Creates a DM channel with recipient_id (multiple for a gc) and returns the channel ID."""
        url = f"{API_BASE}/users/@me/channels"
        payload = {"recipients": [str(id) for id in recipient_ids]}
        r = requests.post(url, headers=self.headers, json=payload)
        if r.status_code == 200:
            channel = r.json()
            return channel["id"]
        else:
            raise Exception(f"Failed to create DM channel: {r.text}")

    def upload_files(self, channel_id: str or int, file_paths: list[str]) -> list[tuple[str, str]]:
        """ Upload files to discord's CDN and return the file URLs. """
        url = f"{API_BASE}/channels/{channel_id}/attachments"
        uploaded_files = []
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                filename = file_path.split("/")[-1]
                file = f.read()
                file_size = len(file)
            payload = {
                'files': [{
                    "filename": filename,
                    "file_size": file_size,
                    "id": str(random.randint(1, 1_000_000)),
                    "is_clip": False,
                }]
            }

            res = requests.post(url, headers=self.headers, json=payload)

            if res.status_code == 200:
                attachments = res.json().get("attachments")[0]
                upload_id = attachments.get("id")
                upload_url = attachments.get("upload_url")
                upload_filename = attachments.get("upload_filename")

                headers = {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(file_size),
                    "Origin": "https://discord.com",
                }

                res2 = requests.put(upload_url, data=file, headers=headers)

                if res2.status_code == 200:
                    uploaded_files.append((filename, upload_filename))
                else:
                    print(f"Failed to upload {filename} to {upload_url}: {res2.status_code}")
            else:
                print(f"Failed to request space for {filename}: {res.text}")

        return uploaded_files

    def create_session(self) -> (websocket.WebSocket, dict):

        # recursive function to keep the connection alive
        def pulse(ws, pulse_rate):
            def keep_alive():
                try:
                    ws.send(dumps({"op": 1, "d": None}))
                except websocket.WebSocketConnectionClosedException or ssl.SSLEOFError:
                    print('Connection closed, stopping heartbeat')
                    return
                pulse(ws, pulse_rate)

            threading.Timer(pulse_rate / 1000, keep_alive).start()

        req = requests.get("https://canary.discordapp.com/api/v9/users/@me", headers=self.headers)

        if req.status_code != 200:
            print(f'Basic request returned Error code {req.status_code}, try refreshing your token')
            quit()

        user_info = req.json()
        username = user_info["username"]
        user_id = user_info["id"]
        print(f'Logged into user {username} ({user_id})')

        ws = websocket.WebSocket()
        ws.connect("wss://gateway.discord.gg/?v=6&encoding=json")
        hello = loads(ws.recv())  # https://discord.com/developers/docs/events/gateway#hello-event
        heartbeat_interval = hello["d"]["heartbeat_interval"]
        print(f'Connected to the Discord Websocket with a heartbeat of {heartbeat_interval}')

        pulse(ws, heartbeat_interval)  # recursive async function, keeps heartbeat going

        # Handshake (https://discord.com/developers/docs/topics/gateway)
        auth = {
            "op": 2,
            "d": {
                "token": self.token,
                "capabilities": 30717,
                "properties": {
                    "os": "Windows",
                    "browser": "Chrome",
                    "device": "",
                    "system_locale": "en-US",
                    "browser_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                    "browser_version": "133.0.0.0",
                    "os_version": "10",
                    "referrer": "",
                    "referring_domain": "",
                    "referrer_current": "",
                    "referring_domain_current": "",
                    "release_channel": "stable",
                    "client_build_number": 372693,
                    "client_event_source": None,
                    "has_client_mods": False
                },
                "presence": {
                    "status": "unknown",
                    "since": 0,
                    "activities": [],
                    "afk": False
                },
                "compress": False,
                "client_state": {
                    "guild_versions": {}
                }
            },
            "s": None,
            "t": None,
        }
        ws.send(dumps(auth))
        status = {
            "op": 3,
            "d": {
                "since": 0,
                "activities": [
                    {
                        "type": 4,
                        "state": '',
                        "name": "Custom Status",
                        "id": "custom",
                    }
                ],
                "status": '',
                "afk": False,
            },
        }
        ws.send(dumps(status))
        print(f'Handshake information sent...')

        ready = loads(ws.recv())  # https://discord.com/developers/docs/events/gateway-events#ready
        connection_info = ready["d"]
        print(f'Handshake established!')

        return ws, connection_info

    def add_listeners(self, func, packet_types: list[PacketType] = None):
        """Adds a listener. If packet_type is provided, the callback will only be triggered on matching packets.
        Otherwise, it triggers on all packets and gets both packet_type and packet_data."""
        # print(f'[{self.info["user"]["username"]}] Adding listener for {func.__name__} on {packet_types}')
        packet_types = packet_types or [None]  # default to all packets if packet_types is None or []
        for packet_type in packet_types:
            self.listeners.append((func, packet_type))

    def remove_listener(self, func):
        """Removes a listener."""
        for listener in self.listeners:
            if listener[0] == func:
                self.listeners.remove(listener)

    def on_packet(self, packet_types: list[PacketType] = None):
        """Decorator to register a listener."""

        def decorator(func):
            self.add_listeners(func, packet_types)
            return func

        return decorator

    def start_listeners(self):
        """
        Creates the websocket connection and starts listening for packets.
        :param do_reconnect: If True, will attempt to reconnect on disconnect.
        """
        # print(f'[{self.info["user"]["username"]}] Starting listeners...')
        while True:
            try:
                raw = self.websocket.recv()
                if not raw:
                    raise Exception(f'Received empty packet: {raw}')  # connection is likely closed
                packet = loads(raw)
                # print(f'[{self.info["user"]["username"]}] Received packet: {packet}')
                self._handle_packet(packet)
            except Exception as e:  # Note: runtime errors (e.g. file not found) will also be caught here with generally unhelpful error messages (due to asyncio)
                print(f'Connection closed ({e})')
                if not self.do_reconnect:
                    break  # stop the loop if do_reconnect is False
                print('Attempting to reconnect...')
                self.websocket, self.info = self.create_session()  # reconnect
                self.print_info()
                print()
                continue

    async def close_session(self):
        """Closes the websocket connection, stops the start_bot loop."""
        self.do_reconnect = False
        await self.websocket.close()

    def _handle_packet(self, packet):
        """Dispatch the packet to the appropriate listeners."""
        packet_type = packet[
            't']  # the type of gateway event (opcode: 0) being sent, REF: https://discord.com/developers/docs/topics/gateway#list-of-intents
        packet_sequence = packet['s']  # the order number in which the event occurred
        packet_opcode = packet[
            'op']  # The type of packet being sent, REF: https://discord.com/developers/docs/topics/opcodes-and-status-codes
        packet_data = packet[
            'd']  # the data in the packet, REF: MORE INFO: https://discord.com/developers/docs/topics/gateway-events#receive-events

        # Optional: Add check for heartbeat ACK (op 11) or other ops if needed
        if packet_opcode == 11:  # Opcode 11: Heartbeat ACK
            # print("Received Heartbeat ACK (skipping)")
            return  # Nothing more to do for ACK

        for func, expected_type in self.listeners:

            if expected_type is None:
                # Listener for all packets receives both the packet type and the data.
                func(packet_data, packet_type)

            elif packet_type == expected_type.value:
                # If a specific packet_type was provided, only call the function if it matches.
                func(packet_data)


if __name__ == '__main__':
    # Initialize the DiscordAccount class with a token
    TOKEN = os.environ['TOKEN']
    client = DiscordAccount(TOKEN, do_reconnect=False)  # Should the bot reconnect after idle disconnect (~15 min of inactivity)?


    # Add listeners using decorators
    @client.on_packet([PacketType.MESSAGE_CREATE, PacketType.MESSAGE_UPDATE])
    def log_new_msgs(packet_data: dict):
        """Logs all sent/edited messages."""
        timestamp = packet_data["timestamp"].split("T")[-1].split("+")[0]
        author = packet_data["author"]["username"]
        message = packet_data["content"]
        attachments = packet_data["attachments"]
        message_id = packet_data["id"]

        attachments_message = f' with {len(attachments)} attachments' if attachments else ""
        print(f'[{timestamp}] Message ({message_id}) from {author}{attachments_message}: {message}')

    @client.on_packet([PacketType.TYPING_START])
    def log_typing(packet_data: dict):
        """Logs typing events."""
        timestamp = packet_data["timestamp"]  # epoch time, no idea why these are not in ISO format, discord = bad?
        user_id = packet_data["user_id"]
        channel_id = packet_data["channel_id"]
        print(f'[{timestamp}] User {user_id} is typing in channel {channel_id}')

    # Add generic listener for all packets
    @client.on_packet()
    def log_all_packets(packet_data: dict, packet_type: PacketType):
        """Logs all packets."""
        print(f'Packet Type: {packet_type} - Data: {str(packet_data)}')


    def ping_response(packet_data: dict):
        """Replies to pings with pongs! Use with PacketType.MESSAGE_CREATE"""
        if packet_data["content"] == "ping":
            client.send_message(packet_data["channel_id"], "pong!", reference_message_id=packet_data["id"])


    # Add listener function directly
    client.add_listeners(ping_response, packet_types=[PacketType.MESSAGE_CREATE])

    # Start listening (blocking call)
    try:
        client.start_listeners()
        print('done?')
    except KeyboardInterrupt:
        print("Exiting...")
        client.close_session()
        print("Session closed!")

    # Note: You can also use asyncio.get_running_loop() + loop.run_in_executor() to run without blocking, e.g.:
    """
    async def main():
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, client.start_listeners)

    asyncio.run(main())
    """