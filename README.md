# Optimist
*Bringing the future of anonymous, accessible mental health resources to platforms of the users.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
**Built at the 2025 Canton Hackathon – 🏆 2nd Place Overall**

---

## Why Optimist?
Talking helps, but many people hesitate to reach out when they feel low.  **Optimist** lives in your Discord DMs as a friendly, non-judgemental companion who:

* Listens & responds with empathy (no therapy-speak or canned advice)
* Remembers past conversations in long-term “memory blocks”
* Gently checks in if you’ve been quiet for a while (opt-in)
* Runs entirely on open-source, locally stored data

Optimist is a powerful, cost-effective tool for learning how to talk about complex emotions and process challenging experiences in a safe, sandboxed environment. 

Available 24/7, Optimist is anonymously accessible friend to anyone with a Discord account.

---

## Demo

> WARNING: This was a demo video recorded for the hackathon, with the theme of combating mental health crises. 
> As such, it contains some sensitive topics and may not be suitable for all audiences. Trigger warning for **Suicide**.

https://github.com/user-attachments/assets/b6265714-400e-4816-ad15-2e0a1e375870

---

## Getting Started

### Prerequisites
* Python 3.11+
* A Discord user token
* An OpenAI API key

### Local Setup
```bash
git clone https://github.com/yourname/optimist.git
cd optimist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

Create a `.env` file or export environment variables before running (no default = required):

| Category     | Name                                  | Default                  | Description                                                                         |
|--------------|---------------------------------------|--------------------------|-------------------------------------------------------------------------------------|
| API Keys     | `OPENAI_API_KEY`                      | —                        | Your OpenAI secret key                                                              |
| API Keys     | `DISCORD_TOKEN`                       | —                        | Discord user token                                                                  |
| Model Config | `COMPLETION_MODEL`                    | `gpt-4.1-mini`           | OpenAI model to use for conversations                                               |
| Model Config | `EMBEDDING_MODEL`                     | `text-embedding-3-small` | OpenAI model to use for embedding                                                   |
| Model Config | `SYSTEM_PROMPT`                       | (see `main.py`)          | Override the base personality                                                       |
| Memory       | `MAX_SUMMARIES_PER_CONVERSATION`      | 10                       | Max number of memory blocks to use in a conversation                                |
| Memory       | `MAX_TURNS_PER_CONVERSATION`          | -1                       | Max number of turns in the active dialogue to use in a conversation (-1 = no limit) |
| Memory       | `MAX_SUMMARIES_FOR_SUMMARIZATION`     | 10                       | How many memory blocks to use when the bot summarizes a conversation                |
| Memory       | `CONVERSATION_MAX_IDLE_AGE_MINS`      | 10                       | Time (minutes) before a conversations is considered "inactive"                      |
| Memory       | `SUMMARY_SWEEPER_INTERVAL_MINS`       | 10                       | Time (minutes) between checks for inactive conversations                            |
| Check-ins    | `PROACTIVE_CHECKIN_INTERVAL_MINS`     | 60                       | Time (minutes) between checks for check-in-eligible people                          |
| Check-ins    | `DEFAULT_PROACTIVE_CHECKIN_IDLE_DAYS` | 1                        | Default time (days) before a user is considered eligible for check-in (see below)   |
| Other        | `AUTO_ACCEPT_FRIENDS`                 | true                     | Whether your bot should automatically accept friend requests from unknown users     |

### Running the Bot

```bash
python main.py
```

And if everything worked, you should see something like this:

![image](https://github.com/user-attachments/assets/cefeee40-4cac-4ce6-9134-31aae12ab6bb)

Shoot the bot a DM to start chatting!

---

## How Memory Works

One of the biggest concerns with using LLMs as long-term companions is the risk of **high costs** and **fragile memory** that comes with linearly scaling token count of multiple conversations. If a user has a chat history with an LLM going back thousands of messages, this scaling can easily become a problem.

1. **Active turns** stay verbatim (configurable count).
2. When conversation is idle for an hour (configurable), `summary_sweeper.py`:
   * Asks the bot to summarize its conversation to a third length (a memory block).
   * Stores `Conversation.summary` + a generated semantic embedding.
   * Deletes raw messages from memory.
3. During memory retrieval, `storage.load_context()`:
   * Gets the semantic embedding of the current conversation.
   * Vector-searches for relevant memory blocks in the database.
   * Prepends up to 10 (configurable) memory blocks to the conversation.
   * Submits the system prompt + memory blocks + active turns to the completion model.

---

## Proactive Check-ins

* A day after the fist conversation, if the user has not already opted in, Optimist politely asks permission to check in on the user (one-time).
* Each user can **opt in/out** by talking to the bot (`toggle_proactive_checkins` tool).
* If enabled, the bot will DM again after a couple of days of silence (depending on user's stated preference)
* The bot will only check in during the user’s detected active hours (to avoid inconvenient/unhealthy habits).

![image](https://github.com/user-attachments/assets/15275829-f733-4733-8dbb-60f43a462cc6)

---

## Public Instance

If you want to talk to Optimist without running it locally, shoot our public instance a DM on Discord!

<img src="https://github.com/user-attachments/assets/739d898d-5a33-4a61-9713-be0b204befa9" alt="drawing" width="400"/>

**Disclaimer:** This is a public instance of the bot, and we cannot guarantee the privacy of your conversations. Please do not share any sensitive information with the bot. We are not responsible for any loss of data or temporary/permanent downtime. Use at your own risk.

---

> "Self-care is the gentle art you weave, and a sacred gift you must believe" - Optimist, out of context
