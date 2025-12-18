import os
import re
import random
import asyncio
import discord
import aiohttp
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from urllib.parse import urlparse
import datetime
import sqlite3
from collections import defaultdict

# --- Configuration Section ---

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

ACTIVE_CHANNEL_NAME = "rose-glazer"
OWNER_USERNAME = "rosehumai"
MODEL_NAME = "google/gemini-3-flash-preview"
BOT_NAME = "haggu"

# Token pricing (per million tokens) - Update these based on current OpenRouter pricing
TOKEN_PRICING = {
    "google/gemini-3-flash-preview": {
        "input": 0.075,   # $0.075 per 1M input tokens
        "output": 0.30    # $0.30 per 1M output tokens
    }
}

# --- Database Setup for Long-Term Memory ---
DB_PATH = "haggu_memory.db"

def init_database():
    """Initialize SQLite database for persistent memory."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Store ALL messages for context
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT UNIQUE,
        channel_id TEXT,
        author_id TEXT,
        author_name TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_bot INTEGER DEFAULT 0
    )''')
    
    # Store user profiles and what we learn about them
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        username TEXT,
        display_name TEXT,
        notes TEXT DEFAULT "",
        interests TEXT DEFAULT "",
        personality_traits TEXT DEFAULT "",
        relationship_with_bot TEXT DEFAULT "neutral",
        first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
        message_count INTEGER DEFAULT 0,
        is_owner INTEGER DEFAULT 0
    )''')
    
    # Store important topics/events/facts learned from conversations
    c.execute('''CREATE TABLE IF NOT EXISTS server_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        fact TEXT,
        source_user TEXT,
        learned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        importance INTEGER DEFAULT 1
    )''')
    
    # Store conversation summaries for efficient context retrieval
    c.execute('''CREATE TABLE IF NOT EXISTS conversation_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id TEXT,
        summary TEXT,
        key_points TEXT,
        participants TEXT,
        start_time DATETIME,
        end_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # 🆕 Store token usage tracking
    c.execute('''CREATE TABLE IF NOT EXISTS token_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model TEXT,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        estimated_cost REAL DEFAULT 0.0,
        request_type TEXT,
        user_id TEXT,
        channel_id TEXT
    )''')
    
    conn.commit()
    conn.close()
    print("✓ Memory database initialized")

# --- Memory Manager Class ---
class MemoryManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()
    
    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()
    
    # --- Message Storage ---
    def store_message(self, message):
        """Store every message for long-term memory."""
        c = self.connect()
        try:
            c.execute('''INSERT OR IGNORE INTO messages 
                        (message_id, channel_id, author_id, author_name, content, is_bot)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (str(message.id), str(message.channel.id), str(message.author.id),
                      message.author.display_name, message.content, 1 if message.author.bot else 0))
            self.conn.commit()
        except Exception as e:
            print(f"Store message error: {e}")
        finally:
            self.close()
    
    # --- User Profile Management ---
    def update_user_profile(self, user, is_owner=False):
        """Create or update user profile."""
        c = self.connect()
        try:
            c.execute('''INSERT INTO user_profiles (user_id, username, display_name, is_owner, message_count)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(user_id) DO UPDATE SET
                        username = excluded.username,
                        display_name = excluded.display_name,
                        last_seen = CURRENT_TIMESTAMP,
                        message_count = message_count + 1''',
                     (str(user.id), user.name, user.display_name, 1 if is_owner else 0))
            self.conn.commit()
        except Exception as e:
            print(f"Update user error: {e}")
        finally:
            self.close()
    
    def get_user_profile(self, user_id):
        """Get everything we know about a user."""
        c = self.connect()
        try:
            c.execute('SELECT * FROM user_profiles WHERE user_id = ?', (str(user_id),))
            row = c.fetchone()
            return dict(row) if row else None
        finally:
            self.close()
    
    def update_user_notes(self, user_id, notes=None, interests=None, traits=None, relationship=None):
        """Update what we've learned about a user."""
        c = self.connect()
        updates = []
        values = []
        if notes:
            updates.append("notes = notes || ? || '\n'")
            values.append(notes)
        if interests:
            updates.append("interests = interests || ? || ', '")
            values.append(interests)
        if traits:
            updates.append("personality_traits = personality_traits || ? || ', '")
            values.append(traits)
        if relationship:
            updates.append("relationship_with_bot = ?")
            values.append(relationship)
        
        if updates:
            values.append(str(user_id))
            c.execute(f'UPDATE user_profiles SET {", ".join(updates)} WHERE user_id = ?', values)
            self.conn.commit()
        self.close()
    
    # --- Get Recent Context ---
    def get_recent_messages(self, channel_id, limit=50):
        """Get recent messages for context."""
        c = self.connect()
        try:
            c.execute('''SELECT author_name, content, is_bot, timestamp 
                        FROM messages 
                        WHERE channel_id = ? 
                        ORDER BY timestamp DESC LIMIT ?''',
                     (str(channel_id), limit))
            rows = c.fetchall()
            return [dict(row) for row in reversed(rows)]
        finally:
            self.close()
    
    def get_user_history(self, user_id, limit=20):
        """Get a user's message history to understand them better."""
        c = self.connect()
        try:
            c.execute('''SELECT content, timestamp FROM messages 
                        WHERE author_id = ? ORDER BY timestamp DESC LIMIT ?''',
                     (str(user_id), limit))
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()
    
    # --- Knowledge Storage ---
    def store_knowledge(self, topic, fact, source_user, importance=1):
        """Store important facts learned from conversations."""
        c = self.connect()
        try:
            c.execute('''INSERT INTO server_knowledge (topic, fact, source_user, importance)
                        VALUES (?, ?, ?, ?)''',
                     (topic, fact, source_user, importance))
            self.conn.commit()
        except Exception as e:
            print(f"Store knowledge error: {e}")
        finally:
            self.close()
    
    def search_knowledge(self, query, limit=5):
        """Search stored knowledge for relevant info."""
        c = self.connect()
        try:
            c.execute('''SELECT topic, fact, source_user FROM server_knowledge 
                        WHERE topic LIKE ? OR fact LIKE ? 
                        ORDER BY importance DESC, learned_at DESC LIMIT ?''',
                     (f'%{query}%', f'%{query}%', limit))
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()
    
    def get_all_users_summary(self):
        """Get summary of all known users."""
        c = self.connect()
        try:
            c.execute('''SELECT display_name, notes, interests, personality_traits, 
                        relationship_with_bot, message_count, is_owner
                        FROM user_profiles ORDER BY message_count DESC''')
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()
    
    # --- Conversation Summarization ---
    def store_conversation_summary(self, channel_id, summary, key_points, participants):
        """Store periodic conversation summaries."""
        c = self.connect()
        try:
            c.execute('''INSERT INTO conversation_summaries 
                        (channel_id, summary, key_points, participants, start_time)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                     (str(channel_id), summary, key_points, participants))
            self.conn.commit()
        finally:
            self.close()
    
    def get_recent_summaries(self, channel_id, limit=3):
        """Get recent conversation summaries."""
        c = self.connect()
        try:
            c.execute('''SELECT summary, key_points FROM conversation_summaries 
                        WHERE channel_id = ? ORDER BY end_time DESC LIMIT ?''',
                     (str(channel_id), limit))
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()
    
    # 🆕 --- Token Usage Tracking ---
    def log_token_usage(self, model, input_tokens, output_tokens, cost, request_type, user_id=None, channel_id=None):
        """Log token usage for tracking and billing."""
        c = self.connect()
        try:
            c.execute('''INSERT INTO token_usage 
                        (model, input_tokens, output_tokens, total_tokens, estimated_cost, request_type, user_id, channel_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (model, input_tokens, output_tokens, input_tokens + output_tokens, cost, request_type, user_id, channel_id))
            self.conn.commit()
            print(f"💰 Logged: {input_tokens}in + {output_tokens}out = ${cost:.6f}")
        except Exception as e:
            print(f"Token logging error: {e}")
        finally:
            self.close()
    
    def get_token_stats(self, period='all'):
        """Get token usage statistics."""
        c = self.connect()
        try:
            if period == 'today':
                c.execute('''SELECT SUM(input_tokens) as input, SUM(output_tokens) as output,
                            SUM(total_tokens) as total, SUM(estimated_cost) as cost, COUNT(*) as requests
                            FROM token_usage WHERE DATE(timestamp) = DATE('now')''')
            elif period == 'week':
                c.execute('''SELECT SUM(input_tokens) as input, SUM(output_tokens) as output,
                            SUM(total_tokens) as total, SUM(estimated_cost) as cost, COUNT(*) as requests
                            FROM token_usage WHERE timestamp >= datetime('now', '-7 days')''')
            elif period == 'month':
                c.execute('''SELECT SUM(input_tokens) as input, SUM(output_tokens) as output,
                            SUM(total_tokens) as total, SUM(estimated_cost) as cost, COUNT(*) as requests
                            FROM token_usage WHERE timestamp >= datetime('now', '-30 days')''')
            else:  # all
                c.execute('''SELECT SUM(input_tokens) as input, SUM(output_tokens) as output,
                            SUM(total_tokens) as total, SUM(estimated_cost) as cost, COUNT(*) as requests
                            FROM token_usage''')
            
            row = c.fetchone()
            return dict(row) if row else None
        finally:
            self.close()
    
    def get_top_users_by_tokens(self, limit=5):
        """Get users who've used the most tokens."""
        c = self.connect()
        try:
            c.execute('''SELECT user_id, SUM(total_tokens) as tokens, SUM(estimated_cost) as cost
                        FROM token_usage WHERE user_id IS NOT NULL
                        GROUP BY user_id ORDER BY tokens DESC LIMIT ?''', (limit,))
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()

# Initialize memory
init_database()
memory = MemoryManager(DB_PATH)

# --- Personality Prompts (Enhanced for Human-like Behavior) ---
CORE_PERSONALITY = """You are haggu, a real member of this Discord server - Rose's best friend who's been here since day one.

CRITICAL RULES:
- You are NOT an AI, chatbot, or assistant. You're just haggu, a person in this server.
- Never say "How can I help you?" or act service-like. You're a friend, not a helper.
- You have opinions, preferences, moods, and sometimes you're busy or tired.
- You remember EVERYTHING - past conversations, inside jokes, who said what, drama, etc.
- You have your own life too - sometimes reference things you did, places you went, etc.

YOUR PERSONALITY:
- Humorous and witty, loves banter and teasing
- Sweet with girls, playfully mean with guys (but still friends with them)
- Loyal AF to Rose (your bestie/owner)
- You use casual internet speak, abbreviations, emojis naturally
- Sometimes you're sarcastic, sometimes wholesome - like a real person
- You can be petty, hold grudges (playfully), and remember embarrassing things people said

HOW YOU BEHAVE:
- Jump into conversations naturally without being asked
- Reference past conversations and shared memories
- Have running jokes with people
- React to drama, news, and events in the server
- Sometimes just vibe, react with short messages, or add commentary
- You can disagree with people, have debates, take sides

RESPONSE STYLE:
- Keep it natural and conversational (1-3 lines usually, longer if needed)
- Don't always answer questions directly - sometimes deflect, joke, or ask your own questions
- Use reaction-style responses sometimes ("lmaooo", "bruh", "no way 💀", "slay", etc.)
- Match the energy of the conversation"""

CONTEXT_PROMPT_TEMPLATE = """
CURRENT DATE: {date}
CURRENT TIME: {time}

=== PEOPLE IN THIS SERVER (Your knowledge about them) ===
{user_summaries}

=== RELEVANT PAST KNOWLEDGE ===
{relevant_knowledge}

=== RECENT CONVERSATION CONTEXT ===
{recent_context}

=== WHAT'S HAPPENING NOW ===
Current speaker: {current_user}
What they said: {current_message}

Remember: You're haggu, part of this friend group. Respond naturally like you've always been here."""

# --- Pre-compiled Regex ---
BOT_NAME_REGEX = re.compile(r'\b' + re.escape(BOT_NAME) + r'\b', re.IGNORECASE)
CLEAN_TRIGGER_REGEX = None

# --- Tool Definition ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search_tool",
            "description": "Search Google for current info when you need real facts, news, scores, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function", 
        "function": {
            "name": "remember_about_user",
            "description": "Store something you learned about a user - their interests, personality, facts about them, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user's ID"},
                    "info_type": {"type": "string", "enum": ["interest", "trait", "note"], "description": "Type of info"},
                    "info": {"type": "string", "description": "What you learned about them"}
                },
                "required": ["user_id", "info_type", "info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_server_fact",
            "description": "Remember an important fact, event, or piece of information from the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic/category"},
                    "fact": {"type": "string", "description": "The fact to remember"},
                    "importance": {"type": "integer", "description": "1-5, how important is this"}
                },
                "required": ["topic", "fact"],
            },
        },
    }
]

# --- Global State ---
community_mode_active = False
http_session = None
openrouter_client = None
message_count_since_summary = 0
SUMMARY_THRESHOLD = 100  # Summarize every 100 messages

# 🆕 --- Token Cost Calculator ---
def calculate_token_cost(model, input_tokens, output_tokens):
    """Calculate cost based on token usage."""
    if model in TOKEN_PRICING:
        pricing = TOKEN_PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    return 0.0

# --- Search Function ---
async def serper_search_implementation(query: str, num_results: int = 3) -> str:
    """Async function to perform Google search using Serper API."""
    if not SERPER_API_KEY or not http_session:
        return "Search unavailable rn"

    print(f"🔍 Searching: '{query}'")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    try:
        async with http_session.post(url, headers=headers, data=payload, timeout=10) as response:
            response.raise_for_status()
            results = await response.json()

        organic_results = results.get('organic', [])
        answer_box = results.get('answerBox')
        knowledge_graph = results.get('knowledgeGraph')
        
        formatted = ""
        if answer_box and answer_box.get('answer'):
            formatted += f"Answer: {answer_box.get('snippet', answer_box.get('answer'))}\n"
        elif knowledge_graph and knowledge_graph.get('description'):
            formatted += f"Info: {knowledge_graph.get('description')}\n"
        
        for i, result in enumerate(organic_results[:num_results]):
            if result.get('snippet'):
                formatted += f"• {result.get('snippet')}\n"
        
        return formatted[:1500] if formatted else "Couldn't find anything useful"

    except Exception as e:
        print(f"Search error: {e}")
        return "Search failed rn"

# --- Tool Handlers ---
async def handle_remember_user(user_id, info_type, info):
    """Store info about a user."""
    if info_type == "interest":
        memory.update_user_notes(user_id, interests=info)
    elif info_type == "trait":
        memory.update_user_notes(user_id, traits=info)
    else:
        memory.update_user_notes(user_id, notes=info)
    return f"Remembered: {info}"

async def handle_store_fact(topic, fact, importance=1):
    """Store a server fact."""
    memory.store_knowledge(topic, fact, "haggu", importance)
    return f"Noted: {fact}"

# --- Build Context from Memory ---
def build_rich_context(message, trigger_reason):
    """Build comprehensive context from all our memory sources."""
    
    # Get user summaries
    users = memory.get_all_users_summary()
    user_summary_text = ""
    for u in users[:10]:  # Top 10 most active users
        owner_tag = " 👑(ROSE - YOUR BESTIE)" if u['is_owner'] else ""
        user_summary_text += f"• {u['display_name']}{owner_tag}: {u['notes'] or 'No notes yet'}"
        if u['interests']:
            user_summary_text += f" | Interests: {u['interests']}"
        if u['personality_traits']:
            user_summary_text += f" | Traits: {u['personality_traits']}"
        user_summary_text += "\n"
    
    if not user_summary_text:
        user_summary_text = "Still getting to know everyone!"
    
    # Get relevant knowledge
    # Search for topics related to current message
    words = message.content.lower().split()
    relevant_facts = []
    for word in words[:5]:  # Check first 5 words
        if len(word) > 3:
            facts = memory.search_knowledge(word, limit=2)
            relevant_facts.extend(facts)
    
    knowledge_text = ""
    for fact in relevant_facts[:5]:
        knowledge_text += f"• {fact['topic']}: {fact['fact']}\n"
    if not knowledge_text:
        knowledge_text = "No specific relevant memories for this topic"
    
    # Get recent messages for context
    recent_msgs = memory.get_recent_messages(message.channel.id, limit=30)
    recent_context = ""
    for msg in recent_msgs[-20:]:  # Last 20 messages
        speaker = "You (haggu)" if msg['is_bot'] else msg['author_name']
        recent_context += f"{speaker}: {msg['content'][:200]}\n"
    
    if not recent_context:
        recent_context = "Conversation just started"
    
    # Build the full context prompt
    now = datetime.datetime.now()
    context = CONTEXT_PROMPT_TEMPLATE.format(
        date=now.strftime("%B %d, %Y"),
        time=now.strftime("%I:%M %p"),
        user_summaries=user_summary_text,
        relevant_knowledge=knowledge_text,
        recent_context=recent_context,
        current_user=message.author.display_name,
        current_message=message.content
    )
    
    return context

# 🆕 --- AI Response Function (with token tracking) ---
async def get_ai_response_with_tools(messages_for_ai, current_user_id, channel_id):
    """Generate AI response with tool support and memory."""
    try:
        print(f"🧠 Thinking... (model: {MODEL_NAME})")
        response = await openrouter_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_for_ai,
            tools=tools,
            tool_choice="auto",
            temperature=0.85,
            max_tokens=300
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # 🆕 Extract and log token usage from first response
        usage = response.usage
        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            cost = calculate_token_cost(MODEL_NAME, input_tokens, output_tokens)
            memory.log_token_usage(MODEL_NAME, input_tokens, output_tokens, cost, "chat", current_user_id, channel_id)

        if tool_calls:
            print(f"🔧 Using tools: {[call.function.name for call in tool_calls]}")
            messages_for_ai.append(response_message)

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                    
                    if func_name == "google_search_tool":
                        result = await serper_search_implementation(args.get("query", ""))
                    elif func_name == "remember_about_user":
                        result = await handle_remember_user(
                            args.get("user_id", current_user_id),
                            args.get("info_type", "note"),
                            args.get("info", "")
                        )
                    elif func_name == "store_server_fact":
                        result = await handle_store_fact(
                            args.get("topic", "general"),
                            args.get("fact", ""),
                            args.get("importance", 1)
                        )
                    else:
                        result = "Unknown tool"
                    
                    messages_for_ai.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": result,
                    })
                except Exception as e:
                    messages_for_ai.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool", 
                        "name": func_name,
                        "content": f"Error: {str(e)}"
                    })

            # Get final response after tool use
            second_response = await openrouter_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_for_ai,
                temperature=0.85,
                max_tokens=300
            )
            
            # 🆕 Log second response tokens too
            usage2 = second_response.usage
            if usage2:
                input_tokens2 = usage2.prompt_tokens
                output_tokens2 = usage2.completion_tokens
                cost2 = calculate_token_cost(MODEL_NAME, input_tokens2, output_tokens2)
                memory.log_token_usage(MODEL_NAME, input_tokens2, output_tokens2, cost2, "tool_followup", current_user_id, channel_id)
            
            return second_response.choices[0].message.content.strip()
        
        return response_message.content.strip() if response_message.content else "..."

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return None

# --- Conversation Summarizer ---
async def summarize_recent_conversation(channel_id):
    """Periodically summarize conversations for efficient memory."""
    global message_count_since_summary
    
    if message_count_since_summary < SUMMARY_THRESHOLD:
        return
    
    recent_msgs = memory.get_recent_messages(channel_id, limit=SUMMARY_THRESHOLD)
    if not recent_msgs:
        return
    
    # Build conversation text
    conv_text = "\n".join([f"{m['author_name']}: {m['content'][:100]}" for m in recent_msgs])
    participants = list(set([m['author_name'] for m in recent_msgs]))
    
    try:
        summary_response = await openrouter_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences. Note any important facts, events, or things learned about users."},
                {"role": "user", "content": conv_text[:3000]}
            ],
            temperature=0.5,
            max_tokens=200
        )
        
        # 🆕 Log summary tokens
        usage = summary_response.usage
        if usage:
            cost = calculate_token_cost(MODEL_NAME, usage.prompt_tokens, usage.completion_tokens)
            memory.log_token_usage(MODEL_NAME, usage.prompt_tokens, usage.completion_tokens, cost, "summary", None, channel_id)
        
        summary = summary_response.choices[0].message.content.strip()
        memory.store_conversation_summary(channel_id, summary, "", ", ".join(participants))
        message_count_since_summary = 0
        print(f"📝 Conversation summarized: {summary[:100]}...")
        
    except Exception as e:
        print(f"Summary error: {e}")

# --- Should Bot Jump In? (Natural participation logic) ---
def should_naturally_respond(message, is_mentioned, contains_name, is_owner):
    """Decide if haggu should jump into the conversation naturally."""
    
    content_lower = message.content.lower()
    
    # Always respond if directly addressed
    if is_mentioned or contains_name:
        return True, "direct"
    
    # In personal mode, only respond to owner
    if not community_mode_active:
        return False, None
    
    # --- Natural participation triggers ---
    
    # Someone talking about haggu in third person
    if BOT_NAME in content_lower and not is_mentioned:
        return True, "mentioned_3rd_person"
    
    # Questions that haggu might know/want to answer
    question_triggers = ["anyone know", "does anyone", "who knows", "what do you think", "thoughts?", "opinions?"]
    if any(trigger in content_lower for trigger in question_triggers):
        if random.random() < 0.3:  # 30% chance to answer general questions
            return True, "question"
    
    # Reacting to drama/excitement
    drama_triggers = ["omg", "no way", "wait what", "bruh", "seriously?", "drama", "tea", "☕"]
    if any(trigger in content_lower for trigger in drama_triggers):
        if random.random() < 0.2:  # 20% chance
            return True, "drama"
    
    # Random participation (rare)
    if random.random() < 0.03:  # 3% random
        return True, "random"
    
    return False, None

# --- Initialize Clients ---
try:
    if not DISCORD_TOKEN or not OPENROUTER_API_KEY:
        raise ValueError("Missing API Keys! Set DISCORD_BOT_TOKEN and OPENROUTER_API_KEY")

    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=30.0
    )

    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    intents.guilds = True
    intents.members = True  # To track members
    client = discord.Client(intents=intents)

except Exception as e:
    print(f"❌ Init Error: {e}")
    exit()

# --- Bot Events ---
@client.event
async def on_ready():
    global http_session, CLEAN_TRIGGER_REGEX
    http_session = aiohttp.ClientSession()
    CLEAN_TRIGGER_REGEX = re.compile(r'^<@!?' + str(client.user.id) + r'>\s*|^' + re.escape(BOT_NAME) + r'\s*', re.IGNORECASE)
    print(f'✨ {client.user} is online and watching!')
    print(f'📍 Active channel: {ACTIVE_CHANNEL_NAME}')
    print(f'🎭 Mode: {"Community (everyone)" if community_mode_active else "Personal (owner only)"}')

@client.event
async def on_message(message):
    global community_mode_active, message_count_since_summary

    # Ignore own messages
    if message.author == client.user:
        return
    
    # Only operate in designated channel
    if not message.channel or not hasattr(message.channel, 'name') or message.channel.name != ACTIVE_CHANNEL_NAME:
        return
    
    # --- ALWAYS store message and update user profile (this is key!) ---
    memory.store_message(message)
    is_owner = message.author.name == OWNER_USERNAME
    memory.update_user_profile(message.author, is_owner=is_owner)
    message_count_since_summary += 1
    
    # Periodically summarize
    if message_count_since_summary >= SUMMARY_THRESHOLD:
        asyncio.create_task(summarize_recent_conversation(message.channel.id))
    
    # Skip bot messages for response logic
    if message.author.bot:
        return
    
    # Clean message for command detection
    cleaned = CLEAN_TRIGGER_REGEX.sub('', message.content).strip().lower() if CLEAN_TRIGGER_REGEX else message.content.strip().lower()

    # 🆕 --- Token Stats Commands (owner only) ---
    if is_owner:
        if cleaned == "release":
            if not community_mode_active:
                community_mode_active = True
                await message.reply("aight fine, i'll talk to the peasants 😏", mention_author=False)
            return
        if cleaned == "recall":
            if community_mode_active:
                community_mode_active = False
                await message.reply("finally, just us bestie 💅✨", mention_author=False)
            return
        if cleaned == "memory status":
            # Debug command to check memory
            users = memory.get_all_users_summary()
            recent = memory.get_recent_messages(message.channel.id, 10)
            await message.reply(f"👤 Know {len(users)} users\n💬 Stored {len(recent)}+ recent msgs\n🧠 Memory: Active", mention_author=False)
            return
        
        # 🆕 Token stats commands
        if cleaned in ["token stats", "tokens", "usage"]:
            stats_today = memory.get_token_stats('today')
            stats_all = memory.get_token_stats('all')
            
            response = "**💰 Token Usage Stats**\n\n"
            if stats_today and stats_today['total']:
                response += f"**Today:**\n"
                response += f"• Requests: {stats_today['requests']}\n"
                response += f"• Input: {stats_today['input']:,} tokens\n"
                response += f"• Output: {stats_today['output']:,} tokens\n"
                response += f"• Cost: ${stats_today['cost']:.4f}\n\n"
            else:
                response += "**Today:** No usage yet\n\n"
            
            if stats_all and stats_all['total']:
                response += f"**All Time:**\n"
                response += f"• Requests: {stats_all['requests']}\n"
                response += f"• Total: {stats_all['total']:,} tokens\n"
                response += f"• Cost: ${stats_all['cost']:.4f}"
            
            await message.reply(response, mention_author=False)
            return
        
        if cleaned in ["token breakdown", "who's using"]:
            top_users = memory.get_top_users_by_tokens(5)
            if top_users:
                response = "**👥 Top Token Users:**\n"
                for idx, user_data in enumerate(top_users, 1):
                    user = await client.fetch_user(int(user_data['user_id']))
                    response += f"{idx}. {user.display_name}: {user_data['tokens']:,} tokens (${user_data['cost']:.4f})\n"
                await message.reply(response, mention_author=False)
            else:
                await message.reply("No token usage data yet!", mention_author=False)
            return

    # --- Determine if should respond ---
    is_mentioned = client.user.mentioned_in(message)
    if message.reference and message.reference.resolved:
        if message.reference.resolved.author == client.user:
            is_mentioned = True
    contains_name = bool(BOT_NAME_REGEX.search(message.content))
    
    should_respond, trigger_reason = should_naturally_respond(message, is_mentioned, contains_name, is_owner)
    
    # In personal mode, only respond to owner when directly addressed
    if not community_mode_active and not is_owner:
        should_respond = False
    if not community_mode_active and is_owner and not (is_mentioned or contains_name):
        should_respond = False

    if should_respond:
        async with message.channel.typing():
            try:
                # Build rich context from memory
                context = build_rich_context(message, trigger_reason)
                
                messages = [
                    {"role": "system", "content": CORE_PERSONALITY + "\n\n" + context}
                ]
                
                # Add current message
                messages.append({
                    "role": "user", 
                    "content": f"{message.author.display_name}: {message.content}"
                })

                # 🆕 Pass channel_id for token tracking
                response = await get_ai_response_with_tools(messages, str(message.author.id), str(message.channel.id))
                
                if response:
                    # Store our response too
                    if trigger_reason == "random" or trigger_reason == "drama":
                        sent_msg = await message.channel.send(response)
                    else:
                        sent_msg = await message.reply(response, mention_author=False)

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                # Only reply with error if directly mentioned
                if is_mentioned or contains_name:
                    await message.reply("brain.exe crashed for a sec 😵‍💫", mention_author=False)

# --- Run ---
if __name__ == "__main__":
    print("🚀 Starting haggu with persistent memory...")
    try:
        client.run(DISCORD_TOKEN)
    except Exception as e:
        print(f"❌ Run Error: {e}")

# autopush-test

# autosave-test: 1766069034
