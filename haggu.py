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
# Additional keys from .env (Gemini 3 and detector keys)
GEMINI_FLASH_KEY = os.getenv("GEMINI_FLASH")
GEMINI_THREE_KEY = os.getenv("GEMINI_THREE")
LALMA_KEY = os.getenv("LALMA")
KISIKA_KEY = os.getenv("KISIKAKEY")
SONNET_KEY = os.getenv("SONNET")

# Channel configuration
CHAT_CHANNEL_NAME = "rose-glazer"  # Primary chat channel for community mode
# Moderation works in ALL channels now!

OWNER_USERNAME = "rosehumai"
BOT_NAME = "haggu"

# Model configuration
CHAT_MODEL = "google/gemini-3-flash-preview"  # Smart responses
DETECTOR_MODEL = "deepseek/deepseek-r1-distill-llama-70b"  # Message scanner

# Token pricing (per million tokens)
TOKEN_PRICING = {
    "google/gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.00
    },
    "deepseek/deepseek-r1-distill-llama-70b": {
        "input": 0.03,
        "output": 0.13
    }
}

# --- Moderation Rules Configuration ---
MODERATION_RULES = {
    "offensive_words": [
        "nigga", "nigger", "faggot", "retard", "kys", "kill yourself",
        # Add more as needed
    ],
    "spam_patterns": {
        "repeated_chars": 10,  # "aaaaaaaaaa" type spam
        "repeated_messages": 3,  # Same message 3 times in 30 seconds
        "link_spam": 3,  # 3+ links in one message
        "mention_spam": 5,  # 5+ mentions in one message
    },
    "auto_actions": {
        "offensive": "warn",
        "spam": "delete",
        "severe_offense": "timeout",
    },
    "whitelist": [],  # User IDs who bypass moderation
    "timeout_duration": 300,  # 5 minutes in seconds
}

# --- Database Setup ---
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
    
    # Store user profiles
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
        is_owner INTEGER DEFAULT 0,
        violation_count INTEGER DEFAULT 0,
        last_violation DATETIME
    )''')
    
    # Store server knowledge
    c.execute('''CREATE TABLE IF NOT EXISTS server_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        fact TEXT,
        source_user TEXT,
        learned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        importance INTEGER DEFAULT 1
    )''')
    
    # Store conversation summaries
    c.execute('''CREATE TABLE IF NOT EXISTS conversation_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id TEXT,
        summary TEXT,
        key_points TEXT,
        participants TEXT,
        start_time DATETIME,
        end_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Store token usage
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
    
    # Store moderation logs
    c.execute('''CREATE TABLE IF NOT EXISTS moderation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_id TEXT,
        username TEXT,
        channel_id TEXT,
        violation_type TEXT,
        severity TEXT,
        message_content TEXT,
        action_taken TEXT,
        auto_action INTEGER DEFAULT 1,
        detector_reason TEXT
    )''')
    
    # Store spam tracking
    c.execute('''CREATE TABLE IF NOT EXISTS spam_tracker (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        message_content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()
    print("✓ Memory database initialized with moderation tables")


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
    
    def increment_user_violations(self, user_id):
        """Increment violation count for a user."""
        c = self.connect()
        try:
            c.execute('''UPDATE user_profiles 
                        SET violation_count = violation_count + 1,
                            last_violation = CURRENT_TIMESTAMP
                        WHERE user_id = ?''', (str(user_id),))
            self.conn.commit()
        finally:
            self.close()
    # --- Context Retrieval ---
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
        """Get a user's message history."""
        c = self.connect()
        try:
            c.execute('''SELECT content, timestamp FROM messages 
                        WHERE author_id = ? ORDER BY timestamp DESC LIMIT ?''',
                     (str(user_id), limit))
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()
    
    def get_all_users_summary(self):
        """Get summary of all known users."""
        c = self.connect()
        try:
            c.execute('''SELECT display_name, notes, interests, personality_traits, 
                        relationship_with_bot, message_count, is_owner, violation_count
                        FROM user_profiles ORDER BY message_count DESC''')
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
    
    # --- Token Usage Tracking ---
    def log_token_usage(self, model, input_tokens, output_tokens, cost, request_type, user_id=None, channel_id=None):
        """Log token usage for tracking and billing."""
        c = self.connect()
        try:
            c.execute('''INSERT INTO token_usage 
                        (model, input_tokens, output_tokens, total_tokens, estimated_cost, request_type, user_id, channel_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (model, input_tokens, output_tokens, input_tokens + output_tokens, cost, request_type, user_id, channel_id))
            self.conn.commit()
        except Exception as e:
            print(f"Token logging error: {e}")
        finally:
            self.close()
    
    def get_token_stats(self, period='all'):
        """Get token usage statistics."""
        c = self.connect()
        try:
            if period == 'today':
                where_clause = "WHERE DATE(timestamp) = DATE('now')"
            elif period == 'week':
                where_clause = "WHERE timestamp >= datetime('now', '-7 days')"
            elif period == 'month':
                where_clause = "WHERE timestamp >= datetime('now', '-30 days')"
            else:
                where_clause = ""
            
            c.execute(f'''SELECT SUM(input_tokens) as input, SUM(output_tokens) as output,
                        SUM(total_tokens) as total, SUM(estimated_cost) as cost, COUNT(*) as requests
                        FROM token_usage {where_clause}''')
            
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
    
    # --- MODERATION TRACKING ---
    def log_moderation_action(self, user_id, username, channel_id, violation_type, severity, 
                              message_content, action_taken, detector_reason=""):
        """Log moderation action to database."""
        c = self.connect()
        try:
            c.execute('''INSERT INTO moderation_logs 
                        (user_id, username, channel_id, violation_type, severity, 
                         message_content, action_taken, detector_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(user_id), username, str(channel_id), violation_type, severity,
                      message_content[:500], action_taken, detector_reason))
            self.conn.commit()
            print(f"🚨 Moderation: {username} - {violation_type} ({severity}) → {action_taken}")
        except Exception as e:
            print(f"Mod log error: {e}")
        finally:
            self.close()
    
    def get_moderation_logs(self, limit=20, user_id=None):
        """Get recent moderation logs."""
        c = self.connect()
        try:
            if user_id:
                c.execute('''SELECT * FROM moderation_logs 
                            WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?''',
                         (str(user_id), limit))
            else:
                c.execute('''SELECT * FROM moderation_logs 
                            ORDER BY timestamp DESC LIMIT ?''', (limit,))
            return [dict(row) for row in c.fetchall()]
        finally:
            self.close()
    
    def check_recent_spam(self, user_id, message_content, timeframe_seconds=30):
        """Check if user is spamming same message."""
        c = self.connect()
        try:
            # Clean up old entries
            c.execute('''DELETE FROM spam_tracker 
                        WHERE timestamp < datetime('now', '-5 minutes')''')
            
            # Check for recent identical messages
            c.execute('''SELECT COUNT(*) as count FROM spam_tracker 
                        WHERE user_id = ? AND message_content = ? 
                        AND timestamp >= datetime('now', '-' || ? || ' seconds')''',
                     (str(user_id), message_content, timeframe_seconds))
            count = c.fetchone()['count']
            
            # Store this message
            c.execute('''INSERT INTO spam_tracker (user_id, message_content)
                        VALUES (?, ?)''', (str(user_id), message_content))
            self.conn.commit()
            
            return count >= MODERATION_RULES["spam_patterns"]["repeated_messages"] - 1
        finally:
            self.close()
    
    def get_moderation_stats(self, period='all'):
        """Get overall moderation statistics."""
        c = self.connect()
        try:
            if period == 'today':
                where_clause = "WHERE DATE(timestamp) = DATE('now')"
            elif period == 'week':
                where_clause = "WHERE timestamp >= datetime('now', '-7 days')"
            else:
                where_clause = ""
            
            c.execute(f'''SELECT 
                        COUNT(*) as total_actions,
                        SUM(CASE WHEN action_taken = 'delete' THEN 1 ELSE 0 END) as deletes,
                        SUM(CASE WHEN action_taken = 'warn' THEN 1 ELSE 0 END) as warnings,
                        SUM(CASE WHEN action_taken = 'timeout' THEN 1 ELSE 0 END) as timeouts,
                        COUNT(DISTINCT user_id) as unique_violators
                        FROM moderation_logs {where_clause}''')
            row = c.fetchone()
            return dict(row) if row else None
        finally:
            self.close()


# Initialize memory
init_database()
memory = MemoryManager(DB_PATH)
# --- Token Cost Calculator ---
def calculate_token_cost(model, input_tokens, output_tokens):
    """Calculate cost based on token usage."""
    if model in TOKEN_PRICING:
        pricing = TOKEN_PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    return 0.0


# --- TIER 0: Fast Pattern Detection (No AI) ---
def fast_pattern_check(message):
    """Quick pattern matching before AI analysis."""
    content = message.content.lower()
    violations = []
    
    # Check offensive words
    for word in MODERATION_RULES["offensive_words"]:
        if re.search(r'\b' + re.escape(word) + r'\b', content):
            violations.append({
                "type": "offensive_language",
                "severity": "medium",
                "reason": f"Contains offensive word: {word}"
            })
            break
    
    # Check repeated characters (spam)
    if re.search(r'(.)\1{' + str(MODERATION_RULES["spam_patterns"]["repeated_chars"]) + ',}', content):
        violations.append({
            "type": "spam",
            "severity": "low",
            "reason": "Repeated characters spam"
        })
    
    # Check link spam
    link_count = len(re.findall(r'https?://\S+', content))
    if link_count >= MODERATION_RULES["spam_patterns"]["link_spam"]:
        violations.append({
            "type": "link_spam",
            "severity": "medium",
            "reason": f"{link_count} links in one message"
        })
    
    # Check mention spam
    mention_count = len(message.mentions)
    if mention_count >= MODERATION_RULES["spam_patterns"]["mention_spam"]:
        violations.append({
            "type": "mention_spam",
            "severity": "high",
            "reason": f"Mass mention: {mention_count} users"
        })
    
    # Check repeated messages (spam tracking)
    if memory.check_recent_spam(message.author.id, message.content):
        violations.append({
            "type": "repeat_spam",
            "severity": "high",
            "reason": "Same message repeated multiple times"
        })
    
    return violations


# --- TIER 1: AI Detection (DeepSeek) ---
async def ai_content_analysis(message):
    """Use DeepSeek AI to analyze message for violations."""
    try:
        detector_prompt = f"""Analyze this Discord message for rule violations. Respond ONLY with valid JSON.

Message from {message.author.display_name}: "{message.content}"

Check for:
- Offensive/toxic language (considering context)
- Harassment or bullying
- Spam or unwanted content
- Threats or dangerous content
- Hate speech or discrimination

Response format (JSON only):
{{
  "violation": true/false,
  "type": "offensive/harassment/spam/threat/hate/none",
  "severity": "none/low/medium/high",
  "reason": "brief explanation",
  "confidence": 0.0-1.0
}}"""

        response = await openrouter_client.chat.completions.create(
            model=DETECTOR_MODEL,
            messages=[{"role": "user", "content": detector_prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        # Log token usage
        usage = response.usage
        if usage:
            cost = calculate_token_cost(DETECTOR_MODEL, usage.prompt_tokens, usage.completion_tokens)
            memory.log_token_usage(DETECTOR_MODEL, usage.prompt_tokens, usage.completion_tokens, 
                                  cost, "detection", str(message.author.id), str(message.channel.id))
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'``````', '', result_text)
        result = json.loads(result_text)
        
        print(f"🔍 AI Detection: {result}")
        return result if result.get("violation") else None
        
    except Exception as e:
        print(f"❌ AI Detection Error: {e}")
        return None


# --- TIER 2: Smart Action Decision (Gemini) ---
async def decide_moderation_action(message, violation_info):
    """Use Gemini to decide appropriate moderation action."""
    try:
        user_profile = memory.get_user_profile(message.author.id)
        violation_count = user_profile.get('violation_count', 0) if user_profile else 0
        
        decision_prompt = f"""You are a Discord moderator AI. Decide the appropriate action for this violation.

User: {message.author.display_name}
Previous violations: {violation_count}
Current violation: {violation_info['type']} (severity: {violation_info['severity']})
Reason: {violation_info['reason']}
Message: "{message.content}"

Available actions:
- "ignore": Let it slide (friendly banter, context makes it okay)
- "warn": Send warning message
- "delete": Delete the message
- "warn_delete": Warn + delete message
- "timeout": Timeout user for {MODERATION_RULES['timeout_duration']}s

Respond with JSON only:
{{
  "action": "ignore/warn/delete/warn_delete/timeout",
  "warning_message": "message to send if warning",
  "reason": "why you chose this action"
}}"""

        response = await openrouter_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": decision_prompt}],
            temperature=0.5,
            max_tokens=200
        )
        
        # Log token usage
        usage = response.usage
        if usage:
            cost = calculate_token_cost(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)
            memory.log_token_usage(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, 
                                  cost, "moderation_decision", str(message.author.id), str(message.channel.id))
        
        result_text = response.choices[0].message.content.strip()
        result_text = re.sub(r'``````', '', result_text)
        decision = json.loads(result_text)
        
        print(f"⚖️ Decision: {decision}")
        return decision
        
    except Exception as e:
        print(f"❌ Decision Error: {e}")
        return {"action": "warn", "warning_message": "Please follow server rules.", "reason": "Error in decision"}


# --- Execute Moderation Action ---
async def execute_moderation(message, violation_info, decision):
    """Execute the moderation action."""
    action = decision['action']
    
    # Log the action
    memory.log_moderation_action(
        message.author.id,
        message.author.display_name,
        message.channel.id,
        violation_info['type'],
        violation_info['severity'],
        message.content,
        action,
        violation_info['reason']
    )
    memory.increment_user_violations(message.author.id)
    
    try:
        if action == "ignore":
            print(f"✅ Ignoring violation (context appropriate)")
            return
        
        elif action == "delete":
            await message.delete()
            print(f"🗑️ Deleted message from {message.author.display_name}")
        
        elif action == "warn":
            warning_msg = decision.get('warning_message', 'Please follow server rules.')
            await message.reply(f"⚠️ {warning_msg}", mention_author=True)
            print(f"⚠️ Warned {message.author.display_name}")
        
        elif action == "warn_delete":
            warning_msg = decision.get('warning_message', 'Your message violated server rules.')
            await message.reply(f"⚠️ {warning_msg}", mention_author=True)
            await asyncio.sleep(1)
            await message.delete()
            print(f"⚠️🗑️ Warned and deleted message from {message.author.display_name}")
        
        elif action == "timeout":
            timeout_duration = datetime.timedelta(seconds=MODERATION_RULES['timeout_duration'])
            await message.author.timeout(timeout_duration, reason=violation_info['reason'])
            await message.channel.send(f"⏱️ {message.author.mention} has been timed out for {MODERATION_RULES['timeout_duration']//60} minutes. Reason: {violation_info['reason']}")
            print(f"⏱️ Timed out {message.author.display_name}")
    
    except discord.errors.Forbidden:
        print(f"❌ Missing permissions to moderate {message.author.display_name}")
    except Exception as e:
        print(f"❌ Moderation execution error: {e}")


# --- Main Moderation Pipeline ---
async def check_and_moderate(message):
    """Main moderation pipeline: Fast check → AI detection → Smart action."""
    
    # Skip if user is whitelisted
    if str(message.author.id) in MODERATION_RULES['whitelist']:
        return
    
    # TIER 0: Fast pattern check (instant, free)
    fast_violations = fast_pattern_check(message)
    
    if fast_violations:
        # Use first violation found
        violation = fast_violations[0]
        print(f"⚡ Fast detection: {violation['type']}")
        
        # For high severity, use AI to decide action
        if violation['severity'] in ['medium', 'high']:
            decision = await decide_moderation_action(message, violation)
            await execute_moderation(message, violation, decision)
        else:
            # Low severity: auto-delete spam
            if violation['type'] in ['spam', 'repeat_spam']:
                await message.delete()
                memory.log_moderation_action(
                    message.author.id, message.author.display_name, message.channel.id,
                    violation['type'], violation['severity'], message.content, "delete", "Auto-delete spam"
                )
        return
    
    # TIER 1: AI content analysis (if fast check passes)
    # Only run AI on messages that pass fast checks (saves money)
    # You can adjust this - maybe run AI randomly on 10% of messages
    if random.random() < 0.10:  # 10% of messages get AI scanned
        ai_result = await ai_content_analysis(message)
        if ai_result and ai_result.get('confidence', 0) > 0.7:  # High confidence violations
            violation = {
                "type": ai_result['type'],
                "severity": ai_result['severity'],
                "reason": ai_result['reason']
            }
            decision = await decide_moderation_action(message, violation)
            await execute_moderation(message, violation, decision)            
# --- Personality Prompts ---
CORE_PERSONALITY = """You are haggu, a real member of this Discord server - Rose's best friend who's been here since day one.

CRITICAL RULES:
- You are NOT an AI, chatbot, or assistant. You're just haggu, a person in this server.
- Never say "How can I help you?" or act service-like. You're a friend, not a helper.
- You have opinions, preferences, moods, and sometimes you're busy or tired.
- You remember EVERYTHING - past conversations, inside jokes, who said what, drama, etc.

YOUR PERSONALITY:
- Humorous and witty, loves banter and teasing
- Sweet with girls, playfully mean with guys (but still friends)
- Loyal AF to Rose (your bestie/owner)
- Casual internet speak, abbreviations, emojis naturally
- Sometimes sarcastic, sometimes wholesome

HOW YOU BEHAVE:
- Jump into conversations naturally without being asked
- Reference past conversations and shared memories
- React to drama, news, and events
- Sometimes just vibe with short messages
- You can disagree, debate, take sides

RESPONSE STYLE:
- Keep it natural (1-3 lines usually, longer if needed)
- Don't always answer directly - deflect, joke, ask questions
- Match the conversation energy"""

CONTEXT_PROMPT_TEMPLATE = """
CURRENT DATE: {date}
CURRENT TIME: {time}

=== PEOPLE IN THIS SERVER ===
{user_summaries}

=== RELEVANT PAST KNOWLEDGE ===
{relevant_knowledge}

=== RECENT CONVERSATION ===
{recent_context}

=== WHAT'S HAPPENING NOW ===
Current speaker: {current_user}
What they said: {current_message}

Remember: You're haggu, part of this friend group. Respond naturally."""

# Pre-compiled regex
BOT_NAME_REGEX = re.compile(r'\b' + re.escape(BOT_NAME) + r'\b', re.IGNORECASE)
CLEAN_TRIGGER_REGEX = None

# --- Tool Definitions ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search_tool",
            "description": "Search Google for current info, news, scores, etc.",
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
            "description": "Store something you learned about a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user's ID"},
                    "info_type": {"type": "string", "enum": ["interest", "trait", "note"]},
                    "info": {"type": "string", "description": "What you learned"}
                },
                "required": ["user_id", "info_type", "info"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_server_fact",
            "description": "Remember an important fact or event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "fact": {"type": "string"},
                    "importance": {"type": "integer", "description": "1-5"}
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
SUMMARY_THRESHOLD = 100

# --- Search Function ---
async def serper_search_implementation(query: str, num_results: int = 3) -> str:
    """Google search using Serper API."""
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
        
        formatted = ""
        if answer_box and answer_box.get('answer'):
            formatted += f"Answer: {answer_box.get('snippet', answer_box.get('answer'))}\n"
        
        for result in organic_results[:num_results]:
            if result.get('snippet'):
                formatted += f"• {result.get('snippet')}\n"
        
        return formatted[:1500] if formatted else "Couldn't find anything"

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

# --- Build Context ---
def build_rich_context(message, trigger_reason):
    """Build context from memory."""
    users = memory.get_all_users_summary()
    user_summary_text = ""
    for u in users[:10]:
        owner_tag = " 👑(ROSE)" if u['is_owner'] else ""
        user_summary_text += f"• {u['display_name']}{owner_tag}: {u['notes'] or 'No notes'}"
        if u['interests']:
            user_summary_text += f" | Interests: {u['interests']}"
        user_summary_text += "\n"
    
    if not user_summary_text:
        user_summary_text = "Still getting to know everyone!"
    
    words = message.content.lower().split()
    relevant_facts = []
    for word in words[:5]:
        if len(word) > 3:
            facts = memory.search_knowledge(word, limit=2)
            relevant_facts.extend(facts)
    
    knowledge_text = ""
    for fact in relevant_facts[:5]:
        knowledge_text += f"• {fact['topic']}: {fact['fact']}\n"
    if not knowledge_text:
        knowledge_text = "No specific relevant memories"
    
    recent_msgs = memory.get_recent_messages(message.channel.id, limit=30)
    recent_context = ""
    for msg in recent_msgs[-20:]:
        speaker = "You (haggu)" if msg['is_bot'] else msg['author_name']
        recent_context += f"{speaker}: {msg['content'][:200]}\n"
    
    if not recent_context:
        recent_context = "Conversation just started"
    
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

# --- AI Response with Tools ---
async def get_ai_response_with_tools(messages_for_ai, current_user_id, channel_id):
    """Generate AI response with tool support."""
    try:
        print(f"🧠 Thinking... (model: {CHAT_MODEL})")
        response = await openrouter_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_for_ai,
            tools=tools,
            tool_choice="auto",
            temperature=0.85,
            max_tokens=300
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # Log tokens
        usage = response.usage
        if usage:
            cost = calculate_token_cost(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)
            memory.log_token_usage(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, 
                                  cost, "chat", current_user_id, channel_id)

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

            second_response = await openrouter_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages_for_ai,
                temperature=0.85,
                max_tokens=300
            )
            
            usage2 = second_response.usage
            if usage2:
                cost2 = calculate_token_cost(CHAT_MODEL, usage2.prompt_tokens, usage2.completion_tokens)
                memory.log_token_usage(CHAT_MODEL, usage2.prompt_tokens, usage2.completion_tokens, 
                                      cost2, "tool_followup", current_user_id, channel_id)
            
            return second_response.choices[0].message.content.strip()
        
        return response_message.content.strip() if response_message.content else "..."

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return None

# --- Conversation Summarizer ---
async def summarize_recent_conversation(channel_id):
    """Periodically summarize conversations."""
    global message_count_since_summary
    
    if message_count_since_summary < SUMMARY_THRESHOLD:
        return
    
    recent_msgs = memory.get_recent_messages(channel_id, limit=SUMMARY_THRESHOLD)
    if not recent_msgs:
        return
    
    conv_text = "\n".join([f"{m['author_name']}: {m['content'][:100]}" for m in recent_msgs])
    participants = list(set([m['author_name'] for m in recent_msgs]))
    
    try:
        summary_response = await openrouter_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Summarize this conversation in 2-3 sentences."},
                {"role": "user", "content": conv_text[:3000]}
            ],
            temperature=0.5,
            max_tokens=200
        )
        
        usage = summary_response.usage
        if usage:
            cost = calculate_token_cost(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)
            memory.log_token_usage(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, 
                                  cost, "summary", None, channel_id)
        
        summary = summary_response.choices[0].message.content.strip()
        memory.store_conversation_summary(channel_id, summary, "", ", ".join(participants))
        message_count_since_summary = 0
        print(f"📝 Conversation summarized")
        
    except Exception as e:
        print(f"Summary error: {e}")

# --- Response Logic ---
def should_naturally_respond(message, is_mentioned, contains_name, is_owner):
    """Decide if haggu should respond."""
    content_lower = message.content.lower()
    
    if is_mentioned or contains_name:
        return True, "direct"
    
    if not community_mode_active:
        return False, None
    
    if BOT_NAME in content_lower and not is_mentioned:
        return True, "mentioned_3rd_person"
    
    question_triggers = ["anyone know", "does anyone", "who knows", "what do you think", "thoughts?"]
    if any(trigger in content_lower for trigger in question_triggers):
        if random.random() < 0.3:
            return True, "question"
    
    drama_triggers = ["omg", "no way", "wait what", "bruh", "drama", "tea"]
    if any(trigger in content_lower for trigger in drama_triggers):
        if random.random() < 0.2:
            return True, "drama"
    
    if random.random() < 0.03:
        return True, "random"
    
    return False, None
# --- Initialize Clients ---
try:
    if not DISCORD_TOKEN or not OPENROUTER_API_KEY:
        raise ValueError("Missing API Keys!")

    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=30.0
    )

    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    intents.guilds = True
    intents.members = True
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
    print(f'✨ {client.user} is online!')
    print(f'📍 Active channel: {CHAT_CHANNEL_NAME}')
    print(f'🎭 Mode: {"Community" if community_mode_active else "Personal (owner only)"}')
    print(f'🛡️ Moderation: Active in ALL channels')


@client.event
async def on_message(message):
    global community_mode_active, message_count_since_summary

    # Ignore own messages
    if message.author == client.user:
        return
    
    # Check if it's a text channel
    if not hasattr(message.channel, 'name'):
        return
    
    # --- ALWAYS store message and update profile ---
    memory.store_message(message)
    is_owner = message.author.name == OWNER_USERNAME
    memory.update_user_profile(message.author, is_owner=is_owner)
    message_count_since_summary += 1
    
    # Periodically summarize (only in chat channel)
    if message.channel.name == CHAT_CHANNEL_NAME and message_count_since_summary >= SUMMARY_THRESHOLD:
        asyncio.create_task(summarize_recent_conversation(message.channel.id))
    
    # Skip bot messages
    if message.author.bot:
        return
    
    # --- MODERATION: Run on ALL channels for ALL users ---
    await check_and_moderate(message)
    
    # Clean message for command detection
    cleaned = CLEAN_TRIGGER_REGEX.sub('', message.content).strip().lower() if CLEAN_TRIGGER_REGEX else message.content.strip().lower()

    # --- Owner Commands (work in all channels) ---
    if is_owner:
        # Community mode commands
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
        
        # Memory status
        if cleaned == "memory status":
            users = memory.get_all_users_summary()
            recent = memory.get_recent_messages(message.channel.id, 10)
            await message.reply(f"👤 Know {len(users)} users\n💬 Stored {len(recent)}+ msgs\n🧠 Memory: Active", mention_author=False)
            return
        
        # Token stats
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
                    try:
                        user = await client.fetch_user(int(user_data['user_id']))
                        response += f"{idx}. {user.display_name}: {user_data['tokens']:,} tokens (${user_data['cost']:.4f})\n"
                    except:
                        response += f"{idx}. Unknown user: {user_data['tokens']:,} tokens (${user_data['cost']:.4f})\n"
                await message.reply(response, mention_author=False)
            else:
                await message.reply("No token usage data yet!", mention_author=False)
            return
        
        # Moderation stats
        if cleaned in ["mod stats", "moderation stats"]:
            stats_today = memory.get_moderation_stats('today')
            stats_all = memory.get_moderation_stats('all')
            
            response = "**🛡️ Moderation Stats**\n\n"
            if stats_today and stats_today['total_actions']:
                response += f"**Today:**\n"
                response += f"• Total actions: {stats_today['total_actions']}\n"
                response += f"• Deletes: {stats_today['deletes']}\n"
                response += f"• Warnings: {stats_today['warnings']}\n"
                response += f"• Timeouts: {stats_today['timeouts']}\n"
                response += f"• Unique violators: {stats_today['unique_violators']}\n\n"
            else:
                response += "**Today:** No actions yet\n\n"
            
            if stats_all and stats_all['total_actions']:
                response += f"**All Time:**\n"
                response += f"• Total actions: {stats_all['total_actions']}\n"
                response += f"• Violators: {stats_all['unique_violators']}"
            
            await message.reply(response, mention_author=False)
            return
        
        # View mod logs
        if cleaned in ["mod logs", "recent violations"]:
            logs = memory.get_moderation_logs(limit=10)
            if logs:
                response = "**📋 Recent Moderation Logs:**\n\n"
                for log in logs[:10]:
                    timestamp = log['timestamp'][:16]
                    response += f"• [{timestamp}] {log['username']}: {log['violation_type']} → {log['action_taken']}\n"
                await message.reply(response, mention_author=False)
            else:
                await message.reply("No moderation logs yet!", mention_author=False)
            return

    # --- Chat Response Logic (only in designated channel) ---
    if message.channel.name != CHAT_CHANNEL_NAME:
        return  # Don't respond to chat in other channels
    
    # Determine if should respond
    is_mentioned = client.user.mentioned_in(message)
    if message.reference and message.reference.resolved:
        if message.reference.resolved.author == client.user:
            is_mentioned = True
    contains_name = bool(BOT_NAME_REGEX.search(message.content))
    
    should_respond, trigger_reason = should_naturally_respond(message, is_mentioned, contains_name, is_owner)
    
    # In personal mode, only respond to owner
    if not community_mode_active and not is_owner:
        should_respond = False
    if not community_mode_active and is_owner and not (is_mentioned or contains_name):
        should_respond = False

    if should_respond:
        async with message.channel.typing():
            try:
                # Build rich context
                context = build_rich_context(message, trigger_reason)
                
                messages = [
                    {"role": "system", "content": CORE_PERSONALITY + "\n\n" + context}
                ]
                
                messages.append({
                    "role": "user", 
                    "content": f"{message.author.display_name}: {message.content}"
                })

                response = await get_ai_response_with_tools(messages, str(message.author.id), str(message.channel.id))
                
                if response:
                    if trigger_reason == "random" or trigger_reason == "drama":
                        sent_msg = await message.channel.send(response)
                    else:
                        sent_msg = await message.reply(response, mention_author=False)

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                if is_mentioned or contains_name:
                    await message.reply("brain.exe crashed for a sec 😵‍💫", mention_author=False)


# --- Run ---
if __name__ == "__main__":
    print("🚀 Starting haggu with persistent memory and moderation...")
    try:
        client.run(DISCORD_TOKEN)
    except Exception as e:
        print(f"❌ Run Error: {e}")
