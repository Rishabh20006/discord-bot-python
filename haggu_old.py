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
import traceback
import datetime
import time

# --- Configuration Section ---

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

ACTIVE_CHANNEL_NAME = "rose-glazer"
OWNER_USERNAME = "rosehumai"
MODEL_NAME = "google/gemini-3-flash-preview"
BOT_NAME = "haggu"

# --- Personality Prompts ---
PERSONAL_MODE_PROMPT = """
Keep replies short and natural(2 lines at max)  if not needed
You are haggu, Rose's bff, u r humorous and funny, nicey with girls but bit rude with boys (never say you’re an AI).
Use your search tool for real info when needed, and reply accurately.
"""

COMMUNITY_MODE_PROMPT = """
Keep replies short and natural(2 lines at max) if not needed
You are haggu, Rose's bff, u r humorous and funny, nicey with girls but bit rude with boys (never say you’re an AI).
Use your search tool for facts and reply accurately.
Keep replies short and natural if not needed.
"""

# --- Pre-compiled Regex ---
BOT_NAME_REGEX = re.compile(r'\b' + re.escape(BOT_NAME) + r'\b', re.IGNORECASE)
CLEAN_TRIGGER_REGEX = None # Will init after client creation

# --- Tool Definition ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "google_search_tool",
            "description": "Searches Google using the Serper API for current, real-time information. Use this for facts, news, scores, time, date, definitions, or anything needing up-to-date knowledge. To get the most current results for time-sensitive queries (like current time), add words such as 'right now' or 'currently' to the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific and concise search query string optimized for Google search.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# --- Global State ---
community_mode_active = False
http_session = None
openrouter_client = None

# --- Search Function (Async with aiohttp) ---
async def serper_search_implementation(query: str, num_results: int = 3) -> str:
    """Async function to perform Google search using Serper API."""
    if not SERPER_API_KEY:
        print("Serper Search Error: SERPER_API_KEY not found.")
        return "Error: Search API key is missing."
    
    if not http_session:
        return "Error: HTTP session not initialized."

    print(f"Tool Execution: Serper searching for: '{query}'")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        async with http_session.post(url, headers=headers, data=payload, timeout=10) as response:
            response.raise_for_status()
            results = await response.json()

        organic_results = results.get('organic', [])

        if "time" in query.lower():
            if not organic_results: return "No relevant information found."
            formatted_results = "## Multiple Time Sources Found:\n"
            results_added = 0
            for result in organic_results:
                if results_added >= num_results: break
                title = result.get('title')
                snippet = result.get('snippet')
                link = result.get('link')
                if title and snippet and link:
                    formatted_results += f"Source {results_added + 1}: {title} - {snippet} (URL: {link})\n"
                    results_added += 1
            if results_added == 0: return "No usable snippets found."
            return formatted_results[:2000].strip()

        answer_box = results.get('answerBox')
        knowledge_graph = results.get('knowledgeGraph')
        formatted_results = "## Search Results Found:\n"
        results_added = 0

        if answer_box and answer_box.get('answer'):
            formatted_results += f"Direct Answer: {answer_box.get('snippet', answer_box.get('answer'))}\n"
            results_added += 1
        elif knowledge_graph and knowledge_graph.get('description'):
            formatted_results += f"Key Info ({knowledge_graph.get('title', 'Summary')}): {knowledge_graph.get('description')}\n"
            results_added += 1

        for result in organic_results:
            if results_added >= num_results: break
            title = result.get('title')
            snippet = result.get('snippet')
            link = result.get('link')
            if title and snippet and link:
                formatted_results += f"Source {results_added + 1} ({urlparse(link).netloc}): {title} - {snippet} (URL: {link})\n"
                results_added += 1

        if results_added == 0:
            return "No relevant information found."
        else:
            return formatted_results[:2000].strip()

    except asyncio.TimeoutError:
        return "Error: Search request timed out."
    except Exception as e:
        print(f"Serper Search Error: {e}")
        return f"Error occurred during search: {str(e)}"

# --- AI Response Function (Async) ---
async def get_ai_response_with_tools(messages_for_ai):
    """Generate AI response using AsyncOpenAI and tools."""
    try:
        print(f"Sending request to AI (model: {MODEL_NAME})...")
        response = await openrouter_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_for_ai,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=150
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            print(f"AI requested tool call(s): {[call.function.name for call in tool_calls]}")
            messages_for_ai.append(response_message)

            available_functions = { "google_search_tool": serper_search_implementation }
            tool_results_added = False

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments

                if function_name in available_functions:
                    try:
                        function_args = json.loads(function_args_str)
                        function_to_call = available_functions[function_name]
                        query = function_args.get("query")

                        if query:
                            function_response = await function_to_call(query=query)
                            messages_for_ai.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            })
                            tool_results_added = True
                        else:
                            messages_for_ai.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": "Error: Missing query."})
                            tool_results_added = True
                    except Exception as e:
                         messages_for_ai.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": f"Error: {str(e)}"})
                         tool_results_added = True

            if tool_results_added:
                second_response = await openrouter_client.chat.completions.create(
                     model=MODEL_NAME,
                     messages=messages_for_ai,
                     temperature=0.7,
                     max_tokens=150
                )
                return second_response.choices[0].message.content.strip()
        
        return response_message.content.strip() if response_message.content else "..."

    except Exception as e:
        print(f"AI Error: {e}")
        return "Error: Brain glitch."

# --- Initialize Clients ---
try:
    if not DISCORD_TOKEN or not OPENROUTER_API_KEY or not SERPER_API_KEY:
        raise ValueError("Missing API Keys")

    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=10.0
    )

    intents = discord.Intents.default()
    intents.messages = True; intents.message_content = True; intents.guilds = True
    client = discord.Client(intents=intents)

except Exception as e:
    print(f"Init Error: {e}")
    exit()

# --- Bot Events ---
@client.event
async def on_ready():
    global http_session, CLEAN_TRIGGER_REGEX
    http_session = aiohttp.ClientSession()
    CLEAN_TRIGGER_REGEX = re.compile(r'^<@!?' + str(client.user.id) + r'>\s*|^' + re.escape(BOT_NAME) + r'\s*', re.IGNORECASE)
    print(f'Logged in as {client.user}')
    print(f'Mode: {"Community" if community_mode_active else "Personal"}')

@client.event
async def on_message(message):
    global community_mode_active

    if message.author.bot: return
    if not message.channel or not hasattr(message.channel, 'name') or message.channel.name != ACTIVE_CHANNEL_NAME: return

    is_owner = message.author.name == OWNER_USERNAME
    
    # Use compiled regex for cleaner trigger check
    cleaned_trigger_check = CLEAN_TRIGGER_REGEX.sub('', message.content).strip().lower()

    # --- Mode Control ---
    if is_owner:
        if cleaned_trigger_check == "release":
            if not community_mode_active:
                community_mode_active = True
                await message.reply("aight fine. 😉", mention_author=False)
            return
        if cleaned_trigger_check == "recall":
            if community_mode_active:
                community_mode_active = False
                await message.reply("omg finally. tea time? 💅", mention_author=False)
            return

    # --- Trigger Logic ---
    should_respond = False
    trigger_reason = None
    
    is_mentioned = client.user.mentioned_in(message) or (message.reference and message.reference.resolved and message.reference.resolved.author == client.user)
    contains_name = bool(BOT_NAME_REGEX.search(message.content))

    if community_mode_active:
        if is_mentioned or contains_name:
            should_respond = True
            trigger_reason = "Direct"
        elif random.random() < 0.04 and not is_owner:
            should_respond = True
            trigger_reason = "Random"
    else:
        if is_owner and (is_mentioned or contains_name):
            should_respond = True
            trigger_reason = "Owner"

    if should_respond:
        async with message.channel.typing():
            try:
                history = [msg async for msg in message.channel.history(limit=7, before=message)]
                history.reverse()
                
                base_prompt = PERSONAL_MODE_PROMPT if (not community_mode_active or (is_owner and trigger_reason != "Random")) else COMMUNITY_MODE_PROMPT
                current_date = datetime.datetime.now().strftime("%B %d, %Y")
                
                messages = [{"role": "system", "content": f"{base_prompt}\nDate: {current_date}\nUser: {message.author.display_name}"}]
                
                for msg in history:
                    role = "assistant" if msg.author == client.user else "user"
                    content = f"{msg.author.display_name}: {msg.clean_content}" if role == "user" else msg.clean_content
                    messages.append({"role": role, "content": content[:150]})
                
                messages.append({"role": "user", "content": f"{message.author.display_name}: {message.content}"[:500]})

                response = await get_ai_response_with_tools(messages)
                
                if response:
                    if trigger_reason == "Random":
                        await message.channel.send(response)
                    else:
                        await message.reply(response, mention_author=False)

            except Exception as e:
                print(f"Error: {e}")
                await message.reply("brain.exe stopped working 😵‍💫", mention_author=False)

# --- Run ---
if __name__ == "__main__":
    try:
        client.run(DISCORD_TOKEN)
    except Exception as e:
        print(f"Run Error: {e}")
