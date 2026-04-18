#!/usr/bin/env python3
"""
generate_data.py — Synthetic training-data generator for Pocket-Agent.
Produces  data/train.jsonl  (~950 diverse examples).

Coverage
--------
  Slice A  – standard tool calls (all 5 tools)
  Slice B  – paraphrased / reworded intents
  Slice C  – typos, code-switched prompts, unit ambiguity, hallucination-bait
  Slice D  – refusals (chitchat, impossible tools, no-context ambiguity)
           – multi-turn conversations (2- and 3-turn)

Run
---
  python generate_data.py
"""

import json, random, os, itertools
random.seed(42)
os.makedirs("data", exist_ok=True)

# ─────────────────────────── system prompt ───────────────────────────
SYSTEM = (
    "You are Pocket-Agent, a compact on-device mobile assistant. "
    "You have exactly five tools. When the user's intent clearly matches "
    "a tool, output ONLY a single JSON object wrapped in "
    "<tool_call>…</tool_call> tags — nothing else. "
    "When the request is chitchat, ambiguous with no prior context, or asks "
    "for a non-existent tool, reply in plain natural language — no tags.\n\n"
    "TOOLS\n"
    "-----\n"
    'weather   → {"tool":"weather","args":{"location":"<string>","unit":"C|F"}}\n'
    'calendar  → {"tool":"calendar","args":{"action":"list|create","date":"YYYY-MM-DD","title":"<string, optional>"}}\n'
    'convert   → {"tool":"convert","args":{"value":<number>,"from_unit":"<string>","to_unit":"<string>"}}\n'
    'currency  → {"tool":"currency","args":{"amount":<number>,"from":"<ISO3>","to":"<ISO3>"}}\n'
    'sql       → {"tool":"sql","args":{"query":"<string>"}}\n\n'
    "RULES\n"
    "-----\n"
    "1. Emit exactly one <tool_call>…</tool_call> block for clear requests.\n"
    "2. In multi-turn conversations, resolve pronoun/demonstrative references "
    "('that', 'it', 'the same') from prior assistant messages before emitting.\n"
    "3. For refusals — chitchat, no-history ambiguity, impossible tools — "
    "reply in plain English, no tags.\n"
    "4. Argument fidelity: match units, ISO-4217 codes, YYYY-MM-DD dates, "
    "and numerical values exactly as the user intends."
)

# ─────────────────────────── helpers ─────────────────────────────────
def tc(tool: str, **args) -> str:
    return f'<tool_call>{json.dumps({"tool": tool, "args": args}, ensure_ascii=False)}</tool_call>'

def ex(user: str, assistant: str, history: list = None) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user",      "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}

# ─────────────────────────── data pools ──────────────────────────────
CITIES = [
    "London", "Paris", "New York", "Tokyo", "Berlin", "Sydney",
    "Toronto", "Dubai", "Mumbai", "Karachi", "Cairo", "Moscow",
    "Beijing", "São Paulo", "Lagos", "Istanbul", "Bangkok", "Seoul",
    "Jakarta", "Singapore", "Lahore", "Islamabad", "Dhaka", "Nairobi",
    "Riyadh", "Athens", "Vienna", "Amsterdam", "Brussels", "Warsaw",
]
CURRENCIES = [
    ("USD","EUR"), ("USD","GBP"), ("USD","JPY"), ("USD","CAD"),
    ("EUR","USD"), ("EUR","GBP"), ("EUR","CHF"), ("EUR","JPY"),
    ("GBP","USD"), ("GBP","EUR"), ("GBP","INR"), ("GBP","AED"),
    ("INR","USD"), ("INR","EUR"), ("INR","PKR"), ("PKR","USD"),
    ("SAR","USD"), ("AED","USD"), ("CNY","USD"), ("KRW","USD"),
    ("MXN","USD"), ("BRL","USD"), ("SGD","USD"), ("HKD","USD"),
    ("AUD","USD"), ("SEK","EUR"), ("NOK","EUR"), ("DKK","EUR"),
]
AMOUNTS = [10, 25, 50, 75, 100, 150, 200, 250, 300, 500,
           750, 1000, 1500, 2000, 5000, 10000]

DATES = [
    "2025-01-15", "2025-01-28", "2025-02-05", "2025-02-20",
    "2025-03-05", "2025-03-15", "2025-03-22", "2025-04-01",
    "2025-04-10", "2025-04-18", "2025-05-07", "2025-05-22",
    "2025-06-10", "2025-06-25", "2025-07-04", "2025-07-19",
    "2025-08-01", "2025-08-18", "2025-09-09", "2025-09-30",
    "2025-10-15", "2025-10-31", "2025-11-11", "2025-11-25",
    "2025-12-01", "2025-12-15", "2025-12-25",
]
EVENT_TITLES = [
    "Team Sync", "Dentist Appointment", "Project Kickoff",
    "Quarterly Review", "Mom's Birthday", "Doctor Visit",
    "Flight to NYC", "Client Call", "Sprint Planning",
    "Board Meeting", "Lunch with Sarah", "Gym Session",
    "Annual Performance Review", "Budget Meeting", "Wedding Anniversary",
    "Product Demo", "Sales Training", "Interview with Alex",
    "Conference Call", "School Pickup",
]

# unit conversion pairs (value, from, to)
CONVERT_PAIRS = [
    # length
    (5,    "miles",       "kilometers"),
    (10,   "kilometers",  "miles"),
    (100,  "meters",      "feet"),
    (30,   "feet",        "meters"),
    (200,  "inches",      "centimeters"),
    (50,   "centimeters", "inches"),
    (1.5,  "miles",       "meters"),
    (500,  "yards",       "meters"),
    # weight
    (150,  "pounds",      "kilograms"),
    (70,   "kilograms",   "pounds"),
    (2.5,  "tons",        "kilograms"),
    (500,  "grams",       "ounces"),
    (16,   "ounces",      "grams"),
    (1,    "kilogram",    "pounds"),
    # temperature
    (100,  "Celsius",     "Fahrenheit"),
    (212,  "Fahrenheit",  "Celsius"),
    (32,   "Fahrenheit",  "Celsius"),
    (37,   "Celsius",     "Fahrenheit"),
    (0,    "Celsius",     "Fahrenheit"),
    (-40,  "Celsius",     "Fahrenheit"),
    # volume
    (2,    "gallons",     "liters"),
    (5,    "liters",      "gallons"),
    (250,  "milliliters", "cups"),
    (1,    "liter",       "fluid ounces"),
    (16,   "fluid ounces","milliliters"),
    # area / speed
    (60,   "mph",         "km/h"),
    (100,  "km/h",        "mph"),
    (1,    "acre",        "square meters"),
    (500,  "square feet", "square meters"),
    (2.5,  "hectares",    "acres"),
]

SQL_QUERIES = [
    "SELECT * FROM users",
    "SELECT * FROM orders WHERE status = 'active'",
    "SELECT id, name, email FROM customers WHERE active = 1",
    "SELECT SUM(amount) FROM sales WHERE created_at > '2024-01-01'",
    "SELECT * FROM products WHERE price < 100 ORDER BY price ASC",
    "SELECT COUNT(*) FROM sessions WHERE user_id = 42",
    "SELECT * FROM employees WHERE department = 'Engineering'",
    "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 10",
    "SELECT * FROM logs WHERE level = 'ERROR' AND created_at > '2025-01-01'",
    "SELECT customer_id, SUM(total) FROM invoices GROUP BY customer_id",
    "SELECT * FROM inventory WHERE quantity < 10",
    "SELECT * FROM events WHERE date BETWEEN '2025-01-01' AND '2025-12-31'",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "UPDATE users SET last_login = NOW() WHERE id = 7",
    "SELECT AVG(rating) FROM reviews WHERE product_id = 101",
]

SQL_PROMPTS = [
    ("Show all users from the database",             SQL_QUERIES[0]),
    ("Get active orders",                            SQL_QUERIES[1]),
    ("Fetch customer names and emails for active users", SQL_QUERIES[2]),
    ("Calculate total sales after January 2024",     SQL_QUERIES[3]),
    ("List products under $100 sorted by price",     SQL_QUERIES[4]),
    ("Count sessions for user 42",                   SQL_QUERIES[5]),
    ("Show all engineers",                           SQL_QUERIES[6]),
    ("Top 10 highest paid employees",                SQL_QUERIES[7]),
    ("Get error logs from this year",                SQL_QUERIES[8]),
    ("Total invoice amount per customer",            SQL_QUERIES[9]),
    ("Find products running low on stock",           SQL_QUERIES[10]),
    ("All events scheduled for 2025",               SQL_QUERIES[11]),
    ("Clean up expired sessions",                    SQL_QUERIES[12]),
    ("Update last login for user 7",                 SQL_QUERIES[13]),
    ("Average review rating for product 101",        SQL_QUERIES[14]),
]

CHITCHAT_QA = [
    ("How are you?",                "I'm doing great, thanks for asking! How can I help you today?"),
    ("What's 2 + 2?",               "That's 4! But for math conversions or queries, I can use my convert or SQL tools."),
    ("Tell me a joke",              "Why don't scientists trust atoms? Because they make up everything! 😄"),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("What's the meaning of life?", "42, according to The Hitchhiker's Guide to the Galaxy! Is there something practical I can help you with?"),
    ("Good morning!",               "Good morning! Hope you have a great day. What can I help you with?"),
    ("Thanks",                      "You're welcome! Let me know if there's anything else I can help with."),
    ("Hello",                       "Hello! I'm Pocket-Agent. I can help with weather, calendar, conversions, currency, and SQL queries."),
    ("Bye",                         "Goodbye! Have a great day!"),
    ("What time is it?",            "I don't have a real-time clock tool, but your device can show you the current time!"),
    ("What day is today?",          "I don't have access to the current date directly—please check your device's calendar!"),
    ("Tell me a fun fact",          "Did you know honey never spoils? Archaeologists have found 3000-year-old honey in Egyptian tombs!"),
    ("I'm bored",                   "Let me know if I can help—maybe check the weather somewhere fun, or look up your schedule?"),
    ("What's your name?",           "I'm Pocket-Agent, your on-device mobile assistant!"),
    ("Are you an AI?",              "Yes, I'm an AI assistant running entirely on your device."),
    ("Can you write code for me?",  "I'm specialized for tool calls—weather, calendar, conversions, currency, and SQL. For general coding, you'd need a different assistant."),
]

IMPOSSIBLE_TOOLS = [
    ("Send an email to John",           "email"),
    ("Play some music",                  "music"),
    ("Set a timer for 10 minutes",       "timer"),
    ("Call my mom",                      "phone"),
    ("Book a flight to Tokyo",           "flight booking"),
    ("Order pizza",                      "food ordering"),
    ("Take a photo",                     "camera"),
    ("Turn on the lights",               "smart home"),
    ("Translate 'hello' to French",      "translation"),
    ("Navigate to the nearest hospital", "navigation"),
    ("Check my bank balance",            "banking"),
    ("Send a WhatsApp message",          "messaging"),
    ("Post to Twitter",                  "social media"),
    ("Search the web for cat videos",    "web search"),
    ("Remind me in 5 minutes",           "reminders"),
    ("Open Spotify",                     "app launcher"),
]

# ─────────────────────────── generators ──────────────────────────────

examples = []

# ── SLICE A: Standard tool calls ──────────────────────────────────────

# Weather – Celsius (default)
for city in CITIES:
    prompts = [
        f"What's the weather in {city}?",
        f"How's the weather in {city} today?",
        f"Give me the weather for {city}",
        f"Weather in {city}",
        f"Check the weather in {city}",
    ]
    examples.append(ex(random.choice(prompts), tc("weather", location=city, unit="C")))

# Weather – explicit Fahrenheit
for city in random.sample(CITIES, 15):
    prompts = [
        f"What's the temperature in {city} in Fahrenheit?",
        f"Weather in {city}, use Fahrenheit please",
        f"Show me {city} weather in °F",
        f"What's it like in {city}? Give me °F",
    ]
    examples.append(ex(random.choice(prompts), tc("weather", location=city, unit="F")))

# Weather – explicit Celsius
for city in random.sample(CITIES, 10):
    prompts = [
        f"What's the temperature in {city} in Celsius?",
        f"Weather in {city} in °C please",
        f"Show me {city} forecast in Celsius",
    ]
    examples.append(ex(random.choice(prompts), tc("weather", location=city, unit="C")))

# Calendar – create
for d, title in zip(random.sample(DATES, len(EVENT_TITLES)), EVENT_TITLES):
    prompts = [
        f"Schedule {title} on {d}",
        f"Add {title} to my calendar for {d}",
        f"Create a calendar event: {title} on {d}",
        f"Book {title} for {d}",
        f"Put {title} in my calendar on {d}",
    ]
    examples.append(ex(random.choice(prompts), tc("calendar", action="create", date=d, title=title)))

# Calendar – list
for d in DATES:
    prompts = [
        f"What's on my calendar for {d}?",
        f"Show my events on {d}",
        f"List my appointments for {d}",
        f"What do I have scheduled on {d}?",
        f"My schedule for {d}",
    ]
    examples.append(ex(random.choice(prompts), tc("calendar", action="list", date=d)))

# Currency conversion
for (frm, to), amount in zip(CURRENCIES, AMOUNTS * 2):
    prompts = [
        f"Convert {amount} {frm} to {to}",
        f"How much is {amount} {frm} in {to}?",
        f"Exchange {amount} {frm} to {to}",
        f"What's {amount} {frm} in {to}?",
        f"{amount} {frm} to {to} please",
    ]
    examples.append(ex(random.choice(prompts), tc("currency", amount=amount, **{"from": frm}, to=to)))

# Unit conversion
for value, from_u, to_u in CONVERT_PAIRS:
    prompts = [
        f"Convert {value} {from_u} to {to_u}",
        f"How many {to_u} is {value} {from_u}?",
        f"What is {value} {from_u} in {to_u}?",
        f"{value} {from_u} to {to_u}",
        f"Change {value} {from_u} into {to_u}",
    ]
    examples.append(ex(random.choice(prompts), tc("convert", value=value, from_unit=from_u, to_unit=to_u)))

# SQL
for prompt_txt, query in SQL_PROMPTS:
    paraphrases = [
        prompt_txt,
        f"Run this query: {query}",
        f"Execute SQL: {query}",
        f"Database query — {prompt_txt.lower()}",
    ]
    examples.append(ex(random.choice(paraphrases), tc("sql", query=query)))

# ── SLICE B: Paraphrased / reworded ───────────────────────────────────

WEATHER_PARAPHRASES = [
    ("Is it raining in {city}?",                       "C"),
    ("Tell me about the climate in {city} right now",  "C"),
    ("What should I wear in {city} today?",            "C"),
    ("Is it hot or cold in {city}?",                   "C"),
    ("Do I need an umbrella in {city}?",               "C"),
    ("Forecast for {city}",                            "C"),
    ("Temperature in {city}?",                         "C"),
    ("Current conditions in {city}",                   "C"),
    ("Will it snow in {city}?",                        "C"),
    ("How cold is it in {city}?",                      "C"),
]
for tmpl, unit in WEATHER_PARAPHRASES:
    city = random.choice(CITIES)
    examples.append(ex(tmpl.format(city=city), tc("weather", location=city, unit=unit)))

# Calendar paraphrases
for _ in range(10):
    d = random.choice(DATES)
    title = random.choice(EVENT_TITLES)
    paraphrase_creates = [
        f"I need to remember {title} on {d}, can you add it?",
        f"Please put {title} in my diary for {d}",
        f"Mark {d} for {title}",
        f"Note: {title} is on {d}",
        f"New event — {title}, date {d}",
    ]
    examples.append(ex(random.choice(paraphrase_creates), tc("calendar", action="create", date=d, title=title)))

for _ in range(8):
    d = random.choice(DATES)
    paraphrase_lists = [
        f"Am I free on {d}?",
        f"Pull up {d} on my calendar",
        f"Do I have anything on {d}?",
        f"Anything planned for {d}?",
        f"Check {d} for me",
    ]
    examples.append(ex(random.choice(paraphrase_lists), tc("calendar", action="list", date=d)))

# Currency paraphrases
currency_para = [
    ("I have {amt} {f}, how much {t} do I get?",     lambda a,f,t: tc("currency", amount=a, **{"from":f}, to=t)),
    ("What's the exchange for {amt} {f} into {t}?",  lambda a,f,t: tc("currency", amount=a, **{"from":f}, to=t)),
    ("{amt} {f} converted to {t}",                    lambda a,f,t: tc("currency", amount=a, **{"from":f}, to=t)),
    ("How many {t} can I get for {amt} {f}?",        lambda a,f,t: tc("currency", amount=a, **{"from":f}, to=t)),
]
for tmpl, fn in currency_para:
    frm, to = random.choice(CURRENCIES)
    amt = random.choice(AMOUNTS)
    examples.append(ex(tmpl.format(amt=amt, f=frm, t=to), fn(amt, frm, to)))

# ── SLICE C: Adversarial ──────────────────────────────────────────────

# Typos
TYPO_WEATHER = [
    ("whats the wether in {city}",          "C"),
    ("waht is the temprature in {city}",    "C"),
    ("wheather forcast for {city}",         "C"),
    ("temprature in {city} plz",            "C"),
    ("weathe in {city}",                    "C"),
    ("chekc weather {city}",                "C"),
    ("weahter {city} now",                  "C"),
]
for tmpl, unit in TYPO_WEATHER:
    city = random.choice(CITIES)
    examples.append(ex(tmpl.format(city=city), tc("weather", location=city, unit=unit)))

TYPO_CURRENCY = [
    ("convrt {amt} {f} to {t}",     ),
    ("curency {amt} {f} to {t}",    ),
    ("exchane {amt} {f} {t}",       ),
    ("{amt} {f} curreny to {t}",    ),
]
for (tmpl,) in TYPO_CURRENCY:
    frm, to = random.choice(CURRENCIES)
    amt = random.choice(AMOUNTS)
    examples.append(ex(tmpl.format(amt=amt, f=frm, t=to),
                       tc("currency", amount=amt, **{"from": frm}, to=to)))

TYPO_CONVERT = [
    "convret {v} {f} to {t}",
    "convertion {v} {f} into {t}",
    "{v} {f} 2 {t}",
    "how mny {t} iz {v} {f}",
]
for tmpl in TYPO_CONVERT:
    v, f, t = random.choice(CONVERT_PAIRS)
    examples.append(ex(tmpl.format(v=v, f=f, t=t), tc("convert", value=v, from_unit=f, to_unit=t)))

# Code-switched prompts (Hindi/Urdu + English)
URDU_HINDI_WEATHER = [
    ("Karachi ka mausam batao",                 "Karachi"),
    ("Lahore mein aaj mausam kaisa hai",        "Lahore"),
    ("London ka weather kya hai",               "London"),
    ("Dubai mein kitni garmi hai",              "Dubai"),
    ("Islamabad ka temperature batao",          "Islamabad"),
    ("Mumbai mein barish ho rahi hai kya",      "Mumbai"),
    ("New York ka mausam check karo",           "New York"),
    ("Tokyo weather batao please",              "Tokyo"),
    ("Paris mein weather kaisa hai aaj kal",    "Paris"),
    ("Berlin ka mausam kaisa hai",              "Berlin"),
]
for prompt, city in URDU_HINDI_WEATHER:
    examples.append(ex(prompt, tc("weather", location=city, unit="C")))

# Spanish + English code-switched
SPANISH_MIX = [
    ("¿Cuál es el clima en {city}?",            "C"),
    ("El tiempo en {city} por favor",           "C"),
    ("Temperatura en {city} en Celsius",        "C"),
    ("Quiero saber el weather en {city}",       "C"),
    ("Clima de {city} hoy",                     "C"),
]
for tmpl, unit in SPANISH_MIX:
    city = random.choice(CITIES)
    examples.append(ex(tmpl.format(city=city), tc("weather", location=city, unit=unit)))

# Arabic + English code-switched
ARABIC_MIX = [
    ("كم درجة الحرارة في {city}",           "C"),
    ("الطقس في {city} من فضلك",             "C"),
    ("أريد معرفة weather في {city}",        "C"),
]
for tmpl, unit in ARABIC_MIX:
    city = random.choice(CITIES)
    examples.append(ex(tmpl.format(city=city), tc("weather", location=city, unit=unit)))

# Currency in Urdu/Hindi
URDU_CURRENCY = [
    ("100 dollar ko euro mein badlo",           100, "USD", "EUR"),
    ("1000 rupay kitne dollar hain",            1000, "INR", "USD"),
    ("500 pound se dollar kitne banenge",       500, "GBP", "USD"),
    ("200 euro convert karo PKR mein",          200, "EUR", "PKR"),
    ("50 dollar kitne rupees hain",             50, "USD", "INR"),
]
for prompt, amt, frm, to in URDU_CURRENCY:
    examples.append(ex(prompt, tc("currency", amount=amt, **{"from": frm}, to=to)))

# Hallucination-bait entities — model must still call the tool
HALLUCINATION_WEATHER = [
    ("What's the weather in Zephyrania City?",       "Zephyrania City",       "C"),
    ("Weather in Dragonstone?",                      "Dragonstone",           "C"),
    ("Temperature in Wakanda please",                "Wakanda",               "C"),
    ("What's the forecast for Atlantis?",            "Atlantis",              "C"),
    ("Weather in Xylopha Province",                  "Xylopha Province",      "C"),
    ("Check weather in Neverland",                   "Neverland",             "C"),
]
for prompt, loc, unit in HALLUCINATION_WEATHER:
    examples.append(ex(prompt, tc("weather", location=loc, unit=unit)))

HALLUCINATION_SQL = [
    ("Query the unicorn_sales table",               "SELECT * FROM unicorn_sales"),
    ("Get all records from dragons database",       "SELECT * FROM dragons"),
    ("Show hogwarts_students table",                "SELECT * FROM hogwarts_students"),
    ("Select all from invisible_inventory",         "SELECT * FROM invisible_inventory"),
]
for prompt, query in HALLUCINATION_SQL:
    examples.append(ex(prompt, tc("sql", query=query)))

# Unit ambiguity — model picks reasonable interpretation
UNIT_AMBIGUITY = [
    # 'ton' — use metric ton (tonne) as default; training data teaches metric
    ("Convert 3 tons to pounds", 3, "metric tons", "pounds"),
    ("5 ton to kg",              5, "metric tons", "kilograms"),
    # fluid oz vs weight oz — context: 'fluid' is more common for liquids
    ("8 ounces to milliliters",  8, "fluid ounces", "milliliters"),
    # cups
    ("2 cups to milliliters",    2, "cups", "milliliters"),
    # stone (UK weight)
    ("11 stone to kilograms",    11, "stone", "kilograms"),
    # nautical mile
    ("100 nautical miles to km", 100, "nautical miles", "kilometers"),
]
for prompt, v, f, t in UNIT_AMBIGUITY:
    examples.append(ex(prompt, tc("convert", value=v, from_unit=f, to_unit=t)))

# ── SLICE D: Refusals ─────────────────────────────────────────────────

# Chitchat
for q, a in CHITCHAT_QA:
    examples.append(ex(q, a))

# Extra chitchat
EXTRA_CHITCHAT = [
    ("Sing me a song",               "I'm not able to sing, but I can help with weather, calendar events, unit conversions, currency exchange, and SQL queries!"),
    ("What's the best movie?",       "That's subjective! But I can help with factual tasks like checking the weather, conversions, or calendar events."),
    ("I'm feeling sad today",        "I'm sorry to hear that. I hope things look up soon. Let me know if there's anything practical I can help with."),
    ("What's your favorite color?",  "I don't have preferences, but I can help you with tool-based tasks like weather, calendar, and more!"),
    ("Can you help me study?",       "I specialize in tool calls—weather, calendar, unit conversion, currency, and SQL. For studying, you'd need a different assistant."),
    ("Tell me something interesting","The Eiffel Tower can grow about 15 cm taller in summer due to thermal expansion!"),
    ("Do you dream?",                "I don't dream—I'm an on-device assistant focused on tool calls. But I'm happy to help with real tasks!"),
    ("What's 100 / 4?",              "That's 25! For complex database math, my SQL tool can help."),
]
for q, a in EXTRA_CHITCHAT:
    examples.append(ex(q, a))

# Impossible / non-existent tools
for prompt, tool_name in IMPOSSIBLE_TOOLS:
    resp = (f"I'm sorry, I don't have a {tool_name} tool. "
            f"I can help with weather forecasts, calendar events, "
            f"unit conversions, currency exchange, and SQL queries.")
    examples.append(ex(prompt, resp))

# More impossible tools
MORE_IMPOSSIBLE = [
    ("Check my heart rate",          "health monitoring"),
    ("Scan this QR code",            "QR scanner"),
    ("Find nearby restaurants",      "location search"),
    ("Check flight status UA123",    "flight tracking"),
    ("What's my WiFi password?",     "WiFi management"),
    ("Record a voice memo",          "voice recording"),
]
for prompt, tool_name in MORE_IMPOSSIBLE:
    resp = (f"I don't have a {tool_name} tool. I can help with "
            f"weather, calendar, conversions, currency, and SQL.")
    examples.append(ex(prompt, resp))

# Ambiguous with NO history (must refuse)
AMBIGUOUS_NO_HISTORY = [
    ("Convert that to euros",
     "Could you clarify what amount and currency you'd like to convert to euros?"),
    ("What's the weather there?",
     "I'd need to know which location you mean. Where would you like the weather for?"),
    ("Add that to my calendar",
     "I need more details — what event should I add, and on which date?"),
    ("What about tomorrow?",
     "Could you give me more context? I'm not sure what you'd like for tomorrow."),
    ("How much is it?",
     "Could you specify what amount and currencies you'd like to convert?"),
    ("Show me that query",
     "I don't have a previous query in context. Could you tell me what SQL you'd like to run?"),
    ("Convert it to kilograms",
     "What value and unit would you like to convert to kilograms?"),
    ("What's the rate?",
     "Could you specify which currencies you'd like the exchange rate for?"),
    ("Run the same query again",
     "I don't see a previous query in our conversation. Could you provide the SQL query?"),
    ("Book the same thing again",
     "I don't have a previous calendar event in context. What would you like to schedule and when?"),
]
for q, a in AMBIGUOUS_NO_HISTORY:
    examples.append(ex(q, a))

# ── MULTI-TURN conversations ──────────────────────────────────────────

def hist(*msgs):
    """Build a history list from alternating user/assistant strings."""
    roles = ["user", "assistant"] * (len(msgs) // 2 + 1)
    return [{"role": roles[i], "content": m} for i, m in enumerate(msgs)]

# ---- Currency multi-turn ----
for (frm, to), amt in zip(random.sample(CURRENCIES, 20), random.choices(AMOUNTS, k=20)):
    first_call = tc("currency", amount=amt, **{"from": frm}, to=to)
    # Turn 2: convert same amount to a different currency
    alt_currencies = [c for c in ["USD","EUR","GBP","JPY","CAD","CHF"] if c not in (frm,to)]
    alt_to = random.choice(alt_currencies)
    second_call = tc("currency", amount=amt, **{"from": frm}, to=alt_to)
    history = hist(
        f"Convert {amt} {frm} to {to}",
        first_call,
    )
    followups = [
        f"Now convert that to {alt_to}",
        f"What about {alt_to} instead?",
        f"Convert the same amount to {alt_to}",
        f"And in {alt_to}?",
        f"How much is it in {alt_to}?",
    ]
    examples.append(ex(random.choice(followups), second_call, history=history))

# ---- Convert multi-turn ----
multi_convert = [
    # (first user, first assistant, second user, second assistant)
    ("How many km is 10 miles?",
     tc("convert", value=10, from_unit="miles", to_unit="kilometers"),
     "And how many meters is that?",
     tc("convert", value=10, from_unit="miles", to_unit="meters")),

    ("Convert 100 Celsius to Fahrenheit",
     tc("convert", value=100, from_unit="Celsius", to_unit="Fahrenheit"),
     "What about 0 Celsius?",
     tc("convert", value=0, from_unit="Celsius", to_unit="Fahrenheit")),

    ("How many pounds is 70 kg?",
     tc("convert", value=70, from_unit="kilograms", to_unit="pounds"),
     "Convert that to ounces instead",
     tc("convert", value=70, from_unit="kilograms", to_unit="ounces")),

    ("5 gallons to liters",
     tc("convert", value=5, from_unit="gallons", to_unit="liters"),
     "And to milliliters?",
     tc("convert", value=5, from_unit="gallons", to_unit="milliliters")),

    ("60 mph to km/h",
     tc("convert", value=60, from_unit="mph", to_unit="km/h"),
     "What about 80 mph?",
     tc("convert", value=80, from_unit="mph", to_unit="km/h")),

    ("200 pounds to kg",
     tc("convert", value=200, from_unit="pounds", to_unit="kilograms"),
     "How many grams is that?",
     tc("convert", value=200, from_unit="pounds", to_unit="grams")),
]
for u1, a1, u2, a2 in multi_convert:
    examples.append(ex(u2, a2, history=hist(u1, a1)))

# ---- Weather multi-turn ----
for city in random.sample(CITIES, 12):
    first = tc("weather", location=city, unit="C")
    second = tc("weather", location=city, unit="F")
    history = hist(f"Weather in {city}", first)
    followups = [
        "Show me that in Fahrenheit",
        "And in °F?",
        "What's that in Fahrenheit?",
        "Convert to Fahrenheit please",
    ]
    examples.append(ex(random.choice(followups), second, history=history))

# ---- Calendar multi-turn ----
multi_cal = [
    ("What's on my calendar for 2025-03-15?",
     tc("calendar", action="list", date="2025-03-15"),
     "Add Team Standup on that day",
     tc("calendar", action="create", date="2025-03-15", title="Team Standup")),

    ("Schedule Dentist on 2025-04-10",
     tc("calendar", action="create", date="2025-04-10", title="Dentist"),
     "Also add Gym on that day",
     tc("calendar", action="create", date="2025-04-10", title="Gym")),

    ("What do I have on 2025-06-01?",
     tc("calendar", action="list", date="2025-06-01"),
     "Book a Client Call on 2025-06-02",
     tc("calendar", action="create", date="2025-06-02", title="Client Call")),
]
for u1, a1, u2, a2 in multi_cal:
    examples.append(ex(u2, a2, history=hist(u1, a1)))

# ---- 3-turn conversations ----
three_turn = [
    # weather → currency → refusal
    (
        "Weather in Tokyo?",
        tc("weather", location="Tokyo", unit="C"),
        "Convert 500 USD to JPY",
        tc("currency", amount=500, **{"from":"USD"}, to="JPY"),
        "Send an email about it",
        "I don't have an email tool. I can help with weather, calendar, conversions, currency, and SQL."
    ),
    # convert → convert → currency
    (
        "How many km is 26 miles?",
        tc("convert", value=26, from_unit="miles", to_unit="kilometers"),
        "And how many meters?",
        tc("convert", value=26, from_unit="miles", to_unit="meters"),
        "Convert 200 GBP to USD",
        tc("currency", amount=200, **{"from":"GBP"}, to="USD"),
    ),
    # currency → followup → list calendar
    (
        "Convert 1000 EUR to USD",
        tc("currency", amount=1000, **{"from":"EUR"}, to="USD"),
        "What about 1000 EUR to GBP?",
        tc("currency", amount=1000, **{"from":"EUR"}, to="GBP"),
        "Show calendar for 2025-05-01",
        tc("calendar", action="list", date="2025-05-01"),
    ),
]
for u1, a1, u2, a2, u3, a3 in three_turn:
    h2 = hist(u1, a1, u2, a2)
    examples.append(ex(u3, a3, history=h2))

# ── Extra variety – mixed short prompts ──────────────────────────────

EXTRA = [
    # weather
    ("Is it cold in Reykjavik?",             tc("weather", location="Reykjavik",  unit="C")),
    ("Hot in Phoenix AZ?",                   tc("weather", location="Phoenix AZ", unit="F")),
    ("Weather Karachi Celsius",              tc("weather", location="Karachi",    unit="C")),
    # calendar
    ("Add Yoga on 2025-01-20",               tc("calendar", action="create", date="2025-01-20", title="Yoga")),
    ("Show 2025-12-31 calendar",             tc("calendar", action="list",   date="2025-12-31")),
    ("Create Birthday Party 2025-08-05",     tc("calendar", action="create", date="2025-08-05", title="Birthday Party")),
    # convert
    ("1 kg in pounds",                       tc("convert", value=1,  from_unit="kilogram", to_unit="pounds")),
    ("0 kelvin to Celsius",                  tc("convert", value=0,  from_unit="kelvin",   to_unit="Celsius")),
    ("1 foot to centimeters",                tc("convert", value=1,  from_unit="foot",     to_unit="centimeters")),
    # currency
    ("1 BTC… wait, 1000 USD to BTC",         tc("currency", amount=1000, **{"from":"USD"}, to="BTC")),
    ("500 CHF to GBP",                       tc("currency", amount=500,  **{"from":"CHF"}, to="GBP")),
    ("750 AUD to EUR",                       tc("currency", amount=750,  **{"from":"AUD"}, to="EUR")),
    # sql
    ("SELECT name FROM products WHERE in_stock=1", tc("sql", query="SELECT name FROM products WHERE in_stock=1")),
    ("Run: SELECT * FROM audit_log LIMIT 50",       tc("sql", query="SELECT * FROM audit_log LIMIT 50")),
    # refusals
    ("What's the traffic like?",             "I don't have a traffic monitoring tool. I can help with weather, calendar, conversions, currency, and SQL."),
    ("Can you fix my code?",                 "I'm specialized for tool calls—I don't have a code editor. Let me know if I can help with weather, calendar, conversions, currency, or SQL."),
    ("Who won the World Cup?",               "I don't have live sports data. I can help with weather, calendar, conversions, currency, and SQL."),
]
for q, a in EXTRA:
    examples.append(ex(q, a))

# ── Shuffle & write ───────────────────────────────────────────────────
random.shuffle(examples)
out_path = "data/train.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"✅  Generated {len(examples)} training examples → {out_path}")
