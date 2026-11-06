"""
Banking Seminar · Day 1 (LangChain) — Step 1: Talk to a model.

The tiniest possible step: send one message to the AI and print the reply.
Everything else today builds on this.

Run it:   python banking/day_1/01_chat.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# The .env file (with your OpenRouter key) lives at the repo root.
load_dotenv(Path(__file__).resolve().parent / ".env")

# OpenRouter speaks the OpenAI API, so we use ChatOpenAI pointed at it.
model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,  # 0 = focused & repeatable
)

# Type your own question, or press Enter to use the example hint.
EXAMPLE = "In one simple sentence, what is a savings account?"
question = input(f"Ask the bank's AI anything [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE

reply = model.invoke(question)

print("\n" + reply.content)
