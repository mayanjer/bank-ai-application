"""
Banking Seminar · Day 1 (LangChain) — Step 2: Prompts & structured output.

Two ideas:
  1. A PROMPT TEMPLATE: a fixed role + a blank we fill in.
  2. STRUCTURED OUTPUT: make the model return tidy FIELDS, not a paragraph.
     Here we turn a plain instruction like "deposit fifty thousand shillings"
     into {action: "deposit", amount: 50000} -- data our code can actually use.

Run it:   python banking/day_1/02_prompts.py
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# 1) A template: a fixed "teller" role + a "{topic}" placeholder.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly bank teller. Explain in 2-3 simple sentences."),
        ("user", "Explain this to a beginner: {topic}"),
    ]
)

# 2) A chain: prompt -> model -> plain text. Read the | as "then".
chain = prompt | model | StrOutputParser()

TOPIC_EXAMPLE = "what is interest on a savings account"
topic = input(f"Banking topic to explain [Enter for: {TOPIC_EXAMPLE}]: ").strip() or TOPIC_EXAMPLE

print("\n=== Templated answer ===")
print(chain.invoke({"topic": topic}))


# 3) Structured output: turn a plain instruction into tidy, typed fields.
#    Using Literal locks "action" to a fixed list, so the model must pick one.
class Transaction(BaseModel):
    action: Literal["balance", "deposit", "withdraw"] = Field(
        description="The kind of banking action the user wants"
    )
    amount: float = Field(description="The amount of money, or 0 if none is mentioned")


SENTENCE_EXAMPLE = "I want to deposit fifty thousand shillings"
sentence = input(
    f"\nA banking instruction [Enter for: {SENTENCE_EXAMPLE}]: "
).strip() or SENTENCE_EXAMPLE

tx = model.with_structured_output(Transaction).invoke(
    "Extract the banking action and amount from this instruction: " + sentence
)

print("\n=== Structured output ===")
print("Action:", tx.action)
print("Amount:", tx.amount)
