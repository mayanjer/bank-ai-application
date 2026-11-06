
"""
Banking Seminar · Day 1 (LangChain) — Step 4: Tools and a simple agent.

A TOOL is just a Python function the model is allowed to call.
An AGENT is a model that decides WHICH tool to call and WHEN, reads the
result, and then writes the final answer.

Here the model can check a balance, deposit, or withdraw money.
(On Day 2 we add a HUMAN approval step before money leaves the account.)

Run it:   python banking/day_1/04_tools.py   (type 'quit' to exit)
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parent / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# The account details live in a plain JSON file -- our "database without a
# database". Open banking/data/account.json and watch the balance change as
# you deposit and withdraw. (On Day 2 we also show a REAL vector database.)
ACCOUNT_FILE = Path(__file__).resolve().parent / "data" / "account.json"


def load_account() -> dict:
    """Read the account details from the JSON file."""
    return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))


def save_account(account: dict) -> None:
    """Write the account details back to the JSON file (saves to disk)."""
    ACCOUNT_FILE.write_text(json.dumps(account, indent=2), encoding="utf-8")


def check_balance() -> str:
    """Return the current account balance."""
    account = load_account()
    return f"Your balance is UGX {account['balance']:,.0f}."


def deposit(amount: float) -> str:
    """Add money to the account. 'amount' must be positive."""
    if amount <= 0:
        return "Deposit amount must be positive."
    account = load_account()
    account["balance"] += amount
    save_account(account)
    return f"Deposited UGX {amount:,.0f}. New balance: UGX {account['balance']:,.0f}."


def withdraw(amount: float) -> str:
    """Take money out of the account."""
    if amount <= 0:
        return "Withdrawal amount must be positive."
    account = load_account()
    if amount > account["balance"]:
        return f"Insufficient funds. Your balance is only UGX {account['balance']:,.0f}."
    account["balance"] -= amount
    save_account(account)
    return f"Withdrew UGX {amount:,.0f}. New balance: UGX {account['balance']:,.0f}."

def read_account_name() -> str:
    """The tool picks the account name"""
    account = load_account()
    return account["name"]

   


agent = create_agent(
    model=model,
    tools=[check_balance, deposit, withdraw, read_account_name],
    system_prompt=(
        "You are a helpful bank assistant. Use a tool to check the balance, "
        "deposit, or withdraw money -- never make up numbers. Confirm what you "
        "did in one short sentence."
    ),
)


def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("Assistant:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    print("Bank Assistant with tools (LangChain). Type 'quit' to exit.")
    print("Examples:  What's my balance?  |  Deposit 100000  |  Withdraw 50000\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        ask(q)
