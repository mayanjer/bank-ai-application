"""
Banking Seminar · Day 2 (LangGraph) — Step 2: The bank assistant as a GRAPH.

Day 1 answered EVERY question with one straight line. A graph can make
DECISIONS. This assistant first ROUTES the message, then takes a different
path depending on what it is:

                       +--> transaction -> (withdraw? -> HUMAN approves) -> do it
   message -> router --+--> policy      -> retrieve from bank docs -> answer
                       +--> sensitive   -> human review (fraud, disputes)

It brings the whole seminar together in one project: routing, structured
output, transactions (tools), RAG, and a human in the loop.

Run it:   python banking/day_2/02_banking_assistant.py   (type 'quit' to exit)
"""

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import warnings

# Keep the workshop terminal clean: hide the "langchain-community is being
# sunset" deprecation notice. The library still works fine.
warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parent / "data"
load_dotenv(ROOT / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# Build the policy retriever once (same idea as Day 1).
_docs = []
for _md in DATA.glob("*.md"):
    _docs.extend(TextLoader(str(_md), encoding="utf-8").load())
_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(_docs)
retriever = BM25Retriever.from_documents(_chunks)
retriever.k = 2

# The account details live in a plain JSON file (no database needed). Open
# banking/data/account.json to watch the balance change as money moves.
ACCOUNT_FILE = Path(__file__).resolve().parent / "data" / "account.json"


def load_account() -> dict:
    """Read the account details from the JSON file."""
    return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))


def save_account(account: dict) -> None:
    """Write the account details back to the JSON file (saves to disk)."""
    ACCOUNT_FILE.write_text(json.dumps(account, indent=2), encoding="utf-8") 
    


# ---------------------------------------------------------------------------
# STATE: the shared data passed between nodes.
# ---------------------------------------------------------------------------
class State(TypedDict):
    message: str
    route: str
    answer: str


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def router(state: State) -> dict:
    """Ask the model to sort the message into one of three buckets."""
    decision = model.invoke(
        [
            (
                "system",
                "Classify the user's message into exactly ONE word:\n"
                "- 'transaction' if they want to check their balance, deposit, "
                "or withdraw money\n"
                "- 'sensitive' if it is about fraud, a lost or stolen card, a "
                "dispute, a complaint, or anything risky\n"
                "- 'policy' for questions about the bank's rules, fees, limits, "
                "interest or opening hours\n"
                "Reply with ONLY that one word.",
            ),
            ("user", state["message"]),
        ]
    ).content.lower()

    route = next((r for r in ("sensitive", "transaction", "policy") if r in decision), "policy")
    print(f"[router] -> {route}")
    return {"route": route}


# A tidy, typed shape for a banking instruction (structured output).
class Intent(BaseModel):
    action: Literal["balance", "deposit", "withdraw"] = Field(
        description="The kind of banking action the user wants"
    )
    amount: float = Field(description="The amount of money, or 0 if none is mentioned")


def do_transaction(state: State) -> dict:
    """Transaction path: understand the instruction, then do it safely."""
    intent = model.with_structured_output(Intent).invoke(
        "Extract the banking action and amount from this message: " + state["message"]
    )
    action, amount = intent.action.lower(), intent.amount
    account = load_account()

    if "balance" in action:
        return {"answer": f"Your balance is UGX {account['balance']:,.0f}."}

    if "deposit" in action:
        if amount <= 0:
            return {"answer": "Please tell me a positive amount to deposit."}
        account["balance"] += amount
        save_account(account)
        return {"answer": f"Deposited UGX {amount:,.0f}. New balance: UGX {account['balance']:,.0f}."}

    if "withdraw" in action:
        if amount <= 0:
            return {"answer": "Please tell me a positive amount to withdraw."}
        if amount > account["balance"]:
            return {"answer": f"Insufficient funds. Your balance is only UGX {account['balance']:,.0f}."}
        # HUMAN IN THE LOOP: a person must approve money leaving the account.
        print(f"\n[APPROVAL NEEDED] The assistant wants to withdraw UGX {amount:,.0f}.")
        ok = input("Approve this withdrawal? (yes/no): ").strip().lower()
        if ok not in {"yes", "y"}:
            return {"answer": "The withdrawal was declined by the human reviewer."}
        account["balance"] -= amount
        save_account(account)
        return {"answer": f"Withdrew UGX {amount:,.0f}. New balance: UGX {account['balance']:,.0f}."}

    return {"answer": "I can check your balance, make a deposit, or make a withdrawal."}


def answer_policy(state: State) -> dict:
    """Policy path: find relevant text, answer grounded in it, cite sources."""
    found = retriever.invoke(state["message"])
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}" for d in found
    )
    ans = model.invoke(
        [
            (
                "system",
                "Answer using ONLY the context and cite the source file. If the "
                "answer is not there, say: 'I could not find this in the bank's "
                "policies.'\n\n" + context,
            ),
            ("user", state["message"]),
        ]
    ).content
    return {"answer": ans}


def human_review(state: State) -> dict:
    """Sensitive path: pause and let a human handle it (fraud, disputes...)."""
    print("\n[HUMAN REVIEW NEEDED] This looks sensitive (e.g. fraud or a dispute).")
    print("Message:", state["message"])
    reply = input("Type an approved response for the customer (or Enter to escalate): ").strip()
    return {"answer": reply or "This needs a human. I have forwarded it to our support team."}


# ---------------------------------------------------------------------------
# EDGES: wire the nodes together, with one DECISION after the router.
# ---------------------------------------------------------------------------
def choose_path(state: State) -> str:
    return {"transaction": "do_transaction", "sensitive": "human_review"}.get(
        state["route"], "answer_policy"
    )


builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("do_transaction", do_transaction)
builder.add_node("answer_policy", answer_policy)
builder.add_node("human_review", human_review)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router", choose_path, ["do_transaction", "answer_policy", "human_review"]
)
builder.add_edge("do_transaction", END)
builder.add_edge("answer_policy", END)
builder.add_edge("human_review", END)

graph = builder.compile()


if __name__ == "__main__":
    print("Sandbox Bank Assistant (LangGraph + human-in-the-loop). Type 'quit' to exit.")
    print("Examples:  What's my balance?  |  Withdraw 200000  |  "
          "What is the withdrawal limit?  |  I lost my card\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        result = graph.invoke({"message": q})
        print("Assistant:", result["answer"], "\n")
