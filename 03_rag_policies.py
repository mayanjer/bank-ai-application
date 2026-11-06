"""
Banking Seminar · Day 1 (LangChain) — Step 3: RAG over the bank's policies.

RAG = Retrieval-Augmented Generation. The shape is always:
  1. Keep a set of approved documents (here: the bank FAQ + policies).
  2. RETRIEVE the pieces most relevant to the question.
  3. Answer from THEM (grounded) and name the source.

We keep it simple: NO embeddings, NO downloads -- just keyword search (BM25).

Run it:   python banking/day_1/03_rag_policies.py
"""

import os
from pathlib import Path

import warnings

# Keep the workshop terminal clean: hide the "langchain-community is being
# sunset" deprecation notice. The library still works fine.
warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent           # repo root (.env lives here)
DATA = Path(__file__).resolve().parent / "data"   # banking/data
load_dotenv(ROOT / ".env")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    temperature=0,
)

# --- Load + split + index the bank's documents ---
docs = []
for md_file in DATA.glob("*.md"):
    docs.extend(TextLoader(str(md_file), encoding="utf-8").load())
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,  # each piece is targeted to be about 500 characters or fewer,
    # although some boundaries may cause exceptions.
    chunk_overlap=50 # neighboring chunks share approximately 50 characters to help preserve context.
).split_documents(docs)
retriever = BM25Retriever.from_documents(chunks) # Looks for relevant words such as daily, withdrawal, and limit.
retriever.k = 2  # return the 2 best-matching chunks


def answer(question: str) -> str:
    found = retriever.invoke(question)
    context = "\n\n".join(
        f"[source: {Path(d.metadata['source']).name}]\n{d.page_content}"
        for d in found
    )
    messages = [
        (
            "system",
            "Answer the question using ONLY the context below. Cite the source "
            "file name. If the answer is not in the context, say: "
            "'I could not find this in the bank's policies.'\n\n" + context,
        ),
        ("user", question),
    ]
    return model.invoke(messages).content


if __name__ == "__main__":
    EXAMPLE = "What is the daily withdrawal limit?"
    q = input(f"Ask about the bank's policies [Enter for: {EXAMPLE}]: ").strip() or EXAMPLE
    print("\nQ:", q)
    print("A:", answer(q))
