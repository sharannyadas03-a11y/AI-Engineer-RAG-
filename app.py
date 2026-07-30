#!/usr/bin/env python3
"""
CLI entry point for the RAG policy assistant.

Usage:
    python app.py                 # interactive chat
    python app.py --ask "..."     # single question, non-interactive
    python app.py --stream        # interactive chat with streamed output
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv not installed; env vars can still be set manually

from src.chunker import load_and_chunk_documents
from src.retriever import HybridRetriever
from src.conversation import Conversation
from src.llm import generate_answer, stream_answer
from src.feedback import record_feedback

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "feedback", "feedback_log.jsonl")


def build_retriever() -> HybridRetriever:
    chunks = load_and_chunk_documents(DATA_DIR)
    return HybridRetriever(chunks)


def answer_question(retriever: HybridRetriever, conversation: Conversation,
                     question: str, doc_filter=None, stream: bool = False):
    retrieval_query = conversation.rewrite_query_for_retrieval(question)
    results = retriever.search(retrieval_query, top_k=5, doc_filter=doc_filter)
    history = conversation.history_for_llm()

    conversation.add_user_turn(question)

    if stream:
        print("Assistant: ", end="", flush=True)
        full = ""
        for token in stream_answer(question, results, history):
            print(token, end="", flush=True)
            full += token
        print()
        answer = full.strip()
    else:
        answer = generate_answer(question, results, history)
        print(f"\nAssistant: {answer}")

    conversation.add_assistant_turn(answer)

    sources = sorted({f"{r.chunk.doc_name} > {r.chunk.section_title}" for r in results})
    if sources:
        print("\nSources consulted:")
        for s in sources:
            print(f"  - {s}")
    else:
        print("\n(No matching sources were found in the document set.)")

    return answer, sources


def interactive_loop(stream: bool = False):
    print("Building index from ./data ...")
    retriever = build_retriever()
    print(f"Indexed {len(retriever.chunks)} chunks from "
          f"{len({c.source_file for c in retriever.chunks})} documents.\n")
    print("Ask a question about expense, travel, finance, or HR policy.")
    print("Commands: ':docs' list documents | ':filter <doc>' restrict search | "
          "':filter clear' | ':feedback up|down' rate last answer | ':quit'\n")

    conversation = Conversation()
    doc_filter = None
    last_q, last_a, last_sources = None, None, []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in (":quit", ":exit"):
            print("Goodbye.")
            break
        if question.lower() == ":docs":
            docs = sorted({c.doc_name for c in retriever.chunks})
            print("Available documents:", ", ".join(docs))
            continue
        if question.lower().startswith(":filter"):
            arg = question[len(":filter"):].strip()
            if not arg or arg.lower() == "clear":
                doc_filter = None
                print("Filter cleared.")
            else:
                doc_filter = [arg]
                print(f"Filtering to documents matching: {arg}")
            continue
        if question.lower().startswith(":feedback"):
            rating = question[len(":feedback"):].strip().lower()
            if rating in ("up", "down") and last_q:
                record_feedback(FEEDBACK_PATH, last_q, last_a, last_sources, rating)
                print(f"Thanks — recorded '{rating}' feedback on the last answer.")
            else:
                print("Usage: ':feedback up' or ':feedback down' (after an answer).")
            continue

        last_a, last_sources = answer_question(
            retriever, conversation, question, doc_filter=doc_filter, stream=stream
        )
        last_q = question
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG Policy Assistant")
    parser.add_argument("--ask", type=str, help="Ask a single question and exit")
    parser.add_argument("--filter", type=str, help="Restrict to a document name")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    args = parser.parse_args()

    if args.ask:
        retriever = build_retriever()
        conversation = Conversation()
        doc_filter = [args.filter] if args.filter else None
        answer_question(retriever, conversation, args.ask, doc_filter=doc_filter,
                         stream=args.stream)
    else:
        interactive_loop(stream=args.stream)


if __name__ == "__main__":
    main()
