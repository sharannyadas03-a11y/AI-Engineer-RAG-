#!/usr/bin/env python3
"""
Minimal Flask web UI for the RAG policy assistant.

This is a thin wrapper around the same retriever/generation/conversation
code used by app.py (CLI) -- no logic is duplicated. Session state (chat
history) is kept server-side in-memory per browser session, which is fine
for a single-user demo/assignment; a production version would move this to
a proper session store (see README -> Improvements).
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv not installed; env vars can still be set manually

from flask import Flask, request, jsonify, session, render_template_string

from src.chunker import load_and_chunk_documents
from src.retriever import HybridRetriever
from src.conversation import Conversation
from src.llm import generate_answer
from src.feedback import record_feedback

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
FEEDBACK_PATH = os.path.join(APP_DIR, "feedback", "feedback_log.jsonl")

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

RETRIEVER = HybridRetriever(load_and_chunk_documents(DATA_DIR))
CONVERSATIONS = {}  # session_id -> Conversation
LAST_ANSWER = {}     # session_id -> {question, answer, sources}

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Policy Assistant (RAG)</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 800px;
         margin: 40px auto; background: #f6f5f3; color: #1f1f1f; }
  h1 { font-size: 1.3rem; }
  #chat { background: white; border-radius: 10px; padding: 16px; min-height: 300px;
          max-height: 60vh; overflow-y: auto; border: 1px solid #e5e2dd; }
  .msg { margin: 10px 0; line-height: 1.45; }
  .user { font-weight: 600; }
  .assistant { white-space: pre-wrap; }
  .sources { font-size: 0.85rem; color: #666; margin-top: 4px; }
  form { display: flex; gap: 8px; margin-top: 12px; }
  input[type=text] { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
  select, button { padding: 10px; border-radius: 8px; border: 1px solid #ccc; background: white; }
  button.fb { padding: 4px 8px; margin-left: 6px; font-size: 0.8rem; cursor: pointer; }
</style>
</head>
<body>
<h1>Employee Policy Assistant (RAG demo)</h1>
<p style="color:#666; font-size:0.9rem;">Answers are grounded only in the loaded policy
documents (Expense, Travel, Finance, Employee Handbook). If no API key is configured this
runs in offline extractive mode.</p>
<div id="chat"></div>
<form id="f">
  <select id="filter">
    <option value="">All documents</option>
    <option>Expense Policy</option>
    <option>Travel Policy</option>
    <option>Finance Policy</option>
    <option>Employee Handbook</option>
  </select>
  <input id="q" type="text" placeholder="Ask a policy question..." autofocus>
  <button type="submit">Ask</button>
</form>
<script>
const chat = document.getElementById('chat');
let lastId = null;
function addMsg(role, text, sources) {
  const d = document.createElement('div');
  d.className = 'msg';
  const who = role === 'user' ? 'You' : 'Assistant';
  d.innerHTML = `<div class="${role}">${who}: ${text.replace(/</g,'&lt;')}</div>`;
  if (sources && sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.textContent = 'Sources: ' + sources.join(' | ');
    d.appendChild(s);
  }
  if (role === 'assistant') {
    const up = document.createElement('button');
    up.className = 'fb'; up.textContent = '👍'; up.onclick = () => sendFeedback('up');
    const down = document.createElement('button');
    down.className = 'fb'; down.textContent = '👎'; down.onclick = () => sendFeedback('down');
    d.appendChild(up); d.appendChild(down);
  }
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}
async function sendFeedback(rating) {
  await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({rating})});
}
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  const filter = document.getElementById('filter').value;
  if (!q) return;
  addMsg('user', q);
  document.getElementById('q').value = '';
  const resp = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({question: q, filter})});
  const data = await resp.json();
  addMsg('assistant', data.answer, data.sources);
});
</script>
</body>
</html>
"""


def get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    sid = get_session_id()
    conv = CONVERSATIONS.setdefault(sid, Conversation())
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip()
    doc_filter = [payload["filter"]] if payload.get("filter") else None

    if not question:
        return jsonify({"answer": "Please enter a question.", "sources": []})

    retrieval_query = conv.rewrite_query_for_retrieval(question)
    results = RETRIEVER.search(retrieval_query, top_k=5, doc_filter=doc_filter)
    history = conv.history_for_llm()

    conv.add_user_turn(question)
    answer = generate_answer(question, results, history)
    conv.add_assistant_turn(answer)

    sources = sorted({f"{r.chunk.doc_name} > {r.chunk.section_title}" for r in results})
    LAST_ANSWER[sid] = {"question": question, "answer": answer, "sources": sources}

    return jsonify({"answer": answer, "sources": sources})


@app.route("/feedback", methods=["POST"])
def feedback():
    sid = get_session_id()
    payload = request.get_json(force=True)
    rating = payload.get("rating")
    last = LAST_ANSWER.get(sid)
    if not last or rating not in ("up", "down"):
        return jsonify({"ok": False}), 400
    record_feedback(FEEDBACK_PATH, last["question"], last["answer"], last["sources"], rating)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
