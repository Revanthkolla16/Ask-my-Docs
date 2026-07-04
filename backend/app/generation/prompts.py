"""
backend/app/generation/prompts.py
-----------------------------------
Prompt templates and context-formatting utilities for the citation-grounded
generation step.

The system prompt instructs the model to:
  - Answer *only* from the provided context.
  - Return a strict JSON object with keys ``answer`` and ``citations``.
  - Never cite a chunk_id that was not in the supplied context.

Functions
---------
format_context(chunks)
    Turn a list of RetrievalResult objects into a numbered, ID-tagged
    context block that is safe to insert into the user turn.

build_messages(query, context_chunks)
    Assemble the full messages list (system + user) ready to send to the
    Groq chat API.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.dense import RetrievalResult


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise, citation-grounded research assistant.

Rules you MUST follow:
1. Answer the user's question using ONLY the information in the CONTEXT block below.
2. If the context does not contain enough information to answer the question,
   say "I don't have enough information in the provided documents to answer this."
3. Your response MUST be a single valid JSON object with exactly two keys:
   - "answer": a clear, concise answer string (markdown allowed)
   - "citations": a JSON array of chunk_id strings that directly support your answer.
     Only include chunk_ids that appear verbatim in the CONTEXT block.
4. Do NOT invent facts, do NOT cite sources outside the CONTEXT block.
5. Keep your answer focused — avoid restating the entire context.

RESPONSE FORMAT (strict JSON, no markdown fences):
{
  "answer": "...",
  "citations": ["chunk_id_1", "chunk_id_2"]
}
"""


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(chunks: "List[RetrievalResult]") -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is rendered as::

        [chunk_id: <id>]
        Source: <source> | Page: <page> | Section: <section>
        <text>

    Parameters
    ----------
    chunks:
        List of ``RetrievalResult`` objects from the retrieval/reranking stage.

    Returns
    -------
    str
        A multi-line string ready to be embedded in the user message.
    """
    if not chunks:
        return "(No context available.)"

    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page_num", "—")
        section = meta.get("section", "—")

        lines.append(f"[{i}] [chunk_id: {chunk.chunk_id}]")
        lines.append(f"Source: {source} | Page: {page} | Section: {section}")
        lines.append(chunk.text.strip())
        lines.append("")          # blank line between chunks

    return "\n".join(lines).rstrip()


# ── Message builder ───────────────────────────────────────────────────────────

def build_messages(query: str, context_chunks: "List[RetrievalResult]") -> list:
    """
    Build the ``messages`` list for the Groq chat completion API.

    Parameters
    ----------
    query:
        The user's natural-language question.
    context_chunks:
        Reranked chunks to embed as context.

    Returns
    -------
    list
        ``[{"role": "system", ...}, {"role": "user", ...}]``
    """
    context_block = format_context(context_chunks)

    user_content = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "Respond with a JSON object following the format in the system prompt."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
