"""Milestone 3 — per-session temp directories, limits and cleanup for the Gradio app.

To implement in M3:

* ``Session`` with a unique id, a temp directory (uploads/ and index/), created_at,
  question_count. ``Session.new(root)``; ``session.path``.
* ``SessionStore`` keeping sessions in memory: ``get(id)``, ``create()``,
  ``cleanup(ttl_hours)`` deleting directories older than the TTL (called on a timer
  and on every new session), ``clear(id)``.
* Limits enforced here (not in the UI code): max files, max MB per file, max total
  pages, max questions per session — all from ``config.Settings``.
* No document text is ever logged. Log only counts, sizes and timings.
"""
