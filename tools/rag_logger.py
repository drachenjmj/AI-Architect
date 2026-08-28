"""rag_logger.py - Rule-based RAG query logging (stdlib only).

Writes one JSON object per line (JSON-Lines) to rag_log.jsonl so the Chroma
DISTANCE_THRESHOLD can be derived empirically before it is activated.
No frameworks, no LLM calls - pure observation.
"""
import json
import sys
from datetime import datetime

RAG_LOG_PATH = "rag_log.jsonl"


class RagLogger:
    """Append-only JSON-Lines logger for RAG retrieval calls."""

    def __init__(self, path: str = RAG_LOG_PATH):
        self._path = path

    def log_query(
        self,
        query: str,
        duration_ms: float,
        chunks_returned: int,
        best_distance,
        source_files: list,
        status: str,
    ) -> None:
        """Append a single retrieval record to the JSON-Lines log file.

        status is one of (str):
        - "success":               KB hit(s) within DISTANCE_THRESHOLD
        - "no_results":            Chroma returned nothing
        - "below_threshold":       Chroma hits existed but all exceeded
                                   DISTANCE_THRESHOLD
        - "web_fallback":          web fallback ran and returned >=1 grounded
                                   chunk (successful web fallback)
        - "web_fallback_empty":    web fallback ran but yielded no usable
                                   grounded chunks
        - "web_fallback_error":    web fallback raised (API/network failure)
        - "web_fallback_disabled": web fallback skipped via the
                                   WEB_FALLBACK_ENABLED kill switch
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "duration_ms": round(duration_ms, 1),
            "chunks_returned": chunks_returned,
            "best_distance": best_distance,
            "source_files": source_files,
            "status": status,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"RagLogger: failed to write {self._path}: {e}", file=sys.stderr)
