"""QMDSearcher — async subprocess wrapper for the QMD re-ranking CLI."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory_index.search import SearchResult


class QMDSearcher:
    """Thin async wrapper around an external QMD binary for LLM-based re-ranking.

    Protocol: send JSON to stdin, read ranked IDs from stdout. Falls back
    silently to the original candidate order on any error or if the binary
    is not installed.
    """

    _TIMEOUT = 10.0  # seconds

    def __init__(self, binary: str) -> None:
        self._binary = binary

    def is_available(self) -> bool:
        """Return True if the QMD binary can be found on PATH (or is an absolute path)."""
        import shutil
        return shutil.which(self._binary) is not None

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Re-rank candidates via the QMD subprocess; fallback on any failure.

        If the binary is unavailable or the subprocess fails, returns
        candidates[:top_k] unchanged.
        """
        if not candidates:
            return []
        if not self.is_available():
            logger.debug("QMD binary '{}' not found, skipping re-rank", self._binary)
            return candidates[:top_k]

        payload = {
            "query": query,
            "candidates": [
                {
                    "id": i,
                    "text": r.text,
                    "source": r.source,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": r.score,
                }
                for i, r in enumerate(candidates)
            ],
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                self._binary,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(json.dumps(payload).encode()),
                timeout=self._TIMEOUT,
            )
            data = json.loads(stdout)
            ranked_ids: list[int] = data["ranked"]
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            logger.warning("QMD reranking timed out, falling back to original order")
            return candidates[:top_k]
        except Exception:
            logger.warning("QMD reranking failed, falling back to original order")
            return candidates[:top_k]

        id_to_result = dict(enumerate(candidates))
        seen: set[int] = set()
        reranked = []
        for rid in ranked_ids:
            if rid in id_to_result and rid not in seen:
                reranked.append(id_to_result[rid])
                seen.add(rid)
        # Append any candidates not mentioned by QMD (safety net)
        for i, r in enumerate(candidates):
            if i not in seen:
                reranked.append(r)
        return reranked[:top_k]
