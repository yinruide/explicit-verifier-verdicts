"""Run logging: JSONL writer, record kinds, run snapshots.

Note: this package is ``rafr.logging``; with the ``src`` layout and absolute imports it
never shadows the standard library ``logging`` module (see README).
"""

from rafr.logging.schema import RecordKind
from rafr.logging.snapshot import write_snapshot
from rafr.logging.writer import JsonlWriter, read_jsonl

__all__ = ["JsonlWriter", "RecordKind", "read_jsonl", "write_snapshot"]
