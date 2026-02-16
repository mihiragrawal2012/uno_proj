from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]  # subject/from/to/file/etc.
