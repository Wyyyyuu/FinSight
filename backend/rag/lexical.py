"""Optional CJK lexical channel; never changes existing dense preprocessing."""
from collections import Counter
import re
import unicodedata

from backend.rag.embedder import SparseVector

VERSION = "cjk-bigram-v1"
_PARTS = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def lexical_tokens(text: str) -> list[str]:
    tokens = []
    for part in _PARTS.findall(unicodedata.normalize("NFKC", text).lower()):
        if "\u3400" <= part[0] <= "\u9fff" and len(part) > 1:
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def lexical_sparse(text: str) -> SparseVector:
    counts = Counter(lexical_tokens(text))
    scale = max(counts.values(), default=1)
    return SparseVector({token: count / scale for token, count in counts.items()})


def lexical_query(text: str) -> str:
    # Only regex-produced alphanumeric/CJK tokens; no user tsquery operators.
    return " | ".join(dict.fromkeys(lexical_tokens(text)))
