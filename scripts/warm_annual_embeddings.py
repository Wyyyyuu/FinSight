"""Download and verify the actual Chinese ONNX embedding model before serving."""

import os
from pathlib import Path

from fastembed import TextEmbedding


def main() -> None:
    cache = Path(
        os.getenv("ANNUAL_REPORT_MODEL_CACHE")
        or Path(__file__).resolve().parents[1] / ".annual-models"
    )
    model = TextEmbedding(
        model_name="BAAI/bge-small-zh-v1.5", cache_dir=str(cache), threads=2
    )
    vectors = list(
        model.embed(["经营现金流下降，主要由于应收账款增加。", "营业收入同比增长。"])
    )
    if len(vectors) != 2 or len(vectors[0]) != 512:
        raise RuntimeError("中文模型未返回预期向量，不能启用语义检索。")
    print(f"BAAI/bge-small-zh-v1.5 就绪，维度 512；缓存：{cache}")


if __name__ == "__main__":
    main()
