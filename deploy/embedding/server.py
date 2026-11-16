"""Local-only FlagEmbedding service. Startup fails if real model loading fails."""
from contextlib import asynccontextmanager
import os
from threading import Lock

import torch
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel

MODEL = "BAAI/bge-m3"
MAX_LENGTH = int(os.getenv("BGE_M3_MAX_LENGTH", "512"))
DEVICE = os.getenv("BGE_M3_DEVICE", "cpu")
lock = Lock()
model = None
revision = ""


@asynccontextmanager
async def lifespan(app):
    global model, revision
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
    path = snapshot_download(MODEL, revision=os.getenv("BGE_M3_REVISION", "main"),
                             allow_patterns=["*.json", "model.safetensors", "pytorch_model.bin",
                                             "*.pt", "sentencepiece.bpe.model"])
    revision = path.rsplit("/", 1)[-1]
    model = BGEM3FlagModel(path, devices=DEVICE, use_fp16=DEVICE != "cpu")
    model.encode(["模型就绪 model ready"], max_length=MAX_LENGTH, return_dense=True, return_sparse=True)
    yield


app = FastAPI(lifespan=lifespan)


class EncodeRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


@app.get("/health")
def health():
    return {"ready": model is not None, "model": "bge-m3", "dim": 1024,
            "revision": f"{revision}-maxlen{MAX_LENGTH}-{DEVICE}", "device": DEVICE,
            "max_length": MAX_LENGTH}


@app.post("/encode")
def encode(request: EncodeRequest):
    with lock, torch.inference_mode():
        result = model.encode(request.texts, batch_size=int(os.getenv("BGE_M3_BATCH_SIZE", "8")),
                              max_length=MAX_LENGTH, return_dense=True, return_sparse=True,
                              return_colbert_vecs=False)
    return {**health(), "dense": result["dense_vecs"].tolist(),
            "sparse": [{str(k): float(v) for k, v in row.items()} for row in result["lexical_weights"]]}
