"""Same cached FP32 cross encoder on CUDA; verify kernel execution at startup."""
from contextlib import asynccontextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort
from fastembed.common.preprocessor_utils import load_tokenizer

lock = Lock()
session = tokenizer = None
identity = {}


def predict(query, documents):
    if len(documents) > 8:
        return [score for i in range(0,len(documents),8) for score in predict(query,documents[i:i+8])]
    encoded = tokenizer.encode_batch([(query, d) for d in documents])
    fields = {'input_ids':'ids', 'attention_mask':'attention_mask','token_type_ids':'type_ids'}
    feed = {node.name:np.asarray([getattr(item,fields[node.name]) for item in encoded],dtype=np.int64)
            for node in session.get_inputs()}
    return [float(x) for x in session.run(None,feed)[0][:,0]]


@asynccontextmanager
async def lifespan(app):
    global session, tokenizer, identity
    ort.preload_dlls(directory='', msvc=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    opts.enable_profiling = True
    opts.profile_file_prefix = '/tmp/cuda-warmup'
    path = Path('/model')
    tokenizer, _ = load_tokenizer(path)
    session = ort.InferenceSession(str(path/'onnx/model.onnx'),sess_options=opts,
                                   providers=[('CUDAExecutionProvider', {'use_tf32':0}), 'CPUExecutionProvider'])
    assert session.get_providers()[0] == 'CUDAExecutionProvider', 'CUDA required; CPU fallback refused'
    session.disable_fallback()
    scores = predict('company revenue', ['company revenue increased','weather forecast'])
    assert all(np.isfinite(scores)) and scores[0] > scores[1]
    profile = json.loads(Path(session.end_profiling()).read_text())
    kernels = [r for r in profile if r.get('args',{}).get('provider')=='CUDAExecutionProvider']
    assert kernels and any(r['args'].get('op_name') in {'MatMul','Gemm','Attention','FusedMatMul'} for r in kernels)
    with (path/'onnx/model.onnx').open('rb') as stream:
        weights = hashlib.file_digest(stream,'sha256').hexdigest()
    identity = {'ready':True,'model':'bge-reranker-base','provider':'CUDAExecutionProvider',
                'onnxruntime':ort.__version__,'weights_sha256':weights,
                'tokenizers':importlib.metadata.version('tokenizers'),
                'fastembed':importlib.metadata.version('fastembed-gpu'),
                'revision':'2cfc18c9415c912f9d8155881c133215df768a70',
                'tf32':False,'batch_size':8,'cuda_kernel_count':len(kernels)}
    yield


app = FastAPI(lifespan=lifespan)


class Request(BaseModel):
    query: str
    documents: list[str] = Field(min_length=1,max_length=64)


@app.get('/health')
def health():
    return identity


@app.post('/rerank')
def rerank(request: Request):
    with lock:
        scores = predict(request.query,request.documents)
    return {**identity,'scores':scores}
