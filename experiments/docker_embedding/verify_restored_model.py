"""Confirm the restored CPU model agrees with vectors produced before reboot."""
import json
import numpy as np
import requests
from evaluate import Cache, DATA, HERE, save, sha

health = requests.get('http://127.0.0.1:8010/health',timeout=5).json()
expected = json.loads((HERE/'cache/embedding_health.json').read_text())
assert health==expected, 'Restored model profile changed'
cache = Cache(health)
data = json.loads(DATA.read_text(encoding='utf-8'))
texts = [d['contextual'] for d in data['documents']]
texts += [q['question'] for q in data['questions']]
sample = sorted(set(t for t in texts if cache.db.execute('SELECT 1 FROM vectors WHERE id=?',(cache.key(t),)).fetchone()),key=sha)[:16]
encoded = cache.encoder.encode(sample)
assert encoded.profile==cache.profile
dense_error, sparse_error = [], []
for text,dense,sparse in zip(sample,encoded.dense,encoded.sparse):
    prior_dense, prior_sparse = cache.get(text)
    dense_error.append(float(np.max(np.abs(prior_dense-np.asarray(dense)))))
    sparse_error.append(max((abs(prior_sparse.get(k,0)-sparse.weights.get(k,0)) for k in prior_sparse.keys()|sparse.weights.keys()),default=0))
assert max(dense_error)<1e-5 and max(sparse_error)<1e-4
result=dict(profile=cache.profile,sample_text_sha256=[sha(t) for t in sample],samples=len(sample),
            max_dense_difference=max(dense_error),max_sparse_difference=max(sparse_error))
save(HERE/'restored_model_check.json',result)
print(result)
