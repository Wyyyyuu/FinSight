"""Recover verified pre-reboot CPU pair scores after checking overlapping reruns."""
import json
import sqlite3
from evaluate import DATA, HERE, reranker_identity, sha, save

old_code = 'ba66492f0d031e20e6529b5e784e109fd9da9a1d5b3d65a744ef97f655596e5d'
expected_profile = '273a9a1704f183df32f5538245ba13ee6d45b3ece468c16c504fac99a4353c09'
current = reranker_identity()
old = {k:current[k] for k in ['files','fastembed']}
old['code'] = old_code
old_profile = sha(json.dumps(old,sort_keys=True))
assert old_profile == expected_profile, 'Old model/code identity no longer matches'
new_profile = sha(json.dumps(current,sort_keys=True))
data = json.loads(DATA.read_text(encoding='utf-8'))
cases = [(q['question'],[q['collection']]) for q in data['questions'] if q['split']=='dev']
cases += [(t['question'],t['collections']) for t in data['multi'] if t['split']=='dev']
db = sqlite3.connect(HERE/'cache/rerank.sqlite',timeout=60)
assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
recovered, differences = {}, []
for query, collections in cases:
    for d in data['documents']:
        if d['collection'] not in collections:
            continue
        suffix = '\n'+query+'\n'+d['contextual']
        prior = db.execute('SELECT score FROM scores WHERE id=?',(sha(old_profile+suffix),)).fetchone()
        if prior is None:
            continue
        key = sha(new_profile+suffix)
        recovered[key] = prior[0]
        fresh = db.execute('SELECT score FROM scores WHERE id=?',(key,)).fetchone()
        if fresh is not None:
            differences.append(abs(prior[0]-fresh[0]))
assert len(recovered)==4217 and len(differences)>=1000
assert max(differences)<0.001, 'CPU batching parity failed; do not reuse scores'
before = db.total_changes
db.executemany('INSERT OR IGNORE INTO scores VALUES (?,?)',recovered.items())
db.commit()
report = dict(old_identity=old,new_identity=current,old_profile=old_profile,new_profile=new_profile,
              matched_old_pairs=len(recovered),overlap_pairs=len(differences),max_abs_difference=max(differences),
              restored_pairs=db.total_changes-before,
              note='Old source reconstructed and hash matched; CPU weights/tokenization unchanged. Pair scores only; no cached timings used as online latency.')
save(HERE/'cache_recovery.json',report)
print({k:v for k,v in report.items() if k.endswith('pairs') or k=='max_abs_difference'})
