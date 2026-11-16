"""Outcome-informed dev repair, reusing frozen graph retrievals (no rebuild).

Protect each explicit facet's first two baseline candidates before graph fusion.
This is development tuning on the same three tasks, not an independent holdout.
"""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.lightrag.multidoc_pilot import (HERE, DATA, choose, EmbeddingCache,
    Engine, fuse_sources, pack, context, text_tokens, save)
from experiments.lightrag.multidoc_report import score_checked
from experiments.lightrag.metrics import retrieval_metrics
from experiments.lightrag.adapters import ChatClient,CallBudget,endpoint_config


def anchored_candidates(base, fused, anchors=2, limit=24):
    if anchors<0 or anchors>limit:
        raise ValueError('Invalid anchor count')
    candidates=[]
    seen=set()
    for d in [*base[:anchors],*fused,*base]:
        if d['id'] not in seen:
            seen.add(d['id'])
            candidates.append(d)
    return candidates[:limit]


async def main():
    parent=Path(json.loads((HERE/'multidoc_latest.json').read_text())['path'])
    original=json.loads(parent.read_text(encoding='utf-8'))
    assert original['status']=='complete'
    tasks,docs=choose(json.loads(DATA.read_text(encoding='utf-8')))
    cache=EmbeddingCache(parent.parent,original['plan']['model'])
    engine=Engine(docs,cache)
    prepared=[]
    for task in tasks:
        trace=json.loads((parent.parent/(task['category']+'-retrieval.json')).read_text(encoding='utf-8'))
        rankings=[]
        for facet in trace['facets']:
            base=engine.ranked(facet['query'],[facet['collection']],'m3_bm25_rrf')[:24]
            fused=fuse_sources(base,facet['retrieval']['resolved_sources'],docs,facet['collection'])
            rankings.append(anchored_candidates(base,fused))
        sources=pack(rankings,docs,True)
        prepared.append((task,sources))
    folder=HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-guarded')
    folder.mkdir(parents=True,exist_ok=False)
    config=endpoint_config()
    assert {c['model'] for c in original['calls']}=={config['model']}
    prereg=json.loads((parent.parent/'scoring_preregistration.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(config['url'].encode()).hexdigest()==prereg['endpoint_url_sha256']
    plan=dict(parent_report=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              rule='Protect first two original hybrid candidates per query-derived facet, then fused candidates; same table expansion, final K8/3000tokens.',
              model=config['model'],candidate_sources={t['id']:[d['id'] for d in ss] for t,ss in prepared},
              status='outcome-informed dev repair on same 3 cases; not a new holdout',
              unit_ambiguity='cross_company facet1 is excluded from the additional 6-facet comparison: source/request/answer monetary scale requires separate original-PDF validation. Sealed 7-facet primary scores remain unchanged.',
              primary_metric_limitation='Exact gold source IDs undercount equivalent table text; keep these scores, but do not call them all-valid-evidence recall or factuality.')
    save(folder/'plan.json',plan)
    client=ChatClient(config,CallBudget(3),timeout=90)
    client.phase='guarded_answer'
    rows=[]
    try:
        for task,sources in prepared:
            supplied={d['id']:d for d in sources}
            request=[{k:f[k] for k in ['id','question','collection']} for f in task['facets']]
            prompt=('Use ONLY supplied evidence to answer every requested item. Return JSON with summary and facts. '
                    'Each fact must contain facet_id, report (exact requested collection), answer (number, or null if '
                    'insufficient evidence), unit, citations (exact source IDs), quotes (objects with source_id and '
                    'text copied literally from that source), calculation. Percent answers use 12 for 12%, not 0.12. '
                    'Keep years, signs, units and restatement bases separate. Summary must be supported by facts.\n'
                    'Request:\n'+task['question']+'\nRequested facets:\n'+json.dumps(request)+
                    '\nEvidence:\n'+''.join(context(d) for d in sources))
            output=json.loads(await client.complete(prompt,response_format={'type':'json_object'}))
            rows.append(dict(id=task['id'],arm='guarded_graph8',supplied=list(supplied),output=output,
                             context_tokens=sum(text_tokens(context(d)) for d in sources)))
            save(folder/'report.json',dict(plan=plan,status='running',rows=rows,calls=client.calls))
            print('GUARDED',len(rows),3,flush=True)
    finally:
        await client.close()
        cache.db.close()
    corrected=deepcopy(tasks)
    overrides=json.loads((HERE/'multidoc_label_audit.json').read_text(encoding='utf-8'))['overrides']
    for override in overrides:
        f=next(f for t in corrected if t['id']==override['task'] for f in t['facets'] if f['id']==override['facet'])
        if 'answer' in override:f['answer']=override['answer']
        f['required']+=override.get('additional_required',[])
    lookup={d['id']:d for d in docs}
    scored=[]
    for row in [*original['rows'],*rows]:
        t=next(t for t in corrected if t['id']==row['id'])
        output={str(f['facet_id']):f for f in row['output']['facts']}
        supplied={s:lookup[s] for s in row['supplied']}
        facts=[score_checked(f,output.get(f['id'],{}),supplied) for f in t['facets']]
        required=sorted({s for f in t['facets'] for s in f['required']})
        scored.append(dict(id=row['id'],arm=row['arm'],facets=facts,
                           metrics=retrieval_metrics(row['supplied'],required,8)))
    summaries={}
    for arm in sorted({r['arm'] for r in scored}):
        selected=[r for r in scored if r['arm']==arm]
        facts=[f for r in selected for f in r['facets']
               if not (r['id'].endswith('cross_company') and f['id']=='1')]
        summaries[arm]=dict(cases=len(selected),numeric_correct=sum(f['numeric_correct'] is True for f in facts),
                           numeric_scored=len(facts),exact_gold_id_recall=mean(r['metrics']['recall_at_k'] for r in selected))
    result=dict(plan=plan,status='complete',rows=rows,calls=client.calls,scored=scored,summary=summaries)
    save(folder/'report.json',result)
    save(HERE/'multidoc_guarded_latest.json',{'path':str(folder/'report.json')})
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    asyncio.run(main())
