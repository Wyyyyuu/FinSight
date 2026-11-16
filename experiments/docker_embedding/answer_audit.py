"""Paired generation check using frozen global24-to-rerank8 evidence."""
import asyncio
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.docker_embedding.evaluate import HERE, DATA, sha, save, context, text_tokens
from experiments.financial_retrieval_v2.answer_audit import score_facet, summarize, number_reference
from experiments.lightrag.adapters import ChatClient, CallBudget, endpoint_config


async def main():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    pointer = json.loads((HERE/'test_latest.json').read_text())
    report = json.loads(Path(pointer['path']).read_text(encoding='utf-8'))
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == report['seal']['dataset_sha256']
    winner = report['selection']['winner']
    docs = {d['id']:d for d in data['documents']}
    singles = sorted([q for q in data['questions'] if q['split']=='test' and number_reference(q.get('answer'))],key=lambda q:sha(q['id']))[:16]
    tasks = []
    for category in ['cross_company','cross_year','three_reports','missing_report']:
        tasks += sorted([t for t in data['multi'] if t['split']=='test' and t['category']==category],key=lambda t:sha(t['id']))[:2]
    cases = [dict(id=q['id'],kind='single',question=q['question'],facets=[{**q,'id':'1','answerable':True}]) for q in singles]
    cases += [dict(id=t['id'],kind='multi',question=t['question'],facets=t['facets']) for t in tasks]
    evidence = {kind:{arm:{r['id']:r['evidence'] for r in report[key][arm]}
                      for arm in ['hash_legacy',winner]}
                for kind,key in [('single','rerank'),('multi','rerank_multi')]}
    folder = HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-answers')
    seal = {'retrieval_sha':sha(Path(pointer['path']).read_text(encoding='utf-8')),
            'sample_ids':[c['id'] for c in cases], 'max_sources':8,'max_context_tokens':3000,
            'code_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scorer_sha':hashlib.sha256((ROOT/'experiments/financial_retrieval_v2/answer_audit.py').read_bytes()).hexdigest(),
            'limitations':['Multi sample retains empty/non-numeric author references; these are explicitly unscored.',
                           'Literal source quotations do not prove full summary entailment.']}
    save(folder/'plan.json',seal)
    client = ChatClient(endpoint_config(),CallBudget(48),timeout=90)
    client.phase = 'docker_embedding_grounded_answer'
    gate = asyncio.Semaphore(2)
    rows = []

    async def answer(case,arm):
        supplied, used = {}, 0
        for sid in evidence[case['kind']][arm][case['id']]:
            d = docs[sid]
            size = text_tokens(context(d))
            if used + size <= 3000:
                supplied[sid]=d
                used += size
        request = [{k:f[k] for k in ['id','question','collection']} for f in case['facets']]
        prompt = ('Use ONLY supplied evidence to answer every requested item. Return JSON with summary and facts. '
                  'Each fact must contain facet_id, report (exact requested collection), answer (number, yes/no for a boolean question, or null if '
                  'insufficient evidence), unit, citations (exact source IDs), quotes (objects with source_id and text '
                  'copied literally from that source), calculation. Percent answers use 12 for 12%, not 0.12. '
                  'Keep report years, signs, units, and restatement bases separate. For a missing report return null '
                  'and empty citations; never substitute another report. Summary must be supported by the facts.\n'
                  'Request:\n'+case['question']+'\nRequested facets:\n'+json.dumps(request,ensure_ascii=False)+
                  '\nEvidence:\n'+''.join(context(d) for d in supplied.values()))
        row = dict(id=case['id'],kind=case['kind'],arm='previous' if arm=='hash_legacy' else 'selected',
                   retrieval_arm=arm,supplied=list(supplied),context_tokens=used)
        async with gate:
            started = time.perf_counter()
            try:
                output = json.loads(await client.complete(prompt,max_tokens=2400,response_format={'type':'json_object'}))
                outputs = {str(f['facet_id']):f for f in output['facts']}
                row.update(output=output,facets=[score_facet(f,outputs.get(f['id'],{}),supplied) for f in case['facets']])
            except Exception as exc:
                row.update(error=type(exc).__name__,facets=[score_facet(f,{},supplied) for f in case['facets']])
            row['elapsed_ms']=(time.perf_counter()-started)*1000
            rows.append(row)
            save(folder/'partial.json',dict(seal=seal,rows=rows,calls=client.calls))
            print('answered',len(rows),48,flush=True)
    try:
        await asyncio.gather(*(answer(c,a) for c in cases for a in ['hash_legacy',winner]))
    finally:
        await client.close()
    result = dict(seal=seal,rows=rows,calls=client.calls,summary=summarize(rows),
                  errors=sum('error' in r for r in rows),model=client.config['model'],
                  tokens=sum((c.get('prompt_tokens') or 0)+(c.get('completion_tokens') or 0) for c in client.calls))
    save(folder/'report.json',result)
    save(HERE/'answer_latest.json',{'path':str(folder/'report.json')})
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':
    asyncio.run(main())
