"""Paired numerical + multi-report summary audit. No model self-grading.

Claims are checked against author numeric answers, requested reports and literal
source quotations. Free-form narrative entailment is NOT inferred from ID validity.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.financial_retrieval_v2.evaluate import HERE, ENC, save, context_text, sha
from experiments.lightrag.adapters import ChatClient, CallBudget, endpoint_config


def number_reference(answer):
    if isinstance(answer,bool) or answer is None:return None
    raw=str(answer).strip().replace(',','').replace('$','').replace('%','')
    if not re.fullmatch(r'-?\d+(?:\.\d+)?',raw):return None
    value=Decimal(raw)
    # Match the precision stated by the author, rather than a hidden tighter rule.
    decimals=len(raw.rsplit('.',1)[1]) if '.' in raw else 0
    tolerance=max(Decimal('0.000001'),Decimal('0.5')*(Decimal(10)**-decimals))
    return value,tolerance


def expected_unit(facet):
    answer=str(facet.get('answer','')).strip();q=facet['question'].lower()
    if answer.endswith('%'):
        return 'percentage_points' if 'percentage point' in q else 'percent'
    if 'ratio' in q:return 'ratio'
    return None  # Do not guess monetary scale from an answer without its unit.


def normalize_unit(value):
    value=str(value).strip().lower().replace(' ','_')
    return {'%':'percent','percentage':'percent','百分比':'percent','百分数':'percent',
            '百分点':'percentage_points','percentage_point':'percentage_points',
            '比例':'ratio','倍':'ratio'}.get(value,value)


def score_facet(facet,output,supplied):
    record=dict(id=facet['id'],answerable=facet['answerable'],reference=facet.get('answer'),
                expected_unit=expected_unit(facet),output=output)
    value=output.get('answer');reference=number_reference(facet.get('answer'))
    try:numeric=Decimal(str(value)) if value is not None and not isinstance(value,bool) else None
    except InvalidOperation:numeric=None
    if numeric is not None and not numeric.is_finite():numeric=None
    record['numeric_correct']=(bool(numeric is not None and abs(numeric-reference[0])<=reference[1])
                               if reference and facet['answerable'] else None)
    record['unit_correct']=(normalize_unit(output.get('unit',''))==record['expected_unit']
                            if record['expected_unit'] and value is not None and facet['answerable'] else None)
    citations=output.get('citations',[])
    if not isinstance(citations,list):citations=[]
    valid=bool(citations) and all(isinstance(c,str) and c in supplied and supplied[c]['collection']==facet['collection'] for c in citations)
    record['citation_ids_valid']=valid
    record['gold_citation_complete']=valid and set(facet['required'])<=set(citations)
    quotes=output.get('quotes',[])
    if not isinstance(quotes,list):quotes=[]
    valid_quotes=bool(quotes)
    quoted_sources=set()
    for quote in quotes:
        if not isinstance(quote,dict):valid_quotes=False;continue
        sid=quote.get('source_id');text=quote.get('text')
        quote_valid=bool(isinstance(sid,str) and sid in supplied and sid in citations and isinstance(text,str)
                         and text.strip() and ' '.join(text.split()) in ' '.join(supplied[sid]['safe'].split()))
        valid_quotes &= quote_valid
        if quote_valid:quoted_sources.add(sid)
    record['quotes_valid']=valid_quotes
    record['report_correct']=output.get('report')==facet['collection']
    record['abstained']=value is None
    record['missing_report_correct']=(value is None and not citations and record['report_correct']) if not facet['answerable'] else None
    record['numeric_and_evidence_correct']=(record['numeric_correct'] is True and record['unit_correct'] is not False
                                            and record['gold_citation_complete'] and record['report_correct'] and valid_quotes
                                            and set(facet['required'])<=quoted_sources)
    return record


def summarize(rows):
    result={}
    for kind in ('single','multi'):
        result[kind]={}
        for arm in ('previous','selected'):
            current=[r for r in rows if r['kind']==kind and r['arm']==arm]
            facets=[f for r in current for f in r.get('facets',[])]
            numeric=[f for f in facets if f['numeric_correct'] is not None]
            units=[f for f in facets if f['unit_correct'] is not None]
            missing=[f for f in facets if f['missing_report_correct'] is not None]
            fully_scorable=[r for r in current if r.get('facets') and all(
                f['numeric_correct'] is not None if f['answerable'] else f['missing_report_correct'] is not None
                for f in r['facets'])]
            result[kind][arm]=dict(cases=len(current),errors=sum('error' in r for r in current),
                numeric_correct=sum(f['numeric_correct'] for f in numeric),numeric_scored=len(numeric),
                unit_correct=sum(f['unit_correct'] for f in units),unit_scored=len(units),
                numeric_and_evidence_correct=sum(f['numeric_and_evidence_correct'] for f in numeric),
                missing_report_correct=sum(f['missing_report_correct'] for f in missing),missing_report_scored=len(missing),
                answered_facets=sum(not f['abstained'] for f in facets),
                quotes_valid=sum(f['quotes_valid'] for f in facets),
                cited_facets=sum(bool(f['output'].get('citations')) for f in facets),
                complete_cases_scored=len(fully_scorable),
                complete_cases_unscored=len(current)-len(fully_scorable),
                complete_cases=sum(all(f['numeric_and_evidence_correct'] if f['answerable'] else f['missing_report_correct'] for f in r['facets']) for r in fully_scorable))
    return result


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retry-errors',type=Path);args=parser.parse_args()
    data=json.loads((HERE/'dataset.json').read_text(encoding='utf-8'))
    report_path=json.loads((HERE/'test_latest.json').read_text(encoding='utf-8'))['path']
    retrieval=json.loads(Path(report_path).read_text(encoding='utf-8'))
    assert retrieval['dataset_sha256']==hashlib.sha256((HERE/'dataset.json').read_bytes()).hexdigest()
    docs={d['id']:d for d in data['documents']}
    questions=sorted([q for q in data['questions'] if q['split']=='test' and number_reference(q.get('answer')) is not None],key=lambda q:sha('v2-answer-sample'+q['id']))[:16]
    tasks=[]
    for cat in ('cross_company','cross_year','three_reports','missing_report'):
        tasks+=sorted([t for t in data['multi'] if t['split']=='test' and t['category']==cat],key=lambda t:sha('v2-summary-sample'+t['id']))[:3]
    cases=[]
    for q in questions:
        cases.append(dict(id=q['id'],kind='single',question=q['question'],facets=[{**q,'id':'1','answerable':True}]))
    cases += [dict(id=t['id'],kind='multi',question=t['question'],facets=t['facets']) for t in tasks]
    by_single={arm:{r['id']:r for r in retrieval['rows'][arm]} for arm in ('previous','selected')}
    by_multi={arm:{r['id']:r for r in retrieval['multi_rows'][name]} for arm,name in [('previous','previous_global'),('selected','selected')]}
    folder=HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-answers');folder.mkdir(parents=True,exist_ok=False)
    prior=json.loads(args.retry_errors.read_text(encoding='utf-8')) if args.retry_errors else None
    seal=dict(dataset_sha256=retrieval['dataset_sha256'],retrieval_report_sha256=hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
              audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sample_ids=[c['id'] for c in cases])
    if prior:assert prior['seal']==seal
    save(folder/'plan.json',seal)
    rows=[r for r in prior['rows'] if 'error' not in r] if prior else []
    retry={(r['id'],r['arm']) for r in prior['rows'] if 'error' in r} if prior else None
    previous_calls=prior['calls'] if prior else []
    client=ChatClient(endpoint_config(),CallBudget(60),timeout=75);client.phase='v2_source_grounded_summary'
    gate=asyncio.Semaphore(2)

    async def answer(case,arm):
        source=(by_single if case['kind']=='single' else by_multi)[arm][case['id']]
        supplied={sid:docs[sid] for sid in source['evidence']}
        context=''.join(context_text(d) for d in supplied.values())
        assert len(ENC.encode(context)) <= (1500 if case['kind']=='single' else 3000)
        # Facet specification is request-derived; labels and answers are never sent.
        request=[{k:f[k] for k in ('id','question','collection')} for f in case['facets']]
        prompt=('Write a concise research summary using ONLY the supplied source evidence. Address every requested item. '
                'Return JSON with summary (text) and facts (one item for each facet). Each fact must contain '
                'facet_id, report (exact requested collection), answer (number or null when evidence is insufficient), '
                'unit (percent, percentage_points, ratio, or explicit original monetary/count unit), '
                'citations (exact source IDs), quotes (objects with source_id and text copied literally from that source), '
                'and calculation (short explanation). Percent answers use 12 for 12%, not 0.12. '
                'Keep report years, signs, units and restatement bases separate. Do not infer facts for a missing report; '
                'return null and empty citations for it, while answering supported items. Never substitute another report. '
                'Do not add facts in summary that are not supported by the facts array.\nRequest:\n'+case['question']+
                '\nRequested facets:\n'+json.dumps(request,ensure_ascii=False)+'\nEvidence:\n'+context)
        record=dict(id=case['id'],kind=case['kind'],arm=arm,supplied=list(supplied))
        async with gate:
            started=time.perf_counter()
            try:
                text=await client.complete(prompt,max_tokens=2400,response_format={'type':'json_object'})
                output=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))
                assert isinstance(output,dict) and isinstance(output.get('facts'),list)
                outputs={str(f['facet_id']):f for f in output['facts'] if isinstance(f,dict) and 'facet_id' in f}
                record.update(output=output,facets=[score_facet(f,outputs.get(f['id'],{}),supplied) for f in case['facets']])
            except Exception as exc:
                record.update(error=type(exc).__name__,facets=[score_facet(f,{},supplied) for f in case['facets']])
            record['elapsed_ms']=(time.perf_counter()-started)*1000;rows.append(record)
            save(folder/'partial.json',dict(seal=seal,rows=rows,calls=previous_calls+client.calls))
            print('answered',len(rows),'/',len(cases)*2,case['kind'],arm,flush=True)
    try:
        await asyncio.gather(*(answer(case,arm) for case in cases for arm in ('previous','selected')
                              if retry is None or (case['id'],arm) in retry))
    finally:await client.close()
    calls=previous_calls+client.calls
    result=dict(status='partial' if any('error' in r for r in rows) else 'complete',seal=seal,model=client.config['model'],
                rows=rows,calls=calls,summary=summarize(rows),
                tokens=sum((c.get('prompt_tokens') or 0)+(c.get('completion_tokens') or 0) for c in calls),
                limitations=['16 numerical single-report cases; 12 constructed multi-report summary cases',
                             'Unknown source answer units remain unscored; numeric-and-evidence is not full unit correctness',
                             'Gold citations are minimal author evidence; alternative valid evidence may be undercounted',
                             'Literal quote validity does not prove entailment of the full narrative summary'])
    save(folder/'report.json',result);save(HERE/'answer_assessment.json',{k:v for k,v in result.items() if k not in ('rows','calls')})
    print(json.dumps(result['summary']),flush=True);print(folder/'report.json',flush=True)


if __name__=='__main__':asyncio.run(main())
