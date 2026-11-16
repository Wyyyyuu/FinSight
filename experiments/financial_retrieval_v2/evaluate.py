"""Frozen dev/test ablation, fixed-4 anchors and separately token-budgeted evidence."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re
from statistics import mean
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import tiktoken
from fastembed import TextEmbedding
from backend.rag.bm25 import BM25Index, reciprocal_rank_fusion
from backend.rag.evidence_pack import pack_evidence, balanced_evidence
from experiments.financial_retrieval.evaluate import sha, sparse_scores
from experiments.lightrag.adapters import SharedEmbedding
from backend.rag.lexical import lexical_tokens
from experiments.lightrag.metrics import retrieval_metrics, percentile

HERE=Path(__file__).resolve().parent
V1=HERE.parent/'financial_retrieval'
ENC=tiktoken.get_encoding('cl100k_base')
CONFIGS={
    'previous':dict(model='zh',field='structured'),
    'header_fix':dict(model='zh',field='safe'),
    'table_context':dict(model='zh',field='contextual'),
    'english_model_only':dict(model='en',field='structured'),
    'english_header':dict(model='en',field='safe'),
    'english_context':dict(model='en',field='contextual'),
}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def identity():
    paths=[HERE/'build_dataset.py',Path(__file__),ROOT/'backend/rag/table_context.py',
           ROOT/'backend/rag/evidence_pack.py',ROOT/'backend/rag/bm25.py',ROOT/'backend/rag/lexical.py',
           V1/'build_dataset.py',V1/'evaluate.py',ROOT/'experiments/lightrag/adapters.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Encoder:
    def __init__(self, model):
        self.model=model
        self.engine=(SharedEmbedding(ROOT/'.annual-models') if model=='zh' else
                     TextEmbedding('BAAI/bge-small-en-v1.5',cache_dir=str(HERE/'models'),threads=2,local_files_only=True))

    def encode(self,texts):
        return (np.asarray(self.engine.encode(texts).dense,dtype=np.float32) if self.model=='zh'
                else np.asarray(list(self.engine.embed(texts,batch_size=16)),dtype=np.float32))


def facet_plan(text):
    # Explicit user-authored report references only. No benchmark labels involved.
    return [{'collection':m.group(1),'question':m.group(2).strip()}
            for m in re.finditer(r'^\s*-\s*\[([^\]\n]+)\]\s*(.+)$',text,re.M)]


def vectors_for(encoder,documents,queries,fields=None):
    path=HERE/'cache'/f'{encoder.model}.npz';path.parent.mkdir(exist_ok=True)
    vectors={}
    inherited=V1/'cache/bge-small-zh-46fbe35-vectors.npz'
    source=path if path.exists() else inherited if encoder.model=='zh' and inherited.exists() else None
    if source:
        with np.load(source,allow_pickle=False) as z:
            vectors=dict(zip(z['keys'].tolist(),z['vectors']))
    if fields is None:
        fields=['structured','safe','contextual']
        if encoder.model=='zh': fields+=['plain']
    texts=list(dict.fromkeys([d[f] for d in documents for f in fields]+queries))
    pending=[t for t in texts if sha(t) not in vectors]
    started=time.perf_counter()
    for i in range(0,len(pending),64):
        batch=pending[i:i+64]
        for text,v in zip(batch,encoder.encode(batch)): vectors[sha(text)]=v
        if i%1024==0 or i+64>=len(pending):
            np.savez_compressed(path,keys=np.asarray(list(vectors)),vectors=np.asarray(list(vectors.values())))
            print('embedding',encoder.model,min(i+64,len(pending)),'/',len(pending),flush=True)
    return vectors,dict(new_texts=len(pending),seconds=time.perf_counter()-started)


def context_text(d):
    page='; '.join(str(loc.get('page') or f"{loc.get('context_uid','context')}:{loc.get('table_row',loc.get('paragraph_index',''))}")
                   for loc in d['locations'])
    return f'[{d["id"]}] Report: {d["collection"]}; source: {page}\n{d["safe"]}\n'


@lru_cache(maxsize=50000)
def text_tokens(text):
    return len(ENC.encode(text))


def token_count(d):
    return text_tokens(context_text(d))


class Retriever:
    def __init__(self,documents,vectors,config):
        self.documents=documents;self.vectors=vectors;self.config=config;self.index={}
        self.by_collection=defaultdict(list)
        for d in documents:self.by_collection[d['collection']].append(d)

    def ranked(self,query,collections,vector=None):
        key=tuple(sorted(set(collections)))
        if key not in self.index:
            docs=[d for c in key for d in self.by_collection.get(c,[])]
            field=self.config['field']
            self.index[key]=(docs,np.asarray([self.vectors[sha(d[field])] for d in docs],dtype=np.float64),
                             BM25Index([d[field] for d in docs]))
        docs,matrix,lexical=self.index[key]
        if not docs:return []
        vector=self.vectors[sha(query)] if vector is None else vector
        dense=matrix@np.asarray(vector,dtype=np.float64)
        if self.config.get('lexical')=='cjk':
            counts=[Counter(lexical_tokens(d['plain'])) for d in docs]
            sparse=sparse_scores(query,counts,lexical_tokens)
        else:sparse=lexical.scores(query)
        order=reciprocal_rank_fusion(dense,sparse,rrf_k=self.config.get('rrf_k',20))
        return [docs[i] for i in order]

    def single(self,q,*,expand=False,vector=None):
        ranks=self.ranked(q['question'],[q['collection']],vector)
        evidence=pack_evidence(ranks,all_documents=ranks,token_count=token_count,budget=1500,expand_tables=expand)
        return ranks[:4],evidence

    def multi(self,text,collections,*,mode,encoder=None):
        plan=facet_plan(text)
        if mode=='global' or not plan:
            vector=encoder.encode([text])[0] if encoder else None
            ranks=self.ranked(text,collections,vector)
            return pack_evidence(ranks,all_documents=ranks,token_count=token_count,budget=3000)
        rankings=[]
        for facet in plan:
            if facet['collection'] not in collections:
                rankings.append([]);continue
            vector=encoder.encode([facet['question']])[0] if encoder else None
            rankings.append(self.ranked(facet['question'],[facet['collection']],vector))
        allowed=[d for c in collections for d in self.by_collection.get(c,[])]
        if mode=='balanced':
            return balanced_evidence(rankings,all_documents=allowed,token_count=token_count,budget=3000)
        # Separate decomposition-only ablation: the first facet can fill budget.
        return pack_evidence([d for rows in rankings for d in rows],all_documents=allowed,
                             token_count=token_count,budget=3000,expand_tables=True)


def metric(ids,gold,k=None):
    gold=set(gold);ids=list(dict.fromkeys(ids));k=len(ids) if k is None else k
    if not gold:return dict(recall_at_k=None,all_evidence=None,ndcg_at_k=None,precision_at_k=None)
    return retrieval_metrics(ids,list(gold),max(1,k))


def single_row(q,anchors,evidence,latencies):
    return dict(id=q['id'],language=q['language'],company=q.get('company',q['group'].split('/')[0]),
                category=q['category'],required=q['required'],anchors=[d['id'] for d in anchors],
                evidence=[d['id'] for d in evidence],anchor=metric([d['id'] for d in anchors],q['required'],4),
                context=metric([d['id'] for d in evidence],q['required']),
                tokens=sum(map(token_count,evidence)),latency_ms=latencies)


def multi_row(task,evidence,latencies):
    ids={d['id'] for d in evidence};available=[f for f in task['facets'] if f['answerable']]
    facet_metrics=[metric(ids,f['required']) for f in available]
    reports={f['collection'] for f in available};covered={d['collection'] for d in evidence}
    required={s for f in available for s in f['required']}
    return dict(id=task['id'],category=task['category'],cluster=task['cluster'],evidence=[d['id'] for d in evidence],
                required=sorted(required),context=metric([d['id'] for d in evidence],required),
                facet_complete=mean(m['all_evidence'] for m in facet_metrics),
                worst_facet_recall=min(m['recall_at_k'] for m in facet_metrics),
                report_coverage=len(reports&covered)/len(reports),
                scope_violations=sum(d['collection'] not in task['collections'] for d in evidence),
                tokens=sum(map(token_count,evidence)),latency_ms=latencies)


def summary(rows,multi=False):
    if not rows:return {}
    result={'questions':len(rows),'context':{m:mean(r['context'][m] for r in rows)
             for m in ('recall_at_k','all_evidence','ndcg_at_k','precision_at_k')},
            'tokens_mean':mean(r['tokens'] for r in rows),'tokens_max':max(r['tokens'] for r in rows),
            'p50_ms':percentile([t for r in rows for t in r['latency_ms']],.5),
            'p95_ms':percentile([t for r in rows for t in r['latency_ms']],.95)}
    if multi:
        result.update({m:mean(r[m] for r in rows) for m in ('facet_complete','worst_facet_recall','report_coverage','scope_violations')})
    else:
        result['anchor']={m:mean(r['anchor'][m] for r in rows) for m in ('recall_at_k','all_evidence','ndcg_at_k','precision_at_k')}
    return result


def evaluate_single(engine,questions,*,expand=False,encoder=None):
    rows=[]
    for q in questions:
        latency=[]
        for _ in range(3 if encoder else 1):
            start=time.perf_counter()
            vector=encoder.encode([q['question']])[0] if encoder else None
            anchors,evidence=engine.single(q,expand=expand,vector=vector)
            latency.append((time.perf_counter()-start)*1000)
        rows.append(single_row(q,anchors,evidence,latency))
    return rows


def evaluate_multi(engine,tasks,mode,encoder=None):
    rows=[]
    for task in tasks:
        latency=[]
        for _ in range(3 if encoder else 1):
            start=time.perf_counter();hits=engine.multi(task['question'],task['collections'],mode=mode,encoder=encoder)
            latency.append((time.perf_counter()-start)*1000)
        rows.append(multi_row(task,hits,latency))
    return rows


def paired_ci(before,after,path,cluster):
    old={r['id']:r for r in before};groups=defaultdict(list)
    def value(r):
        for p in path:r=r[p]
        return r
    for r in after:groups[r[cluster]].append(value(r)-value(old[r['id']]))
    parts=list(groups.values());rng=np.random.default_rng(20260915)
    draws=[mean(v for i in rng.integers(0,len(parts),len(parts)) for v in parts[i]) for _ in range(2000)]
    return dict(delta=mean(v for row in parts for v in row),lower=float(np.quantile(draws,.025)),upper=float(np.quantile(draws,.975)),clusters=len(parts))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['dev','test','regression']);args=parser.parse_args()
    dataset_path=HERE/'dataset.json';data=json.loads(dataset_path.read_text(encoding='utf-8'))
    dataset_sha=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    selected_path=HERE/'selection.json'
    if args.mode!='dev':
        selection=json.loads(selected_path.read_text(encoding='utf-8'))
        assert selection['dataset_sha256']==dataset_sha and selection['code_sha256']==identity(),'Frozen experiment changed'
    questions=[q for q in data['questions'] if q['split']==args.mode]
    tasks=[t for t in data['multi'] if t['split']==args.mode]
    collections={q['collection'] for q in questions}|{c for t in tasks for c in t['collections']}
    docs=[d for d in data['documents'] if d['collection'] in collections]
    queries=[q['question'] for q in questions]+[t['question'] for t in tasks]+[f['question'] for t in tasks for f in facet_plan(t['question'])]
    encoders={m:Encoder(m) for m in ('zh','en')}
    vectors={};embedding_cost={}
    for m,e in encoders.items():
        fields=None
        if args.mode=='regression':
            fields=['structured','plain'] if m=='zh' else []
            winner_config=CONFIGS[selection['single']]
            if winner_config['model']==m:fields=list(dict.fromkeys(fields+[winner_config['field']]))
        vectors[m],embedding_cost[m]=vectors_for(e,docs,queries,fields=fields)
    engines={name:Retriever(docs,vectors[c['model']],c) for name,c in CONFIGS.items()}
    if args.mode=='dev':
        candidates={name:summary(evaluate_single(engine,questions)) for name,engine in engines.items()}
        base=candidates['previous']['anchor']
        safe=[name for name,s in candidates.items() if all(s['anchor'][m]>=base[m]-1e-12
               for m in ('recall_at_k','all_evidence','ndcg_at_k'))]
        winner=max(safe,key=lambda n:mean(candidates[n]['anchor'][m] for m in ('recall_at_k','all_evidence','ndcg_at_k')))
        packed={str(expand):summary(evaluate_single(engines[winner],questions,expand=expand)) for expand in (False,True)}
        expansion=max((False,True),key=lambda x:mean(packed[str(x)]['context'][m] for m in ('recall_at_k','all_evidence')))
        multi_results={mode:summary(evaluate_multi(engines[winner],tasks,mode),multi=True) for mode in ('global','decomposed','balanced')}
        multi_mode=max(multi_results,key=lambda n:mean([multi_results[n]['context']['recall_at_k'],multi_results[n]['facet_complete'],multi_results[n]['worst_facet_recall']]))
        selection=dict(status='frozen_before_fresh_test',dataset_sha256=dataset_sha,code_sha256=identity(),
                       created=datetime.now(timezone.utc).isoformat(),single=winner,expand_tables=expansion,multi_mode=multi_mode,
                       candidates=candidates,packing=packed,multi_candidates=multi_results,embedding_cost=embedding_cost,
                       model_manifest=json.loads((HERE/'model_manifest.json').read_text(encoding='utf-8')),
                       selection_rule='Only 100 dev questions and 20 dev composites; non-regression vs previous for fixed4 recall/complete/nDCG; max mean. Packing and summary modes selected separately on dev.')
        save(selected_path,selection);print(json.dumps(selection,ensure_ascii=True),flush=True);return
    configs={name:name for name in ('previous','header_fix','table_context','english_model_only')}
    if args.mode=='regression':configs={'previous':'previous'}
    configs['selected']=selection['single']
    rows={};summaries={}
    # All retained ablations are fixed before heldout scores are inspected.
    for name,config in configs.items():
        engine=engines[config];e=encoders[CONFIGS[config]['model']]
        subset=[q for q in questions if args.mode!='regression' or q['language']=='en']
        current=evaluate_single(engine,subset,expand=(selection['expand_tables'] if name=='selected' else False),encoder=e)
        rows[name]=current;summaries[name]=summary(current)
        print('evaluated',args.mode,name,len(current),flush=True)
    multi_rows={}
    if tasks:
        variants={'previous_global':('previous','global'),'new_global':(selection['single'],'global'),
                  'decomposed':(selection['single'],'decomposed'),'selected':(selection['single'],selection['multi_mode'])}
        for name,(config,mode) in variants.items():
            multi_rows[name]=evaluate_multi(engines[config],tasks,mode,encoders[CONFIGS[config]['model']])
    if args.mode=='regression':
        # Chinese stays on the prior query-language route; don't force English weights/model.
        chinese=[q for q in questions if q['language']=='zh']
        engine=Retriever(docs,vectors['zh'],dict(model='zh',field='plain',lexical='cjk',rrf_k=60))
        chinese_rows=evaluate_single(engine,chinese,encoder=encoders['zh'])
        for name in ('previous','selected'):
            rows[name]=[r for r in rows[name] if r['language']=='en']+chinese_rows
            summaries[name]=summary(rows[name])
    report=dict(status='complete',split=args.mode,selection=selection,dataset_sha256=dataset_sha,no_reranker=True,
                budgets=dict(single_anchor_k=4,single_context_tokens=1500,multi_context_tokens=3000),
                rows=rows,summary=summaries,multi_rows=multi_rows,
                multi_summary={name:summary(r,multi=True) for name,r in multi_rows.items()},embedding_cost=embedding_cost,
                by_language={name:{lang:summary([r for r in values if r['language']==lang]) for lang in ('en','zh')}
                             for name,values in rows.items()},
                delta_ci={m:paired_ci(rows['previous'],rows['selected'],['anchor',m],'company')
                          for m in ('recall_at_k','all_evidence','ndcg_at_k')})
    if multi_rows:
        report['multi_delta_ci']={m:paired_ci(multi_rows['previous_global'],multi_rows['selected'],[m],'cluster')
                                  for m in ('facet_complete','report_coverage','worst_facet_recall')}
        report['multi_by_category']={name:{cat:summary([r for r in values if r['category']==cat],True)
                                           for cat in sorted({r['category'] for r in values})}
                                    for name,values in multi_rows.items()}
    out=HERE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+args.mode)
    save(out/'report.json',report)
    save(HERE/f'{args.mode}_assessment.json',{k:v for k,v in report.items() if k not in ('rows','multi_rows')})
    save(HERE/f'{args.mode}_latest.json',{'path':str(out/'report.json')})
    print(json.dumps({'single':summaries,'multi':report['multi_summary']},ensure_ascii=True),flush=True)
    print('report',out/'report.json',flush=True)


if __name__=='__main__':main()
