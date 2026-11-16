"""Fresh company-disjoint finance tasks plus exposed v1 regression cases.

Composite questions are constructed tasks, not original summary annotations.
Source answers are only labels. Corpus construction never reads qa fields.
"""
from __future__ import annotations
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.rag.table_context import render_table_row, contextual_table_row
from experiments.financial_retrieval.build_dataset import render_row_legacy

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'financial_retrieval'
SEED = 'finance-v2-company-disjoint-20260915'


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def ordered(rows, key=lambda x: x):
    return sorted(rows, key=lambda x: sha(SEED + str(key(x))))


def corpus(group, records):
    collection = 'finqa:' + group
    documents, aliases = {}, {}
    pages = {r['filename']: r for r in records}
    for page, r in sorted(pages.items()):
        blocks = [(f'text_{i}', text, text, text, text, 'text', None)
                  for i, text in enumerate(r['pre_text'] + r['post_text'])]
        blocks += [(f'table_{i}', render_row_legacy(r['table'], i, False),
                    render_row_legacy(r['table'], i, True), render_table_row(r['table'], i),
                    contextual_table_row(r['table'], i, r['pre_text']), 'table', i)
                   for i in range(len(r['table']))]
        for key, plain, legacy, safe, contextual, kind, row_index in blocks:
            if not any(c.isalnum() for c in plain):
                continue
            sid = 'F' + sha(collection + '\n' + legacy.strip())[:18]
            aliases[(page, key)] = sid
            if sid in documents:
                documents[sid]['locations'].append({'page': page, 'evidence_key': key})
                if kind=='table':
                    documents[sid]['table_ids']=sorted(set(documents[sid]['table_ids']+[page]))
                    documents[sid]['parent_rows'][page]=row_index
                    documents[sid]['table_id']=None  # An ambiguous shared row must not select a parent.
                    documents[sid]['contextual']=documents[sid]['safe']
                continue
            documents[sid] = dict(id=sid, collection=collection, language='en', kind=kind,
                                 plain=plain.strip(), structured=legacy.strip(), safe=safe.strip(),
                                 contextual=contextual.strip(), group=group,
                                 company=group.split('/')[0], year=group.split('/')[-1],
                                 table_id=page if kind == 'table' else None, row_index=row_index,
                                 table_ids=[page] if kind=='table' else [],
                                 parent_rows={page:row_index} if kind=='table' else {},
                                 locations=[{'page': page, 'evidence_key': key}])
    return list(documents.values()), aliases


def make_question(row, split, aliases):
    qa = row['qa']; group = row['filename'].rsplit('/', 1)[0]
    return dict(id='finqa:' + row['id'], question=qa['question'].strip(), split=split,
                dataset='FinQA-fresh', language='en', group=group, company=group.split('/')[0],
                collection='finqa:' + group, source_page=row['filename'],
                answer=qa.get('answer'), numeric_answer=qa.get('exe_ans'), program=qa.get('program'),
                required=sorted({aliases[(row['filename'], k)] for k in qa['gold_inds']}),
                category='multi_evidence' if len(qa['gold_inds']) > 1 else 'single_evidence')


def composites(questions):
    tasks = []
    for split in ('dev', 'test'):
        by_company = defaultdict(lambda: defaultdict(list))
        for q in questions:
            if q['split'] == split:
                by_company[q['company']][q['collection']].append(q)
        companies = sorted(by_company)
        for offset in range(0, len(companies)-1, 2):
            pair = companies[offset:offset+2]
            collections = [c for company in pair for c in sorted(by_company[company])]
            if len(collections) < 4:
                continue
            candidates = {c: ordered([q for company in pair for q in by_company[company].get(c, [])], key=lambda q:q['id'])
                          for c in collections}
            # Include other reports as distractors, selected from the same split.
            distractors = [c for company in companies if company not in pair for c in sorted(by_company[company])][:2]
            variants = [('cross_company', [collections[0], collections[2]]),
                        ('cross_year', collections[:2]),
                        ('three_reports', collections[:3]),
                        ('missing_report', [collections[0], collections[2]])]
            for vi, (category, chosen) in enumerate(variants):
                facets = [candidates[c][vi % len(candidates[c])] for c in chosen]
                allowed = list(collections + distractors)
                if category == 'missing_report':
                    allowed.remove(chosen[-1])
                text = 'Write a research summary addressing every item below. Keep each report and period separate; cite evidence for every finding. If a requested report is unavailable, say that item cannot be answered. Do not combine amounts across incompatible units.\n'
                text += '\n'.join(f'- [{q["collection"]}] {q["question"]}' for q in facets)
                tasks.append(dict(id=f'{split}:multi:{offset//2}:{category}', question=text, split=split,
                                  category=category, cluster='|'.join(pair), collections=allowed,
                                  facets=[dict(id=str(i+1), collection=q['collection'], question=q['question'],
                                               source_question=q['id'], required=q['required'], answer=q['answer'],
                                               program=q['program'], numeric_answer=q['numeric_answer'],
                                               answerable=q['collection'] in allowed)
                                          for i,q in enumerate(facets)],
                                  annotation='constructed composite of author questions; not original summary ground truth'))
    return tasks


def main():
    old = json.loads((V1/'dataset.json').read_text(encoding='utf-8'))
    raw = [r for split in ('train','dev','test') for r in json.loads((V1/'raw'/f'finqa_dataset_{split}.json').read_text(encoding='utf-8'))]
    groups = defaultdict(list)
    for r in raw:
        if r.get('qa',{}).get('gold_inds'):
            groups[r['filename'].rsplit('/',1)[0]].append(r)
    seen = {q['group'].split('/')[0] for q in old['questions'] if q['dataset']=='FinQA'}
    eligible = defaultdict(list)
    for g, rows in groups.items():
        if len(rows)>=5 and g.split('/')[0] not in seen:
            eligible[g.split('/')[0]].append(g)
    companies = ordered([c for c, reports in eligible.items() if len(reports)>=2])[:30]
    selected = {g: ('dev' if i<10 else 'test') for i,c in enumerate(companies) for g in ordered(eligible[c])[:2]}
    documents, questions = [], []
    for g, split in selected.items():
        docs, aliases = corpus(g,groups[g]); documents += docs
        questions += [make_question(r,split,aliases) for r in ordered(groups[g],key=lambda r:r['id'])[:5]]
    # Exposed old tests are regression-only; never used to select the new config.
    regression = [{**q,'split':'regression'} for q in old['questions'] if q['split']=='test']
    old_groups = {q['group'] for q in regression if q['dataset']=='FinQA'}
    for g in sorted(old_groups):
        docs,_ = corpus(g,groups[g]); documents += docs
    for d in old['documents']:
        if d['language']=='zh':
            documents.append({**d,'safe':d['structured'],'contextual':d['structured'],
                              'group':d['collection'], 'company':d['collection'], 'year':'',
                              'table_id':None,'row_index':None})
    data = dict(version=SEED, documents=documents, questions=questions+regression,
                multi=composites(questions), old_dataset_sha256=hashlib.sha256((V1/'dataset.json').read_bytes()).hexdigest(),
                source_manifest=json.loads((V1/'source_manifest.json').read_text(encoding='utf-8')),
                selection_rule='30 unseen companies with >=2 eligible reports; SHA256 order; first10 dev/next20 test; two reports/company; five questions/report',
                limitations=['QA-bearing pages only, not full annual reports','Composed questions are not human annotated free-form summary tasks',
                             'Chinese cases and old English cases are exposed regressions, not fresh heldout'])
    docs = {d['id']:d for d in documents}
    assert len(docs)==len(documents)
    assert not ({q['company'] for q in questions if q['split']=='dev'} & {q['company'] for q in questions if q['split']=='test'})
    assert not (seen & set(companies))
    assert all(set(q['required'])<=docs.keys() for q in data['questions'])
    (HERE/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'documents':len(documents),'questions':dict(Counter(q['split'] for q in data['questions'])),
                      'multi':dict(Counter(t['split'] for t in data['multi']))}))


if __name__=='__main__': main()
