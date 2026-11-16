"""Build a compact assessment from completed, sealed experiment reports."""
import json
import hashlib
from pathlib import Path
from statistics import mean, median
from collections import Counter
from evaluate import HERE, DATA, save, sha, percentile


def load_latest(name):
    return json.loads(Path(json.loads((HERE/(name+'_latest.json')).read_text())['path']).read_text(encoding='utf-8'))


def main():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    labels = {q['id']:q for q in data['questions']}
    test, regression = load_latest('test'), load_latest('regression')
    winner = test['selection']['winner']
    assert regression['seal']==test['seal']
    result = {'winner':winner,'seal':test['seal'],'groups':{},'multi':{}}
    collection_sizes = Counter(d['collection'] for d in data['documents'])
    eligible = [collection_sizes[q['collection']] for q in data['questions'] if q['split']=='test']
    result['english_test_search_space'] = dict(min=min(eligible),median=median(eligible),max=max(eligible),
                                             mean=mean(eligible),corpus_le_24=sum(n<=24 for n in eligible))
    for group, report, language in [('English test',test,'en'),('English regression',regression,'en'),('Chinese regression',regression,'zh')]:
        values = {}
        for arm in ['hash_legacy',winner]:
            rows = [r for r in report['rows'][arm] if labels[r['id']]['language']==language]
            reranked = [r for r in report['rerank'][arm] if labels[r['id']]['language']==language]
            candidate_gold = sum(len(set(r['candidates']) & set(labels[r['id']]['required'])) for r in rows)
            final_gold = sum(len(set(r['evidence']) & set(labels[r['id']]['required'])) for r in reranked)
            values[arm] = {'questions':len(rows),
                'candidate_gold_total':candidate_gold,'final_gold_total':final_gold,
                'reranker_evidence_retention':final_gold/candidate_gold if candidate_gold else None,
                'retrieve24':{m:mean(r['top24'][m] for r in rows) for m in rows[0]['top24']},
                'retrieve8':{m:mean(r['top8'][m] for r in rows) for m in rows[0]['top8']},
                'rerank8':{m:mean(r['top8'][m] for r in reranked) for m in reranked[0]['top8']}}
        result['groups'][group] = values
    for arm in ['hash_legacy',winner]:
        rows = test['multi'][arm]
        rr = test['rerank_multi'][arm]
        result['multi'][arm] = {'tasks':len(rows),
            'global8':{m:mean(r['global8'][m] for r in rows) for m in rows[0]['global8']},
            'global_rerank8':{m:mean(r['top8'][m] for r in rr) for m in rr[0]['top8']},
            'balanced3000':{m:mean(r['balanced3000'][m] for r in rows) for m in rows[0]['balanced3000']},
            'balanced_facet_complete':mean(r['facet_complete'] for r in rows),
            'max_context_tokens':max(r['tokens'] for r in rows)}
    answers = load_latest('answer')
    test_path = Path(json.loads((HERE/'test_latest.json').read_text())['path'])
    assert answers['seal']['retrieval_sha']==sha(test_path.read_text(encoding='utf-8'))
    runtime = json.loads((HERE/'runtime_assessment.json').read_text())
    assert runtime['seal']['selection']==test['selection']
    assert runtime['seal']['model']==test['seal']['model']
    assert runtime['seal']['reranker']==test['seal']['reranker']
    assert runtime['seal']['code_sha256']==hashlib.sha256((HERE/'runtime_check.py').read_bytes()).hexdigest()
    result['generation'] = answers['summary']
    result['runtime'] = runtime['summary']
    result['runtime_by_language'] = {}
    for language in ['en','zh']:
        result['runtime_by_language'][language] = {}
        for arm in ['hash_legacy',winner]:
            measured = [r for r in runtime['rows'] if r['arm']==arm and r['language']==language and r['repeat']>0]
            result['runtime_by_language'][language][arm] = {
                'warm_samples':len(measured),
                **{name:{'p50':percentile([r[name] for r in measured],.5),
                         'p95':percentile([r[name] for r in measured],.95)}
                   for name in ['retrieve_ms','rerank_ms','total_ms']}}
    failed_singles = [r for r in answers['rows'] if r['kind']=='single' and r['arm']=='selected'
                      and any(f['numeric_correct'] is False for f in r['facets'])]
    gold_supplied_failures = sum(set(labels[r['id']]['required'])<=set(r['supplied']) for r in failed_singles)
    result['generation_diagnostics'] = {'selected_single_failures':len(failed_singles),
                                        'all_labeled_gold_supplied':gold_supplied_failures}
    save(HERE/'assessment.json',result)
    lines = ['# Docker BGE-M3 evaluation','',f'Selected on dev100: `{winner}`.', '',
             'Same candidate K=24, final K=8, RRF constant=60; all comparisons use the same corpus and source field.', '',
             f'English test eligible passages per question: median {median(eligible)}, range {min(eligible)}–{max(eligible)}; {sum(n<=24 for n in eligible)}/200 questions have at most 24 passages. Report scope is already known.', '',
             '| Split | Method | Recall@24 | Recall@8 before rerank | Recall@8 after rerank | Complete evidence@8 | nDCG@8 |',
             '|---|---|---:|---:|---:|---:|---:|']
    pct = lambda v:f'{100*v:.2f}%'
    for name, arms in result['groups'].items():
        for arm,r in arms.items():
            lines.append('| '+' | '.join([name+' (n='+str(r['questions'])+')',arm,
               pct(r['retrieve24']['recall_at_k']),pct(r['retrieve8']['recall_at_k']),
               pct(r['rerank8']['recall_at_k']),pct(r['rerank8']['all_evidence']),pct(r['rerank8']['ndcg_at_k'])])+' |')
    lines += ['', '## English test ablation before reranking', '',
              '| Method | Recall@8 | Recall@24 | nDCG@8 |', '|---|---:|---:|---:|']
    for arm,r in test['summary'].items():
        lines.append(f'| {arm} | {pct(r["top8"]["recall_at_k"])} | {pct(r["top24"]["recall_at_k"])} | {pct(r["top8"]["ndcg_at_k"])} |')
    lines += ['', 'The upgrade combines a semantic encoder and BM25. The hash+BM25 arm isolates lexical improvements; its final rerank8 was not evaluated. These gains must not all be attributed to the encoder alone.']
    lines += ['', '## Multi-report tasks', '',
              '| Method | Tasks | Global recall@8 before rerank | Global rerank8 recall | Global rerank8 complete evidence | Balanced 3000-token recall (no rerank) | Balanced complete evidence |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in result['multi'].items():
        lines.append(f'| {arm} | {r["tasks"]} | {pct(r["global8"]["recall_at_k"])} | {pct(r["global_rerank8"]["recall_at_k"])} | {pct(r["global_rerank8"]["all_evidence"])} | {pct(r["balanced3000"]["recall_at_k"])} | {pct(r["balanced3000"]["all_evidence"])} |')
    lines += ['', '## Generated answers (paired DeepSeek sample)', '',
              '| Group | Method | Numeric correct | Numeric + required citation + quote checks | Complete cases |',
              '|---|---|---:|---:|---:|']
    for kind,arms in result['generation'].items():
        for arm,r in arms.items():
            lines.append(f'| {kind} | {arm} | {r["numeric_correct"]}/{r["numeric_scored"]} | {r["numeric_and_evidence_correct"]}/{r["numeric_scored"]} | {r["complete_cases"]}/{r["complete_cases_scored"]} |')
    lines += ['', '## Runtime (uncached real service, small sample)', '',
              'Total means retrieval plus reranking only. Ingestion, cold model loading, and answer generation are excluded; ingestion and repeat0 measurements remain in the raw runtime artifact.', '',
              '| Language | Method | Warm samples | Retrieval p50 / p95 | Rerank p50 / p95 | Total p50 / p95 |',
              '|---|---|---:|---:|---:|---:|']
    for language,arms in result['runtime_by_language'].items():
        for arm,r in arms.items():
            timing = lambda k:f'{r[k]["p50"]/1000:.3f}s / {r[k]["p95"]/1000:.3f}s'
            lines.append(f'| {language} | {arm} | {r["warm_samples"]} | {timing("retrieve_ms")} | {timing("rerank_ms")} | {timing("total_ms")} |')
    lines += ['', '## Limits', '',
              '- Previously exposed FinQA data: this is a controlled replay, not a fresh unseen holdout.',
              '- Report scope is supplied; corpus contains QA-bearing pages, not full-PDF search across the whole business KB.',
              '- Multi-report tasks combine author questions; missing-report evidence is scored on answerable facets only.',
              '- Balanced 3000-token contexts can contain more than 8 passages and are a separate retrieval-only experiment.',
              '- Single/multi-evidence labels count gold snippets; they do not certify factual vs true multi-hop reasoning.',
              '- Generation sample is small. Non-numeric/empty references remain unscored; quote matching does not establish full narrative factuality.',
              f'- Gold snippet recall does not guarantee all arithmetic operands or table headers are present. {gold_supplied_failures}/{len(failed_singles)} failed selected single-question answers received all labeled gold snippets.',
              '- Balanced context packing is a separate experiment; the deployed main path remains global retrieval24 -> rerank8.',
              '- GPU path remains paused and unvalidated after the reboot. All final measurements use CPU.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
