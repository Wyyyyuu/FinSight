"""Report raw and independently audited labels without changing experiment inputs."""
from copy import deepcopy
import hashlib
import json
import re
import importlib.metadata
from statistics import mean
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.lightrag.multidoc_pilot import HERE, DATA, ARMS, choose, save
from experiments.lightrag.metrics import retrieval_metrics
from experiments.financial_retrieval_v2.answer_audit import score_facet


def scale_matches(question, unit):
    question=question.lower()
    actual=str(unit or '').lower().replace('_',' ')
    if re.search(r'\bin (?:[a-z.]+ )?millions\b',question):
        return bool(re.search(r'\b(?:millions?|mn)\b',actual))
    if re.search(r'\bin (?:[a-z.]+ )?billions\b',question):
        return bool(re.search(r'\b(?:billions?|bn)\b',actual))
    return None


def score_checked(facet, output, supplied):
    record=score_facet(facet,output,supplied)
    record['monetary_scale_correct']=scale_matches(facet['question'],output.get('unit'))
    record['numeric_and_evidence_correct'] &= record['monetary_scale_correct'] is not False
    return record


def main():
    path = Path(json.loads((HERE/'multidoc_latest.json').read_text())['path'])
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status']=='complete' and len(result['rows'])==15
    plan = result['plan']
    prereg_path=path.parent/'scoring_preregistration.json'
    prereg=json.loads(prereg_path.read_text(encoding='utf-8'))
    if 'reporter_sha256' in prereg:
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==prereg['reporter_sha256']
    assert prereg['answer_rows_at_sealing']==0 and prereg['phase']=='graph_build'
    assert hashlib.sha256((path.parent/'plan.json').read_bytes()).hexdigest()==prereg['plan_sha256']
    assert prereg['tiktoken']==importlib.metadata.version('tiktoken')
    assert {c['model'] for c in result['calls']}=={prereg['llm_model']}
    for relative,digest in prereg['files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==plan['dataset_sha256']
    # Preserve the executed source version even after the reusable runner is fixed.
    assert hashlib.sha256((path.parent/'runner_snapshot.py').read_bytes()).hexdigest()==plan['code_sha256']
    for relative, digest in plan['dependencies'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    tasks, docs = choose(json.loads(DATA.read_text(encoding='utf-8')))
    assert [t['id'] for t in tasks]==plan['sample_ids']
    assert hashlib.sha256(json.dumps([(d['id'],d['contextual']) for d in docs]).encode()).hexdigest()==plan['source_sha256']
    audit = json.loads((HERE/'multidoc_label_audit.json').read_text(encoding='utf-8'))
    assert audit['dataset_sha256']==plan['dataset_sha256']
    corrected = deepcopy(tasks)
    for override in audit['overrides']:
        facet = next(f for t in corrected if t['id']==override['task'] for f in t['facets'] if f['id']==override['facet'])
        if 'answer' in override:
            assert facet['answer']==override['original_answer']
            facet['answer']=override['answer']
        if 'additional_required' in override:
            assert facet['required']==override['original_required']
            facet['required']+=override['additional_required']
    lookup={d['id']:d for d in docs}
    assessments={}
    for label, samples in [('original',tasks),('audited',corrected)]:
        evaluations=[]
        for row in result['rows']:
            task=next(t for t in samples if t['id']==row['id'])
            required=sorted({s for f in task['facets'] if f['answerable'] for s in f['required']})
            supplied={s:lookup[s] for s in row['supplied']}
            output={str(f['facet_id']):f for f in row['output']['facts']}
            evaluations.append(dict(id=row['id'],arm=row['arm'],
                metrics=retrieval_metrics(row['supplied'],required,8),
                facets=[score_checked(f,output.get(f['id'],{}),supplied) for f in task['facets']]))
        scores={}
        for arm in ARMS:
            rows=[r for r in evaluations if r['arm']==arm]
            assert len(rows)==3
            facts=[f for r in rows for f in r['facets']]
            numeric=[f for f in facts if f['numeric_correct'] is not None]
            scores[arm]=dict(recall=mean(r['metrics']['recall_at_k'] for r in rows),
                complete=mean(r['metrics']['all_evidence'] for r in rows),
                precision=mean(r['metrics']['precision_at_k'] for r in rows),
                ndcg=mean(r['metrics']['ndcg_at_k'] for r in rows),
                numeric_correct=sum(f['numeric_correct'] for f in numeric), numeric_scored=len(numeric),
                numeric_and_evidence=sum(f['numeric_and_evidence_correct'] for f in numeric))
        assessments[label]=dict(scores=scores,rows=evaluations)
    costs={}
    for phase in sorted({c['phase'] for c in result['calls']}):
        calls=[c for c in result['calls'] if c['phase']==phase]
        costs[phase]=dict(calls=len(calls),errors=sum(not c['ok'] for c in calls),
            usage_known=sum(c['prompt_tokens'] is not None and c['completion_tokens'] is not None for c in calls),
            input_tokens=sum(c['prompt_tokens'] for c in calls) if all(c['prompt_tokens'] is not None for c in calls) else None,
            output_tokens=sum(c['completion_tokens'] for c in calls) if all(c['completion_tokens'] is not None for c in calls) else None)
    artifact=dict(report=str(path), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256((HERE/'multidoc_label_audit.json').read_bytes()).hexdigest(),
                  assessments=assessments,costs=costs,build=result['build'],
                  scoring_preregistration_sha256=hashlib.sha256(prereg_path.read_bytes()).hexdigest())
    stopped_path=HERE/'runs/20260915T060808Z-multidoc/report.json'
    if stopped_path.exists():
        stopped=json.loads(stopped_path.read_text(encoding='utf-8'))
        artifact['stopped_row_pilot']=dict(status=stopped['status'],calls=len(stopped['calls']),
            tokens=sum((c['prompt_tokens'] or 0)+(c['completion_tokens'] or 0) for c in stopped['calls']),
            answers=len(stopped['rows']),reason=stopped['reason'])
    save(HERE/'multidoc_assessment.json',artifact)
    lines=['# LightRAG 多文档隔离实验','',
        '实际语料：FinQA 开发集中语料最小的一组，501 条原文、6 份报告（含干扰报告）；3 道构造的复合问题，7 个数值子问题。选择依据是语料规模，没有根据本次得分挑题。', '',
        '统一 BGE-M3 CPU、报告元数据前缀、原文来源、最多 8 条证据、3000 token 上限。所有方案均关闭精细重排；线上配置未变。图谱描述只用于找候选，不作为生成上下文。', '',
        f'图谱索引按同报告同页聚合为 {plan["graph_blocks"]} 块，每块保留原始行 ID，最多 480 个 cl100k token，不触发 LightRAG 的 512-token 切分。普通混合检索仍按原始行索引。BGE-M3 服务的子词长度上限为 512，与 cl100k 分词不是同一单位。', '',
        '五组：普通全局 BGE-M3+BM25；按问题明确指定的报告/子问题均衡取证；LightRAG naive 向量块召回；LightRAG mix 图谱+向量块召回；把图谱找到的原文按报告过滤后与 BM25+向量候选用 RRF60 融合，再均衡取证。两个 LightRAG 直接组共享索引和原文解析规则：先按召回块顺序，在块内用同一 BGE-M3+BM25 排序，最多取24条原文候选再截取8条。因此 naive 行级结果也包含共同的词法排序，不能称作纯向量行检索。两种均衡方案使用同样的表格邻行补充策略。', '']
    names=dict(zip(ARMS,['普通全局混合','子问题均衡取证','LightRAG naive','LightRAG mix','图谱辅助均衡取证']))
    for label,title in [('original','原始标签'),('audited','复核修正标签')]:
        lines += ['## '+title,'', '| 方案 | Recall@8 | 证据完整题数 | 数值正确 | 数值、量级及引用检查通过 |',
                  '|---|---:|---:|---:|---:|']
        for arm,s in assessments[label]['scores'].items():
            lines.append(f'| {names[arm]} | {s["recall"]:.2%} | {round(s["complete"]*3)}/3 | {s["numeric_correct"]}/{s["numeric_scored"]} | {s["numeric_and_evidence"]}/{s["numeric_scored"]} |')
        lines += ['']
    lines += ['## 标注复核','',
              '在本次检索/生成结果出现前记录两处修正：2013 到 2014 年摊销额从 58 降到 24，百分比变化应为 −58.6%，原参考答案漏负号；固定期限投资占比的必要原文漏标了分子 24704，补回同一表格的那一行。原始与修正分数都保留，修正内容没有进入任何模型输入。详见 [标注复核](multidoc_label_audit.json)。','',
              '评分快照在建图阶段、0份答案时固定：标注修正、指标实现、分词器版本和生成模型身份均校验。对于题目明确要求 millions/billions 的三个子问题，额外检查输出单位的金额量级；其余四个子问题检查百分数单位。此检查不等同于完整货币换算或叙述事实性审计。','',
              '## 成本与限制','',
              f'- 图谱建库耗时 {result["build"]["seconds"]:.1f} 秒；原始建库状态 {result["build"]["status"]}。',
              *[f'- {phase}: {c["calls"]} 次 LLM 请求，输入 {c["input_tokens"]} / 输出 {c["output_tokens"]} tokens，失败 {c["errors"]}。' for phase,c in costs.items()],
              '- 向量缓存被共享；本轮耗时不能当作在线检索延迟对比。图谱建库含抽取、实体/关系向量化及落盘。',
              '- 三道开发集题目不是新的盲测集，也不是严格的关系多跳问题，不能外推为生产准确率或统计显著收益。',
              '- 标注完整不保证算术所需信息都齐全；数字与字面引文检查不等同于整段总结的事实正确率。',
              '- LightRAG 全局模式先按块排序再解析原始行，取 Top8；均衡模式使用查询里显式给出的子问题与报告范围。普通均衡先筛报告再检索；图谱子查询另外写明报告名称。两组获得相同范围信息，但查询表达和过滤阶段不同。',
              '- 块索引与原始行索引的粒度不同：图谱增量主要看 mix 对 naive，以及图谱辅助均衡对普通均衡；不可把相对旧 hash 的全部收益归给图谱。',
              '- 原主流程、年报索引、Docker 配置、重排开关均未被本轮修改。', '',
              '原始结果路径：`'+str(path)+'`。可运行 `experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/multidoc_report.py` 校验快照并重新生成本报告。']
    if artifact.get('stopped_row_pilot'):
        stopped=artifact['stopped_row_pilot']
        lines += ['', f'另有一次逐行建图试运行在任何检索/回答产生前被主动停止：{stopped["calls"]} 次 LLM 调用、{stopped["tokens"]} tokens、0 份回答。原索引、调用记录与脚本快照保留；其成本不混入聚合块实验耗时，也不作为效果样本。']
    (HERE/'MULTIDOC_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({label:value['scores'] for label,value in assessments.items()},ensure_ascii=False))


if __name__=='__main__':
    main()
