from backend.rag.reranker import RerankerService


def test_grouped_cpu_scores_are_restored_to_sources(monkeypatch):
    calls = []
    class Model:
        def rerank(self, query, documents, *, batch_size):
            calls.append((documents, batch_size))
            return [float(len(d)) for d in documents]
    service = RerankerService(force_backend='bge-reranker-onnx')
    service._onnx_model = Model()
    service._available = True
    monkeypatch.setenv('RAG_RERANKER_BATCH_SIZE','4')
    result = service.rerank('q', [{'id':'a','content':'longest'},
                               {'id':'b','content':'x'},{'id':'c','content':'four'}], top_n=3)
    assert calls == [(['x','four','longest'],4)]
    assert [(r['id'],r['rerank_score']) for r in result] == [('a',7),('c',4),('b',1)]
