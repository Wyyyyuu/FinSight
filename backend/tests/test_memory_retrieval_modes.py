from backend.rag.hybrid_service import HybridRAGService, RAGDocument


def test_bm25_statistics_are_collection_scoped(monkeypatch):
    monkeypatch.setenv('RAG_LEXICAL_MODE', 'bm25')
    service = HybridRAGService.for_testing()
    def doc(collection, sid, content):
        return RAGDocument(collection=collection,source_id=sid,content=content,scope='ephemeral')
    service.ingest_documents([doc('ws:a','revenue','营业收入同比增长'),doc('ws:a','noise','weather rain')])
    before = service.hybrid_search('营业收入',collection='ws:a',top_k=2)
    service.ingest_documents([doc('ws:b',str(i),'营业收入') for i in range(20)])
    after = service.hybrid_search('营业收入',collection='ws:a',top_k=2)
    assert [(d['source_id'],d['sparse_score']) for d in before] == [(d['source_id'],d['sparse_score']) for d in after]
    assert after[0]['source_id']=='revenue' and after[0]['sparse_score'] > 0


def test_dense_only_has_no_lexical_rank(monkeypatch):
    monkeypatch.setenv('RAG_LEXICAL_MODE','dense_only')
    service = HybridRAGService.for_testing()
    service.ingest_documents([RAGDocument(collection='ws:a',source_id='a',content='revenue growth',scope='ephemeral')])
    hit = service.hybrid_search('revenue',collection='ws:a',top_k=1)[0]
    assert hit['dense_rank']==1 and hit['sparse_rank'] is None
    assert hit['rrf_score']==1/(service.rrf_k+1)
