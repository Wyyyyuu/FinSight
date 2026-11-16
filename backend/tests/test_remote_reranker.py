import pytest
import requests
from backend.rag.reranker import RerankerService


class Response:
    def __init__(self, data):
        self.data=data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def test_remote_scores_sort_with_source_identity(monkeypatch):
    monkeypatch.setattr(requests,'post',lambda *a,**kw:Response({
        'model':'bge-reranker-base','provider':'CUDAExecutionProvider','scores':[-1,3]}))
    docs=[{'source_id':'a','content':'noise'},{'source_id':'b','content':'revenue','metadata':{'year':2024}}]
    result=RerankerService(force_backend='bge-reranker-http').rerank('revenue',docs,top_n=1)
    assert result==[{**docs[1],'rerank_score':3.}]
    assert 'rerank_score' not in docs[1]


@pytest.mark.parametrize('payload',[
    {'model':'wrong','provider':'CUDAExecutionProvider','scores':[1.]},
    {'model':'bge-reranker-base','provider':'CPUExecutionProvider','scores':[1.]},
    {'model':'bge-reranker-base','provider':'CUDAExecutionProvider','scores':[]},
    {'model':'bge-reranker-base','provider':'CUDAExecutionProvider','scores':[float('nan')]},
])
def test_remote_failure_does_not_fabricate_rerank_scores(monkeypatch,payload):
    monkeypatch.setattr(requests,'post',lambda *a,**kw:Response(payload))
    docs=[{'source_id':'a','content':'evidence'}]
    result=RerankerService(force_backend='bge-reranker-http').rerank('query',docs,top_n=1)
    assert result==docs and 'rerank_score' not in result[0]
