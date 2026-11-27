"""
02 Retrieval: Validate the configured production retrieval stack
Module: 02 Retrieval
Author: Debanjali Ghosh

"""

import sys, json, time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src import retrieval as r
from src.observability import flush_traces
import os

def main():
    assert os.getenv('RERANKER_ENABLED','').lower()=='true'
    assert Path(os.getenv('CORPUS_PATH','')).name=='corpus_medical_v2_200.jsonl'
    with open('data/evaluation/retrieval_gold.jsonl',encoding='utf-8') as f: gold=[json.loads(l) for l in f if l.strip()]
    rows=[]
    for g in gold:
        t=time.perf_counter(); result=r.hybrid_search(g['question'],5); elapsed=time.perf_counter()-t
        assert len(result)==5 and all(x.doc_id.startswith('mlp_') for x in result)
        hits=[i+1 for i,x in enumerate(result) if x.doc_id in g['expected_source_ids']]
        row=dict(id=g['id'],rank=min(hits) if hits else None,latency_s=elapsed,doc_ids=[x.doc_id for x in result]); rows.append(row); print(json.dumps(row),flush=True)
    assert len(r._corpus)==2341 and r._embeddings is not None and r._reranker is not None
    q=gold[0]['question']
    with patch.object(r,'_rerank',side_effect=RuntimeError('simulated')): fallback=r.hybrid_search(q,5)
    with patch.dict(os.environ,{'RERANKER_ENABLED':'false'}): hybrid=r.hybrid_search(q,5)
    assert [x.chunk_id for x in fallback]==[x.chunk_id for x in hybrid]
    with patch.object(r,'embed_texts',side_effect=RuntimeError('simulated')), patch.object(r,'_rerank',side_effect=AssertionError('must skip')) as mocked:
        sparse=r.hybrid_search(q,5); assert len(sparse)==5 and not mocked.called
    summary={f'recall_at_{k}':sum(x['rank'] is not None and x['rank']<=k for x in rows)/len(rows) for k in (1,3,5)}
    summary['mrr_at_5']=sum(1/x['rank'] if x['rank'] else 0 for x in rows)/len(rows); summary['fallback_checks']='passed'; summary['cold_latency_s']=rows[0]['latency_s']; summary['warm_mean_latency_s']=sum(x['latency_s'] for x in rows[1:])/(len(rows)-1)
    out=Path('reports/evaluation/production_retrieval'); out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__':
    try: main()
    finally: flush_traces()
