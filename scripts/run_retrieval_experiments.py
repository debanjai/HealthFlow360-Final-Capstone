"""Run from project root: python3 scripts/run_retrieval_experiments.py."""
import sys, json, hashlib, time, csv, os, argparse
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from rank_bm25 import BM25Okapi
from src.ingest import build_corpus, load_corpus
from src.model_clients import embed_texts
from src.observability import span, update, flush_traces

def read(p):
    with p.open(encoding='utf-8') as f: return [json.loads(l) for l in f if l.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def matrix(texts, path):
    if path.exists(): a=np.load(path,allow_pickle=False)
    else:
        a=np.asarray(embed_texts(texts),dtype=np.float32)
        if a.shape!=(len(texts),384) or not np.isfinite(a).all(): raise ValueError('Invalid embeddings')
        tmp=path.with_suffix('.tmp.npy'); np.save(tmp,a); tmp.replace(path)
    if a.shape!=(len(texts),384) or not np.isfinite(a).all(): raise ValueError('Invalid cached embeddings')
    return a

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--skip-reranker',action='store_true'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    data=Path('data'); out=Path('reports/evaluation/retrieval_experiments')/('smoke' if args.smoke else 'full'); out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache'; cache.mkdir(exist_ok=True)
    parent=data/'medlineplus_v2_documents.jsonl'; active=data/'corpus_medical_v2.jsonl'; manifest=json.loads((data/'corpus_medical_v2_manifest.json').read_text()); goldpath=data/'evaluation/retrieval_gold.jsonl'
    assert sha(parent)==manifest['documents_sha256'] and sha(active)==manifest['corpus_sha256'], 'Corpus hash mismatch'
    docs=read(parent); gold=read(goldpath); ids={d['doc_id'] for d in docs}; assert len(ids)==len(docs)==750
    assert len({g['id'] for g in gold})==len(gold) and all(set(g['expected_source_ids'])<=ids and g['expected_source_ids'] for g in gold)
    if args.smoke: gold=gold[:3]
    model=os.getenv('EMBEDDING_MODEL','BAAI/bge-small-en-v1.5'); assert model==manifest['embedding_model']
    config={'parent_sha256':sha(parent),'active_sha256':sha(active),'gold_sha256':sha(goldpath),'model':model,'overlap':50,'candidate_k':20,'reranker':'BAAI/bge-reranker-base','smoke':args.smoke,'skip_reranker':args.skip_reranker}
    (out/'run_config.json').write_text(json.dumps(config,indent=2)); key=digest(config)[:16]
    reranker=None
    if not args.skip_reranker:
        from sentence_transformers import CrossEncoder
        reranker=CrossEncoder(config['reranker'],device='cpu',max_length=512)
    rows=[]; builds=[]
    for size in ([400] if args.smoke else [400,300,200]):
        start=time.perf_counter(); chunks=load_corpus(active) if size==400 else build_corpus(docs,chunk_size=size,overlap=50)
        texts=[c.text for c in chunks]; bm=BM25Okapi([t.lower().split() for t in texts]); bm_build=time.perf_counter()-start
        p=cache/f'{size}_{key}_{digest(texts)[:16]}.npy'; cached=p.exists(); start=time.perf_counter(); emb=matrix(texts,p); dense_build=time.perf_counter()-start
        builds.append(dict(chunk_size=size,chunks=len(chunks),documents=len({c.doc_id for c in chunks}),unique_texts=len(set(texts)),bm25_and_chunk_build_s=bm_build,dense_load_or_build_s=dense_build,cache_hit=cached))
        norms=np.linalg.norm(emb,axis=1)
        for i,g in enumerate(gold,1):
            q=g['question']; expected=set(g['expected_source_ids']); timings={}; scores={}
            with span('retrieval_experiment',metadata={'case_id':g['id'],'chunk_size':size}) as obs:
                t=time.perf_counter(); b=bm.get_scores(q.lower().split()); bn=b/(b.max()+1e-8) if b.max()>0 else b; scores['bm25']=np.argsort(bn)[::-1]; timings['bm25']=time.perf_counter()-t
                t=time.perf_counter(); qe=np.asarray(embed_texts([q])[0]); assert qe.shape==(384,) and np.isfinite(qe).all(); d=emb@qe/(norms*np.linalg.norm(qe)+1e-8); scores['dense']=np.argsort(d)[::-1]; timings['dense']=time.perf_counter()-t
                t=time.perf_counter(); dn=(d-d.min())/(d.max()-d.min()+1e-8); h=.4*bn+.6*dn; scores['hybrid']=np.argsort(h)[::-1]; timings['hybrid']=timings['bm25']+timings['dense']+time.perf_counter()-t
                if reranker is not None:
                    t=time.perf_counter(); candidates=scores['hybrid'][:20]; rs=np.asarray(reranker.predict([(q,texts[j]) for j in candidates],show_progress_bar=False)); assert rs.shape==(len(candidates),) and np.isfinite(rs).all(); scores['hybrid_reranked']=candidates[np.argsort(rs)[::-1]]; timings['hybrid_reranked']=timings['hybrid']+time.perf_counter()-t
                for method,order in scores.items():
                    retrieved=[chunks[j].doc_id for j in order[:5]]; hits=[j+1 for j,x in enumerate(retrieved) if x in expected]; row=dict(chunk_size=size,method=method,id=g['id'],expected_source_ids=sorted(expected),retrieved_doc_ids=retrieved,retrieved_chunk_ids=[chunks[j].chunk_id for j in order[:5]],latency_s=timings[method],mrr_at_5=1/min(hits) if hits else 0)
                    for k in (1,3,5): row[f'recall_at_{k}']=len(expected.intersection(retrieved[:k]))/len(expected)
                    rows.append(row)
                update(obs,output={'methods':len(scores)})
            print(f'{size} tokens | {i}/{len(gold)} | {g["id"]}',flush=True)
        # Save completed configuration, including partial experiment results.
        (out/'per_query_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (out/'index_builds.json').write_text(json.dumps(builds,indent=2))
        summary=[]
        for s,m in sorted({(r['chunk_size'],r['method']) for r in rows}):
            group=[r for r in rows if r['chunk_size']==s and r['method']==m]; a={'chunk_size':s,'method':m,'n':len(group)}
            for metric in ('recall_at_1','recall_at_3','recall_at_5','mrr_at_5','latency_s'): a['mean_'+metric]=float(np.mean([r[metric] for r in group]))
            a['p95_latency_s']=float(np.percentile([r['latency_s'] for r in group],95)); summary.append(a)
        with (out/'retrieval_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
        print(json.dumps(summary,indent=2))
        flush_traces()
if __name__=='__main__':
    try: main()
    finally: flush_traces()
