"""Run from project root; smoke and full results are separate."""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env', override=True)
from src.prompt_experiments import predict, LABELS
from src.observability import span, update, trace_details, flush_traces
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

STRATEGIES = ('zero_shot','few_shot','decomposition')
def read(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--strategy',choices=(*STRATEGIES,'all'),default='all')
    args=p.parse_args()
    base=ROOT/'data/evaluation'
    manifest=json.loads((base/'pubmedqa_manifest.json').read_text())
    for name,meta in manifest['files'].items():
        if sha(base/name)!=meta['sha256']: raise ValueError('Hash mismatch: '+name)
    rows=read(base/'pubmedqa_eval.jsonl'); examples=read(base/'pubmedqa_fewshot.jsonl')
    assert Counter(r['gold_answer'] for r in rows)==dict.fromkeys(LABELS,30)
    assert Counter(r['gold_answer'] for r in examples)==dict.fromkeys(LABELS,2)
    assert not ({r['id'] for r in rows}&{r['id'] for r in examples})
    if args.smoke: rows=[next(r for r in rows if r['gold_answer']==label) for label in LABELS]
    model=os.getenv('INFERENCE_MODEL','Qwen/Qwen3-8B')
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
    out=ROOT/'reports/evaluation/prompt_experiments'/('smoke' if args.smoke else 'full'); out.mkdir(parents=True,exist_ok=True)
    config={'model':model,'temperature':0.0,'dataset_hash':sha(base/'pubmedqa_eval.jsonl'),'fewshot_hash':sha(base/'pubmedqa_fewshot.jsonl'),'prompt_code_hash':sha(ROOT/'src/prompt_experiments.py'),'runner_hash':sha(Path(__file__)),'ids':[r['id'] for r in rows]}
    cfg=out/'run_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config: raise ValueError('Existing run configuration differs; archive output directory first')
    cfg.write_text(json.dumps(config,indent=2))
    try:
        for strategy in STRATEGIES if args.strategy=='all' else [args.strategy]:
            path=out/f'{strategy}_predictions.jsonl'
            done={r['id'] for r in read(path)} if path.exists() else set()
            for i,row in enumerate(rows,1):
                if row['id'] in done: continue
                stats={'model_calls':0,'estimated_input_tokens':0}; start=time.perf_counter()
                with span('pubmedqa_experiment',metadata={'strategy':strategy,'case_id':row['id']}) as obs:
                    trace_id,_=trace_details(obs)
                    try: label=predict(strategy,row,examples,tokenizer,stats)
                    except Exception as exc:
                        print(f'Interrupted: {strategy} {row["id"]}: {type(exc).__name__}',flush=True)
                        raise
                    result={'id':row['id'],'gold':row['gold_answer'],'prediction':label,'latency_s':time.perf_counter()-start,'trace_id':trace_id,**stats}
                    update(obs,output={'prediction':label,'correct':label==row['gold_answer']})
                with path.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(result)+'\n'); f.flush(); os.fsync(f.fileno())
                print(f'{strategy} {i}/{len(rows)}: {label} | {result["latency_s"]:.2f}s',flush=True)
                time.sleep(1.1)
    finally: flush_traces()
    summary=[]; reports={}
    for strategy in STRATEGIES:
        path=out/f'{strategy}_predictions.jsonl'
        if not path.exists(): continue
        records=read(path)
        if len(records)!=len(rows) or {r['id'] for r in records}!={r['id'] for r in rows}: continue
        y=[r['gold'] for r in records]; pred=[r['prediction'] for r in records]
        report=classification_report(y,pred,labels=list(LABELS),output_dict=True,zero_division=0)
        lat=[r['latency_s'] for r in records]
        summary.append({'strategy':strategy,'n':len(y),'accuracy':accuracy_score(y,pred),'macro_f1':report['macro avg']['f1-score'],'macro_precision':report['macro avg']['precision'],'macro_recall':report['macro avg']['recall'],'invalid_rate':pred.count('invalid')/len(y),'mean_latency_s':float(np.mean(lat)),'p95_latency_s':float(np.percentile(lat,95)),'model_calls':sum(r['model_calls'] for r in records),'estimated_input_tokens':sum(r['estimated_input_tokens'] for r in records)})
        reports[strategy]={'classification_report':report,'confusion_matrix':confusion_matrix(y,pred,labels=[*LABELS,'invalid']).tolist(),'matrix_labels':[*LABELS,'invalid']}
    if summary:
        with (out/'metrics_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        (out/'classification_reports.json').write_text(json.dumps(reports,indent=2))
        print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
