"""Prepare PubMedQA for HealthFlow360 without calling inference APIs."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data/evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per-class', type=int, default=30)
    parser.add_argument('--few-shot-per-class', type=int, default=2)
    parser.add_argument('--revision', default=None)
    args = parser.parse_args()
    if args.per_class < 1 or args.few_shot_per_class < 1:
        parser.error('Sample sizes must be positive')
    from datasets import load_dataset
    dataset_id = 'qiaojin/PubMedQA'
    config = 'pqa_labeled'
    ds = load_dataset(dataset_id, config, split='train', revision=args.revision)
    records = []
    seen = set()
    for row in ds:
        pubid = str(row['pubid']).strip()
        label = row['final_decision'].strip().lower()
        question = row['question'].strip()
        context = '\n'.join(row['context']['contexts']).strip()
        if not pubid or not question or not context or label not in {'yes', 'no', 'maybe'} or pubid in seen:
            continue
        seen.add(pubid)
        records.append(dict(id=f'pubmedqa_{pubid}', pubid=pubid, dataset='PubMedQA', question=question, context=context, gold_answer=label, reference_explanation=row['long_answer'], url=f'https://pubmed.ncbi.nlm.nih.gov/{pubid}/', category='biomedical_reasoning'))
    rng = random.Random(args.seed)
    evaluation, examples = [], []
    for label in ('yes', 'no', 'maybe'):
        pool = sorted((r for r in records if r['gold_answer'] == label), key=lambda r:r['id'])
        rng.shuffle(pool)
        n = args.per_class + args.few_shot_per_class
        if len(pool) < n:
            raise ValueError(f'Insufficient {label} records: {len(pool)} < {n}')
        evaluation.extend(pool[:args.per_class])
        examples.extend(pool[args.per_class:n])
    rng.shuffle(evaluation)
    rng.shuffle(examples)
    assert not ({r['id'] for r in evaluation} & {r['id'] for r in examples})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset=dataset_id, subset=config, source_split='train', requested_revision=args.revision, dataset_fingerprint=ds._fingerprint, original_records=len(ds), retained_records=len(records), seed=args.seed, partition_note='Custom stratified subsets of Hugging Face labeled split; not official PubMedQA test partition.', files={})
    for name, rows in [('pubmedqa_labeled.jsonl', records), ('pubmedqa_eval.jsonl', evaluation), ('pubmedqa_fewshot.jsonl', examples)]:
        path = args.output_dir / name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        counts = dict(Counter(r['gold_answer'] for r in rows))
        manifest['files'][name] = dict(records=len(rows), label_distribution=counts, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(f'{name}: {len(rows)} records | {counts}')
    path = args.output_dir / 'pubmedqa_manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Manifest: {path}')
    print('PASS: Evaluation and few-shot IDs are disjoint.')
    print('No changes to active RAG corpus. No inference calls.')


if __name__ == '__main__':
    main()
