import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from src.model_clients import embed_texts  # loads project environment
from src.ingest import build_corpus, save_corpus, validate_embedding_texts

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import os
    data=Path('data'); parent=data/'medlineplus_v2_documents.jsonl'; baseline=data/'corpus_medical_v2.jsonl'
    manifest=json.loads((data/'corpus_medical_v2_manifest.json').read_text())
    if sha(parent)!=manifest['documents_sha256'] or sha(baseline)!=manifest['corpus_sha256']: raise ValueError('Frozen input hash mismatch')
    if os.getenv('EMBEDDING_MODEL','BAAI/bge-small-en-v1.5')!=manifest['embedding_model']: raise ValueError('Embedding model mismatch')
    with parent.open(encoding='utf-8') as f: docs=[json.loads(l) for l in f if l.strip()]
    chunks=build_corpus(docs,chunk_size=200,overlap=50); texts=[c.text for c in chunks]
    validate_embedding_texts(texts)
    if (len(chunks),len(set(texts)),len({c.doc_id for c in chunks}))!=(2341,2340,750): raise ValueError('Chunk counts differ from benchmark; investigate before activation')
    dest=data/'corpus_medical_v2_200.jsonl'; tmp=dest.with_suffix('.tmp'); save_corpus(chunks,tmp); tmp.replace(dest)
    result=dict(parent_sha256=sha(parent),baseline_sha256=sha(baseline),corpus_sha256=sha(dest),documents=750,chunks=len(chunks),unique_texts=len(set(texts)),chunk_size=200,overlap=50,embedding_model=manifest['embedding_model'])
    (data/'corpus_medical_v2_200_manifest.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
