#!/usr/bin/env python3
"""Build a versioned MedlinePlus corpus without modifying the active corpus."""
"""
1 Medical Corpus: Build a versioned MedlinePlus corpus without modifying the active corpus
Module: 1 Medical Corpus
Author: Debanjali Ghosh

"""

import argparse, hashlib, io, json, random, re, sys, zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

PAGE = 'https://medlineplus.gov/xml.html'
ATTRIBUTION = 'Source: MedlinePlus, National Library of Medicine.'
class HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.parts=[]; self.links=[]
    def handle_data(self, data): self.parts.append(data)
    def handle_starttag(self, tag, attrs):
        if tag in ('p','li','br','div','h2','h3'): self.parts.append(' ')
        if tag == 'a': self.links.append(dict(attrs).get('href',''))
    def handle_endtag(self, tag):
        if tag in ('p','li','div','h2','h3'): self.parts.append(' ')
def sha(data): return hashlib.sha256(data).hexdigest()
def fetch(url):
    with urlopen(url, timeout=90) as r: return r.read()
def parse_xml(raw):
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.xml')]
            if len(names)!=1: raise ValueError('Expected exactly one XML file in archive')
            raw=z.read(names[0])
    root=ET.fromstring(raw); docs=[]; seen=set(); ids=set(); urls=set()
    for t in root.findall('health-topic'):
        if t.get('language','').lower()!='english': continue
        s=t.find('full-summary'); title=t.get('title','').strip(); url=t.get('url',''); ident=t.get('id','')
        if s is None or not title or not ident or urlparse(url).hostname!='medlineplus.gov': continue
        h=HTML(); h.feed((s.text or '')+''.join(ET.tostring(c,encoding='unicode') for c in s)); text=' '.join(''.join(h.parts).split())
        key=sha(text.casefold().encode())
        if not text or key in seen or ident in ids or url in urls: continue
        seen.add(key); ids.add(ident); urls.add(url)
        docs.append(dict(doc_id='mlp_'+ident,title=title,url=url,text=text))
    return docs, root.attrib

def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--source',type=Path,help='Previously downloaded XML or ZIP'); p.add_argument('--target',type=int,default=750); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    if not 500<=a.target<=2000: p.error('target must be 500..2000')
    root=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(root))
    from dotenv import load_dotenv
    load_dotenv(root/'.env',override=True)
    from src.ingest import build_corpus, validate_embedding_texts, embedding_tokenizer, embedding_limit
    import os
    out=root/'data'; out.mkdir(exist_ok=True)
    archive=out/'medlineplus_v2_source.zip'; raw_path=out/'medlineplus_v2_documents.jsonl'; corpus=out/'corpus_medical_v2.jsonl'; manifest=out/'corpus_medical_v2_manifest.json'
    if any(x.exists() for x in (raw_path,corpus,manifest)): raise SystemExit('Version v2 files already exist; move them aside before rebuilding.')
    url=None
    if a.source: raw=a.source.read_bytes()
    else:
        h=HTML(); h.feed(fetch(PAGE).decode('utf-8'))
        links=[urljoin(PAGE,x) for x in h.links if re.search(r'mplus_topics_.*\.zip$',x,re.I)]
        links=[x for x in links if urlparse(x).scheme=='https' and urlparse(x).hostname=='medlineplus.gov']
        if not links: raise SystemExit('No archive link found. Download ZIP from '+PAGE+' and use --source PATH.')
        url=links[0]; raw=fetch(url)
    docs, meta=parse_xml(raw)
    if len(docs)<a.target: raise SystemExit(f'Only {len(docs)} usable topics; requested {a.target}.')
    selected=random.Random(a.seed).sample(sorted(docs,key=lambda d:d['doc_id']),a.target); selected.sort(key=lambda d:d['doc_id'])
    chunks=build_corpus(selected,chunk_size=400,overlap=50)
    validate_embedding_texts([c.text for c in chunks])
    if len({c.doc_id for c in chunks})!=a.target or len({c.chunk_id for c in chunks})!=len(chunks): raise ValueError('Document/chunk count validation failed')
    model=os.getenv('EMBEDDING_MODEL','BAAI/bge-small-en-v1.5'); tok=embedding_tokenizer(model)
    lengths=[len(tok.encode(c.text,add_special_tokens=True,verbose=False)) for c in chunks]
    raw_bytes=''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in selected).encode(); chunk_bytes=''.join(c.model_dump_json()+'\n' for c in chunks).encode()
    info=dict(attribution=ATTRIBUTION,source_page=PAGE,download_url=url,local_source=str(a.source) if a.source else None,source_sha256=sha(raw),xml_metadata=meta,eligible_documents=len(docs),documents=a.target,chunks=len(chunks),unique_chunk_texts=len({c.text.casefold() for c in chunks}),seed=a.seed,chunk_size=400,overlap=50,embedding_model=model,embedding_limit=embedding_limit(),max_chunk_tokens=max(lengths),documents_sha256=sha(raw_bytes),corpus_sha256=sha(chunk_bytes))
    if not a.source: archive.write_bytes(raw)
    raw_path.write_bytes(raw_bytes); corpus.write_bytes(chunk_bytes); manifest.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2)); print('Build complete. Active corpus unchanged.')
if __name__=='__main__': main()
