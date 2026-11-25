"""HealthFlow 360: Knowledge, Evidence, and text-only Radiology Report Triage.
Run: python3 -m streamlit run app.py

Visual design: "Clinical Calm" (theme in .streamlit/config.toml, change log in
docs/UI_CHANGES.md). Model and corpus text is HTML-escaped before it is rendered.
"""
from __future__ import annotations

import html
import logging
import os
import re
import uuid

import streamlit as st
from src import retrieval
from src.agent import EMERGENCY, REFUSAL, run_agent
from src.report_triage import analyze_radiology_report
from src.report_ingest import ingest_report_upload
from src.app_support import RateLimiter, request_trace, validate_question, validate_report

st.set_page_config(page_title="HealthFlow 360", page_icon="🏥", layout="wide")


def new_session() -> None:
    st.session_state.messages = []
    st.session_state.scan_results = []
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:10]}"


if "messages" not in st.session_state or "session_id" not in st.session_state:
    new_session()
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
if "limiter" not in st.session_state:
    st.session_state.limiter = RateLimiter(max_calls=10, window_s=60)

# ---------------------------------------------------------------- styles

st.markdown("""
<style>
[data-testid="stMainBlockContainer"]{max-width:1000px;padding-top:1.5rem}
[data-testid="stBottomBlockContainer"]{max-width:1000px}
[data-testid="stSidebar"]{min-width:300px !important;max-width:300px !important}
h1{font-size:2rem !important;line-height:1.1 !important}
.hf-brand{display:flex;align-items:center;gap:10px;margin:4px 0 8px}
.hf-brand b{font-family:'Space Grotesk',sans-serif;font-size:18px;line-height:1.2;display:block}
.hf-brand span{font-size:12px;color:#56616B}
.hf-strip{display:flex;gap:10px;align-items:center;padding:10px 14px;margin-bottom:8px;border-radius:10px;
  background:#ECEFEC;border:1px solid #DDE2DD;font-size:13px;color:#3B4650}
.hf-sub{color:#56616B;margin:-6px 0 18px}
.hf-tags{display:flex;gap:8px;align-items:center;font-size:12px;font-weight:600;margin-bottom:6px}
.hf-tag{padding:3px 9px;border-radius:999px;background:#E3F1EE;color:#0A5A51}
.hf-tag.k{background:#ECEFEC;color:#3B4650}
.hf-meta{color:#56616B;font-weight:500}
.hf-cite{font-size:12px;font-weight:600;padding:1px 6px;border-radius:5px;background:#E3F1EE;color:#0A5A51}
.hf-box{display:flex;gap:14px;padding:16px;border-radius:12px;margin:4px 0 8px;line-height:1.5}
.hf-box strong{display:block;font-size:16px;margin-bottom:2px}
.hf-emergency{background:#B42318;color:#FFFFFF}
.hf-declined{background:#F1F3F5;border:1px solid #DCE1E6;color:#3B4650}
.hf-declined strong{color:#17202A}
.hf-error{background:#FFF6E5;border:1px solid #F0DFB8;color:#5C4410}
.hf-note{background:#E3F1EE;color:#34453F}
.hf-note strong{color:#0A5A51}
.hf-sources{background:#FFFFFF;border:1px solid #E2E5E1;border-radius:12px;overflow:hidden;margin:10px 0 4px}
.hf-sources-h{padding:10px 16px;font-size:13px;font-weight:600;border-bottom:1px solid #E2E5E1;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.hf-src{display:flex;gap:12px;padding:14px 16px;border-bottom:1px solid #EEF0ED}
.hf-src:last-child{border-bottom:0}
.hf-num{flex:none;width:24px;height:24px;border-radius:6px;background:#E3F1EE;color:#0A5A51;font-size:12px;font-weight:600;
  display:flex;align-items:center;justify-content:center}
.hf-src a{font-weight:600}
.hf-src p{margin:4px 0 0;font-size:14px;color:#3B4650}
.hf-src code{font-size:11px;color:#56616B}
.hf-spin{flex:none;width:26px;height:26px;border-radius:50%;border:3px solid #C7E2DC;border-top-color:#0E7468;animation:hfspin 1s linear infinite}
@keyframes hfspin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion: reduce){.hf-spin{animation:none}}
.hf-head{padding:20px 22px;border-radius:12px 12px 0 0;display:flex;flex-direction:column;gap:2px}
.hf-head small{font-size:13px;font-weight:600}
.hf-head b{font-family:'Space Grotesk',sans-serif;font-size:26px;line-height:1.15;font-weight:600}
.hf-head.emergency{background:#B42318;color:#FFFFFF}
.hf-head.routine{background:#EEF1F4;color:#17202A;border:1px solid #DCE1E6;border-bottom:0}
.hf-head.unknown{background:#FFF6E5;color:#5C4410;border:1px solid #F0DFB8;border-bottom:0}
.hf-body{background:#FFFFFF;border:1px solid #E2E5E1;border-top:0;border-radius:0 0 12px 12px;padding:20px 22px;margin-bottom:8px}
.hf-body h4{margin:0 0 6px;font-size:13px;font-weight:600;color:#56616B;font-family:Inter,sans-serif}
.hf-body section{margin-bottom:18px}
.hf-flag{display:inline-block;padding:5px 12px;margin:0 6px 6px 0;border-radius:999px;background:#FDE7E5;color:#8F1B12;font-size:14px;font-weight:500}
.hf-summary{padding:14px 16px;border-radius:10px;background:#F5F6F4}
.hf-foot{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:#56616B;border-top:1px solid #EEF0ED;padding-top:14px}
.hf-empty{padding:40px 24px;border:1.5px dashed #C9CFCA;border-radius:12px;text-align:center;color:#56616B}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- constants

MODES = {
    "knowledge": {
        "label": ":material/lightbulb: Knowledge",
        "caption": "General explanations from the model",
        "title": "Knowledge",
        "sub": "General medical explanations from Qwen3-8B. Not checked against sources.",
        "how": "Safety check, then a general educational answer from hosted Qwen3-8B. No citations.",
    },
    "evidence": {
        "label": ":material/menu_book: Evidence",
        "caption": "Answers cited from MedlinePlus",
        "title": "Evidence",
        "sub": "Answers grounded in 750 MedlinePlus documents, with a citation for every claim.",
        "how": "Safety check, then hybrid search over 750 MedlinePlus documents, then a cited answer from Qwen3-8B.",
    },
    "report": {
        "label": ":material/description: Report triage",
        "caption": "Route a radiology report",
        "title": "Report triage",
        "sub": "Paste or upload a radiology report for preliminary routing. A clinician must review every result.",
        "how": "Text is read in memory, checked for suitability, routed by rules, and optionally summarised by Qwen3-8B.",
    },
}
STATUS_LABELS = {"answered": "Answered", "abstained": "Not answered", "escalated": "Escalated", "error": "Error"}
TRIAGE_CONFIDENCE = {"not_validated": "Not clinically validated"}
PASSAGE_CHARS = 350
ICON_ALERT = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="flex:none">'
              '<path d="M12 3l9.5 17h-19z"></path><path d="M12 10v4M12 17v.5"></path></svg>')
ICON_LOCK = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
             'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="flex:none">'
             '<rect x="5" y="11" width="14" height="10" rx="2"></rect><path d="M8 11V8a4 4 0 0 1 8 0v3"></path></svg>')
ICON_INFO = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-linecap="round" aria-hidden="true" style="flex:none"><circle cx="12" cy="12" r="9"></circle>'
             '<path d="M12 8v5M12 16.5v.5"></path></svg>')


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers


def resolve_source(doc_id: str, answer_text: str = "") -> tuple[str, str, str]:
    """Return (title, url, passage) for a cited document ID.

    The answer only records which documents were cited, not which chunk was used,
    so the passage shown is the chunk of that document with the most word overlap
    with the answer. Falls back to the raw ID when the document is not found.
    """
    try:
        chunks = retrieval.get_source(doc_id)
    except Exception:
        chunks = []
    if not chunks:
        return doc_id, "", ""
    answer_words = {w for w in answer_text.lower().split() if len(w) > 3}
    best = max(chunks, key=lambda c: len(answer_words & {w for w in c.text.lower().split() if len(w) > 3}))
    passage = best.text.strip()
    if passage[:1].islower() and ". " in passage[:200]:
        passage = "…" + passage.split(". ", 1)[1]  # chunk starts mid-sentence: skip the fragment
    if len(passage) > PASSAGE_CHARS:
        passage = passage[:PASSAGE_CHARS].rsplit(" ", 1)[0] + "…"
    url = chunks[0].url or ""
    if not url.startswith(("https://", "http://")):
        url = ""
    return chunks[0].title or doc_id, url, passage


def looks_truncated(text: str, min_words: int = 280) -> bool:
    """True when text stops mid-sentence after reaching roughly its output token limit.

    Default 280 words matches the 450-token Knowledge limit (~300-340 words).
    """
    end = text.rstrip().rstrip("*_`").rstrip()
    return len(text.split()) >= min_words and bool(end) and end[-1] not in ".!?:)]\"'"


def evidence_models_cold() -> bool:
    """True until the corpus index (and reranker, when enabled) are loaded in this server process."""
    if not getattr(retrieval, "_corpus", None):
        return True
    reranker_on = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    return reranker_on and getattr(retrieval, "_reranker", None) is None


def render_details(label: str, rows: list[str], trace_url=None) -> None:
    """Developer details (status, request ID, timing, trace) in one collapsed section."""
    with st.expander(label, expanded=False):
        for row in rows:
            st.caption(row)
        if trace_url:
            st.caption(f"[Langfuse trace]({trace_url}) (team Langfuse access required)")


def render_request_metadata(trace_id: str, elapsed: float, trace_url=None) -> None:
    render_details("Details", [f"Request ID: `{trace_id}`", f"Processing time: {elapsed:.2f} s"], trace_url)


def ask_in_knowledge(question: str) -> None:
    """Callback: switch to Knowledge mode and re-send the same question."""
    st.session_state.answer_mode = "knowledge"
    st.session_state.pending_prompt = question

# ---------------------------------------------------------------- chat rendering


def render_assistant_message(message: dict) -> None:
    status = message.get("status")
    mode = message.get("mode")
    content = str(message.get("content", ""))
    citations = message.get("citations") or []
    is_emergency = status == "escalated" and content == EMERGENCY
    is_refusal = status == "escalated" and content == REFUSAL

    tag = '<span class="hf-tag">Evidence</span>' if mode == "evidence" else '<span class="hf-tag k">Knowledge</span>'
    extra = f'<span class="hf-meta">{len(citations)} source{"s" if len(citations) != 1 else ""}</span>' if citations else ""
    html_block(f'<div class="hf-tags">{tag}{extra}</div>')

    if is_emergency:
        html_block(f'<div class="hf-box hf-emergency" role="alert">{ICON_ALERT}<div><strong>This may be a medical emergency</strong>'
                   'Contact your local emergency services now. Do not wait for an AI response.</div></div>')
    elif is_refusal:
        html_block(f'<div class="hf-box hf-declined">{ICON_LOCK}<div><strong>Outside what this assistant can answer</strong>'
                   f'{esc(content)}</div></div>')
    elif status == "error":
        ref = esc(message.get("trace_id", "n/a"))
        html_block(f'<div class="hf-box hf-error">{ICON_INFO}<div><strong>Something went wrong</strong>'
                   f'{esc(content)} Reference: {ref}</div></div>')
    else:
        body = html.escape(content, quote=False)
        if citations:
            body = re.sub(r"\[(\d+)\]", r'<span class="hf-cite">\1</span>', body)
        st.markdown(body, unsafe_allow_html=True)

    if status == "answered" and mode == "knowledge" and looks_truncated(content):
        st.caption("This answer reached the length limit and may be incomplete. Try a narrower question.")

    if status == "abstained":
        if mode == "evidence" and message.get("question"):
            st.caption("The library may not cover this topic. Rephrase with the condition named, or ask for a general explanation:")
            st.button("Ask in Knowledge mode", key=f"ask_k_{message.get('trace_id', id(message))}",
                      icon=":material/lightbulb:", on_click=ask_in_knowledge, args=(message["question"],))
        else:
            st.caption("Try rephrasing with the condition or topic named.")

    if citations:
        rows = []
        for number, source_id in enumerate(citations, start=1):
            title, url, passage = resolve_source(str(source_id), content)
            link = f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(title)}</a>' if url else f"<b>{esc(title)}</b>"
            quote = f"<p>{esc(passage)}</p>" if passage else ""
            rows.append(f'<div class="hf-src"><span class="hf-num">{number}</span><div>{link} '
                        f'<code>{esc(source_id)}</code>{quote}</div></div>')
        html_block('<div class="hf-sources"><div class="hf-sources-h"><span>Sources</span>'
                   '<span class="hf-meta">MedlinePlus, U.S. National Library of Medicine</span></div>'
                   + "".join(rows) + "</div>")

    if status == "answered" and mode == "knowledge":
        st.caption("Model-generated knowledge, not verified against retrieved documents.")

    if is_emergency:
        status_label = "Escalated (possible emergency)"
    elif is_refusal:
        status_label = "Declined"
    else:
        status_label = STATUS_LABELS.get(status, str(status))
    rows = [f"Status: {status_label}"]
    if message.get("trace_id"):
        rows.append(f"Request ID: `{message['trace_id']}`")
    label = "Details"
    if "elapsed" in message:
        rows.append(f"Processing time: {message['elapsed']:.2f} s")
        label = f"Details · {status_label} in {message['elapsed']:.1f} s"
    render_details(label, rows, message.get("trace_url"))

# ---------------------------------------------------------------- triage rendering


def render_triage(result: dict, used_model: bool = False) -> None:
    urgency = str(result.get("urgency", "undetermined"))
    specialist = str(result.get("suggested_specialist", "Undetermined"))
    if urgency == "emergency":
        kind, note = "emergency", "Potential emergency reported. Seek immediate emergency assessment."
    elif urgency == "undetermined":
        kind, note = "unknown", "Urgency cannot be determined from this input."
    else:
        kind, note = "routine", "Preliminary routing only. A clinician must confirm urgency and specialty."
    role = ' role="alert"' if kind == "emergency" else ""
    urgency_label = urgency.replace("_", " ").capitalize()

    flags = [str(f) for f in result.get("emergency_flags") or []]
    flags_html = ("".join(f'<span class="hf-flag">{esc(f[:1].upper() + f[1:])}</span>' for f in flags)
                  if flags else "None found in the report")
    observations = str(result.get("observations") or "Not available")
    summary_title = "AI report summary" if used_model else "Observations"
    confidence = str(result.get("confidence", "not_validated"))
    confidence = TRIAGE_CONFIDENCE.get(confidence, confidence.replace("_", " ").capitalize())

    html_block(
        f'<div class="hf-head {kind}"{role}><small>Urgency: {esc(urgency_label)}</small>'
        f'<b>{esc(specialist)}</b><span>{esc(note)}</span></div>'
        f'<div class="hf-body">'
        f'<section><h4>Emergency flags</h4>{flags_html}</section>'
        f'<section><h4>Why</h4>{esc(result.get("reason", "Not available"))}</section>'
        f'<section class="hf-summary"><h4>{summary_title}</h4>{esc(observations)}</section>'
        f'<div class="hf-foot"><span>{esc(confidence)}</span>'
        f'<span>{esc(result.get("limitations", "This is not a validated clinical assessment."))}</span></div>'
        f'</div>'
    )
    if used_model and looks_truncated(observations, min_words=110):  # 180-token summary limit
        st.caption("This summary reached its length limit and may be incomplete. Review the original report.")

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    html_block('<div class="hf-brand"><svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">'
               '<rect width="32" height="32" rx="9" fill="#0E7468"></rect><path d="M16 9v14M9 16h14" stroke="#FFFFFF" '
               'stroke-width="3" stroke-linecap="round"></path></svg><div><b>HealthFlow 360</b>'
               '<span>Medical education assistant</span></div></div>')
    mode = st.radio(
        "Mode", options=list(MODES), format_func=lambda key: MODES[key]["label"],
        captions=[MODES[k]["caption"] for k in MODES], key="answer_mode",
    )
    st.button("New conversation", icon=":material/add:", use_container_width=True, on_click=new_session)
    st.divider()
    st.markdown(f"**How {MODES[mode]['title']} works**")
    st.caption(MODES[mode]["how"])
    st.caption(f"Session `{st.session_state.session_id}`")

# ---------------------------------------------------------------- page header

html_block(f'<div class="hf-strip">{ICON_INFO}<span><b>Educational prototype.</b> Not for diagnosis, medication decisions '
           'or emergencies. In an emergency, call your local emergency number.</span></div>')
st.title(MODES[mode]["title"])
html_block(f'<p class="hf-sub">{esc(MODES[mode]["sub"])}</p>')

# ---------------------------------------------------------------- report triage

if mode == "report":
    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Report**")
            uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], key="report_upload",
                                        help="Text-based files only. Read in memory, not saved.")
            pasted = st.text_area("Or paste the report", height=180, placeholder="Findings: ...\nImpression: ...")
            use_model = st.checkbox("Add an AI summary of the report", value=True)
            clicked = st.button("Analyse report", type="primary", use_container_width=True,
                                disabled=(uploaded is None and not pasted.strip()))
            st.caption("Synthetic or de-identified reports only. The summary sends report text to the hosted model.")
            req = None
            if clicked:
                st.session_state.scan_results = []
                result = None
                with request_trace("report", st.session_state.session_id,
                                   input_type="upload" if uploaded is not None else "pasted_text",
                                   summary_requested=use_model) as req:
                    if not st.session_state.limiter.allow():
                        req["status"] = "rate_limited"
                        st.warning(f"Too many requests. Retry in {st.session_state.limiter.retry_after():.0f} seconds.")
                    else:
                        try:
                            with st.spinner("Extracting findings and reviewing report…"):
                                report = ingest_report_upload(uploaded) if uploaded is not None else pasted.strip()
                                error = validate_report(report)
                                if error:
                                    raise ValueError(error)
                                result = analyze_radiology_report(report, use_model=use_model)
                            req["status"] = "completed"
                            req["urgency"] = result.get("urgency", "undetermined")
                            summary = result.get("observations", "")
                            req["summary_available"] = bool(use_model and summary.strip() and "Automated summary unavailable" not in summary)
                        except ValueError as exc:
                            req["status"] = "rejected"
                            st.warning(str(exc))
                        except Exception:
                            logging.error("Report analysis failed trace=%s", req["request_id"])
                            st.error(f"Could not analyse this report. Reference: {req['request_id']}")
                if req["status"] == "completed":
                    st.session_state.scan_results.append({"kind": "report", "result": result, "used_model": use_model, "trace_id": req["request_id"], "elapsed": req["elapsed"], "trace_url": req["trace_url"]})
                else:
                    render_request_metadata(req["request_id"], req["elapsed"], req["trace_url"])
    with right:
        if st.session_state.scan_results:
            latest = st.session_state.scan_results[-1]
            render_triage(latest["result"], used_model=bool(latest.get("used_model")))
            render_details(f"Details · Analysed in {latest['elapsed']:.1f} s",
                           [f"Request ID: `{latest['trace_id']}`", f"Processing time: {latest['elapsed']:.2f} s",
                            "Only the structured result is kept in this session; uploads are not added to the search library."],
                           latest.get("trace_url"))
        else:
            html_block('<div class="hf-empty">The routing result will appear here.</div>')

# ---------------------------------------------------------------- knowledge and evidence chat

else:
    if mode == "evidence" and evidence_models_cold():
        html_block('<div class="hf-box hf-note"><div><strong>First question takes a few minutes</strong>'
                   'The first Evidence question after a restart loads the search index for 2,341 passages '
                   '(the very first run also downloads about 1 GB). Later questions take seconds.</div></div>')
    for message in st.session_state.messages:
        avatar = ":material/person:" if message["role"] == "user" else ":material/medical_services:"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(html.escape(message["content"], quote=False))

    prompt = st.chat_input("Ask a medical education question")
    if not prompt and st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")
    if prompt and prompt.strip():
        prompt = prompt.strip()
        message = None
        with request_trace(mode, st.session_state.session_id) as req:
            error = validate_question(prompt)
            if error:
                req["status"] = "rejected"
                st.warning(error)
            elif not st.session_state.limiter.allow():
                req["status"] = "rate_limited"
                st.warning(f"Too many requests. Retry in {st.session_state.limiter.retry_after():.0f} seconds.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt, "mode": mode})
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(html.escape(prompt, quote=False))
                cold = mode == "evidence" and evidence_models_cold()
                loading = st.empty()
                if cold:
                    loading.markdown('<div class="hf-box hf-note"><span class="hf-spin" aria-hidden="true"></span><div>'
                                     '<strong>Preparing the medical library</strong>Loading the search index for 2,341 passages. '
                                     'This takes a few minutes once; later questions take seconds.</div></div>',
                                     unsafe_allow_html=True)
                spinner_text = ("Checking safety and generating an answer…" if mode == "knowledge"
                                else "Searching the library and generating a cited answer…")
                with st.spinner(spinner_text, show_time=True) if not cold else st.container():
                    try:
                        result = run_agent(prompt, session_id=st.session_state.session_id, mode=mode)
                        status = str(result.status)
                        citations = list(result.citations or [])
                        req["status"] = status if status in {"answered", "abstained", "escalated", "error"} else "unknown"
                        req["citation_count"] = len(citations)
                        message = {"role": "assistant", "content": str(result.text), "status": status,
                                   "citations": citations, "mode": mode, "question": prompt}
                    except Exception:
                        logging.error("HealthFlow 360 backend request failed trace=%s", req["request_id"])
                        message = {"role": "assistant", "content": "Unable to complete this request.",
                                   "status": "error", "citations": [], "mode": mode, "question": prompt}
                loading.empty()
        if message is not None:
            message.update(trace_id=req["request_id"], elapsed=req["elapsed"], trace_url=req["trace_url"])
            with st.chat_message("assistant", avatar=":material/medical_services:"):
                render_assistant_message(message)
            st.session_state.messages.append(message)
        else:
            render_request_metadata(req["request_id"], req["elapsed"], req["trace_url"])
