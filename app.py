"""HealthFlow 360: Knowledge, Evidence, and text-only Radiology Report Triage.
Run: python3 -m streamlit run app.py
"""
from __future__ import annotations

import logging
import uuid

import streamlit as st
from src.agent import run_agent
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

LABELS = {
    "knowledge": "🧠 Knowledge (no RAG)",
    "evidence": "📚 Evidence (RAG)",
    "report": "📄 Radiology Report Triage",
}

with st.sidebar:
    st.header("🏥 HealthFlow 360")
    mode = st.radio(
        "Mode", options=list(LABELS), format_func=lambda key: LABELS[key],
        key="answer_mode",
    )
    st.divider()
    st.subheader("Model")
    st.write("**Qwen3-8B · Hosted API** (text generation)")
    st.caption("Knowledge, Evidence, and report summaries use hosted Qwen. Embeddings use a separate service.")
    st.subheader("Pipeline")
    st.write({
        "knowledge": "Safety check → Hosted Qwen3-8B → Educational answer",
        "evidence": "Safety check → Hybrid retrieval → Hosted Qwen3-8B → Cited answer",
        "report": "PDF/DOCX → Private text ingestion → Suitability validation → Rule-based routing → Optional hosted Qwen3-8B summary",
    }[mode])
    st.divider()
    st.button("🗑️ New conversation", use_container_width=True, on_click=new_session)
    st.caption(f"Session: `{st.session_state.session_id}`")

st.title("HealthFlow 360")
st.warning(
    "**Educational research prototype only.** Not for diagnosis, medication decisions, "
    "or emergency triage. A qualified clinician must review reports. "
    "If someone may be experiencing a medical emergency, seek emergency care immediately."
)


def render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("status"):
        st.caption(f"Status: {message['status']}")
    if message.get("trace_id"):
        if "elapsed" in message:
            render_request_metadata(message["trace_id"], message["elapsed"], message.get("trace_url"))
        else:
            st.caption(f"Request ID: `{message['trace_id']}`")
    citations = message.get("citations") or []
    if citations:
        with st.expander(f"📚 Sources ({len(citations)})"):
            for source_id in citations:
                st.code(str(source_id), language=None)
    if message.get("mode") == "knowledge":
        st.caption("Model-generated knowledge, not verified against retrieved documents.")
    if message.get("status") == "escalated":
        st.error("Clinician review or urgent professional attention required.")
    elif message.get("status") == "abstained":
        st.info("Insufficient information for a supported answer.")
    elif message.get("status") == "error":
        st.error("Request failed. Check application logs.")


def render_request_metadata(trace_id: str, elapsed: float, trace_url=None) -> None:
    st.caption(f"Request ID: `{trace_id}` · Processing time: {elapsed:.2f} s")
    if trace_url:
        st.markdown(f"[Open Langfuse trace]({trace_url})")


def render_triage(result: dict) -> None:
    urgency = str(result.get("urgency", "undetermined"))
    specialist = str(result.get("suggested_specialist", "Undetermined"))
    if urgency == "emergency":
        st.error("Potential emergency reported. Seek immediate emergency medical assessment.")
    elif urgency == "undetermined":
        st.warning("Urgency cannot be determined from this input.")
    else:
        st.info("Preliminary routing only; clinical review is required.")
    col1, col2 = st.columns(2)
    col1.metric("Suggested destination", specialist)
    col2.metric("Urgency", urgency.replace("_", " ").title())
    st.markdown("**Reason**")
    st.write(result.get("reason", "Not available"))
    st.markdown("**Observations**")
    st.write(result.get("observations", "Not available"))
    if result.get("emergency_flags"):
        st.markdown("**Emergency flags**")
        st.write(", ".join(map(str, result["emergency_flags"])))
    st.caption(f"Confidence: {result.get('confidence', 'not_validated')}")
    with st.expander("Limitations and safety information", expanded=True):
        st.write(result.get("limitations", "This is not a validated clinical assessment."))


if mode == "report":
    st.subheader("📄 Radiology Report Triage")
    st.caption("Upload a text-based PDF/DOCX or paste report text. Files are processed in memory and are not indexed into the RAG corpus.")
    uploaded = st.file_uploader("Upload radiology report", type=["pdf", "docx"], key="report_upload")
    pasted = st.text_area("Or paste the radiology report", height=180, placeholder="Findings: ...\nImpression: ...")
    st.caption("Use synthetic or appropriately authorized de-identified reports only. Hosted summaries send report text to the inference service.")
    use_model = st.checkbox("Include hosted Qwen3-8B summary", value=True)
    if st.button("Analyze report", type="primary", disabled=(uploaded is None and not pasted.strip())):
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
                    st.error(f"Could not analyze this report. Reference: {req['request_id']}")
        if req["status"] == "completed":
            st.session_state.scan_results.append({"kind": "report", "result": result, "trace_id": req["request_id"], "elapsed": req["elapsed"], "trace_url": req["trace_url"]})
        else:
            render_request_metadata(req["request_id"], req["elapsed"], req["trace_url"])
    if st.session_state.scan_results:
        st.divider()
        st.subheader("Latest report result")
        latest = st.session_state.scan_results[-1]
        render_triage(latest["result"])
        render_request_metadata(latest["trace_id"], latest["elapsed"], latest.get("trace_url"))
        st.caption("Only the structured result is retained in this Streamlit session; uploaded documents are not saved to the RAG corpus.")
else:
    st.caption("Choose Knowledge for hosted Qwen3-8B answers or Evidence for Qwen3-8B answers grounded in retrieved documents.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a medical education question…")
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
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.spinner("Checking safety and generating an answer…" if mode == "knowledge" else "Retrieving evidence and generating an answer…"):
                    try:
                        result = run_agent(prompt, session_id=st.session_state.session_id, mode=mode)
                        status = str(result.status)
                        citations = list(result.citations or [])
                        req["status"] = status if status in {"answered", "abstained", "escalated", "error"} else "unknown"
                        req["citation_count"] = len(citations)
                        message = {"role": "assistant", "content": str(result.text), "status": status, "citations": citations, "mode": mode}
                    except Exception:
                        logging.error("HealthFlow 360 backend request failed trace=%s", req["request_id"])
                        message = {"role": "assistant", "content": "Unable to complete this request. Check application logs.", "status": "error", "citations": [], "mode": mode}
        if message is not None:
            message.update(trace_id=req["request_id"], elapsed=req["elapsed"], trace_url=req["trace_url"])
            with st.chat_message("assistant"):
                render_assistant_message(message)
            st.session_state.messages.append(message)
        else:
            render_request_metadata(req["request_id"], req["elapsed"], req["trace_url"])
