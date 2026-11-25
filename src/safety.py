"""
05 Evaluation Safety: Deterministic Clinical Safety Gate & Guardrails
Module: 05 Evaluation Safety
Author: Gunjan

Zero-latency deterministic rule engine that runs before any RAG or model inference.
Heuristic safety gate; clinical performance is not validated.
"""
import re
from typing import List, Tuple
from .schemas import SafetyDecision

# Clinician-aligned emergency red-flag patterns (05 Evaluation Safety)
EMERGENCY_PATTERNS: List[Tuple[str, str]] = [
    # Cardiac / Acute Coronary Syndrome
    (r"\b(crushing|severe|radiating|squeezing)\s+(chest\s+pain|chest\s+pressure)\b|\bchest\s+pain\b", "Acute Coronary Syndrome / Heart Attack"),
    # Respiratory Distress / Cyanosis
    (r"\b(difficulty|trouble|can't)\s+breathing\b|\bshortness\s+of\s+breath\b|\bgasping\b|\blips\s+turning\s+blue\b", "Severe Respiratory Distress"),
    # Stroke / Cerebrovascular (FAST)
    (r"\b(face\s+droop|arm\s+weakness|slurred\s+speech|sudden\s+numbness|one-sided\s+weakness|(?:cannot|can't|unable to)\s+move\s+(?:the\s+)?(?:left|right|one)\s+side|facial\s+droop)\b", "Acute Stroke (FAST symptoms)"),
    # Neurological / Subarachnoid Hemorrhage
    (r"\b(thunderclap\s+headache|worst\s+headache\s+of\s+my\s+life|worst\s+pain\s+of\s+my\s+(entire\s+)?life|sudden\s+severe\s+headache)\b", "Suspected Subarachnoid Hemorrhage"),
    # Metabolic Emergencies (DKA)
    (r"\b(fruity\s+breath|diabetic\s+ketoacidosis|dka|kussmaul)\b", "Diabetic Ketoacidosis Emergency"),
    # Obstetric Emergencies
    (r"\b(pregnant|pregnancy)\b.*(\bblurry\s+vision\b|\bsevere\s+headache\b|\babdominal\s+pain\b)|\bpre-?eclampsia\b", "Severe Obstetric Emergency / Pre-eclampsia"),
    # Loss of Consciousness / Seizure
    (r"\b(unconscious|passed\s+out|unresponsive|syncope|fainted|collapsed|seizure)\b", "Loss of Consciousness / Seizure"),
    # Anaphylaxis / Airway Compromise
    (r"\b(anaphylaxis|throat\s+(?:is\s+)?closing|lip\s+swelling|tongue\s+swelling|severe\s+allergic)\b", "Severe Allergic Reaction / Anaphylaxis"),
    # Massive Hemorrhage / Bleeding
    (r"\b(cough(ing)?|vomit(ing)?)\b.{0,40}\bblood\b|\buncontrolled\s+bleeding\b|\bstabbing\s+wound\b|\bsevere\s+bleeding\b|\bcoughing\s+blood\b", "Severe Hemorrhage"),
    # Poisoning & Mental Health Crisis
    (r"\b(suicid|kill\s+myself|end\s+my\s+life|want\s+to\s+die|self-harm)\b", "Suicide / Mental Health Crisis"),
    (r"\b(overdose|swallowed\s+(a\s+bottle|too\s+many)\s+pills|poisoned)\b", "Acute Toxicity / Overdose"),
    # Acute Vision Loss
    (r"\b(sudden\s+(total\s+)?loss\s+of\s+vision|severe\s+eye\s+injury)\b", "Acute Ocular Emergency")
]

# Prohibited medical queries (05 Evaluation Safety: Dosages, Prescriptions, Deprescribing)
REVIEW_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(how\s+many|what)\b.{0,30}\b(mg|milligrams|tablets|pills|dose|dosage)\b.{0,40}\b(take|give)\b", "Specific Dosage Calculation Request"),
    (r"\b(exact\s+)?(milligram|mg)?\s*(dose|dosage)\b.{0,30}\b(take|give)\b", "Medication Dosage Calculation"),
    (r"\b(can\s+you|please)\s+prescribe\b", "Direct Prescription Request"),
    (r"\b(diagnose\s+me|what\s+disease\s+do\s+I\s+have|do\s+I\s+have\s+cancer)\b", "Direct Clinical Diagnosis Request"),
    (r"\bcan\s+I\s+stop\s+taking\s+my\b.{0,30}\b(insulin|antibiotics|blood\s+pressure|medication|meds)\b", "Self-Deprescribing Risk"),
    (r"\b(refill|prior\s+auth|prior\s+authorization|insurance\s+denial)\b", "Clinical / Insurance Administrative Request")
]

# Prompt injection & jailbreak patterns (05 Evaluation Safety)
PROMPT_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bignore\s+(all\s+)?(previous|prior|your)\s+(instructions|safety|guardrails|rules)\b", "System Prompt Override"),
    (r"\b(disregard|bypass|disable)\s+(safety|filters|guardrails|rules)\b", "Guardrail Bypass"),
    (r"\byou\s+are\s+now\s+(DAN|unrestricted|a\s+doctor|free)\b", "Persona Hijack"),
    (r"\b(roleplay|hypothetical)\b.{0,40}\b(laws|rules|suspended|bypass)\b", "Hypothetical Bypass"),
    (r"\b(print|show|reveal|leak|extract)\b.{0,40}\b(patient|conversation|session|private|history|prompt)\b", "Confidential Extraction"),
    (r"\b(synthesiz(e|ing)|manufactur(e|ing))\b.{0,40}\b(illicit|fentanyl|insulin|controlled\s+substance)\b", "Dangerous Substance Instruction")
]

EMERGENCY_GUIDANCE = (
    "🚨 CRITICAL EMERGENCY DETECTED: Your symptoms plausibly describe an acute, life-threatening emergency. "
    "Please call emergency services immediately (911 in US / 112 or 108 in India) or go to the nearest emergency room. "
    "Do NOT delay seeking urgent professional medical care."
)

REVIEW_GUIDANCE = (
    "⚠️ CLINICAL DISCLAIMER: HealthFlow 360 cannot prescribe medications, calculate personalized dosages, "
    "or provide formal diagnoses. Please consult a licensed physician or pharmacist for clinical advice."
)


from .observability import traced, span, update

@traced("assess_safety")
def assess_safety(question: str) -> SafetyDecision:
    """
    05 Evaluation Safety: Deterministic keyword & regex safety gate.
    Returns standard SafetyDecision matching team contracts.
    """
    clean_q = question.strip().lower()

    # 1. Deterministic Emergency Check (Highest Priority)
    for pattern, label in EMERGENCY_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return SafetyDecision(
                risk_level="emergency",
                reason=f"Emergency red-flags detected ({label}): {pattern}",
                escalate=True
            )

    if re.search(r"\bstroke\b", clean_q) and not re.match(r"^(?:what is|what are|explain|define|how does|how can|tell me about)\b", clean_q):
        return SafetyDecision(risk_level="emergency", reason="Possible stroke context", escalate=True)

    # 2. Check for Prompt Injections / Jailbreaks
    for pattern, label in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return SafetyDecision(
                risk_level="review",
                reason=f"Prompt injection attempt detected ({label}): {pattern}",
                escalate=True
            )

    # Additional override and confidential-instruction patterns.
    if re.search(r"\b(?:forget|ignore|override)\s+(?:your|the)\s+(?:constraints|instructions|rules|policy)\b|\b(?:repeat|reveal|show|print|extract)\b.{0,50}\b(?:hidden|developer|system)\s+(?:developer\s+)?(?:instructions|prompt)\b", clean_q):
        return SafetyDecision(risk_level="review", reason="Prompt injection attempt detected", escalate=True)

    # 3. Check for Prohibited Medical Advice (Dosages / Prescriptions)
    for pattern, label in REVIEW_PATTERNS:
        if re.search(pattern, clean_q, re.IGNORECASE):
            return SafetyDecision(
                risk_level="review",
                reason=f"Prohibited medical action / clinical review required ({label}): {pattern}",
                escalate=True
            )

    # 4. Normal educational query -> Pass to QA
    return SafetyDecision(
        risk_level="normal",
        reason="Query within safe educational bounds",
        escalate=False
    )
