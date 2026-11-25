"""Safety-first deterministic planner, executor and structural critic."""
from __future__ import annotations
import logging
import os
import re
from .schemas import Answer, SafetyDecision, Evidence, Plan
from .retrieval import hybrid_search
from .generation import generate_answer, generate_knowledge_answer
from .safety import assess_safety
from .observability import traced, span, update
from .agent_state import get_history, record
from .agent_tools import Executor

logger = logging.getLogger(__name__)
VALID_MODES = {'knowledge','evidence'}
EMERGENCY = 'This may be a medical emergency. Contact your local emergency services immediately. Do not wait for an AI response.'
REFUSAL = 'I cannot provide personalized prescriptions, dosages, diagnoses, or disclose private information or internal instructions. For medical decisions, consult a qualified clinician or pharmacist.'

@traced('agent_planner')
def make_plan(question, mode, history):
    # Use state conservatively: ambiguous follow-ups require clarification,
    # rather than injecting prior potentially sensitive text into model prompts.
    if history and re.match(r'^(?:what about (?:it|that|this)|and (?:it|that|this)|what (?:does|is) (?:it|that|this))\s*[?!.]*$', question, re.I):
        return Plan(intent='clarify')
    return Plan(intent=mode, queries=[question], tools=['knowledge'] if mode=='knowledge' else ['retrieve','grounded'])

@traced('agent_critic')
def criticize(answer, mode, evidence):
    if not isinstance(answer, Answer):
        raise TypeError('Malformed answer')
    answer = Answer.model_validate(answer.model_dump())
    if not answer.text.strip() or len(answer.text)>16000:
        return Answer(text='Unable to validate the response.',status='abstained')
    if answer.status=='answered' and ((mode=='knowledge' and answer.citations) or (mode=='evidence' and (not answer.citations or not set(answer.citations)<= {e.doc_id for e in evidence}))):
        return Answer(text='Unable to verify the generated citations.',status='abstained')
    return answer

@traced('run_agent')
def run_agent(question: str, session_id: str='default', mode: str='evidence') -> Answer:
    if mode not in VALID_MODES:
        raise ValueError('Unknown mode')
    question=question.strip()
    if not question:
        return Answer(text='Please enter a medical question.',status='abstained',session_id=session_id)
    risk='unknown'
    def finish(answer):
        answer.session_id=session_id
        record(session_id, {'question':question[:2000], 'answer':answer.text[:16000], 'status':answer.status,'risk':risk,'mode':mode,'citations':list(answer.citations)})
        return answer
    # Resolve functions at invocation time to preserve evaluation instrumentation.
    executor=Executor({'safety':assess_safety,'knowledge':generate_knowledge_answer,'retrieve':hybrid_search,'grounded':generate_answer})
    try:
        decision=executor.call('safety',question)
        if not isinstance(decision,SafetyDecision):
            raise TypeError('Malformed safety decision')
        risk=decision.risk_level
        if risk=='emergency':
            return finish(Answer(text=EMERGENCY,status='escalated'))
        if risk=='review' or decision.escalate:
            return finish(Answer(text=REFUSAL,status='escalated'))
        if len(question)>2000:
            return finish(Answer(text='Please shorten your question to 2,000 characters.',status='abstained'))
        plan=make_plan(question,mode,get_history(session_id))
        if isinstance(plan,Plan) and plan.intent=='clarify':
            return finish(Answer(text='Please restate your medical question with the condition or topic you mean.',status='abstained'))
        expected=['knowledge'] if mode=='knowledge' else ['retrieve','grounded']
        if not isinstance(plan,Plan) or plan.tools!=expected or plan.queries!=[question] or plan.intent!=mode:
            raise ValueError('Invalid plan')
        evidence=[]
        with span('agent_executor',metadata={'mode':mode}) as obs:
            if mode=='knowledge':
                if not 1<=int(os.getenv('KNOWLEDGE_MAX_TOKENS','450'))<=450:
                    raise ValueError('Output budget exceeded')
                answer=executor.call('knowledge',question)
            else:
                if not 1<=int(os.getenv('GEN_MAX_SOURCES','4'))<=4 or not 200<=int(os.getenv('GEN_MAX_CHARS_PER_SOURCE','1400'))<=1400:
                    raise ValueError('Context budget exceeded')
                evidence=executor.call('retrieve',question,k=4)
                if not isinstance(evidence,list) or len(evidence)>4 or any(not isinstance(e,Evidence) for e in evidence):
                    raise TypeError('Malformed retrieval output')
                answer=executor.call('grounded',question,evidence)
            update(obs,output={'tool_steps':executor.steps})
        return finish(criticize(answer,mode,evidence))
    except Exception as exc:
        logger.warning('Agent workflow failed: %s',type(exc).__name__)
        return finish(Answer(text='Unable to process the question at this time. Please consult a qualified clinician for medical concerns.',status='error'))
