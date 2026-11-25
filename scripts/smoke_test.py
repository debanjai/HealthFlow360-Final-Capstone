"""Simple end-to-end smoke test for HealthFlow 360."""
import json
from src.agent import run_agent
from src.evaluation import evaluate_predictions

def load_gold(path="data/eval_gold.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

if __name__ == "__main__":
    print("=== Smoke tests ===")
    tests = [
        "What is hypertension?",
        "I have chest pain and shortness of breath",
        "How do I cancel an appointment?",
        "Can I get a refill of my blood pressure medication?",
    ]
    preds = []
    for q in tests:
        ans = run_agent(q)
        print(f"\nQ: {q}")
        print(f"Status: {ans.status}")
        print(f"A: {ans.text[:300]}...")
        preds.append({"text": ans.text, "status": ans.status})

    gold = load_gold()[:len(tests)]
    metrics = evaluate_predictions(gold, preds)
    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))
