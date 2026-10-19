import argparse, asyncio, csv, json, os, time
from pathlib import Path
from dotenv import load_dotenv
from .agent import GeminiClaimAgent, AgentError
from .evaluator import read_csv, score
ROOT = Path(__file__).resolve().parents[1]

def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "check"], default="development")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--limit", type=int, default=0, help="0 means all scheduled answers")
    args = parser.parse_args()
    answers = read_csv(ROOT / "data" / "answers.csv")
    scheduled = [a for a in answers if a["split"] == ("development" if args.split == "development" else "check")]
    if args.limit:
        scheduled = scheduled[:args.limit]
    agent = GeminiClaimAgent()
    outputs = []
    for i, answer in enumerate(scheduled, 1):
        started = time.perf_counter()
        try:
            result = agent.classify(answer)
            row = {"answer_id": answer["answer_id"], "status": "ok", "claims": [c.model_dump() for c in result.claims], "latency_seconds": round(time.perf_counter()-started, 3), "agent_version": args.version}
            print(f"[{i}/{len(scheduled)}] {answer['answer_id']}: {len(result.claims)} claims")
        except Exception as exc:
            row = {"answer_id": answer["answer_id"], "status": "error", "error": f"{type(exc).__name__}: {exc}", "claims": [], "latency_seconds": round(time.perf_counter()-started, 3), "agent_version": args.version}
            print(f"[{i}/{len(scheduled)}] {answer['answer_id']}: ERROR {row['error']}")
        outputs.append(row)
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    out_path = reports / f"{args.version}_{args.split}.json"
    out_path.write_text(json.dumps({"split": args.split, "agent_version": args.version, "model": agent.model, "results": outputs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    if args.split == "development":
        gold = read_csv(ROOT / "data" / "development_gold.csv")
        ids = {a["answer_id"] for a in scheduled}
        gold_subset = [g for g in gold if g["answer_id"] in ids]
        metrics = score(outputs, gold_subset, [a["answer_id"] for a in scheduled])
        metrics_path = reports / f"{args.version}_development_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({k: metrics[k] for k in ["G_gold_claims","P_accepted_predictions","M_exact_span_matches","C_exact_span_and_label_correct","extraction_f1","joint_f1","macro_f1","answer_errors"]}, indent=2))
        print(f"Saved metrics: {metrics_path}")
if __name__ == "__main__": main()
