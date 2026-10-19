import csv
from pathlib import Path
from app.fact_tool import FactCatalogue, UnknownBrandError
from app.evaluator import score
ROOT = Path(__file__).resolve().parents[1]

def test_csv_counts_and_headers():
    def rows(name):
        with (ROOT/'data'/name).open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
    assert len(rows('answers.csv')) == 50
    assert len(rows('brand_facts.csv')) == 132
    assert len(rows('development_gold.csv')) == 376

def test_fact_lookup():
    cat = FactCatalogue(ROOT/'data'/'brand_facts.csv')
    assert len(cat.get_brand_facts('cleartrail')['facts']) > 0
    try: cat.get_brand_facts('not-a-brand')
    except UnknownBrandError: pass
    else: assert False, 'Unknown brand should be rejected'

def test_scorer_arithmetic_example():
    gold = []
    claims = []
    labels = ["supported", "contradicted", "unknown"]
    for i in range(10):
        label = labels[i % 3]
        gold.append({"answer_id": "A01", "claim_id": f"g{i}", "start_char": i * 10,
                     "end_char": i * 10 + 5, "expected_label": label})
        if i < 8:
            # Seven labels are correct; the eighth matched claim has the wrong label.
            predicted_label = label if i < 7 else ("unknown" if label != "unknown" else "supported")
            claims.append({"claim_id": f"p{i}", "start_char": i * 10,
                           "end_char": i * 10 + 5, "label": predicted_label})
    # Two extra predictions make P=10; only 8 of the 10 gold spans are matched.
    claims.extend([
        {"claim_id": "extra1", "start_char": 200, "end_char": 205, "label": "supported"},
        {"claim_id": "extra2", "start_char": 210, "end_char": 215, "label": "unknown"},
    ])
    result = score([{"answer_id": "A01", "status": "ok", "claims": claims}], gold, ["A01"])
    assert result["G_gold_claims"] == 10
    assert result["P_accepted_predictions"] == 10
    assert result["M_exact_span_matches"] == 8
    assert result["C_exact_span_and_label_correct"] == 7
    assert result["extraction_f1"] == 0.8
    assert result["joint_f1"] == 0.7
