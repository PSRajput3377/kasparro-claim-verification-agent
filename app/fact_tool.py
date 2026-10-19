"""Read-only, deterministic access to the supplied catalogue."""
import csv
from datetime import date
from pathlib import Path
from typing import Any

FACTS_PATH = Path(__file__).resolve().parents[1] / "data" / "brand_facts.csv"
FACT_VERSION = "speaker-catalogue-v4"

class UnknownBrandError(ValueError):
    pass

class FactCatalogue:
    def __init__(self, path: Path = FACTS_PATH):
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))
        self.brand_ids = {r["brand_id"] for r in self.rows}
        self.fact_ids = {r["fact_id"] for r in self.rows}

    def get_brand_facts(self, brand_id: str) -> dict[str, Any]:
        if brand_id not in self.brand_ids:
            raise UnknownBrandError(f"Unknown brand_id: {brand_id}")
        return {"fact_version": FACT_VERSION, "brand_id": brand_id,
                "facts": [r for r in self.rows if r["brand_id"] == brand_id]}

    def lookup(self, brand_id: str, as_of: str = "2026-10-01") -> dict[str, Any]:
        """Convenience filter for date-valid facts; keeps all conditions/units visible."""
        payload = self.get_brand_facts(brand_id)
        target = date.fromisoformat(as_of)
        active = []
        for row in payload["facts"]:
            start = date.fromisoformat(row["valid_from"]) if row.get("valid_from") else None
            end = date.fromisoformat(row["valid_to"]) if row.get("valid_to") else None
            if (start is None or start <= target) and (end is None or target <= end):
                active.append(row)
        return {**payload, "facts": active, "as_of": as_of}
