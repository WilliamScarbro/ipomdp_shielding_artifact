"""Derive the TaxiNetV2 single-estimate perception CSVs from the bundled DNN output.

The conformal-comparison experiment (reproduce-conformal) needs the axis-wise
single-estimate (true, predicted) perception CSVs
    .../artifacts/compiler/lib/acc90/real_cte_single_pred_acc90.csv
    .../artifacts/compiler/lib/acc90/real_he_single_pred_acc90.csv
Upstream these are emitted by recreate_taxinet_v2_perception_artifacts.py from the
external cp-control TaxiNet DNN. They are exactly the argmax point predictions, which
are already committed in artifacts/perception/taxinet_point_estimates.csv, so we derive
them here without needing the external DNN or image dataset. Idempotent.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ipomdp_shielding" / "CaseStudies" / "TaxiNetV2" / "artifacts"
POINT_ESTIMATES = PKG / "perception" / "taxinet_point_estimates.csv"
OUT_DIR = PKG / "compiler" / "lib" / "acc90"
CTE_OUT = OUT_DIR / "real_cte_single_pred_acc90.csv"
HE_OUT = OUT_DIR / "real_he_single_pred_acc90.csv"


def main() -> None:
    with POINT_ESTIMATES.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CTE_OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cte", "cte_pred"])
        for r in rows:
            writer.writerow([r["true_cte_idx"], r["pred_cte_idx"]])

    with HE_OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["he", "he_pred"])
        for r in rows:
            writer.writerow([r["true_he_idx"], r["pred_he_idx"]])

    print(f"Wrote {CTE_OUT} ({len(rows)} rows)")
    print(f"Wrote {HE_OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
