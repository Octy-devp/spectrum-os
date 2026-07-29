"""FALSIFY-009 實驗腳本的 smoke test（scripts/falsify009_rupture_experiment.py）。

以小樣本（--n-synth 5，固定種子）跑完整管線，驗證：
- 腳本零退出、報告 JSON 落盤且 schema 完整（三任務 + 四判決）；
- 關鍵不變量與正式實驗一致：標記窗口候選層全消失、陰性對照全通過、
  路 B 兩 sector 均無特異性、R2 駁回條件未命中。

依賴 ECC 唯讀數據（warfare 樹），缺席時 skip（與 test_quantum.py 同政策）。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "falsify009_rupture_experiment.py"
ECC_WARFARE_FIELDS = "/home/octy/projects/ECC/index/warfare/data/fields"

needs_ecc = pytest.mark.skipif(
    not os.path.exists(ECC_WARFARE_FIELDS),
    reason="ECC data sources not available",
)


@needs_ecc
def test_falsify009_smoke(tmp_path):
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--n-synth", "5", "--output", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))

    # schema
    for key in ["method", "task1_artifact_share", "task2_slutsky",
                "task3_negative_control", "verdicts"]:
        assert key in report, key
    for tree in ["experiment-track0", "anti-intervention-war-1915",
                 "anti-intervention-war-cycle2"]:
        assert tree in report["task1_artifact_share"], tree
    for sector in ["ha_tension", "return_mood"]:
        assert sector in report["task2_slutsky"]["sectors"], sector

    # 任務 1：1915 樹偽影統計與標記窗口候選層消失
    t1 = report["task1_artifact_share"]
    assert t1["anti-intervention-war-1915"]["counts"]["identical_text"] == 16
    assert set(t1["anti-intervention-war-1915"]["counts"]["template_switch_points"]) == {
        "rus/r05", "rus/r08", "dsr/r05", "dsr/r08",
    }
    for per_fac in t1["marked_windows"].values():
        for res in per_fac.values():
            assert res["vanished_as_candidate"] is True

    # 任務 2：路 B 兩 sector 均 ≤ 合成 p95（固定種子下確定性）
    for sector, res in report["task2_slutsky"]["sectors"].items():
        assert res["real_gt_p95"] is False, sector
        assert res["empirical_p_synth_ge_real"] > 0.0, sector
        # 陽性對照至少一個分數版本過 p95 → 檢測器有鑑別力
        pc = res["positive_control"]
        assert pc["plain_gt_p95"] or pc["robust_gt_p95"], sector

    # 任務 3：陰性對照全通過
    assert report["task3_negative_control"]["all_pass"] is True

    # 判決：R1 候選層成立、R2 駁回不成立、R3 路 B 降級、R4 通過不成立
    v = report["verdicts"]
    assert v["R1_return_to_design"]["triggered_as_candidates"] is True
    assert v["R2_theory_rejected"]["triggered"] is False
    assert v["R3_path_b_demotion"]["path_b_demoted"] is True
    assert v["R4_pass"]["triggered"] is False
    assert "退回設計階段" in v["final"]
