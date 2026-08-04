"""收束持久層（convergence store）— T19 companion（PLAN-23 §11.7 / §12.7 落盤）。

把「人機收束」的狀態**寫到檔案**——不是 hardcode 路徑，而是**呼叫方指向哪就寫到哪**
（仿 warfare 的 dashboard / round snapshot / .bak 備份模式）：

- ``save_dashboard``：把 ``convergence_view`` 渲染出的文字樹/markdown 寫成 dashboard 檔
  （對應 warfare ``round-{N}-dashboard.md``）。
- ``save_field_log`` / ``load_field_log``：``FieldLog``（selected + unselected 疊加）
  序列化為 JSON 寫檔 / 讀回（對應 warfare 的場歷史 log 落盤）。
- ``save_round_snapshot`` / ``load_round_snapshot``：每輪收束的輕量快照
  （view + collapse + path + field_log_entry + state_log_entry），可跨 session 續接。
- ``.bak`` 自動備份：每次寫入前把舊檔複製為 ``<path>.bak``（仿
  ``save_oob_to_profiles`` 的「寫入前自動備份，支援上一輪狀態復原」）。

🔴 世界無關：零 ECC 路徑 / 零世界特定詞。所有路徑都是呼叫方顯式傳入的參數——
本模組不引用任何世界線，只做「指向哪就寫到哪」的機械落盤。
依賴面：numpy-only（spectrum-os 依賴契約）——用 stdlib ``json``，不引進 yaml。

關係（下游消費 T17a/T19 既有產物，不碰其既有函數）：
- ``convergence_view``（convergence.py，T19）渲染文字樹 → ``save_dashboard`` 寫檔。
- ``FieldLog``（contracts.py，T17a）場的歷史 log → ``save_field_log`` 落盤。
- ``probe_converge_round``（convergence.py，T19）一輪收束 → 回傳 dict 可直接餵
  ``save_round_snapshot``。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from spectrum_os.contracts import FieldLog

#: 快照檔案前綴（對應 warfare ``snapshot-{N}.yaml``；此處用 JSON 保持 numpy-only）。
_SNAPSHOT_PREFIX = "converge-round"

#: 快照容許的頂層鍵（防「整包 dump」——只落盤收束需要的狀態）。
_SNAPSHOT_KEYS: tuple[str, ...] = (
    "round", "view", "collapse", "path", "field_log_entry", "state_log_entry",
)


# ---------------------------------------------------------------------------
# 低階：寫檔 + .bak 備份
# ---------------------------------------------------------------------------

def _backup(path: Path) -> None:
    """寫入前自動備份——把舊檔複製為 ``<path>.bak``（仿 save_oob_to_profiles）。"""
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def _write_json(path: Path, payload: Any) -> Path:
    """原子寫 JSON（先 .bak 備份，再寫新檔）。回傳寫入路徑。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def _write_text(path: Path, text: str) -> Path:
    """原子寫文字檔（dashboard）。先 .bak 備份，再寫新檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# dashboard：收束視圖落盤
# ---------------------------------------------------------------------------

def save_dashboard(view: str, path: str | Path) -> Path:
    """把收束視圖（``convergence_view`` 的文字樹/markdown）寫成 dashboard 檔。

    ``view``：``convergence_view(..., format="text"|"markdown")`` 的輸出字串。
    ``path``：呼叫方指定的落盤位置（檔名自訂，如 ``round-3-dashboard.md``）。

    每次寫入前自動 .bak 備份舊檔。回傳寫入路徑。
    """
    if not isinstance(view, str):
        raise TypeError(f"view 必須是 str（convergence_view 的 text/markdown 輸出），got {type(view).__name__}")
    return _write_text(Path(path), view)


# ---------------------------------------------------------------------------
# FieldLog：場歷史 log 落盤 / 讀回
# ---------------------------------------------------------------------------

def save_field_log(field_log: FieldLog, path: str | Path) -> Path:
    """把 ``FieldLog``（selected + unselected 疊加）序列化為 JSON 寫檔。

    ``field_log``：``FieldLog`` 實例（contracts.py，T17a）。寫入
    ``field_log.to_dict()``——含 field_id / entries（每筆 layer + selected +
    unselected，**未選不刪除**，維持疊加）/ meta。每次寫入前 .bak 備份。
    回傳寫入路徑。
    """
    if not isinstance(field_log, FieldLog):
        raise TypeError(
            f"field_log 必須是 FieldLog 實例（contracts.py），got {type(field_log).__name__}"
        )
    return _write_json(Path(path), field_log.to_dict())


def load_field_log(path: str | Path) -> FieldLog:
    """從 JSON 檔讀回 ``FieldLog``（跨 session 續接用）。

    ``path``：先前 ``save_field_log`` 寫出的檔案。回傳重建的 ``FieldLog``。
    若檔案不存在 → ``FileNotFoundError``。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"field log 檔不存在：{p}")
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "field_id" not in data:
        raise ValueError(f"{p} 不是合法的 FieldLog JSON（缺 field_id）")
    field_log = FieldLog(
        field_id=str(data["field_id"]),
        entries=list(data.get("entries", [])),
        meta=dict(data.get("meta", {})),
    )
    return field_log


# ---------------------------------------------------------------------------
# round snapshot：每輪收束的輕量快照（可跨 session 續接）
# ---------------------------------------------------------------------------

def _snapshot_path(out_dir: str | Path, round_num: int) -> Path:
    return Path(out_dir) / f"{_SNAPSHOT_PREFIX}-{round_num:02d}.json"


def save_round_snapshot(
    round_result: dict,
    round_num: int,
    out_dir: str | Path,
) -> Path:
    """每輪收束結束後寫輕量快照（對應 warfare ``snapshot-{N}.yaml``）。

    ``round_result``：``probe_converge_round`` 的回傳 dict——只取
    ``_SNAPSHOT_KEYS`` 允許的頂層鍵（view / collapse / path / field_log_entry /
    state_log_entry），避免把整包 expanded/tree 落盤（保持輕量）。
    ``round_num``：快照編號（檔名 ``converge-round-{N:02d}.json``）。
    ``out_dir``：呼叫方指定的快照目錄。

    每次寫入前 .bak 備份舊檔。回傳寫入路徑。
    """
    if not isinstance(round_result, dict):
        raise TypeError(
            f"round_result 必須是 probe_converge_round 的回傳 dict，got {type(round_result).__name__}"
        )
    if round_num < 1:
        raise ValueError(f"round_num 必須 ≥ 1，got {round_num}")
    snapshot = {k: round_result.get(k) for k in _SNAPSHOT_KEYS if k in round_result}
    snapshot["round"] = round_num
    return _write_json(_snapshot_path(out_dir, round_num), snapshot)


def load_round_snapshot(round_num: int, out_dir: str | Path) -> dict:
    """從快照檔讀回某輪的收束狀態（跨 session 續接用）。

    回傳 dict（含 round / view / collapse / path / field_log_entry /
    state_log_entry 等鍵，凡快照有者）。檔案不存在 → ``FileNotFoundError``。
    """
    p = _snapshot_path(out_dir, round_num)
    if not p.exists():
        raise FileNotFoundError(f"round 快照不存在：{p}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def list_rounds(out_dir: str | Path) -> list[int]:
    """列出 out_dir 下已存在的收束輪次（依檔名 ``converge-round-{N}.json``）。"""
    out = Path(out_dir)
    rounds: list[int] = []
    if not out.exists():
        return rounds
    for p in out.glob(f"{_SNAPSHOT_PREFIX}-*.json"):
        stem = p.stem
        num = stem.rsplit("-", 1)[-1]
        if num.isdigit():
            rounds.append(int(num))
    return sorted(rounds)


__all__ = [
    "save_dashboard",
    "save_field_log",
    "load_field_log",
    "save_round_snapshot",
    "load_round_snapshot",
    "list_rounds",
]
