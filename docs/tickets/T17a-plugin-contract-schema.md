# T17a — 插件契約 Schema（OS 層·世界無關）

> **工單**：T17a（spectrum-os 待辦表）
> **日期**：2026-08-04
> **定位**：OS 層設計的核心——定義「世界要接入 OS 推演，必須提供的插件接口契約」
> **產物**：`spectrum_os/contracts.py` 新增世界接入 dataclass + 驗證（世界無關）
> **衡量**：一個世界（β₁/γ/anak-world）能填滿全部插件 → OS 就能推演它

---

## 一、目標

把「世界接入 OS 的 7 個插件」中屬於 **OS 層契約**的部分定義為 dataclass：
- ① 場 folder（FieldSpec）
- ② 行動者卡（ActorCard）
- ③ situation payload（SituationSpec——升級現有扁平 dict）
- ⑦ 場的歷史 log（FieldLog）

④⑤（SourceSpec/約束場）已有；⑥（世界 system prompt）屬 T16 剩餘。

## 二、設計原則（世界無關硬紅線）

1. **零 ECC 路徑/世界特定詞**——schema 只能有「世界要填的欄位」，不能有「這個世界的值」
2. **可驗證**——每個 dataclass 有 `__post_init__` 驗證（仿 `RoleTrajectory`）
3. **鏡像 warfare 但抽象**——warfare 的 `field.yaml`/`spiral.yaml`/`rounds/` 是實例；schema 是它們的通用形狀
4. **向後相容**——現有 `_build_layer_payload` 吃的扁平 `situation` dict 必須仍可接受（schema 是它的升級，非取代）

## 三、Schema 草稿

### 3.1 FieldSpec（①場 folder）

```python
@dataclass
class FieldSpec:
    """場 folder 的 OS 契約——一次推演的完整定義（世界無關）。"""
    field_id: str                      # 場的唯一 id（如 "july-crisis-1914"）
    worldline: str                     # 世界線標籤（β₁/γ/...——值由世界填）
    temporal_scope: dict               # {start, end?}——時間範圍（值由世界填）
    spatial_scope: dict                # {primary?, nodes: [...]}——空間範圍
    actors: list[str]                  # 參與的行動者 id（對應 ActorCard）
    params: dict                       # 場參數（manpower_caps/faction_modifiers...）
    triggers: dict = field(default_factory=dict)   # 殘差觸發器（Mode A→B）
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.field_id or not self.field_id.strip():
            raise ValueError("field_id required")
        if not self.worldline or not self.worldline.strip():
            raise ValueError("worldline required")
        if not self.actors:
            raise ValueError("actors cannot be empty")
```

### 3.2 ActorCard（②行動者卡——打破 N=5「5 柱」的關鍵）

```python
@dataclass
class ActorCard:
    """行動者卡 OS 契約——一個行動者的結構化剖面（世界無關）。

    N=5「5 柱」根源 = situation 只給國家名、不給結構 → LLM 每國一條主線。
    本卡提供「內在張力」多主線 → LLM 每行動者可展開多條路。
    """
    actor_id: str                      # 行動者 id（如 "austria-hungary"）
    name: str                          # 顯示名
    inherited_conditions: dict         # 承繼條件：制度/階級/財政/外交承諾（結構）
    field_coordinates: dict            # 場域座標（Bourdieu 5D 或自訂軸）
    internal_tensions: list[dict]      # 內在張力（多主線！）[{"axis", "forces", "stakes"}]
    temporal_states: dict              # 時間演化 {date: {關鍵欄位變化}}
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actor_id or not self.actor_id.strip():
            raise ValueError("actor_id required")
        if not self.internal_tensions:
            raise ValueError("internal_tensions cannot be empty — 每行動者至少一條內在張力線")
```

### 3.3 SituationSpec（③situation payload 升級）

```python
@dataclass
class SituationSpec:
    """situation payload OS 契約——digest + local_texture + 6D 向量的結構化版本。"""
    digest: str                        # 處境敘述（世界填）
    local_texture: dict                # 結構化參數（世界填：動員表/鐵路/距離...）
    vector_6d: dict | None = None      # 6D 向量（可選——缺失時 OS 機械層回退）
    actors: dict[str, ActorCard] = field(default_factory=dict)  # 行動者卡集合

    def __post_init__(self) -> None:
        if not self.digest or not self.digest.strip():
            raise ValueError("digest required")
```

### 3.4 FieldLog（⑦場的歷史 log）

```python
@dataclass
class FieldLog:
    """場的歷史 log OS 契約——已坍縮選擇 + unselected 可能性（維持疊加）。"""
    field_id: str
    entries: list[dict] = field(default_factory=list)   # [{layer, selected, unselected[], ts}]
    meta: dict = field(default_factory=dict)

    def add_entry(self, layer: int, selected: dict, unselected: list[dict]) -> None:
        self.entries.append({"layer": layer, "selected": selected,
                             "unselected": unselected})
```

## 四、驗收

1. **測試**：新增 `tests/test_contracts.py` 覆蓋 4 個 dataclass 的驗證（空/缺欄位 → raise）
2. **世界無關檢定**：`contracts.py` 無 ECC 路徑/世界特定詞（grep 檢定）
3. **向後相容**：現有 `_build_layer_payload` 的扁平 situation dict 仍可接受（測試確認）
4. **全套測試全綠**（現 556 → 560+）

## 五、施工步驟

```
N1: 在 contracts.py 新增 4 個 dataclass + 驗證（FieldSpec/ActorCard/SituationSpec/FieldLog）
N2: 新增 tests/test_contracts.py（驗證 + 向後相容 + 世界無關 grep 檢定）
N3: 跑全套測試確認全綠
→ debug: 修任何失敗
→ 人機收束：schema 交給人類檢視（欄位是否貼近需求）
```

## 六、明確不做（後話）

- ❌ ECC 的文件格式對應（β₁ 的場 folder 實例）——那是 T17b，世界層
- ❌ 行動者卡的實際內容（6 國卡）——T17b
- ❌ probe_expand_layer 拆解——T18
- ❌ 收束 UX——T19
