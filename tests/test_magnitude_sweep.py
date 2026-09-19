"""回歸掃尾：kernel 預設律狀態饋入讀數的幅度量家法全清——H4 揭出的最後兩處。

家法（kernel 對**自己構造的預設律**讀數夾制；substrate ctx 保持 raw）：
  - b55d6ba  c_source 主項     ``c_eff·C``        → ``c_eff·max(C, 0)``
  - 42a0de3  k_inflow 預設項   ``η·μ_E·P``        → ``η·μ_E·max(P, 0)``
  - 本檔     k_decay 預設項    ``γ_K·K``          → ``γ_K·max(K, 0)``
             c_source 阻塞項   ``κ_C·(1−Φ)``      → ``κ_C·(1−clip(Φ, 0, 1))``

事故（2026-09-19，H4 同家法潛伏雷審計）：
  1. ``γ_K > 2/dt`` 時 RK4 中間步 K 下潛 < 0（無流入時連續律 dK = −γ_K·K
     單調指向 0，是自癒流；步末 clamp_nonneg 兜底），預設折舊律裸讀 stage
     隨之為負，被 ``_driver_value`` 防護誤爆
     （ValueError: substrate driver k_decay returned a negative value）。
  2. ``κ_Φ·drive·dt/2 > 1`` 時 RK4 中間步 Φ 越上界 > 1（logistic 連續流單調
     指向 Φ* ≤ 1，是自癒流；步末 clip 兜底），``κ_C·(1−Φ)`` 裸讀 stage 為負，
     κ_C > 0 且主項小時 c_source 總額為負，同樣誤爆
     （ValueError: substrate driver c_source returned a negative value）。

分界契約（同 42a0de3 裁定）：kernel 只夾制自己構造的預設律讀數；substrate
的 ``k_decay_fn``／``c_source_fn`` 仍收到原始 ``ctx["K"]``／``ctx["Phi"]``
（世界有權看到真實狀態，含 RK4 中間步的暫時越界值），返回負值照樣大聲爆炸
——防護不松。至此預設律中所有**狀態饋入**的 driver 讀數（4 處）已全部夾制；
φ_drive／φ_suppress／issuance／verified_growth 不饋入狀態，無此暴露面。
"""
import numpy as np
import pytest

from spectrum_os.kernel.forces import FastChannelConfig, ForceFieldDynamics


def _engine_5d(**kw) -> ForceFieldDynamics:
    R0 = np.array([1.5, 1.2, 0.9])
    C0 = np.array([0.4, 0.1, 0.3])
    fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1)
    return ForceFieldDynamics(R0, C0, fast_channel=fc, **kw)


class TestKDecayMagnitude:
    """k_decay 預設律 ``γ_K·K`` 把 K 讀作幅度量（≥0）。"""

    def test_negative_intermediate_k_does_not_raise(self):
        """直接喂負 K stage 進 8 分量 RHS——修前在此 ValueError，修後必須有限。"""
        fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1,
                               eta=1.0, mu_e=3.0, gamma_k=3.0)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]),
                                 fast_channel=fc, latent_force=1.0)
        neg_k = np.full_like(eng._K, -0.25)
        out = eng._morphology_core(
            0.0, eng._R, eng._C, eng._P_pool, eng._P_dist,
            eng._Phi, neg_k, eng._M, eng._Eta)
        d_k = out[5]
        for comp in out:
            assert np.all(np.isfinite(comp))
        # 恢復流：K < 0 時夾制後的折舊讀數歸零（不再往下拉），流入
        # η·μ_E·P_pool > 0 ⇒ dK > 0（向零回升）；步末 clamp_nonneg 再兜底
        assert np.all(d_k[np.asarray(neg_k) < 0] > 0)

    def test_step_survives_strong_decay_and_stays_nonnegative(self):
        """γ_K·dt/2 > 1（3.0·1.0/2 > 1）：修前 step 0 即炸，修後整支積分存活。"""
        fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1,
                               eta=0.0, mu_e=0.0, gamma_k=3.0, k0=0.5)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]), fast_channel=fc)
        for _ in range(60):
            eng.step(1.0)
        assert np.all(np.isfinite(eng._R)) and np.all(np.isfinite(eng._K))
        assert np.all(eng._K >= 0.0)
        # 自癒語義：無流入時 K 單調衰減到 0（被步末夾制駐留在 0）
        assert np.all(eng._K <= 1e-12)

    def test_substrate_k_decay_fn_negative_still_raises_and_ctx_raw(self):
        """防護不松：substrate 明式供律回負值照樣爆炸；ctx["K"] 維持原始觀測量。"""
        eng = _engine_5d()
        seen_ctx_k = {}

        def k_decay_fn(t, ctx):
            seen_ctx_k["K"] = np.asarray(ctx["K"]).copy()
            return np.full_like(eng._K, -0.5)

        eng._fc["k_decay_fn"] = k_decay_fn
        neg_k = np.full_like(eng._K, -0.25)
        with pytest.raises(ValueError, match="k_decay"):
            eng._morphology_core(
                0.0, eng._R, eng._C, eng._P_pool, eng._P_dist,
                eng._Phi, neg_k, eng._M, eng._Eta)
        # ctx 未夾制：fn 看到的是原始（負）stage K
        assert np.all(seen_ctx_k["K"] == neg_k)


class TestCSourcePhiShareMagnitude:
    """c_source 阻塞項 ``κ_C·(1−Φ)`` 把 Φ 讀作份額（clip 到 [0,1]，步末同界）。"""

    def test_phi_above_one_stage_does_not_raise(self):
        """直接喂 Φ>1 stage 進 RHS——修前 κ_C>0 時 c_source 為負誤爆。

        growth_c=0、C0=0 ⇒ 主項恆 0；修後 c_source = κ_C·(1−clip(Φ)) = 0，
        dC 有限。同時驗證 dPhi < 0：logistic 連續流把越界 stage 拉回界內
        （自癒），kernel 夾的是讀數不是動力學。
        """
        fc = FastChannelConfig(phi0=0.65, phi_drive=1.0, phi_suppress=0.0,
                               kappa_phi=3.0, kappa_c=1.0)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.0, 0.0, 0.0]),
                                 growth_c=0.0, fast_channel=fc)
        phi_over = np.full_like(eng._Phi, 1.175)
        out = eng._morphology_core(
            0.0, eng._R, eng._C, eng._P_pool, eng._P_dist,
            phi_over, eng._K, eng._M, eng._Eta)
        d_c, d_phi = out[1], out[4]
        for comp in out:
            assert np.all(np.isfinite(comp))
        assert np.all(d_c == 0.0)            # 阻塞源在份額讀數下歸零，不是負爆
        assert np.all(d_phi < 0.0)           # 連續流自癒：Φ>1 時 dΦ<0 拉回界內

    def test_step_survives_strong_drive_and_phi_stays_in_bounds(self):
        """κ_Φ·drive·dt/2 > 1（3.0·1.0/2 > 1）：修前 step 0 即炸，修後存活。

        積分語義不變：步末 clip 仍把 Φ 駐留在 [0,1]，向 Φ*=1 單調收斂。
        """
        fc = FastChannelConfig(phi0=0.65, phi_drive=1.0, phi_suppress=0.0,
                               kappa_phi=3.0, kappa_c=1.0)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]),
                                 growth_c=0.0, fast_channel=fc)
        for _ in range(60):
            eng.step(1.0)
        assert np.all(np.isfinite(eng._R)) and np.all(np.isfinite(eng._C))
        assert np.all(eng._Phi >= 0.0) and np.all(eng._Phi <= 1.0)
        assert np.all(eng._Phi >= 0.65 - 1e-9)   # 單調逼近 Φ* = 1

    def test_substrate_c_source_fn_negative_still_raises_and_ctx_raw(self):
        """防護不松：substrate 明式供律回負值照樣爆炸；ctx["Phi"] 維持原始觀測量。"""
        fc = FastChannelConfig(phi0=0.65, phi_drive=1.0, phi_suppress=0.0,
                               kappa_phi=3.0, kappa_c=1.0)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]),
                                 growth_c=0.0, fast_channel=fc)
        seen_ctx_phi = {}

        def c_source_fn(t, ctx):
            seen_ctx_phi["Phi"] = np.asarray(ctx["Phi"]).copy()
            return np.full_like(eng._C, -0.5)

        eng._fc["c_source_fn"] = c_source_fn
        phi_over = np.full_like(eng._Phi, 1.175)
        with pytest.raises(ValueError, match="c_source"):
            eng._morphology_core(
                0.0, eng._R, eng._C, eng._P_pool, eng._P_dist,
                phi_over, eng._K, eng._M, eng._Eta)
        # ctx 未夾制：fn 看到的是原始（越界）stage Φ
        assert np.all(seen_ctx_phi["Phi"] == phi_over)
