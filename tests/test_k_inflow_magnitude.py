"""回歸：k_inflow 把 P_pool 讀作幅度量（≥0）——RK4 中間步負下潛不得炸掉健康積分。

事故（2026-09-19，政策×fast5 車道）：
``ECC_FAST_CHANNEL=measured``＋``SB_DYNKC=1`` 同開的 policy×fast5 run 中，
P_pool 抽取較強（λ+η·μ_E+ρ > 2/dt）時 RK4 中間步的 P_pool 暫時 < 0，預設
流入律 ``η·μ_E·P_pool`` 讀原始 stage 陣列隨之為負，被 ``_driver_value`` 防護
攔下（ValueError: substrate driver k_inflow returned a negative value）。

連續動力學本身自癒：P < 0 時 −λ·P 與 −ρ·P 是恢復流（dP > 0），步末
``clamp_nonneg`` 再兜底。夾制的是「異化流入」這個幅度量的定義域
（b55d6ba C 源律夾制的同一家法；s()/tension()/panic weight 的 magnitude
讀法），不是動力學。分界契約：kernel 只夾制**自己構造的預設律**的讀數；
substrate 的 ``k_inflow_fn`` 仍收到原始 ``ctx["P"]``（世界有權看到真實
狀態），其返回值由 ``_driver_value`` 防護把關——明式負律照樣大聲爆炸，
防護不松。若世界的流入律自身對 stage 下潛不健壯，該世界應在自己的
substrate 層夾制自己的讀數（同家法），kernel 不代勞。
"""
import numpy as np
import pytest

from spectrum_os.kernel.forces import FastChannelConfig, ForceFieldDynamics


def _engine_5d(**kw) -> ForceFieldDynamics:
    R0 = np.array([1.5, 1.2, 0.9])
    C0 = np.array([0.4, 0.1, 0.3])
    fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1)
    return ForceFieldDynamics(R0, C0, fast_channel=fc, **kw)


class TestKInflowMagnitude:
    def test_negative_intermediate_p_does_not_raise(self):
        """直接喂負 P 進 8 分量 RHS——修前在此 ValueError，修後必須有限。"""
        fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1,
                               eta=1.0, mu_e=3.0, lam=0.5)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]), fast_channel=fc)
        neg_p = np.full_like(eng._P_pool, -0.1)
        out = eng._morphology_core(
            0.0, eng._R, eng._C, neg_p, eng._P_dist,
            eng._Phi, eng._K, eng._M, eng._Eta)
        d_p = out[2]
        for comp in out:
            assert np.all(np.isfinite(comp))
        # 恢復流：負 P 在 λ drain 下 dP > 0（向零回升），不是發散
        assert np.all(d_p[np.asarray(neg_p) < 0] > 0)

    def test_step_survives_strong_extraction_and_stays_nonnegative(self):
        """強抽取整支積分存活——dt=1.0（a·dt/2=1.75>1，修前 step 0 即炸）。"""
        fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1,
                               eta=1.0, mu_e=3.0, lam=0.5)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]),
                                 fast_channel=fc, latent_force=1.0)
        for _ in range(60):
            eng.step(1.0)
        assert np.all(np.isfinite(eng._P_pool)) and np.all(np.isfinite(eng._K))
        assert np.all(eng._P_pool >= 0.0)
        assert np.all(eng._K >= 0.0)

    def test_substrate_k_inflow_fn_negative_still_raises(self):
        """防護不松：substrate 明式供律回負值，照樣大聲爆炸。

        分界契約（2026-09-19 裁定）：ctx["P"] 維持原始觀測量——本測試同時
        記錄 fn 實收的 ctx["P"]，assert kernel 未夾制 ctx（substrate 有權
        看到真實狀態，含 RK4 中間步的暫時負值）。
        """
        eng = _engine_5d()
        seen_ctx_p = {}

        def k_inflow_fn(t, ctx):
            seen_ctx_p["P"] = np.asarray(ctx["P"]).copy()
            return np.full_like(eng._P_pool, -0.5)

        eng._fc["k_inflow_fn"] = k_inflow_fn
        neg_p = np.full_like(eng._P_pool, -0.1)
        with pytest.raises(ValueError, match="k_inflow"):
            eng._morphology_core(
                0.0, eng._R, eng._C, neg_p, eng._P_dist,
                eng._Phi, eng._K, eng._M, eng._Eta)
        # ctx 未夾制：fn 看到的是原始（負）stage P
        assert np.all(seen_ctx_p["P"] == neg_p)
