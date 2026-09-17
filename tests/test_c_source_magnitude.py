"""回歸：C 源律把 C 讀作幅度量（≥0）——RK4 中間步負下潛不得炸掉健康積分。

事故（2026-09-17，fast5 車道首飛即撞）：
``ECC_FAST_CHANNEL`` + ``ECC_PHI_FEEDBACK`` 場景下，α_rc sink 較強時 RK4 中間
步的 C 暫時 < 0，預設源律 ``c_eff·C`` 隨之為負，被 ``_driver_value`` 防護攔下
（ValueError: substrate driver c_source returned a negative value）。

連續動力學本身自癒：C < 0 時 ``α_rc·R·C`` 是恢復流（dC > 0），步末
``clamp_nonneg`` 再兜底。夾制的是「制度再生產」這個幅度量的定義域
（與 s()/tension()/panic weight 的 magnitude 讀法同一家法），不是動力學；
substrate 供應的 ``c_source_fn`` 若回負值仍照樣大聲爆炸——防護不松。
"""
import numpy as np
import pytest

from spectrum_os.kernel.forces import FastChannelConfig, ForceFieldDynamics


def _engine_5d(**kw) -> ForceFieldDynamics:
    R0 = np.array([1.5, 1.2, 0.9])
    C0 = np.array([0.4, 0.1, 0.3])
    fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1)
    return ForceFieldDynamics(R0, C0, fast_channel=fc, **kw)


class TestCSourceMagnitude:
    def test_negative_intermediate_c_does_not_raise(self):
        """直接喂負 C 進 8 分量 RHS——修前在此 ValueError，修後必須有限。"""
        eng = _engine_5d()
        neg_c = np.full_like(eng._C, -0.1)
        out = eng._morphology_core(
            0.0, eng._R, neg_c, eng._P_pool, eng._P_dist,
            eng._Phi, eng._K, eng._M, eng._Eta)
        d_c = out[1]
        assert np.all(np.isfinite(d_c))
        # 恢復流：負 C 在 α_rc sink 下 dC > 0（向零回升），不是發散
        assert np.all(d_c[np.asarray(neg_c) < 0] > 0)

    def test_step_survives_strong_sink_and_stays_nonnegative(self):
        fc = FastChannelConfig(phi_drive=0.65, phi_suppress=0.1, alpha_rc=5.0)
        eng = ForceFieldDynamics(np.array([1.5, 1.2, 0.9]),
                                 np.array([0.4, 0.1, 0.3]), fast_channel=fc)
        for _ in range(60):
            eng.step(1.0 / 12)
        assert np.all(np.isfinite(eng._R)) and np.all(np.isfinite(eng._C))
        assert np.all(eng._C >= 0.0)

    def test_substrate_c_source_fn_negative_still_raises(self):
        """防護不松：substrate 明式供律回負值，照樣大聲爆炸。"""
        eng = _engine_5d()
        eng._fc["c_source_fn"] = lambda t, ctx: np.full_like(eng._C, -0.5)
        with pytest.raises(ValueError, match="c_source"):
            eng._morphology_core(
                0.0, eng._R, eng._C, eng._P_pool, eng._P_dist,
                eng._Phi, eng._K, eng._M, eng._Eta)
