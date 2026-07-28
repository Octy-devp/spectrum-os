"""extras/ — α₁ analysis tools (PLAN-23 §四-D).

These are **not** β₁-native tools. The β₁ kernel (`kernel/wave.py`) reasons in
moving averages and autocorrelation; this package gives *us* (α₁ analysts)
Fourier and PCA views of the same data. See PLAN-23 §一 kernel/ vs extras/.
"""

from .fft_spectrum import fft_decompose
from .pca_decomp import pca_decompose

__all__ = ["fft_decompose", "pca_decompose"]
