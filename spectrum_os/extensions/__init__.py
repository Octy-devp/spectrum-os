"""Spectrum OS — domain extensions layer.

Domain models *built on* the world-agnostic kernel. The kernel
(:mod:`spectrum_os.kernel`) knows only R/C/P_R/Φ/K and the substrate slots;
everything world- or domain-specific lives here, so the kernel stays lean and
portable (docs/PLAN.md: "世界無關").

Import rule: ``extensions`` may import ``kernel``; ``kernel`` must never import
``extensions``. Each module here receives its world data root from the caller —
a machine-specific default would silently bind the model to one worldline.
"""
