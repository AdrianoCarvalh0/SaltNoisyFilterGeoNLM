"""Public experiment-specific entry points for near-extreme impulse noise."""

import numpy as np

from .anlm_functions import run_anlm_pipeline
from .geonlm_medians_functions import run_geonlm_medians_pipeline
from .salt_filters import aswmf_filter


def _validate_tolerance(impulse_tolerance):
    if isinstance(impulse_tolerance, bool) or not isinstance(
        impulse_tolerance, (int, np.integer)
    ):
        raise TypeError("impulse_tolerance must be an integer.")
    tolerance = int(impulse_tolerance)
    if not 0 <= tolerance <= 127:
        raise ValueError("impulse_tolerance must be between 0 and 127.")
    return tolerance


def aswmf_impulse_tolerance_filter(image, impulse_tolerance=4, **kwargs):
    """Run the ASWMF variant used by the tolerance experiment."""
    tolerance = _validate_tolerance(impulse_tolerance)
    return aswmf_filter(
        np.asarray(image, dtype=np.float32),
        impulse_tolerance=tolerance,
        **kwargs,
    )


def run_ianlm_impulse_tolerance_pipeline(
    *, img_original, h_base, img_noisy, f, t, mult, impulse_tolerance=4, **kwargs
):
    """Run IANLM with near-extreme switching and candidate rejection."""
    tolerance = _validate_tolerance(impulse_tolerance)
    return run_anlm_pipeline(
        img_original=img_original,
        h_base=h_base,
        img_noisy=img_noisy,
        f=f,
        t=t,
        mult=mult,
        impulse_tolerance=tolerance,
        **kwargs,
    )


def run_ghnlm_impulse_tolerance_pipeline(
    *, img_original, h_base, img_noisy, f, t, mult, impulse_tolerance=4, **kwargs
):
    """Run GHNLM with the same near-extreme impulse definition as IANLM."""
    tolerance = _validate_tolerance(impulse_tolerance)
    return run_geonlm_medians_pipeline(
        img_original=img_original,
        h_base=h_base,
        img_noisy=img_noisy,
        f=f,
        t=t,
        mult=mult,
        impulse_tolerance=tolerance,
        **kwargs,
    )
