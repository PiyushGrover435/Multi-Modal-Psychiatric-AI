"""
Tier 1.5 — Acoustic Biomarker Validator
========================================
Activated ONLY when Tier 1 outputs High Risk.
Performs real librosa-based feature extraction on the provided audio file path:
  - MFCCs (13 coefficients → mean)
  - Jitter  : frame-to-frame F0 pitch instability (%)
  - Shimmer : amplitude variation between consecutive voiced frames (dB)
  - Flat-affect score: low MFCC energy variance → flattened vocal expression
Returns real mathematical scores, not mock values.
"""

import os
import numpy as np

try:
    import librosa
    import librosa.effects
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ── Thresholds (Calibrated for Laptop Microphones) ───────────────────────────────
JITTER_THRESHOLD    = 3.5   # % — Raised from 1.0% due to consumer mic noise floor
SHIMMER_THRESHOLD   = 8.5  # dB — Raised from 0.35dB to account for laptop static
FLAT_AFFECT_THRESHOLD = 0.15  # MFCC variance ratio — below this is flat affect


def _compute_jitter(f0: np.ndarray) -> float:
    """
    Jitter = mean absolute difference of consecutive F0 periods / mean F0 period.
    Returns value in percent (%).
    """
    voiced = f0[f0 > 0]
    if len(voiced) < 2:
        return 0.0
    periods = 1.0 / voiced
    abs_diffs = np.abs(np.diff(periods))
    jitter = (np.mean(abs_diffs) / np.mean(periods)) * 100.0
    return float(jitter)


def _compute_shimmer(y: np.ndarray, sr: int, f0: np.ndarray) -> float:
    """
    Shimmer = mean absolute log amplitude difference between consecutive voiced frames.
    Approximated using RMS over short frames aligned with voiced F0 frames.
    Returns value in dB.
    """
    frame_length = 2048
    hop_length   = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms[rms > 1e-8]   # remove silence frames
    if len(rms) < 2:
        return 0.0
    log_rms = 20.0 * np.log10(rms)
    shimmer = float(np.mean(np.abs(np.diff(log_rms))))
    return shimmer


def _compute_flat_affect(mfccs: np.ndarray) -> tuple[bool, float]:
    """
    Flat affect is estimated via the variance ratio of the first MFCC coefficient
    (energy-related). Low temporal variance → monotone, flat vocal expression.
    Returns (detected: bool, score: float).
    """
    mfcc1_variance = float(np.var(mfccs[0]))
    # Normalize against an expected baseline variance (~50 for typical speech)
    normalized_variance = mfcc1_variance / 50.0
    flat = normalized_variance < FLAT_AFFECT_THRESHOLD
    return flat, round(normalized_variance, 4)


def tier15_validate_audio(audio_path: str | None) -> dict:
    """
    Real Acoustic Biomarker Extraction (Tier 1.5).
    Activated ONLY when Tier 1 outputs High Risk.

    Args:
        audio_path: Absolute or relative path to a .wav audio file.
                    Must not be None — callers must validate before calling.

    Returns:
        {
          'validation_passed': bool,          # True if anomalies detected
          'biomarker_scores': {
              'jitter':       str,             # e.g. "2.34%"
              'shimmer':      str,             # e.g. "0.48 dB"
              'flat_affect':  str,             # "Detected" | "Normal"
              'mfcc_variance_ratio': float,    # raw normalized variance
          },
          'error': str | None                 # set if extraction failed
        }
    """
    # ── Guard: librosa must be available ──────────────────────────────────────
    if not LIBROSA_AVAILABLE:
        return {
            "validation_passed": False,
            "biomarker_scores": {
                "jitter":       "N/A",
                "shimmer":      "N/A",
                "flat_affect":  "N/A",
                "mfcc_variance_ratio": 0.0,
            },
            "error": "librosa is not installed. Run: pip install librosa",
        }

    # ── Guard: valid file path must be provided ────────────────────────────────
    if audio_path is None or not os.path.exists(audio_path):
        return {
            "validation_passed": False,
            "biomarker_scores": {
                "jitter":       "N/A",
                "shimmer":      "N/A",
                "flat_affect":  "N/A",
                "mfcc_variance_ratio": 0.0,
            },
            "error": "No valid audio file provided. Please record or upload a .wav file.",
        }

    try:
        # ── 1. Load audio (mono, native sample rate) ─────────────────────────
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # ── 1.5. Apply Spectral Gating (Noise Reduction) ─────────────────────────
        try:
            import noisereduce as nr
            y = nr.reduce_noise(y=y, sr=sr)
        except ImportError:
            pass # Graceful fallback if not installed

        # ── 2. MFCC extraction (13 coefficients) ─────────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # shape: (13, T)

        # ── 3. F0 (pitch) extraction via pyin ────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # 65 Hz — below human fundamental
            fmax=librosa.note_to_hz('C7'),   # 2093 Hz — above human fundamental
            sr=sr,
        )
        f0 = np.nan_to_num(f0, nan=0.0)

        # ── 4. Compute biomarkers ─────────────────────────────────────────────
        jitter_val  = _compute_jitter(f0)
        shimmer_val = _compute_shimmer(y, sr, f0)
        flat_detected, mfcc_var_ratio = _compute_flat_affect(mfccs)

        # ── 5. Validate: anomaly if ANY biomarker exceeds clinical threshold ──
        jitter_anomaly  = jitter_val  > JITTER_THRESHOLD
        shimmer_anomaly = shimmer_val > SHIMMER_THRESHOLD
        validation_passed = flat_detected or jitter_anomaly or shimmer_anomaly

        return {
            "validation_passed": bool(validation_passed),
            "biomarker_scores": {
                "jitter":       f"{jitter_val:.2f}%",
                "shimmer":      f"{shimmer_val:.3f} dB",
                "flat_affect":  "Detected" if flat_detected else "Normal",
                "mfcc_variance_ratio": mfcc_var_ratio,
            },
            "error": None,
        }

    except Exception as exc:
        return {
            "validation_passed": False,
            "biomarker_scores": {
                "jitter":       "Error",
                "shimmer":      "Error",
                "flat_affect":  "Error",
                "mfcc_variance_ratio": 0.0,
            },
            "error": f"Librosa extraction failed: {exc}",
        }
