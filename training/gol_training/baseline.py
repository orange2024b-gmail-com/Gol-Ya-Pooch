"""Python port of the Phase 2 HMM BeliefFilter baseline for offline comparison."""

from __future__ import annotations

import math
from typing import Sequence


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sigmoid(x: float) -> float:
    if x >= 20.0:
        return 1.0 - 1e-9
    if x <= -20.0:
        return 1e-9
    return 1.0 / (1.0 + math.exp(-x))


DEFAULT_HMM_PARAMS = {
    "init_certainty": 0.99,
    "apart_transfer_rate_per_sec": 0.002,
    "contact_base_rate_per_sec": 0.14,
    "contact_duration_slope_per_sec": 0.45,
    "max_contact_diffusion_per_step": 0.22,
    "partial_opening_handover_prob": 0.66,
    "sighting_log_odds_weight": 4.5,
    "compactness_log_odds_weight": 1.8,
    "curl_log_odds_weight": 1.1,
    "compactness_delta_log_odds_weight": 1.8,
    "pre_contact_weak_cue_attenuation": 0.05,
    "min_likelihood": 0.05,
}


def run_hmm_belief_filter_on_sequence(
    raw_entries: Sequence[tuple[float, Sequence[float]]],
    initial_side: str,
) -> float:
    """
    Runs the exact 2-state Bayesian HMM forward filter over raw `(t_ms, vec[24])` frames
    and returns final `P(gol in LEFT)`.
    """
    cert = _clamp(DEFAULT_HMM_PARAMS["init_certainty"], 0.5, 0.999)
    p_left = cert if initial_side == "LEFT" else 1.0 - cert
    p_right = 1.0 - p_left
    last_ts: float | None = None
    has_had_contact = False

    for ts_ms, vec in raw_entries:
        dt_sec = (
            _clamp((ts_ms - last_ts) / 1000.0, 1.0 / 120.0, 0.35)
            if last_ts is not None
            else 1.0 / 30.0
        )
        last_ts = ts_ms

        left_det = vec[1] > 0.5
        right_det = vec[2] > 0.5
        left_curl = vec[3]
        right_curl = vec[4]
        left_comp = vec[5]
        right_comp = vec[6]
        left_comp_delta = vec[7]
        right_comp_delta = vec[8]
        left_partial = vec[15]
        right_partial = vec[16]
        left_sighting = vec[17]
        right_sighting = vec[18]
        in_contact = vec[19] > 0.5
        contact_dur_sec = vec[21] * 5.0
        low_quality = vec[22] > 0.5

        if in_contact:
            has_had_contact = True

        # 1. Predict step
        if not in_contact:
            p_apart = _clamp(
                DEFAULT_HMM_PARAMS["apart_transfer_rate_per_sec"] * dt_sec, 0.0, 0.02
            )
            t_l_to_r = p_apart
            t_r_to_l = p_apart
        else:
            rate = DEFAULT_HMM_PARAMS["contact_base_rate_per_sec"] + (
                DEFAULT_HMM_PARAMS["contact_duration_slope_per_sec"]
                * min(2.5, contact_dur_sec)
            )
            base_diff = _clamp(
                rate * dt_sec, 0.0, DEFAULT_HMM_PARAMS["max_contact_diffusion_per_step"]
            )
            t_l_to_r = base_diff
            t_r_to_l = base_diff
            if left_partial > 0.0:
                boost = DEFAULT_HMM_PARAMS["partial_opening_handover_prob"] * max(
                    0.35, left_partial
                )
                t_l_to_r = _clamp(t_l_to_r + boost, 0.0, 0.72)
            if right_partial > 0.0:
                boost = DEFAULT_HMM_PARAMS["partial_opening_handover_prob"] * max(
                    0.35, right_partial
                )
                t_r_to_l = _clamp(t_r_to_l + boost, 0.0, 0.72)

        pred_left = p_left * (1.0 - t_l_to_r) + p_right * t_r_to_l
        pred_right = p_right * (1.0 - t_r_to_l) + p_left * t_l_to_r

        # 2. Observation step
        sighting_diff = left_sighting - right_sighting
        has_sighting = abs(sighting_diff) >= 0.2
        if low_quality and not has_sighting:
            lik_left = 0.5
            lik_right = 0.5
        else:
            log_odds = DEFAULT_HMM_PARAMS["sighting_log_odds_weight"] * sighting_diff
            if not low_quality and left_det and right_det:
                weak_scale = (
                    1.0
                    if has_had_contact
                    else DEFAULT_HMM_PARAMS["pre_contact_weak_cue_attenuation"]
                )
                comp_diff = left_comp - right_comp
                curl_diff = left_curl - right_curl
                delta_diff = left_comp_delta - right_comp_delta
                log_odds += weak_scale * (
                    DEFAULT_HMM_PARAMS["compactness_log_odds_weight"] * comp_diff
                    + DEFAULT_HMM_PARAMS["curl_log_odds_weight"] * curl_diff
                    + DEFAULT_HMM_PARAMS["compactness_delta_log_odds_weight"] * delta_diff
                )
            lik_left = _clamp(
                _sigmoid(log_odds),
                DEFAULT_HMM_PARAMS["min_likelihood"],
                1.0 - DEFAULT_HMM_PARAMS["min_likelihood"],
            )
            lik_right = 1.0 - lik_left

        unnorm_left = pred_left * lik_left
        unnorm_right = pred_right * lik_right
        total = unnorm_left + unnorm_right
        if total > 1e-12:
            p_left = _clamp(unnorm_left / total, 0.001, 0.999)
            p_right = 1.0 - p_left

    return p_left
