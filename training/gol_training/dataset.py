"""Dataset loader, numeric-only privacy validator, synthetic multi-player generator, and player-grouped splits."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .baseline import run_hmm_belief_filter_on_sequence
from .features import (
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_SAMPLE_RATE_HZ,
    TEMPORAL_SEQ_LEN,
    resample_raw_feature_entries,
)

FORBIDDEN_KEY_RE = re.compile(
    r"(image|imagedata|frameimage|video|videoblob|canvas|pixel|pixels|rgba|rgb|bgr|png|jpeg|jpg|webp|webm|mp4|blob|dataurl|base64|screenshot|mediastream|framebytes)",
    re.IGNORECASE,
)
FORBIDDEN_STRING_RE = re.compile(r"^(data:image/|data:video/|blob:)", re.IGNORECASE)


class PrivacySchemaError(ValueError):
    """Raised when an input log file contains forbidden media/image/video fields."""


def assert_numeric_only_payload(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise PrivacySchemaError(f"Non-finite numeric value at {path}")
        return
    if isinstance(value, str):
        if FORBIDDEN_STRING_RE.match(value.strip()):
            raise PrivacySchemaError(f"Forbidden media URI at {path}")
        if len(value) > 4096:
            raise PrivacySchemaError(f"Oversized string ({len(value)}) at {path}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            assert_numeric_only_payload(item, f"{path}[{idx}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if FORBIDDEN_KEY_RE.search(str(key)):
                raise PrivacySchemaError(f"Forbidden media key '{key}' at {path}.{key}")
            assert_numeric_only_payload(child, f"{path}.{key}")


@dataclass(frozen=True)
class RoundSample:
    round_id: str
    player_id: str
    initial_side: str
    label_left: int
    transfer_count: int
    sequence: np.ndarray  # shape (TEMPORAL_SEQ_LEN, TEMPORAL_FEATURE_DIM), float32
    hmm_p_left: float


def _count_transfers_from_timeline(events: Sequence[dict[str, Any]]) -> int:
    # Estimate transfer count from contact episodes when explicit count isn't in logs
    contacts = sum(1 for ev in events if ev.get("type") == "CONTACT_START")
    partials = sum(1 for ev in events if ev.get("type") == "PARTIAL_OPENING" and ev.get("duringContact"))
    if partials > 0:
        return partials
    return contacts


def load_exported_logs(json_path: str | Path) -> list[RoundSample]:
    """Loads exported JSON round logs produced by `RoundLogStore.exportLogsJson`."""
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert_numeric_only_payload(raw, "export")
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1 or not isinstance(raw.get("rounds"), list):
        raise ValueError("Expected schemaVersion=1 and rounds list in exported JSON")

    samples: list[RoundSample] = []
    for r in raw["rounds"]:
        round_id = str(r["id"])
        player_id = str(r.get("profileId", "default-player"))
        initial_side = "RIGHT" if r.get("initialShownHand") == "RIGHT" else "LEFT"
        truth = "RIGHT" if r.get("verifiedGroundTruth") == "RIGHT" else "LEFT"
        label_left = 1 if truth == "LEFT" else 0

        raw_entries_json = r.get("rawFrameFeatures") or []
        raw_entries: list[tuple[float, list[float]]] = [
            (float(item["tMs"]), [float(x) for x in item["v"]])
            for item in raw_entries_json
        ]

        if "resampledSequence" in r and isinstance(r["resampledSequence"], list):
            seq = np.asarray(r["resampledSequence"], dtype=np.float32)
            if seq.shape != (TEMPORAL_SEQ_LEN, TEMPORAL_FEATURE_DIM):
                seq = resample_raw_feature_entries(raw_entries)
        else:
            seq = resample_raw_feature_entries(raw_entries)

        final_guess = r.get("finalGuess") or {}
        if raw_entries:
            hmm_p_left = run_hmm_belief_filter_on_sequence(raw_entries, initial_side)
        else:
            hmm_p_left = float(final_guess.get("hmmPLeft", 0.5))

        events = r.get("eventTimeline") or []
        transfer_count = _count_transfers_from_timeline(events)

        samples.append(
            RoundSample(
                round_id=round_id,
                player_id=player_id,
                initial_side=initial_side,
                label_left=label_left,
                transfer_count=transfer_count,
                sequence=seq,
                hmm_p_left=hmm_p_left,
            )
        )
    return samples


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def generate_synthetic_dataset(
    seed: int = 42,
    num_players: int = 12,
    rounds_per_player: int = 48,
) -> list[RoundSample]:
    """
    Generates a deterministic multi-player synthetic dataset of hiding-phase trajectories
    spanning 0, 1, and 2+ transfers, fake contacts, stealthy handovers, and occasional occlusions.
    All features are causally derived from simulated hand kinematics with zero label leakage.
    """
    rng = np.random.default_rng(seed)
    samples: list[RoundSample] = []

    for p_idx in range(num_players):
        player_id = f"player-{p_idx + 1:02d}"
        # Player-specific kinematic signature
        base_curl = float(rng.uniform(0.72, 0.88))
        base_comp = float(rng.uniform(0.64, 0.80))
        gol_squeeze_offset = float(rng.uniform(0.01, 0.055))
        handover_dip_mag = float(rng.uniform(0.05, 0.28))
        fake_decoy_twitch = float(rng.uniform(0.03, 0.085))
        base_speed = float(rng.uniform(0.08, 0.26))
        player_hand_bias = float(rng.normal(0.0, 0.035))  # camera angle / dominant hand curl asymmetry

        for r_idx in range(rounds_per_player):
            transfers = r_idx % 3  # balanced 0, 1, 2 transfers
            if transfers == 2 and (r_idx // 3) % 4 == 3:
                transfers = 3  # include 3-transfer cases in 2+ bucket
            initial_side = "LEFT" if (p_idx + r_idx) % 2 == 0 else "RIGHT"
            fake_contacts = int(rng.integers(0, 3)) if transfers < 3 else int(rng.integers(0, 2))
            has_occlusion = bool(rng.random() < 0.22)
            has_sighting = bool(rng.random() < 0.07)
            round_l_bias = player_hand_bias + float(rng.normal(0.0, 0.025))
            round_r_bias = -player_hand_bias + float(rng.normal(0.0, 0.025))

            total_frames = int(rng.integers(60, 155))  # ~2.0s - 5.2s at 30fps
            dt_ms = 33.333

            # Build non-overlapping contact episodes inside [12, total_frames - 14]
            n_episodes = transfers + fake_contacts
            episode_types: list[bool] = [True] * transfers + [False] * fake_contacts
            rng.shuffle(episode_types)

            episodes: list[dict[str, Any]] = []
            if n_episodes > 0:
                usable_start = 12
                usable_end = total_frames - 14
                slot = max(14, (usable_end - usable_start) // n_episodes)
                holder_sim = initial_side
                for ep_i, is_real in enumerate(episode_types):
                    ep_start = usable_start + ep_i * slot + int(rng.integers(0, max(1, slot // 5)))
                    dur_frames = int(rng.integers(12, 22)) if is_real else int(rng.integers(14, 28))
                    ep_end = min(total_frames - 6, ep_start + dur_frames)
                    if ep_end - ep_start < 8:
                        continue
                    mid = (ep_start + ep_end) // 2
                    episodes.append(
                        {
                            "start": ep_start,
                            "end": ep_end,
                            "mid": mid,
                            "is_real": is_real,
                            "giver": holder_sim,
                        }
                    )
                    if is_real:
                        holder_sim = "RIGHT" if holder_sim == "LEFT" else "LEFT"

            # Actual ground truth determined purely by real transfers
            real_transfers = sum(1 for ep in episodes if ep["is_real"])
            final_side = initial_side
            for _ in range(real_transfers):
                final_side = "RIGHT" if final_side == "LEFT" else "LEFT"

            occ_start = int(rng.integers(20, max(21, total_frames - 18))) if has_occlusion else -1
            occ_end = occ_start + int(rng.integers(2, 6)) if has_occlusion else -1
            sighting_frame = int(rng.integers(total_frames - 16, total_frames - 4)) if has_sighting else -1

            raw_entries: list[tuple[float, list[float]]] = []
            left_comp_hist: list[float] = []
            right_comp_hist: list[float] = []
            cur_holder = initial_side
            in_contact_state = False
            contact_start_ms = 0.0

            for f in range(total_frames):
                ts_ms = f * dt_ms
                active_ep = None
                for ep in episodes:
                    if ep["start"] <= f <= ep["end"]:
                        active_ep = ep
                        break
                    if ep["is_real"] and f == ep["end"] + 1:
                        cur_holder = "RIGHT" if ep["giver"] == "LEFT" else "LEFT"

                # Also update cur_holder right after midpoint during real transfer
                if active_ep and active_ep["is_real"] and f > active_ep["mid"] + 1:
                    cur_holder = "RIGHT" if active_ep["giver"] == "LEFT" else "LEFT"

                # Inter-hand distance
                if active_ep:
                    ramp_in = min(1.0, (f - active_ep["start"] + 1) / 4.0)
                    ramp_out = min(1.0, (active_ep["end"] - f + 1) / 4.0)
                    u = min(ramp_in, ramp_out)
                    approach = u * u * (3.0 - 2.0 * u)
                    dist_norm = max(0.14, 1.45 - 1.25 * approach + float(rng.normal(0.0, 0.025)))
                else:
                    dist_norm = max(0.65, 1.45 + float(rng.normal(0.0, 0.06)))

                # Contact hysteresis matching FeatureExtractor
                if not in_contact_state and dist_norm <= 0.42:
                    in_contact_state = True
                    contact_start_ms = ts_ms
                elif in_contact_state and dist_norm > 0.52:
                    in_contact_state = False

                contact_dur_ms = max(0.0, ts_ms - contact_start_ms) if in_contact_state else 0.0

                # Kinematics for LEFT and RIGHT (with realistic camera-angle bias & noise)
                l_curl = base_curl + round_l_bias + float(rng.normal(0.0, 0.018))
                r_curl = base_curl + round_r_bias + float(rng.normal(0.0, 0.018))
                l_comp = base_comp + round_l_bias + (gol_squeeze_offset if cur_holder == "LEFT" else 0.0) + float(rng.normal(0.0, 0.02))
                r_comp = base_comp + round_r_bias + (gol_squeeze_offset if cur_holder == "RIGHT" else 0.0) + float(rng.normal(0.0, 0.02))

                l_partial = 0.0
                r_partial = 0.0
                l_speed = base_speed + float(rng.uniform(0.0, 0.09))
                r_speed = base_speed + float(rng.uniform(0.0, 0.09))
                l_jitter = float(rng.uniform(0.04, 0.14)) + (0.045 if cur_holder == "LEFT" else 0.0)
                r_jitter = float(rng.uniform(0.04, 0.14)) + (0.045 if cur_holder == "RIGHT" else 0.0)

                if active_ep:
                    l_speed += 0.35
                    r_speed += 0.35
                    dist_to_mid = abs(f - active_ep["mid"])
                    if active_ep["is_real"]:
                        # Real transfer: giver relaxes & dips curl around midpoint, receiver slows & tightens after
                        if dist_to_mid <= 2:
                            dip = handover_dip_mag * (1.0 - 0.22 * dist_to_mid)
                            if active_ep["giver"] == "LEFT":
                                l_curl -= dip
                                l_comp -= dip * 0.85
                                l_speed *= 0.72
                                if dip >= 0.088 and f == active_ep["mid"] + 1:
                                    l_partial = _clamp(dip / 0.16, 0.35, 1.0)
                            else:
                                r_curl -= dip
                                r_comp -= dip * 0.85
                                r_speed *= 0.72
                                if dip >= 0.088 and f == active_ep["mid"] + 1:
                                    r_partial = _clamp(dip / 0.16, 0.35, 1.0)
                        elif f > active_ep["mid"] + 1:
                            receiver = "RIGHT" if active_ep["giver"] == "LEFT" else "LEFT"
                            if receiver == "LEFT":
                                l_comp += 0.055
                                l_curl += 0.035
                                l_jitter += 0.03
                            else:
                                r_comp += 0.055
                                r_curl += 0.035
                                r_jitter += 0.03
                    else:
                        # Fake contact: symmetrical tension or decoy twitch that can fool thresholded HMM
                        if dist_to_mid <= 2:
                            if cur_holder == "LEFT":
                                # Decoy hand opens slightly or holder feints without transferring
                                r_curl -= fake_decoy_twitch * 0.85
                                l_curl -= fake_decoy_twitch * 0.35
                                r_comp += 0.03
                                if fake_decoy_twitch > 0.055 and f == active_ep["mid"] + 1 and rng.random() < 0.5:
                                    l_partial = 0.42  # false-alarm feint dip on holder!
                            else:
                                l_curl -= fake_decoy_twitch * 0.85
                                r_curl -= fake_decoy_twitch * 0.35
                                l_comp += 0.03
                                if fake_decoy_twitch > 0.055 and f == active_ep["mid"] + 1 and rng.random() < 0.5:
                                    r_partial = 0.42  # false-alarm feint dip on holder!

                l_curl = _clamp(l_curl, 0.0, 1.0)
                r_curl = _clamp(r_curl, 0.0, 1.0)
                l_comp = _clamp(l_comp, 0.0, 1.0)
                r_comp = _clamp(r_comp, 0.0, 1.0)

                l_prev_mean = float(np.mean(left_comp_hist[-8:])) if left_comp_hist else l_comp
                r_prev_mean = float(np.mean(right_comp_hist[-8:])) if right_comp_hist else r_comp
                l_delta = _clamp(l_comp - l_prev_mean, -1.0, 1.0)
                r_delta = _clamp(r_comp - r_prev_mean, -1.0, 1.0)
                left_comp_hist.append(l_comp)
                right_comp_hist.append(r_comp)

                l_tip_norm = _clamp(1.25 - 0.85 * l_comp, 0.1, 3.0) / 3.0
                r_tip_norm = _clamp(1.25 - 0.85 * r_comp, 0.1, 3.0) / 3.0

                is_occ = occ_start <= f <= occ_end
                l_det = 0.0 if (is_occ and f % 2 == 0) else 1.0
                r_det = 0.0 if (is_occ and f % 2 == 1) else 1.0
                low_q = 1.0 if is_occ else 0.0

                l_sight = 0.82 if (f == sighting_frame and final_side == "LEFT") else 0.0
                r_sight = 0.82 if (f == sighting_frame and final_side == "RIGHT") else 0.0

                vec = [
                    1.0 if initial_side == "LEFT" else -1.0,
                    l_det,
                    r_det,
                    l_curl,
                    r_curl,
                    l_comp,
                    r_comp,
                    l_delta,
                    r_delta,
                    l_tip_norm,
                    r_tip_norm,
                    _clamp(l_speed, 0.0, 10.0) / 10.0,
                    _clamp(r_speed, 0.0, 10.0) / 10.0,
                    _clamp(l_jitter, 0.0, 5.0) / 5.0,
                    _clamp(r_jitter, 0.0, 5.0) / 5.0,
                    l_partial,
                    r_partial,
                    l_sight,
                    r_sight,
                    1.0 if in_contact_state else 0.0,
                    _clamp(dist_norm, 0.0, 3.0) / 3.0,
                    _clamp(contact_dur_ms, 0.0, 5000.0) / 5000.0,
                    low_q,
                    float(f) / max(1.0, float(total_frames - 1)),
                ]
                raw_entries.append((ts_ms, vec))

            seq = resample_raw_feature_entries(
                raw_entries,
                seq_len=TEMPORAL_SEQ_LEN,
                sample_rate_hz=TEMPORAL_SAMPLE_RATE_HZ,
            )
            hmm_p_left = run_hmm_belief_filter_on_sequence(raw_entries, initial_side)

            samples.append(
                RoundSample(
                    round_id=f"{player_id}-r{r_idx + 1:03d}",
                    player_id=player_id,
                    initial_side=initial_side,
                    label_left=1 if final_side == "LEFT" else 0,
                    transfer_count=real_transfers,
                    sequence=seq,
                    hmm_p_left=hmm_p_left,
                )
            )

    return samples


def split_by_player(
    samples: Sequence[RoundSample],
    train_ratio: float = 0.58,
    val_ratio: float = 0.17,
    seed: int = 42,
) -> tuple[list[RoundSample], list[RoundSample], list[RoundSample]]:
    """
    Splits dataset strictly by `player_id` into `(train, val, test)` with zero player leakage.
    """
    players = sorted({s.player_id for s in samples})
    if len(players) < 3:
        raise ValueError(f"Need at least 3 distinct players for train/val/test split, got {len(players)}")

    rng = np.random.default_rng(seed)
    shuffled_players = list(players)
    rng.shuffle(shuffled_players)

    n_players = len(shuffled_players)
    n_train = max(1, int(round(n_players * train_ratio)))
    n_val = max(1, int(round(n_players * val_ratio)))
    if n_train + n_val >= n_players:
        n_train = max(1, n_players - 2)
        n_val = 1

    train_players = set(shuffled_players[:n_train])
    val_players = set(shuffled_players[n_train : n_train + n_val])
    test_players = set(shuffled_players[n_train + n_val :])

    if not train_players.isdisjoint(val_players) or not train_players.isdisjoint(test_players) or not val_players.isdisjoint(test_players):
        raise RuntimeError("Player split leakage detected")

    train = [s for s in samples if s.player_id in train_players]
    val = [s for s in samples if s.player_id in val_players]
    test = [s for s in samples if s.player_id in test_players]
    return train, val, test
