"""Tests for player-grouped splitting and privacy-safe exported log loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gol_training.dataset import (
    PrivacySchemaError,
    generate_synthetic_dataset,
    load_exported_logs,
    split_by_player,
)
from gol_training.features import TEMPORAL_FEATURE_DIM, TEMPORAL_SEQ_LEN


def test_player_grouped_split_has_zero_leakage() -> None:
    samples = generate_synthetic_dataset(seed=7, num_players=10, rounds_per_player=12)
    train, val, test = split_by_player(samples, seed=7)

    train_players = {s.player_id for s in train}
    val_players = {s.player_id for s in val}
    test_players = {s.player_id for s in test}

    assert len(train_players) >= 1
    assert len(val_players) >= 1
    assert len(test_players) >= 1
    assert train_players.isdisjoint(val_players)
    assert train_players.isdisjoint(test_players)
    assert val_players.isdisjoint(test_players)
    assert len(train) + len(val) + len(test) == len(samples)


def test_load_exported_logs_rejects_image_or_video_fields(tmp_path: Path) -> None:
    bad_payload = {
        "schemaVersion": 1,
        "rounds": [
            {
                "id": "r1",
                "profileId": "p1",
                "initialShownHand": "LEFT",
                "verifiedGroundTruth": "LEFT",
                "frameImage": "data:image/png;base64,AAAA",
            }
        ],
    }
    bad_path = tmp_path / "bad_export.json"
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")

    with pytest.raises(PrivacySchemaError):
        load_exported_logs(bad_path)


def test_load_exported_logs_parses_valid_numeric_export(tmp_path: Path) -> None:
    vec_left = [1.0, 1.0, 1.0, 0.85, 0.8, 0.78, 0.7, 0.01, 0.0, 0.2, 0.25, 0.05, 0.05, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    assert len(vec_left) == TEMPORAL_FEATURE_DIM

    good_payload = {
        "schemaVersion": 1,
        "exportedAtIso": "2026-11-13T16:00:00Z",
        "profileId": "player-export-1",
        "consentEnabled": True,
        "rounds": [
            {
                "schemaVersion": 1,
                "id": "rl-1",
                "profileId": "player-export-1",
                "recordedAtMs": 1700000000000,
                "levelId": 1,
                "ruleSetId": "classic-1v1",
                "initialShownHand": "LEFT",
                "resampledSequence": [[0.0] * TEMPORAL_FEATURE_DIM for _ in range(TEMPORAL_SEQ_LEN)],
                "rawFrameFeatures": [
                    {"tMs": 0.0, "v": vec_left},
                    {"tMs": 100.0, "v": vec_left},
                ],
                "eventTimeline": [],
                "beliefTrace": [],
                "finalGuess": {"guess": "LEFT", "probability": 0.9, "pLeft": 0.9, "hmmPLeft": 0.9, "temporalPLeft": None, "behaviorPLeft": 0.5, "dominantSource": "TRACKING"},
                "verifiedGroundTruth": "LEFT",
                "timings": {"hidingDurationMs": 100, "frameCount": 2, "sampleRateHz": 10, "temporalInferenceMs": None, "totalProcessingMs": 2},
            }
        ],
    }
    good_path = tmp_path / "good_export.json"
    good_path.write_text(json.dumps(good_payload), encoding="utf-8")

    loaded = load_exported_logs(good_path)
    assert len(loaded) == 1
    assert loaded[0].player_id == "player-export-1"
    assert loaded[0].label_left == 1
    assert loaded[0].sequence.shape == (TEMPORAL_SEQ_LEN, TEMPORAL_FEATURE_DIM)
