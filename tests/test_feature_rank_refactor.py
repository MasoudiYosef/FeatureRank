from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.autoencoder_feature_selection import (
    save_sample_weighted_contributions,
    save_top_percent_features_by_abs_max_weight,
)
from src.config import ExperimentConfig


class _FakeLayer:
    def __init__(self, weights: np.ndarray) -> None:
        self._weights = weights

    def get_weights(self) -> list[np.ndarray]:
        return [self._weights]


class _FakeAutoencoder:
    def __init__(self, weights: np.ndarray) -> None:
        self._layer = _FakeLayer(weights)

    def get_layer(self, name: str) -> _FakeLayer:
        assert name == "enc_dense_1"
        return self._layer


class FeatureRankRefactorTests(unittest.TestCase):
    def test_experiment_config_stays_compact(self) -> None:
        self.assertEqual(
            [field.name for field in fields(ExperimentConfig)],
            [
                "dataset_name",
                "task",
                "feature_percent",
                "random_state",
                "encoding_dim",
                "target_column",
                "id_column",
                "cluster_k",
                "save_details",
            ],
        )

    def test_sample_weighted_contribution_and_ranking_formula(self) -> None:
        X_train = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        weights = np.array([[2.0, -1.0], [0.5, -3.0]], dtype=np.float32)
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            contribution_path = tmp_path / "first_layer_W_list.csv"
            contribution_df = save_sample_weighted_contributions(
                _FakeAutoencoder(weights),
                X_train,
                ["a", "b"],
                contribution_path,
            )

            expected = np.mean(
                np.abs(X_train[:, :, np.newaxis] * weights[np.newaxis, :, :]), axis=0
            )
            np.testing.assert_allclose(
                np.asarray(contribution_df["weight_list"].tolist()), expected
            )

            selected = save_top_percent_features_by_abs_max_weight(
                contribution_path,
                ["a", "b"],
                50.0,
                tmp_path / "selected_features.csv",
            )
            self.assertEqual(selected["feature_name"].tolist(), ["b"])
            self.assertEqual(
                pd.read_csv(tmp_path / "selected_features.csv")[
                    "feature_name"
                ].tolist(),
                ["b"],
            )


if __name__ == "__main__":
    unittest.main()
