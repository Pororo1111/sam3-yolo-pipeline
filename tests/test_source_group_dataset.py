from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipeline import dataset, labeler, source_groups


class SourceGroupsTests(unittest.TestCase):
    def test_source_id_supports_new_and_legacy_frame_names(self):
        self.assertEqual(
            source_groups.source_id_from_stem("frame_yt001_00042"),
            "yt001",
        )
        self.assertEqual(
            source_groups.source_id_from_stem("frame_00042"),
            source_groups.LEGACY_SOURCE_ID,
        )


class GroupDatasetTests(unittest.TestCase):
    def test_explicit_validation_source_never_leaks_into_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_dir = root / "raw_frames"
            labels_dir = root / "labels"
            images_dir = root / "images"
            yaml_path = root / "dataset.yaml"
            manifest_path = root / "split_manifest.json"
            frames_dir.mkdir()
            labels_dir.mkdir()

            for source_id in ("yt001", "yt002"):
                for index in range(2):
                    stem = f"frame_{source_id}_{index:05d}"
                    (frames_dir / f"{stem}.jpg").write_bytes(b"image")
                    (labels_dir / f"{stem}.txt").write_text(
                        "0 0.5 0.5 0.2 0.2",
                        encoding="utf-8",
                    )

            with (
                mock.patch.object(dataset, "FRAMES_DIR", frames_dir),
                mock.patch.object(dataset, "LABELS_DIR", labels_dir),
                mock.patch.object(dataset, "IMAGES_DIR", images_dir),
                mock.patch.object(dataset, "YAML_PATH", yaml_path),
                mock.patch.object(
                    dataset,
                    "SPLIT_MANIFEST_PATH",
                    manifest_path,
                ),
            ):
                outputs = list(
                    dataset.build_dataset(
                        "Hardhat, Safety Vest",
                        0.2,
                        False,
                        ["yt002"],
                    )
                )

            train_names = {path.name for path in (images_dir / "train").glob("*.jpg")}
            val_names = {path.name for path in (images_dir / "val").glob("*.jpg")}
            self.assertTrue(all("yt001" in name for name in train_names))
            self.assertTrue(all("yt002" in name for name in val_names))
            self.assertFalse(train_names & val_names)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_sources"], ["yt001"])
            self.assertEqual(manifest["val_sources"], ["yt002"])
            self.assertIn("train 소스: yt001", outputs[-1])

    def test_single_source_is_rejected_instead_of_frame_level_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_dir = root / "raw_frames"
            labels_dir = root / "labels"
            frames_dir.mkdir()
            labels_dir.mkdir()
            (frames_dir / "frame_yt001_00000.jpg").write_bytes(b"image")
            (labels_dir / "frame_yt001_00000.txt").write_text(
                "0 0.5 0.5 0.2 0.2",
                encoding="utf-8",
            )

            with (
                mock.patch.object(dataset, "FRAMES_DIR", frames_dir),
                mock.patch.object(dataset, "LABELS_DIR", labels_dir),
            ):
                outputs = list(
                    dataset.build_dataset("Hardhat", 0.2, False, [])
                )

            self.assertIn("최소 2개 소스", outputs[-1])


class SourceAwareLabelerTests(unittest.TestCase):
    def test_relabeling_one_source_preserves_other_source_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_dir = root / "raw_frames"
            labels_dir = root / "labels"
            frames_dir.mkdir()
            labels_dir.mkdir()
            image = np.zeros((8, 8, 3), dtype=np.uint8)

            for source_id in ("yt001", "yt002"):
                stem = f"frame_{source_id}_00000"
                cv2.imwrite(str(frames_dir / f"{stem}.jpg"), image)
                (labels_dir / f"{stem}.txt").write_text(
                    f"old-{source_id}",
                    encoding="utf-8",
                )

            inferred = (
                np.zeros((8, 8, 3), dtype=np.uint8),
                ["0 0.5 0.5 0.2 0.2"],
                1,
            )
            with (
                mock.patch.object(labeler, "FRAMES_DIR", frames_dir),
                mock.patch.object(labeler, "LABELS_DIR", labels_dir),
                mock.patch.object(labeler, "_get_predictor", return_value=object()),
                mock.patch.object(
                    labeler,
                    "_infer_and_overlay",
                    return_value=inferred,
                ),
            ):
                list(labeler.label("white hardhat", 0.25, ["yt001"]))

            self.assertEqual(
                (labels_dir / "frame_yt001_00000.txt").read_text(),
                "0 0.5 0.5 0.2 0.2",
            )
            self.assertEqual(
                (labels_dir / "frame_yt002_00000.txt").read_text(),
                "old-yt002",
            )


if __name__ == "__main__":
    unittest.main()
