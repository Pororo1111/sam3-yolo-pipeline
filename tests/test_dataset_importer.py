from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import yaml

from pipeline import dataset_importer, trainer


def _write_dataset(
    root: Path,
    *,
    names: list[str] | None = None,
    invalid_label: str | None = None,
) -> Path:
    names = names or ["cone"]
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{split}.jpg").write_bytes(b"fixture")
        label = invalid_label if split == "train" and invalid_label else "0 0.5 0.5 0.4 0.4"
        (label_dir / f"{split}.txt").write_text(label, encoding="utf-8")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


class DatasetImporterTests(unittest.TestCase):
    def test_trainer_uses_fast_mps_data_loading_options(self):
        self.assertEqual(trainer._data_loading_options("mps"), (4, "ram"))
        self.assertEqual(trainer._data_loading_options("cpu"), (0, False))
        self.assertEqual(trainer._data_loading_options("auto"), (0, False))

    def test_parse_roboflow_project_and_version_urls(self):
        self.assertEqual(
            dataset_importer.parse_roboflow_universe_url(
                "https://universe.roboflow.com/example/team-project"
            ),
            ("example", "team-project", None),
        )
        self.assertEqual(
            dataset_importer.parse_roboflow_universe_url(
                "https://universe.roboflow.com/example/team-project/dataset/7"
            ),
            ("example", "team-project", 7),
        )

    def test_parse_roboflow_url_rejects_other_hosts(self):
        with self.assertRaisesRegex(
            dataset_importer.DatasetImportError,
            "universe.roboflow.com",
        ):
            dataset_importer.parse_roboflow_universe_url(
                "https://example.com/workspace/project"
            )

    def test_roboflow_project_url_downloads_latest_and_registers(self):
        class FakeDownload:
            def __init__(self, location):
                self.location = location

        class FakeVersion:
            def __init__(self, number):
                self.version = str(number)

            def download(self, model_format, location, overwrite=False):
                self.model_format = model_format
                self.overwrite = overwrite
                _write_dataset(Path(location))
                return FakeDownload(location)

        class FakeProject:
            def __init__(self):
                self.items = [FakeVersion(1), FakeVersion(4), FakeVersion(2)]

            def versions(self):
                return self.items

            def version(self, number):
                return next(item for item in self.items if int(item.version) == number)

        with tempfile.TemporaryDirectory() as directory:
            project = FakeProject()
            with mock.patch.object(
                dataset_importer,
                "_roboflow_project",
                return_value=project,
            ):
                records, added, version = dataset_importer.download_roboflow_universe(
                    "https://universe.roboflow.com/example/safety-cones",
                    download_root=Path(directory),
                    api_key="secret",
                )

            self.assertEqual(version, 4)
            self.assertEqual(len(records), 1)
            self.assertEqual(len(added), 1)
            self.assertIn("v4", added[0]["source"])
            self.assertTrue(Path(added[0]["yaml"]).is_file())

    def test_validate_roboflow_style_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = _write_dataset(Path(directory) / "안전 라바콘")

            report = dataset_importer.validate_dataset(yaml_path)

            self.assertEqual(report["classes"], ["cone"])
            self.assertEqual(report["train_images"], 1)
            self.assertEqual(report["val_images"], 1)
            self.assertEqual(report["test_images"], 1)
            self.assertEqual(report["boxes"], 3)
            self.assertEqual(report["missing_labels"], 0)

    def test_invalid_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = _write_dataset(
                Path(directory) / "dataset",
                invalid_label="2 0.5 0.5 0.4 0.4",
            )

            with self.assertRaisesRegex(
                dataset_importer.DatasetImportError,
                "클래스 ID",
            ):
                dataset_importer.validate_dataset(yaml_path)

    def test_train_val_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = _write_dataset(Path(directory) / "dataset")
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            data["val"] = data["train"]
            yaml_path.write_text(
                yaml.safe_dump(data, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                dataset_importer.DatasetImportError,
                "train과 val",
            ):
                dataset_importer.validate_dataset(yaml_path)

    def test_compatible_datasets_create_combined_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_dataset(root / "first", names=["cone", "person"])
            second = _write_dataset(root / "second", names=["cone", "person"])

            combined, description = dataset_importer.prepare_training_data(
                [str(first), str(second)],
                output_dir=root / "combined",
            )
            data = yaml.safe_load(combined.read_text(encoding="utf-8"))

            self.assertEqual(len(data["train"]), 2)
            self.assertEqual(len(data["val"]), 2)
            self.assertEqual(data["names"], {0: "cone", 1: "person"})
            self.assertIn("2개", description)

    def test_different_class_order_is_remapped_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_dataset(root / "first", names=["cone", "person"])
            second = _write_dataset(root / "second", names=["person", "cone"])
            original_label = (
                second.parent / "train" / "labels" / "train.txt"
            ).read_text(encoding="utf-8")

            combined, description = dataset_importer.prepare_training_data(
                [str(first), str(second)],
                output_dir=root / "combined",
            )
            data = yaml.safe_load(combined.read_text(encoding="utf-8"))
            staged_second_label = next(
                (combined.parent / "labels" / "train" / "dataset_001").glob("*.txt")
            )
            report = dataset_importer.validate_dataset(combined)

            self.assertEqual(data["names"], {0: "cone", 1: "person"})
            self.assertTrue(staged_second_label.read_text(encoding="utf-8").startswith("1 "))
            self.assertEqual(
                (second.parent / "train" / "labels" / "train.txt").read_text(
                    encoding="utf-8"
                ),
                original_label,
            )
            self.assertEqual(report["train_images"], 2)
            self.assertIn("클래스 ID 재매핑", description)
            self.assertTrue((combined.parent / ".complete.yaml").is_file())

            cached, _ = dataset_importer.prepare_training_data(
                [str(first), str(second)],
                output_dir=root / "combined",
            )
            self.assertEqual(cached, combined)

    def test_class_mapping_change_invalidates_staging_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_dataset(root / "first", names=["a", "b", "c"])
            second = _write_dataset(root / "second", names=["b", "a"])

            before, _ = dataset_importer.prepare_training_data(
                [str(first), str(second)],
                output_dir=root / "combined",
            )
            second_data = yaml.safe_load(second.read_text(encoding="utf-8"))
            second_data["names"] = ["c", "b"]
            second.write_text(
                yaml.safe_dump(second_data, sort_keys=False),
                encoding="utf-8",
            )
            after, _ = dataset_importer.prepare_training_data(
                [str(first), str(second)],
                output_dir=root / "combined",
            )

            self.assertNotEqual(before, after)
            self.assertTrue(before.is_file())
            self.assertTrue(after.is_file())

    def test_hardlink_alias_across_datasets_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_dataset(root / "first")
            second = _write_dataset(root / "second")
            first_image = first.parent / "train" / "images" / "train.jpg"
            second_image = second.parent / "train" / "images" / "train.jpg"
            second_image.unlink()
            try:
                second_image.hardlink_to(first_image)
            except OSError as exc:
                self.skipTest(f"hardlink를 만들 수 없는 파일시스템입니다: {exc}")

            with self.assertRaisesRegex(
                dataset_importer.DatasetImportError,
                "같은 이미지",
            ):
                dataset_importer.prepare_training_data([str(first), str(second)])

    def test_image_copy_fallback_when_hardlink_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            destination = root / "destination.jpg"
            source.write_bytes(b"image")

            with mock.patch.object(dataset_importer.os, "link", side_effect=OSError):
                method = dataset_importer._link_or_copy(source, destination)

            self.assertEqual(method, "copy")
            self.assertEqual(destination.read_bytes(), b"image")

    def test_multiple_zip_archives_can_be_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = []
            for name in ("first", "second"):
                dataset_root = root / f"{name}-source" / name
                _write_dataset(dataset_root)
                archive = root / f"{name}.zip"
                with zipfile.ZipFile(archive, "w") as zip_file:
                    for path in dataset_root.rglob("*"):
                        if path.is_file():
                            zip_file.write(path, path.relative_to(dataset_root.parent))
                archives.append(str(archive))

            records, added = dataset_importer.register_archives(
                archives,
                extract_root=root / "extracted",
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(len(added), 2)
            self.assertTrue(all(Path(record["yaml"]).is_file() for record in records))

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../outside.txt", "unsafe")

            with self.assertRaisesRegex(
                dataset_importer.DatasetImportError,
                "안전하지 않은 경로",
            ):
                dataset_importer.register_archives(
                    [str(archive)],
                    extract_root=root / "extracted",
                )
            self.assertFalse((root / "outside.txt").exists())

    def test_trainer_rejects_an_explicit_empty_selection(self):
        outputs = list(
            trainer.train(
                1,
                320,
                1,
                10,
                "cpu",
                dataset_yamls=[],
            )
        )

        self.assertEqual(len(outputs), 2)
        self.assertIn("통합 준비", outputs[0])
        self.assertIn("한 개 이상 선택", outputs[-1])


if __name__ == "__main__":
    unittest.main()
