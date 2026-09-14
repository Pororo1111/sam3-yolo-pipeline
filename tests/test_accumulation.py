"""누적 수집부터 학습 구성까지의 데이터 보존 회귀 테스트."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipeline import data_store, dataset, extractor, labeler, media, source_groups


class AccumulationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.frames = self.root / 'raw_frames'
        self.labels = self.root / 'labels'
        self.frames.mkdir()
        self.labels.mkdir()
        for module, values in [
            (extractor, {'OUT_DIR': self.frames}),
            (labeler, {'FRAMES_DIR': self.frames, 'LABELS_DIR': self.labels}),
            (dataset, {'FRAMES_DIR': self.frames, 'LABELS_DIR': self.labels,
                       'IMAGES_DIR': self.root / 'images', 'YAML_PATH': self.root / 'dataset.yaml',
                       'SPLIT_MANIFEST_PATH': self.root / 'split_manifest.json'}),
        ]:
            for key, value in values.items():
                self.stack.enter_context(mock.patch.object(module, key, value))

    def frame(self, source, index=0, label=None):
        path = self.frames / f'frame_{source}_{index:05d}.jpg'
        cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
        if label is not None:
            (self.labels / (path.stem + '.txt')).write_text(label)
        return path

    def infer(self, lines):
        self.stack.enter_context(mock.patch.object(labeler, '_get_predictor', return_value=object()))
        return self.stack.enter_context(mock.patch.object(labeler, '_infer_and_overlay',
            return_value=(np.zeros((8, 8, 3), dtype=np.uint8), lines, len(lines))))

    def test_repeated_capture_preserves_images_labels_and_manifest(self):
        source = self.root / 'input.jpg'
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
        list(extractor.capture(media.SOURCE_IMAGES, '', 1, [str(source)]))
        old = next(self.frames.glob('*.jpg'))
        original = old.read_bytes()
        label = self.labels / (old.stem + '.txt')
        label.write_text('0 0.5 0.5 0.2 0.2')
        list(extractor.capture(media.SOURCE_IMAGES, '', 1, [str(source)]))
        self.assertEqual(len(list(self.frames.glob('*.jpg'))), 2)
        self.assertEqual(old.read_bytes(), original)
        self.assertEqual(label.read_text(), '0 0.5 0.5 0.2 0.2')
        records = data_store.read_json(self.root / 'sources.json', {})['sources']
        self.assertEqual([r['id'] for r in records], ['images001', 'images002'])
        self.assertTrue(all(r['status'] == 'complete' and r['frame_count'] == 1 for r in records))

    def test_close_generator_keeps_partial_capture_record(self):
        source = self.root / 'input.jpg'
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
        capture = extractor.capture(media.SOURCE_IMAGES, '', 1, [str(source)] * 3)
        next(capture)
        next(capture)
        capture.close()
        record = data_store.read_json(self.root / 'sources.json', {})['sources'][0]
        self.assertEqual(record['status'], 'interrupted')
        self.assertEqual(record['frame_count'], 1)

    def test_prompt_order_maps_to_stable_ids_and_keeps_alias_after_rename(self):
        self.assertEqual(data_store.map_prompts(self.root, ['person', 'car']), [0, 1])
        data_store.rename_classes(self.root, ['사람', '차량'])
        self.frame('new')
        self.infer(['0 0.5 0.5 0.2 0.2', '1 0.5 0.5 0.2 0.2'])
        list(labeler.label('car, cone', 0.25, ['new']))
        self.assertEqual([l.split()[0] for l in (self.labels / 'frame_new_00000.txt').read_text().splitlines()], ['1', '2'])
        self.assertEqual([c['name'] for c in dataset.scan_classes('cone, car')], ['사람', '차량', 'cone'])

    def test_unknown_legacy_ids_block_mapping_until_named(self):
        self.frame('legacy', label='2 0.5 0.5 0.2 0.2')
        with self.assertRaises(ValueError):
            data_store.map_prompts(self.root, ['car'])
        self.assertEqual(len(dataset.scan_classes()), 3)
        data_store.rename_classes(self.root, ['person', 'car', 'cone'])
        self.assertEqual(data_store.map_prompts(self.root, ['cone']), [2])

    def test_pending_mode_skips_empty_labels_and_resumes_after_close(self):
        self.frame('new', 0, '')
        self.frame('new', 1)
        self.frame('new', 2)
        inferred = self.infer([])
        generator = labeler.label('car', 0.25, ['new'])
        next(generator)
        next(generator)
        generator.close()
        self.assertEqual(inferred.call_count, 1)
        list(labeler.label('car', 0.25, ['new']))
        self.assertEqual(inferred.call_count, 2)
        self.assertEqual(len(list(self.labels.glob('*.txt'))), 3)

    def test_model_failure_preserves_relabel_targets(self):
        self.frame('old', label='0 0.5 0.5 0.2 0.2')
        (self.root / 'dataset.yaml').write_text('names: [car]\n')
        with mock.patch.object(labeler, '_get_predictor', side_effect=RuntimeError('failure')):
            result = list(labeler.label('car', 0.25, ['old'], '선택 소스 재라벨링'))
        self.assertIn('모델 로딩 실패', result[-1][1])
        self.assertEqual((self.labels / 'frame_old_00000.txt').read_text(), '0 0.5 0.5 0.2 0.2')

    def test_build_keeps_assignments_and_excludes_pending(self):
        for source in ('a', 'b'):
            self.frame(source, label='0 0.5 0.5 0.2 0.2')
        list(dataset.build_dataset('car', .2, False, ['b']))
        self.frame('a', 1)
        self.frame('c', label='')
        list(dataset.build_dataset('car', .2, False))
        split = data_store.read_json(self.root / 'split_manifest.json', {})
        self.assertEqual(split['val_sources'], ['b'])
        self.assertEqual(split['train_sources'], ['a', 'c'])
        self.assertFalse((self.root / 'images/train/frame_a_00001.jpg').exists())
        self.assertTrue((self.root / 'labels/train/frame_c_00000.txt').exists())

    def test_same_video_url_variants_cannot_cross_splits(self):
        for source in ('a', 'b', 'c'):
            self.frame(source, label='0 0.5 0.5 0.2 0.2')
        data_store.save_json(self.root / 'sources.json', {'sources': [
            {'id': 'a', 'type': 'YouTube URL', 'value': 'https://youtu.be/abc'},
            {'id': 'b', 'type': 'YouTube URL', 'value': 'https://www.youtube.com/watch?v=abc&t=12'},
        ]})
        list(dataset.build_dataset('car', .2, False, ['b']))
        split = data_store.read_json(self.root / 'split_manifest.json', {})
        self.assertEqual(split['val_sources'], ['a', 'b'])
        self.assertEqual(split['train_sources'], ['c'])

    def test_failed_build_and_cancel_preserve_previous_output(self):
        for source in ('a', 'b'):
            self.frame(source, label='0 0.5 0.5 0.2 0.2')
        list(dataset.build_dataset('car', .2, False, ['b']))
        previous = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(dataset.shutil, 'copy', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                list(dataset.build_dataset('car', .2, False, ['a']))
        generator = dataset.build_dataset('car', .2, False, ['a'])
        next(generator)
        generator.close()
        current = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(previous, current)

    def test_publish_failure_rolls_back_folders_yaml_and_cache(self):
        for source in ('a', 'b'):
            self.frame(source, label='0 0.5 0.5 0.2 0.2')
        list(dataset.build_dataset('car', .2, False, ['b']))
        (self.labels / 'train.cache').write_bytes(b'old-cache')
        previous = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(data_store, 'save_json', side_effect=OSError('manifest failed')):
            with self.assertRaises(OSError):
                list(dataset.build_dataset('car', .2, False, ['a']))
        current = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(previous, current)

    def test_legacy_sources_are_registered_without_renaming(self):
        legacy = self.frame('images001')
        original = legacy.read_bytes()
        extractor._controller.begin()
        with extractor._source_record('images', media.SOURCE_IMAGES, '업로드 이미지') as source_id:
            self.assertEqual(source_id, 'images002')
        records = data_store.read_json(self.root / 'sources.json', {})['sources']
        self.assertEqual(records[0]['status'], 'legacy')
        self.assertEqual(legacy.read_bytes(), original)

    def test_empty_source_selection_does_not_label_all(self):
        self.frame('a')
        inferred = self.infer([])
        list(labeler.label('car', .25, []))
        inferred.assert_not_called()


if __name__ == '__main__':
    unittest.main()
