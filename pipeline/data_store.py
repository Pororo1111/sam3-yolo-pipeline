"""누적 데이터의 원자적 메타데이터 저장과 영구 클래스 ID 관리."""
import json
import os
import tempfile
from pathlib import Path
import yaml


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def save_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def classes(root: Path):
    path = root / 'classes.json'
    if path.exists():
        return read_json(path, {})
    config = root / 'dataset.yaml'
    names = (yaml.safe_load(config.read_text(encoding='utf-8')) or {}).get('names', {}) if config.exists() else {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    used = set()
    for label in (root / 'labels').glob('*.txt'):
        for line in label.read_text().splitlines():
            if line.strip():
                used.add(int(line.split()[0]))
    size = max(set(names) | used, default=-1) + 1
    entries = [{'name': names.get(i, f'class_{i}'), 'aliases': [names[i]] if i in names else [], 'resolved': i in names} for i in range(size)]
    return {'version': 1, 'classes': entries}


def map_prompts(root, prompts):
    data = classes(root)
    entries = data['classes']
    if any(not entry['resolved'] for entry in entries):
        raise ValueError('기존 클래스의 이름을 데이터셋 검토 단계에서 먼저 지정하세요.')
    mapping = []
    for prompt in prompts:
        matches = [i for i, entry in enumerate(entries) if prompt == entry['name'] or prompt in entry['aliases']]
        if len(matches) > 1:
            raise ValueError(f'중복 클래스 이름: {prompt}')
        if not matches:
            entries.append({'name': prompt, 'aliases': [prompt], 'resolved': True})
            matches = [len(entries) - 1]
        mapping.append(matches[0])
    save_json(root / 'classes.json', data)
    return mapping


def rename_classes(root, names):
    data = classes(root)
    if len(names) != len(data['classes']) or any(not n or ',' in n for n in names) or len(set(names)) != len(names):
        raise ValueError('클래스 이름은 비어 있거나 중복될 수 없으며 쉼표를 포함할 수 없습니다.')
    for i, name in enumerate(names):
        if any(name in entry['aliases'] for j, entry in enumerate(data['classes']) if j != i):
            raise ValueError(f'다른 클래스의 기존 프롬프트와 겹치는 이름: {name}')
    for entry, name in zip(data['classes'], names):
        entry['resolved'] = entry['resolved'] or name != entry['name']
        entry['name'] = name
        if name not in entry['aliases']:
            entry['aliases'].append(name)
    save_json(root / 'classes.json', data)
