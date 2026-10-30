# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""Typed pipeline options shared by the Python API and command line."""

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Optional, Tuple

import yaml

SEARCH_METHODS = (
    'RADIUS_SEARCH', 'POSE_GRAPH_ASSOCIATION', 'IMAGE_RETRIEVAL',
    'TRACK_2D_OVERLAP_IR')

RUNNER_FIELDS = (
    'skip_cuvslam',
    'override_frames_meta_file',
    'skip_pose_graph',
    'skip_data_association',
    'feature_type',
    'min_inter_frame_distance',
    'min_inter_frame_rotation_degrees',
    'num_threads',
    'optimize_intrinsics',
    'optimize_extrinsics',
    'feature_extractor_batch_size',
    'feature_matching_batch_size',
    'downsampling_matches',
)
NATIVE_FIELDS = {
    'search_method': ('match_pair_select_config.pb.txt', ('search_method', )),
    'search_translation_threshold_meters': (
        'match_pair_select_config.pb.txt',
        ('search_translation_threshold_meters', )),
    'search_rotation_threshold_degrees': (
        'match_pair_select_config.pb.txt',
        ('search_rotation_threshold_degrees', )),
    'max_frame_pair_size_per_file':
    ('match_pair_select_config.pb.txt', ('max_frame_pair_size_per_file', )),
    'mapping_num_threads': ('vision_mapping_config.pb.txt', ('num_threads', )),
    'ba_num_threads': (
        'vision_mapping_config.pb.txt',
        ('bundle_adjustment_config', 'num_threads')),
}


@dataclass(frozen=True)
class CuSFMConfig:
    """None inherits the existing runner or native configuration-file default.

    GPU IDs are indices into the devices visible through CUDA_VISIBLE_DEVICES.
    Matcher process count is a concurrency limit, not a batch-size setting.
    """

    skip_cuvslam: Optional[bool] = None
    override_frames_meta_file: Optional[str] = None
    skip_pose_graph: Optional[bool] = None
    skip_data_association: Optional[bool] = None
    feature_type: Optional[str] = None
    min_inter_frame_distance: Optional[float] = None
    min_inter_frame_rotation_degrees: Optional[float] = None
    search_method: Optional[str] = None
    search_translation_threshold_meters: Optional[float] = None
    search_rotation_threshold_degrees: Optional[float] = None
    mapping_num_threads: Optional[int] = None
    ba_num_threads: Optional[int] = None
    matcher_num_threads: Optional[int] = None
    matcher_processes: Optional[int] = None
    matcher_gpu_ids: Optional[Tuple[int, ...]] = None
    max_frame_pair_size_per_file: Optional[int] = None
    num_threads: Optional[int] = None
    optimize_intrinsics: Optional[bool] = None
    optimize_extrinsics: Optional[bool] = None
    feature_extractor_batch_size: Optional[int] = None
    feature_matching_batch_size: Optional[int] = None
    downsampling_matches: Optional[bool] = None

    def __post_init__(self):
        for key in ('skip_cuvslam', 'skip_pose_graph', 'skip_data_association',
                    'optimize_intrinsics', 'optimize_extrinsics',
                    'downsampling_matches'):
            value = getattr(self, key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f'{key} must be boolean')
        for key in ('min_inter_frame_distance',
                    'min_inter_frame_rotation_degrees',
                    'search_translation_threshold_meters',
                    'search_rotation_threshold_degrees'):
            value = getattr(self, key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(
                    value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
            if value < 0 or (key == 'search_translation_threshold_meters'
                             and value == 0):
                raise ValueError(f'{key} is outside its valid range')
            if 'degrees' in key and value > 180:
                raise ValueError(f'{key} must not exceed 180 degrees')
        for key in ('mapping_num_threads', 'ba_num_threads',
                    'matcher_num_threads', 'matcher_processes', 'num_threads',
                    'max_frame_pair_size_per_file',
                    'feature_extractor_batch_size',
                    'feature_matching_batch_size'):
            value = getattr(self, key)
            if value is None:
                continue
            minimum = 0 if key.endswith('batch_size') else 1
            if isinstance(
                    value,
                    bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        if self.override_frames_meta_file is not None and not isinstance(
                self.override_frames_meta_file, str):
            raise ValueError('override_frames_meta_file must be a string path')
        if self.search_method is not None:
            if self.search_method not in SEARCH_METHODS:
                raise ValueError('Unknown search_method')
        if self.feature_type is not None and (not isinstance(
                self.feature_type, str) or not self.feature_type.strip()):
            raise ValueError('feature_type must be a nonempty string')
        if self.matcher_gpu_ids is not None:
            try:
                ids = tuple(self.matcher_gpu_ids)
            except TypeError as error:
                raise ValueError(
                    'matcher_gpu_ids must be a sequence') from error
            if not ids or any(
                    isinstance(i, bool) or not isinstance(i, int) or i < 0
                    for i in ids) or len(set(ids)) != len(ids):
                raise ValueError(
                    'matcher_gpu_ids must contain distinct nonnegative integers'
                )
            object.__setattr__(self, 'matcher_gpu_ids', ids)

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        with path.open() as stream:
            data = json.load(
                stream) if path.suffix == '.json' else yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError('Pipeline configuration must be an object')
        known = {f.name for f in fields(cls)}
        if set(data) - known:
            raise ValueError(
                f'Unknown pipeline settings: {sorted(set(data) - known)}')
        return cls(**data)

    def with_overrides(self, **kwargs):
        return replace(
            self, **{
                k: v
                for k, v in kwargs.items() if v is not None
            })

    def to_dict(self):
        return asdict(self)

    def runner_overrides(self):
        return {
            k: getattr(self, k)
            for k in RUNNER_FIELDS if getattr(self, k) is not None
        }


# A small scalar editor for native textproto files. It tracks message scopes,
# retains original formatting/comments, and rejects missing or duplicate fields.
# Native schemas are distributed with the binaries rather than Python bindings.
_TOKEN = re.compile(
    r'\s+|\#[^\n]*|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}:<>]|[^\s{}:<>]+'
)


def _scalar_span(text, path):
    tokens = [
        m for m in _TOKEN.finditer(text)
        if not m[0].isspace() and not m[0].startswith(('#', '//'))
    ]
    stack, found = [], []
    i = 0
    while i < len(tokens):
        value = tokens[i][0]
        if value in ('}', '>'):
            if not stack:
                raise ValueError('Unbalanced native configuration')
            stack.pop()
        elif re.fullmatch(r'[A-Za-z_]\w*', value) and i + 1 < len(tokens):
            j = i + 1
            if tokens[j][0] == ':':
                j += 1
            if j < len(tokens) and tokens[j][0] in ('{', '<'):
                stack.append(value)
                i = j
            elif i + 2 < len(tokens) and tokens[i + 1][0] == ':':
                if tuple(stack) + (value, ) == tuple(path):
                    found.append(tokens[i + 2].span())
                i += 2
        i += 1
    if stack or len(found) != 1:
        raise ValueError(
            f'Expected exactly one native scalar {".".join(path)}')
    return found[0]


def _native_scalar(text, path, value=None):
    start, end = _scalar_span(text, path)
    if value is None:
        raw = text[start:end].rstrip(',;')
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return text[:start] + str(value) + text[end:]


def prepare_native_config(source_dir, output_dir, config, dry_run=False):
    """Return an immutable, per-run native config directory and resolved values."""
    source = Path(source_dir)
    if not source.is_dir():
        raise ValueError(
            f'Native configuration directory does not exist: {source}')
    originals = {
        str(p.relative_to(source)): p.read_bytes()
        for p in sorted(source.rglob('*')) if p.is_file()
    }
    rendered = dict(originals)
    effective = {}
    for name, (filename, field) in NATIVE_FIELDS.items():
        text = rendered[filename].decode('utf-8')
        override = getattr(config, name)
        if override is not None:
            text = _native_scalar(text, field, override)
            rendered[filename] = text.encode('utf-8')
        effective[name] = _native_scalar(text, field)
    digests = {
        n: hashlib.sha256(raw).hexdigest()
        for n, raw in rendered.items()
    }
    identifier = hashlib.sha256(json.dumps(
        digests, sort_keys=True).encode()).hexdigest()[:16]
    target = Path(output_dir) / 'effective_configs' / identifier
    if not dry_run:
        for name, raw in rendered.items():
            p = target / name
            if p.exists():
                if p.read_bytes() != raw:
                    raise ValueError(
                        f'Previously captured native config changed: {p}')
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(raw)
    return target, effective, digests
