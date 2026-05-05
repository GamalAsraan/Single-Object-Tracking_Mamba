"""
AerialTracking — MTC-AIC4 single-object aerial tracking dataset (JPEG-based).

Layout (under `root`):
    {dataset}/{seq_name}/000000.jpg ...   (zero-indexed, 6-digit)
    {dataset}/{seq_name}/annotation.txt   (CSV: x,y,w,h per frame)
    metadata/contestant_manifest.json

Manifest schema (top-level dict):
    {
      "train":     { "dataset1/Car_video_2": {dataset, seq_name, n_frames, ...}, ... },
      "public_lb": { "dataset1/Car_video":   {dataset, seq_name, n_frames, ...}, ... }
    }

This class only consumes the "train" partition. The "public_lb" partition is
used at inference time (Phase 8/9), not for training.
"""

import os
import json
import random
from collections import OrderedDict

import numpy as np
import torch

from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader
from lib.train.admin.environment import env_settings


# Known-bad train sequences (verified during Phase 5 recon over all 255 seqs):
#   * dataset4/bike1     — 0 jpgs on disk vs 3085 expected
#   * dataset4/bird1     — 0 jpgs on disk vs 2437 expected
#   * dataset3/group4_1  — annotation.txt malformed at row 161 (col count changes)
_KNOWN_BAD_SEQS = frozenset({
    "dataset4/bike1",
    "dataset4/bird1",
    "dataset3/group4_1",
})

# Fallback manifest location (videos-only dataset) in case JPEG root has none
_MANIFEST_FALLBACK = "/kaggle/input/datasets/gamalasran/aic-mtc-4/metadata/contestant_manifest.json"


class AerialTracking(BaseVideoDataset):
    """MTC-AIC4 single-object aerial tracking dataset."""

    def __init__(self,
                 root=None,
                 image_loader=jpeg4py_loader,
                 split='train',
                 seq_ids=None,
                 data_fraction=None,
                 val_fraction=0.1,
                 split_seed=42):
        """
        Args:
            root           : Path to JPEG-extracted MTC-AIC4 root. Falls back to
                             env_settings().aerial_dir when None.
            image_loader   : Callable(path) -> np.ndarray(H,W,3) uint8 RGB.
                             Defaults to jpeg4py_loader; sampler passes opencv_loader.
            split          : 'train' | 'val' | 'all' | None.
            seq_ids        : Optional list of indices to override the split.
            data_fraction  : Optional float in (0,1] to subsample sequences.
            val_fraction   : Fraction of sequences held out for validation (deterministic).
            split_seed     : Seed for the deterministic train/val partition.
        """
        root = env_settings().aerial_dir if root is None else root
        super().__init__('AERIAL', root, image_loader)

        # ---- locate manifest --------------------------------------------------
        manifest_path = os.path.join(root, 'metadata', 'contestant_manifest.json')
        if not os.path.isfile(manifest_path):
            if os.path.isfile(_MANIFEST_FALLBACK):
                manifest_path = _MANIFEST_FALLBACK
            else:
                raise FileNotFoundError(
                    f"contestant_manifest.json not found at {manifest_path} or {_MANIFEST_FALLBACK}"
                )
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if 'train' not in manifest:
            raise KeyError(f"Manifest missing 'train' key. Keys: {list(manifest.keys())}")

        # ---- build candidate sequence list, filter known-bad ------------------
        all_keys = sorted(manifest['train'].keys())
        all_keys = [k for k in all_keys if k not in _KNOWN_BAD_SEQS]

        # ---- deterministic 90/10 train/val split ------------------------------
        rng = random.Random(split_seed)
        shuffled = list(all_keys)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_set = set(shuffled[:n_val])

        if split == 'train':
            keys = [k for k in all_keys if k not in val_set]
        elif split == 'val':
            keys = [k for k in all_keys if k in val_set]
        elif split in ('all', None):
            keys = list(all_keys)
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', 'all', or None.")

        # ---- seq_ids / data_fraction overrides --------------------------------
        if seq_ids is not None:
            keys = [keys[i] for i in seq_ids]
        if data_fraction is not None:
            n = max(1, int(len(keys) * data_fraction))
            keys = random.sample(keys, n)

        self.sequence_list = keys
        # Cache per-seq manifest entries for fast lookup
        self.manifest = {k: manifest['train'][k] for k in self.sequence_list}

        # Single-class dataset (no per-target class labels in MTC-AIC4)
        self.class_list = ['aerial_target']
        self.seq_per_class = {'aerial_target': list(range(len(self.sequence_list)))}

        self.split = split
        self._manifest_path = manifest_path

    # ------------------------------------------------------------------ basics
    def get_name(self):
        return 'aerial'

    def is_video_sequence(self):
        return True

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        # We infer occlusion from invalid bbox dims (w<=0 or h<=0)
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class.get(class_name, [])

    # ----------------------------------------------------------- path helpers
    def _get_sequence_path(self, seq_id):
        key = self.sequence_list[seq_id]
        entry = self.manifest[key]
        return os.path.join(self.root, entry['dataset'], entry['seq_name'])

    def _get_annotation_path(self, seq_id):
        key = self.sequence_list[seq_id]
        entry = self.manifest[key]
        return os.path.join(self.root, entry['annotation_path'])

    def _get_frame_path(self, seq_path, frame_id):
        # Frames are zero-indexed, 6-digit pad: 000000.jpg, 000001.jpg, ...
        return os.path.join(seq_path, '{:06d}.jpg'.format(frame_id))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    # ----------------------------------------------------------- annotations
    def _read_bb_anno(self, seq_id):
        ann_path = self._get_annotation_path(seq_id)
        gt = np.loadtxt(ann_path, delimiter=',', dtype=np.float32)
        if gt.ndim == 1:
            gt = gt[None, :]
        if gt.shape[1] != 4:
            raise ValueError(
                f"Annotation {ann_path} has {gt.shape[1]} cols, expected 4 (x,y,w,h)"
            )
        return torch.from_numpy(gt)  # (N, 4) float32

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)
        # valid: positive width AND height (matches Got10k convention)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # No separate occlusion file; visibility ≡ validity for this dataset
        visible = valid.clone().byte()
        return {
            'bbox': bbox,
            'valid': valid,
            'visible': visible,
        }

    # ------------------------------------------------------------- frame API
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # obj_meta keys mirror Got10k so downstream code (which sometimes
        # reads object_class_name) doesn't KeyError.
        obj_meta = OrderedDict({
            'object_class_name': 'aerial_target',
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None,
        })
        return frame_list, anno_frames, obj_meta

    # ----------------------------------------------------------- introspection
    def get_class_name(self, seq_id):
        return 'aerial_target'

    def __repr__(self):
        return (f"AerialTracking(split={self.split!r}, "
                f"num_sequences={len(self.sequence_list)}, "
                f"root={self.root!r})")