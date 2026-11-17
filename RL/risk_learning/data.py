"""Episode feature shards and deterministic group-disjoint training partitions."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .model import FEATURE_SPEC, FORMAT_VERSION


def save_episode(path, features, labels, frames, metadata):
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    frame = np.asarray(frames, dtype=np.int64)
    if x.shape != (len(y), 521) or frame.shape != y.shape or len(y) == 0:
        raise ValueError("Expected non-empty aligned episode features, labels and frame indices")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Invalid feature/label sample")
    if not metadata.get("scene_group") or not metadata.get("encoder_signature") or not metadata.get("reference_config"):
        raise ValueError("Scene group, encoder signature and reference configuration are mandatory")
    meta = dict(metadata, feature_spec=FEATURE_SPEC, format_version=FORMAT_VERSION)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash never leaves a final-name shard that looks complete.
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=x, labels=y, frames=frame, metadata=json.dumps(meta, sort_keys=True))
    temporary.replace(path)


def inspect_shards(data_dir):
    paths = sorted(Path(data_dir).rglob("*.npz"))
    if not paths:
        raise ValueError("No risk feature shards found")
    records = []
    contract = None
    for path in paths:
        with np.load(str(path), allow_pickle=False) as shard:
            meta = json.loads(str(shard["metadata"].item()))
            x, y = shard["features"], shard["labels"]
            if meta.get("format_version") != FORMAT_VERSION or meta.get("feature_spec") != FEATURE_SPEC:
                raise ValueError("Feature format mismatch: {}".format(path))
            current = (meta.get("encoder_signature"), meta.get("reference_config"))
            if not meta.get("scene_group") or not current[0] or not current[1]:
                raise ValueError("Incomplete provenance: {}".format(path))
            if contract is not None and contract != current:
                raise ValueError("Do not mix different frozen encoders or reference definitions")
            contract = current
            if len(y) == 0 or x.shape != (len(y), 521) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
                raise ValueError("Invalid shard: {}".format(path))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(dict(path=str(path.resolve()), sha256=digest, samples=len(y), metadata=meta))
    return records


def split_groups(records, seed=1):
    groups = sorted({r["metadata"]["scene_group"] for r in records})
    if len(groups) < 3:
        raise ValueError("Need at least three distinct scene groups for train/validation/test; do not split frames")
    groups = np.random.RandomState(seed).permutation(groups).tolist()
    n_test = max(1, int(round(len(groups) * 0.15)))
    n_val = max(1, int(round(len(groups) * 0.15)))
    assignment = {g: "test" for g in groups[:n_test]}
    assignment.update({g: "validation" for g in groups[n_test:n_test+n_val]})
    assignment.update({g: "train" for g in groups[n_test+n_val:]})
    return {name: [r for r in records if assignment[r["metadata"]["scene_group"]] == name]
            for name in ("train", "validation", "test")}


class FeatureDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.ends = np.cumsum([r["samples"] for r in records]).tolist()
        self.cache = OrderedDict()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_id = bisect_right(self.ends, index)
        if shard_id not in self.cache:
            with np.load(self.records[shard_id]["path"], allow_pickle=False) as shard:
                self.cache[shard_id] = (shard["features"], shard["labels"])
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard_id)
        row = index - (self.ends[shard_id-1] if shard_id else 0)
        x, y = self.cache[shard_id]
        return torch.from_numpy(x[row].copy()), torch.tensor(float(y[row]), dtype=torch.float32)
