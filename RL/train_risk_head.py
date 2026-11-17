"""Train/test the frozen-feature risk head without CARLA or simulator imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from risk_learning.data import FeatureDataset, inspect_shards, split_groups
from risk_learning.model import FEATURE_SPEC, FORMAT_VERSION, RiskHead, load_head


class ShardShuffleSampler:
    """Shuffle shard and row order without decompressing a shard per random row."""
    def __init__(self, records, seed):
        self.records, self.random = records, np.random.RandomState(seed)

    def __len__(self):
        return sum(r["samples"] for r in self.records)

    def __iter__(self):
        offsets = np.cumsum([0] + [r["samples"] for r in self.records])
        for shard in self.random.permutation(len(self.records)):
            for row in self.random.permutation(self.records[shard]["samples"]):
                yield int(offsets[shard] + row)


def metrics(head, loader, device, optimizer=None):
    head.train(optimizer is not None)
    squared, absolute, count = 0.0, 0.0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            predicted = head(features)
            loss = torch.mean((predicted - labels) ** 2)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite risk loss")
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        error = predicted.detach() - labels
        squared += float((error ** 2).sum().item())
        absolute += float(error.abs().sum().item())
        count += labels.numel()
    if not count:
        raise ValueError("Empty data partition")
    return dict(mse=squared / count, mae=absolute / count, samples=count)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf8")


def train(args):
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer/training parameters")
    out = Path(args.output_dir)
    if (out / "risk_head.pt").exists() or (out / "split_manifest.json").exists():
        raise ValueError("Choose a new output directory; existing training records are not overwritten")
    records = inspect_shards(args.data_dir)
    partitions = split_groups(records, args.split_seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(format_version=FORMAT_VERSION, split_seed=args.split_seed, partitions=partitions)
    write_json(out / "split_manifest.json", manifest)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    head = RiskHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(FeatureDataset(partitions["train"]), batch_size=args.batch_size,
                              sampler=ShardShuffleSampler(partitions["train"], args.seed), num_workers=0)
    validation_loader = DataLoader(FeatureDataset(partitions["validation"]), batch_size=args.batch_size, num_workers=0)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        train_result = metrics(head, train_loader, device, optimizer)
        validation_result = metrics(head, validation_loader, device)
        row = dict(epoch=epoch, train_mse=train_result["mse"], validation_mse=validation_result["mse"], validation_mae=validation_result["mae"])
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_result["mse"] < best:
            best, stale = validation_result["mse"], 0
            payload = dict(format_version=FORMAT_VERSION, feature_spec=FEATURE_SPEC,
                           encoder_signature=records[0]["metadata"]["encoder_signature"],
                           reference_config=records[0]["metadata"]["reference_config"],
                           state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
                           epoch=epoch, validation=validation_result, train_config=vars(args),
                           manifest_sha256=hashlib.sha256((out / "split_manifest.json").read_bytes()).hexdigest())
            temporary = out / "risk_head.pt.tmp"
            torch.save(payload, str(temporary))
            temporary.replace(out / "risk_head.pt")
        else:
            stale += 1
        with (out / "training_history.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if stale >= args.patience:
            break
    write_json(out / "training_summary.json", dict(best_validation_mse=best, epochs_run=len(history),
               partition_samples={name: sum(r["samples"] for r in rows) for name, rows in partitions.items()},
               note="Test data is held out and is not evaluated during checkpoint selection."))
    print("Saved {}".format(out / "risk_head.pt"))


def test(args):
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    partitions = manifest["partitions"]
    groups = {name: {r["metadata"]["scene_group"] for r in rows} for name, rows in partitions.items()}
    if groups["test"] & (groups["train"] | groups["validation"]) or groups["train"] & groups["validation"]:
        raise ValueError("Scene-group leakage in split manifest")
    records = partitions["test"]
    if not records:
        raise ValueError("Empty test partition")
    for record in records:
        if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Test data changed since the manifest was written")
    head, checkpoint = load_head(args.checkpoint, records[0]["metadata"]["encoder_signature"], torch.device(args.device))
    if checkpoint["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Checkpoint and split manifest do not match")
    loader = DataLoader(FeatureDataset(records), batch_size=args.batch_size, num_workers=0)
    result = metrics(head, loader, torch.device(args.device))
    result["meaning"] = "Agreement with a continuous surrogate reference, not collision probability calibration"
    out = Path(args.output)
    if out.exists():
        raise ValueError("Test output already exists; choose a new output filename")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(json.dumps(result, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data_dir", required=True)
    training.add_argument("--output_dir", required=True)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--weight_decay", type=float, default=1e-5)
    training.add_argument("--split_seed", type=int, default=1)
    training.add_argument("--seed", type=int, default=1)
    testing = sub.add_parser("test")
    testing.add_argument("--checkpoint", required=True)
    testing.add_argument("--manifest", required=True)
    testing.add_argument("--output", required=True)
    for child in (training, testing):
        child.add_argument("--device", default="cpu")
        child.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args) if args.command == "train" else test(args)
