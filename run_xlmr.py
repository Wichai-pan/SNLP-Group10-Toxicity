from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from toxicity.data import Example, infer_lang, read_tsv
from toxicity.metrics import compute_metrics, print_report
from toxicity.utils import DeviceInfo, ensure_dir, pick_device, seed_everything


@dataclass(frozen=True)
class Preset:
    max_length: int
    epochs: int
    train_batch_size: int
    grad_accum_steps: int
    lr: float


PRESETS: dict[str, Preset] = {
    "fast": Preset(max_length=128, epochs=1, train_batch_size=8, grad_accum_steps=2, lr=2e-5),
    "score": Preset(max_length=192, epochs=3, train_batch_size=8, grad_accum_steps=4, lr=1e-5),
}


def write_predictions_tsv(ids: list[str], preds: list[int], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "label"])
        for example_id, pred in zip(ids, preds):
            w.writerow([example_id, int(pred)])


class TorchDataset:
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {"id": ex.id, "text": ex.text, "label": ex.label, "lang": ex.lang}


def make_collate(
    *,
    tokenizer: Any,
    max_length: int,
    with_labels: bool,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [b["text"] for b in batch]
        ids = [b["id"] for b in batch]
        langs = [b.get("lang") or infer_lang(b["id"]) for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out: dict[str, Any] = dict(enc)
        out["ids"] = ids
        out["langs"] = langs
        if with_labels:
            labels = [int(b["label"]) for b in batch]  # type: ignore[arg-type]
            import torch

            out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out

    return collate


def to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


@dataclass
class EvalResult:
    y_true: list[int]
    y_pred: list[int]
    langs: list[str]
    macro_f1: float


def evaluate(
    *,
    model: Any,
    dataloader: Any,
    device: Any,
) -> EvalResult:
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    langs: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, device)
            outputs = model(
                **{k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()
            y_pred.extend([int(p) for p in preds])
            y_true.extend([int(l) for l in labels])
            langs.extend(list(batch["langs"]))

    m = compute_metrics(y_true, y_pred)
    return EvalResult(y_true=y_true, y_pred=y_pred, langs=langs, macro_f1=m.f1_macro)


def predict(
    *,
    model: Any,
    dataloader: Any,
    device: Any,
) -> tuple[list[str], list[int]]:
    import torch

    model.eval()
    all_ids: list[str] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            ids = list(batch["ids"])
            batch = to_device(batch, device)
            outputs = model(**{k: v for k, v in batch.items() if k in {"input_ids", "attention_mask"}})
            preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()
            all_ids.extend(ids)
            all_pred.extend([int(p) for p in preds])
    return all_ids, all_pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune XLM-R for multilingual toxicity detection.")
    ap.add_argument("--train_tsv", type=str, default="train.tsv")
    ap.add_argument("--dev_tsv", type=str, default="dev.tsv")
    ap.add_argument("--test_tsv", type=str, default="test.tsv")

    ap.add_argument("--model_name", type=str, default="xlm-roberta-base")
    ap.add_argument("--preset", type=str, choices=sorted(PRESETS.keys()), default="fast")

    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves memory, slower). Default: auto-enable on MPS.",
    )
    ap.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (may OOM on smaller devices).",
    )
    ap.add_argument(
        "--mps_empty_cache_steps",
        type=int,
        default=20,
        help="For MPS, call torch.mps.empty_cache() every N optimizer steps (0 disables).",
    )

    ap.add_argument("--train_batch_size", type=int, default=None)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--grad_accum_steps", type=int, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--amp", action="store_true", help="Force AMP on (CUDA recommended).")
    ap.add_argument("--no_amp", action="store_true", help="Force AMP off.")

    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_dev", type=int, default=None)
    ap.add_argument("--limit_test", type=int, default=None)

    ap.add_argument("--train_on", type=str, choices=["train", "train+dev"], default="train")
    ap.add_argument("--out_dir", type=str, default="artifacts/xlmr_best")
    ap.add_argument(
        "--hf_home",
        type=str,
        default="artifacts/hf_home",
        help="Project-local Hugging Face cache root (avoids using global caches).",
    )

    ap.add_argument("--predict_test", action="store_true")
    ap.add_argument("--out_tsv", type=str, default="artifacts/pred_xlmr_test.tsv")
    ap.add_argument(
        "--predict_only",
        action="store_true",
        help="Skip training and load model/tokenizer from --out_dir for dev/test runs.",
    )

    args = ap.parse_args()

    preset = PRESETS[args.preset]
    max_length = int(args.max_length or preset.max_length)
    epochs = int(args.epochs or preset.epochs)
    used_default_train_batch = args.train_batch_size is None
    used_default_accum = args.grad_accum_steps is None
    train_batch_size = int(args.train_batch_size or preset.train_batch_size)
    grad_accum_steps = int(args.grad_accum_steps or preset.grad_accum_steps)
    lr = float(args.lr or preset.lr)

    seed_everything(args.seed, seed_torch=True)
    ensure_dir("artifacts")

    device_info: DeviceInfo
    if args.device == "auto":
        device_info = pick_device(None)
    else:
        device_info = pick_device(args.device)

    hf_home = Path(args.hf_home)
    transformers_cache = hf_home / "transformers"
    hf_home.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    # Keep caches inside the project directory.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch_device = torch.device(device_info.device)

    default_amp = device_info.backend == "cuda"
    use_amp = default_amp
    if args.amp:
        use_amp = True
    if args.no_amp:
        use_amp = False

    print("== Device ==")
    print(f"backend={device_info.backend}  device={torch_device}  amp={use_amp}")

    if device_info.backend == "mps" and used_default_train_batch:
        # MPS is often more memory-constrained/fragmentation-prone. Reduce per-step batch by default.
        train_batch_size = max(1, train_batch_size // 2)
        if used_default_accum:
            grad_accum_steps *= 2
        print(f"Adjusted for MPS: train_batch_size={train_batch_size}  grad_accum_steps={grad_accum_steps}")

    model_source = args.out_dir if args.predict_only else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, cache_dir=str(transformers_cache))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=2, cache_dir=str(transformers_cache)
    )
    model.to(torch_device)
    model.config.use_cache = False
    auto_gc = device_info.backend == "mps"
    enable_gc = (args.gradient_checkpointing or auto_gc) and not args.no_gradient_checkpointing
    if enable_gc:
        model.gradient_checkpointing_enable()
        print("gradient_checkpointing=on")
    else:
        print("gradient_checkpointing=off")

    dev_examples = read_tsv(args.dev_tsv, has_label=True)
    if args.limit_dev:
        dev_examples = dev_examples[: int(args.limit_dev)]
    dev_ds = TorchDataset(dev_examples)
    dev_loader = DataLoader(
        dev_ds,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(tokenizer=tokenizer, max_length=max_length, with_labels=True),
    )

    if args.predict_only:
        print(f"predict_only=on (loaded from: {args.out_dir})")
        ev = evaluate(model=model, dataloader=dev_loader, device=torch_device)
        print_report(title="DEV (xlmr) predict_only", y_true=ev.y_true, y_pred=ev.y_pred, langs=ev.langs)

        if args.predict_test:
            test_examples = read_tsv(args.test_tsv, has_label=False)
            if args.limit_test:
                test_examples = test_examples[: int(args.limit_test)]
            test_ds = TorchDataset(test_examples)
            test_loader = DataLoader(
                test_ds,
                batch_size=int(args.eval_batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=make_collate(tokenizer=tokenizer, max_length=max_length, with_labels=False),
            )
            ids, preds = predict(model=model, dataloader=test_loader, device=torch_device)
            write_predictions_tsv(ids, preds, args.out_tsv)
            print(f"Wrote test predictions to: {args.out_tsv}")
        return

    train_examples = read_tsv(args.train_tsv, has_label=True)
    if args.limit_train:
        train_examples = train_examples[: int(args.limit_train)]

    if args.train_on == "train+dev":
        train_examples = list(train_examples) + list(dev_examples)
        print("NOTE: train_on=train+dev; dev evaluation/checkpoint selection is skipped.")

    train_ds = TorchDataset(train_examples)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate(tokenizer=tokenizer, max_length=max_length, with_labels=True),
    )

    total_updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = max(1, total_updates_per_epoch * epochs)
    warmup_steps = int(total_steps * float(args.warmup_ratio))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_info.backend == "cuda")

    best_f1 = -1.0
    best_state: Optional[dict[str, torch.Tensor]] = None

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", dynamic_ncols=True)
        accum = 0
        epoch_loss = 0.0
        t0 = time.time()

        for batch in pbar:
            batch = to_device(batch, torch_device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += float(loss.detach().cpu().item())
            accum += 1

            if accum % grad_accum_steps == 0:
                if float(args.max_grad_norm) > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if device_info.backend == "mps" and int(args.mps_empty_cache_steps) > 0:
                    if global_step % int(args.mps_empty_cache_steps) == 0:
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass

            if global_step > 0:
                pbar.set_postfix(loss=epoch_loss / max(1, (accum / grad_accum_steps)))

        if accum % grad_accum_steps != 0:
            if float(args.max_grad_norm) > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if device_info.backend == "mps" and int(args.mps_empty_cache_steps) > 0:
                if global_step % int(args.mps_empty_cache_steps) == 0:
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

        dt = time.time() - t0
        print(f"\nEpoch {epoch} done: loss={epoch_loss:.4f}  steps={global_step}  time={dt:.1f}s")

        if args.train_on == "train":
            ev = evaluate(model=model, dataloader=dev_loader, device=torch_device)
            print_report(title=f"DEV (xlmr) epoch={epoch}", y_true=ev.y_true, y_pred=ev.y_pred, langs=ev.langs)
            if ev.macro_f1 > best_f1:
                best_f1 = ev.macro_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"New best: f1_macro={best_f1:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_on == "train":
        if best_state is None:
            raise RuntimeError("No checkpoint was selected (unexpected).")
        model.load_state_dict(best_state)
        print(f"\nSaving best checkpoint with dev f1_macro={best_f1:.4f} to: {out_dir}")
    else:
        print(f"\nSaving final checkpoint to: {out_dir}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if args.predict_test:
        test_examples = read_tsv(args.test_tsv, has_label=False)
        if args.limit_test:
            test_examples = test_examples[: int(args.limit_test)]
        test_ds = TorchDataset(test_examples)
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate(tokenizer=tokenizer, max_length=max_length, with_labels=False),
        )
        ids, preds = predict(model=model, dataloader=test_loader, device=torch_device)
        write_predictions_tsv(ids, preds, args.out_tsv)
        print(f"Wrote test predictions to: {args.out_tsv}")


if __name__ == "__main__":
    main()
