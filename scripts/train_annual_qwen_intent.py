"""Train a local Qwen3 intent classifier and export CPU INT8 ONNX."""

import argparse
import gc
import hashlib
import json
import tempfile
from itertools import cycle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, Qwen3Model

from backend.annual_reports.intent import LABELS, MAX_TOKENS, normalize_intent_question
from scripts import annual_intent_boundary_data, annual_intent_v3_data
from scripts.train_annual_intent import (
    TRAIN,
    TRAIN_METRICS,
    V2_TRAIN,
    V2_VALIDATION,
    calibrate_logits,
    dataset,
)

MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
WEIGHTS_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
COMPANIES = ("华星科技", "江海实业", "星河能源", "晨光材料", "青原电气", "山川设备", "东湾信息")
PAIRED_PROFILE = "rslora_paired_int8"


class QwenFeatures(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        # Build the causal mask explicitly so ONNX can trace dynamic sequence lengths.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        blocked = (positions[None, :, None] < positions[None, None, :]) | (attention_mask[:, None, :] == 0)
        dtype = self.encoder.get_input_embeddings().weight.dtype
        mask = torch.zeros_like(blocked, dtype=dtype).masked_fill(blocked, torch.finfo(dtype).min)
        hidden = self.encoder(input_ids=input_ids, attention_mask={"full_attention": mask.unsqueeze(1)},
                              use_cache=False).last_hidden_state
        last = (attention_mask.sum(dim=1) - 1).clamp_min(0)
        return hidden.gather(1, last[:, None, None].expand(-1, 1, hidden.shape[-1])).squeeze(1)


def export_backbone(backbone, fp32):
    model = QwenFeatures(backbone).eval().requires_grad_(False)
    sample = torch.ones((1, 12), dtype=torch.int64)
    print("Exporting the Qwen backbone to ONNX", flush=True)
    torch.onnx.export(model, (sample, sample), str(fp32), input_names=["input_ids", "attention_mask"],
                      output_names=["features"], dynamic_axes={"input_ids": {0: "batch", 1: "tokens"},
                      "attention_mask": {0: "batch", 1: "tokens"}, "features": {0: "batch"}},
                      external_data=True, opset_version=17, dynamo=False)


def quantize_features(fp32, artifact):
    quantized = artifact / "features.onnx"
    print("Quantizing Qwen linear layers to INT8", flush=True)
    # ORT shape inference makes a second copy of the full model in a temporary directory.
    previous_tempdir = tempfile.tempdir
    try:
        tempfile.tempdir = str(fp32.parent.resolve())
        quantize_dynamic(fp32, quantized, op_types_to_quantize=["MatMul"], per_channel=True,
                         reduce_range=True, weight_type=QuantType.QInt8, use_external_data_format=True)
    finally:
        tempfile.tempdir = previous_tempdir
    return quantized


def encode(session, data, label):
    ids, mask, targets = (item.numpy() for item in data.tensors)
    lengths = mask.sum(axis=1)
    batches = np.argsort(lengths)
    feature_batches = []
    for offset in range(0, len(batches), 32):
        index = batches[offset:offset + 32]
        length = int(lengths[index].max())
        feature_batches.append(session.run(["features"], {"input_ids": ids[index, :length],
                                                          "attention_mask": mask[index, :length]})[0])
        if (offset + 32) % 256 == 0 or offset + 32 >= len(batches):
            print(f"{label}: {min(offset + 32, len(batches))}/{len(batches)}", flush=True)
    vectors = torch.from_numpy(np.concatenate(feature_batches)[np.argsort(batches)])
    return nn.functional.normalize(vectors), torch.from_numpy(targets)


def attach_head(features_path, head, *, normalize=True):
    graph = onnx.load_model(features_path, load_external_data=False)
    graph.graph.initializer.extend([
        numpy_helper.from_array(head.weight.detach().numpy(), "intent_weight"),
        numpy_helper.from_array(head.bias.detach().numpy(), "intent_bias"),
    ])
    head_input = "features"
    if normalize:
        head_input = "intent_features"
        graph.graph.node.append(helper.make_node("LpNormalization", ["features"], [head_input], axis=1, p=2))
    graph.graph.node.append(
        helper.make_node("Gemm", [head_input, "intent_weight", "intent_bias"], ["logits"], transB=1)
    )
    del graph.graph.output[:]
    graph.graph.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(LABELS)]))
    model_path = features_path.with_name("model.onnx")
    onnx.save_model(graph, model_path)
    features_path.unlink()
    return model_path


def expected_calibration_error(logits, labels):
    confidence, predictions = logits.softmax(dim=-1).max(dim=-1)
    bins = (confidence * 10).long().clamp(max=9)
    errors = (predictions == labels).float() - confidence
    return float(torch.zeros(10).scatter_add_(0, bins, errors).abs().sum() / labels.numel())


def fit_temperature(logits, labels):
    temperatures = torch.linspace(-2, 2, 201).exp()
    scaled = logits.unsqueeze(0) / temperatures[:, None, None]
    targets = labels[None, :, None].expand(len(temperatures), -1, -1)
    losses = -scaled.log_softmax(dim=-1).gather(-1, targets).mean(dim=(1, 2))
    best = int(losses.argmin())
    temperature = float(temperatures[best])
    return temperature, {"calibration_temperature": temperature,
                         "calibration_nll_before": float(nn.functional.cross_entropy(logits, labels)),
                         "calibration_nll_after": float(losses[best]),
                         "calibration_ece_before": expected_calibration_error(logits, labels),
                         "calibration_ece_after": expected_calibration_error(logits / temperature, labels)}


def calibrate_artifact(model_path, questions, validation, options, *, temperature_scaling=False):
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    ids, masks, labels = validation.tensors
    logits = []
    # Dynamic INT8 scales depend on the input batch. Match the runtime's single-question shape.
    for tokens, mask in zip(ids.numpy(), masks.numpy(), strict=True):
        length = int(mask.sum())
        inputs = {"input_ids": tokens[None, :length], "attention_mask": mask[None, :length]}
        logits.append(session.run(["logits"], inputs)[0][0])
    del session
    logits = torch.from_numpy(np.stack(logits))
    details = {}
    if temperature_scaling:
        temperature, details = fit_temperature(logits, labels)
        logits = logits / temperature
        graph = onnx.load_model(model_path, load_external_data=False)
        # Fold the inverse temperature into the head to keep the ONNX runtime contract unchanged.
        for tensor in graph.graph.initializer:
            if tensor.name in {"intent_weight", "intent_bias"}:
                tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor) / temperature, tensor.name))
        onnx.save_model(graph, model_path)
    return {**calibrate_logits(questions, labels, logits, minimum_confidence=0.5), **details}


def pair_margin_loss(logits, labels):
    """Penalize the weaker side of each direct/open pair until both margins reach one."""
    direct_label = labels[:, 0, None]
    open_label = LABELS.index("open")
    direct, open_question = logits.unbind(dim=1)
    direct_margin = direct.gather(1, direct_label).squeeze(1) - direct[:, open_label]
    open_margin = open_question[:, open_label] - open_question.gather(1, direct_label).squeeze(1)
    return (1.0 - torch.minimum(direct_margin, open_margin)).relu().mean()


def read_adapter_details(adapter_dir, profile):
    details = json.loads((adapter_dir / "intent_training.json").read_text(encoding="utf-8"))
    if (details["base_model"], details["revision"], details["training_profile"]) != (MODEL, REVISION, profile):
        raise ValueError("Adapter checkpoint does not match the selected Qwen base and training profile")
    return details


def train_lora(checkpoint, training, questions, validation, device, epochs, *, rslora=False, adapter_dir=None,
               paired_data=None, init_adapter=None):
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    backbone = Qwen3Model.from_pretrained(checkpoint, local_files_only=True, dtype=dtype,
                                          attn_implementation="sdpa").to(device)
    first_layer = len(backbone.layers) - 12
    target_modules = [f"layers.{index}.self_attn.{name}" for index in range(first_layer, len(backbone.layers))
                      for name in ("q_proj", "v_proj")]
    rank, alpha = (16, 8) if rslora else (8, 16)
    if init_adapter is None:
        encoder = get_peft_model(
            backbone, LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules, use_rslora=rslora))
    else:
        read_adapter_details(init_adapter, "rslora_int8")
        encoder = PeftModel.from_pretrained(backbone, init_adapter, is_trainable=True, local_files_only=True)
    head = nn.Linear(backbone.config.hidden_size, len(LABELS)).to(device)
    if init_adapter is not None:
        head.load_state_dict(load_file(init_adapter / "intent_head.safetensors"))
    model = QwenFeatures(encoder)
    adapters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    lr_scale = 0.1 if init_adapter is not None else 1.0
    optimizer = torch.optim.AdamW([{"params": adapters, "lr": 3e-4 * lr_scale},
                                   {"params": head.parameters(), "lr": 1e-3 * lr_scale}])
    training_loader = DataLoader(training, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=32)
    if paired_data is not None:
        pair_training, pair_selection = paired_data
        pair_loader = DataLoader(pair_training, batch_size=8, shuffle=True)
        pair_validation_loader = DataLoader(pair_selection, batch_size=16)
    trainable = sum(parameter.numel() for parameter in adapters) + sum(p.numel() for p in head.parameters())
    print(f"Training {trainable:,} LoRA and head parameters on {device} for {epochs} epochs", flush=True)

    def batch_logits(ids, mask):
        length = int(mask.sum(dim=1).max())
        features = model(ids[:, :length].to(device), mask[:, :length].to(device))
        return head(features.float())

    best_rank = ()
    for epoch in range(0 if paired_data is not None else 1, epochs + 1):
        if epoch:
            model.train()
            pair_batches = cycle(pair_loader) if paired_data is not None else None
            for step, (ids, mask, labels) in enumerate(training_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(batch_logits(ids, mask), labels.to(device))
                if pair_batches is not None:
                    pair_ids, pair_mask, pair_labels = next(pair_batches)
                    pair_logits = batch_logits(pair_ids.flatten(0, 1), pair_mask.flatten(0, 1))
                    pair_targets = pair_labels.to(device)
                    loss = (loss + 0.5 * nn.functional.cross_entropy(pair_logits, pair_targets.flatten())
                            + 0.25 * pair_margin_loss(pair_logits.reshape(-1, 2, len(LABELS)), pair_targets))
                loss.backward()
                nn.utils.clip_grad_norm_([*adapters, *head.parameters()], 1.0)
                optimizer.step()
                if step % 32 == 0:
                    print(f"Epoch {epoch}/{epochs}, batch {step}/{len(training_loader)}, loss {loss.item():.4f}", flush=True)
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([batch_logits(ids, mask).cpu() for ids, mask, _ in validation_loader])
        scores = calibrate_logits(questions, validation.tensors[-1], logits, minimum_confidence=0.5)
        fast_routes = scores["validation_correct_fast_routes"]
        epoch_rank = (fast_routes, -scores["validation_open_as_fixed"], scores["validation_accuracy"])
        if paired_data is not None:
            with torch.inference_mode():
                pair_logits = torch.cat([batch_logits(ids.flatten(0, 1), mask.flatten(0, 1)).cpu()
                                         for ids, mask, _ in pair_validation_loader])
            pair_prediction = pair_logits.argmax(dim=-1).reshape(-1, 2)
            paired_correct = int((pair_prediction == pair_selection.tensors[-1]).all(dim=1).sum())
            scores["validation_pairs_both_correct"] = paired_correct
            if epoch == 0:
                coverage_floor = fast_routes - 2
            epoch_rank = (fast_routes >= coverage_floor, paired_correct, *epoch_rank)
        if epoch_rank > best_rank:
            best_rank = epoch_rank
            best_adapter = {name: tensor.detach().cpu().clone()
                            for name, tensor in get_peft_model_state_dict(encoder).items()}
            best_head = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            selected_epoch = epoch
            selected_scores = scores
        print(f"Epoch {epoch}/{epochs} validation: {json.dumps(scores)}", flush=True)

    set_peft_model_state_dict(encoder, best_adapter)
    head.load_state_dict(best_head)
    details = {"training_profile": "rslora_int8" if rslora else "lora_int8", "trainable_parameters": trainable,
               "lora_rank": rank, "lora_alpha": alpha, "lora_target_layers": len(target_modules) // 2,
               "epochs_trained": epochs, "selected_epoch": selected_epoch}
    if paired_data is not None:
        details.update(training_profile=PAIRED_PROFILE, training_pairs=len(pair_training),
                       selection_pairs=len(pair_selection), paired_selection=selected_scores)
    if adapter_dir is not None:
        from safetensors.torch import save_file

        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=adapter_dir.parent) as temporary:
            saved = Path(temporary) / "checkpoint"
            encoder.save_pretrained(saved, safe_serialization=True)
            save_file(best_head, saved / "intent_head.safetensors")
            checkpoint_details = {"base_model": MODEL, "revision": REVISION, "training_questions": len(training),
                                  "selection_questions": len(validation), **details}
            (saved / "intent_training.json").write_text(json.dumps(checkpoint_details, indent=2), encoding="utf-8")
            saved.rename(adapter_dir)
        print(f"Saved reusable adapter and classifier to {adapter_dir}", flush=True)
    return encoder, head, details


def load_lora(checkpoint, adapter_dir, *, rslora, paired=False):
    from peft import PeftModel
    from safetensors.torch import load_file

    profile = PAIRED_PROFILE if paired else "rslora_int8" if rslora else "lora_int8"
    details = read_adapter_details(adapter_dir, profile)
    backbone = Qwen3Model.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32,
                                          attn_implementation="eager")
    encoder = PeftModel.from_pretrained(backbone, adapter_dir, local_files_only=True)
    head = nn.Linear(backbone.config.hidden_size, len(LABELS))
    head.load_state_dict(load_file(adapter_dir / "intent_head.safetensors"))
    return encoder.merge_and_unload(safe_merge=True), head, details


def build_pair_dataset(tokenizer, pairs, years, companies=COMPANIES):
    rows = []
    labels = []
    for operation, direct, restricted in pairs:
        for year in years:
            for index, metric in enumerate(TRAIN_METRICS):
                values = {"year": year, "metric": metric, "company": companies[(year + index) % len(companies)]}
                rows.extend(normalize_intent_question(template.format(**values)) for template in (direct, restricted))
                labels.append((LABELS.index(operation), LABELS.index("open")))
    encoded = tokenizer(rows, padding=True, max_length=MAX_TOKENS, truncation=True, return_tensors="pt")
    return rows, TensorDataset(encoded.input_ids.reshape(len(labels), 2, -1),
                               encoded.attention_mask.reshape(len(labels), 2, -1), torch.tensor(labels))


def build_datasets(tokenizer, *, rslora):
    train_templates = {label: TRAIN[label] + V2_TRAIN[label] + annual_intent_v3_data.TRAIN[label] for label in LABELS}
    dev_templates = {label: V2_VALIDATION[label] + annual_intent_v3_data.VALIDATION[label] for label in LABELS}
    calibration_templates = dev_templates
    if rslora:
        train_templates = {label: templates + annual_intent_boundary_data.pair_templates(
            annual_intent_boundary_data.TRAIN_PAIRS, label) for label, templates in train_templates.items()}
        calibration_templates = {label: templates[1::2] + annual_intent_boundary_data.pair_templates(
            annual_intent_boundary_data.CALIBRATION_PAIRS, label) for label, templates in dev_templates.items()}
        dev_templates = {label: templates[::2] for label, templates in dev_templates.items()}
    train = dataset(tokenizer, train_templates, (2018, 2021, 2022, 2025), COMPANIES, normalize=normalize_intent_question)
    selection = dataset(tokenizer, dev_templates, (2019, 2023), normalize=normalize_intent_question)
    calibration = (dataset(tokenizer, calibration_templates, (2019, 2023), normalize=normalize_intent_question)
                   if rslora else selection)
    return train, selection, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the pinned Qwen3-0.6B-Base checkpoint")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--lora", action="store_true", help="Fine-tune attention adapters and the classifier head")
    parser.add_argument("--rslora", action="store_true", help="Use rank-stabilized adapters, boundary data and calibration")
    parser.add_argument("--paired", action="store_true", help="Continue rsLoRA using an intent margin on matched pairs")
    parser.add_argument("--init-adapter", type=Path, help="Starting rsLoRA checkpoint for a new paired training run")
    parser.add_argument("--adapter-dir", type=Path, help="Save the trained adapter, or reload it to retry export")
    parser.add_argument("--train-only", action="store_true", help="Write a new adapter checkpoint without exporting ONNX")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=4, help="Number of LoRA training epochs")
    args = parser.parse_args()
    args.rslora = args.rslora or args.paired
    use_lora = args.lora or args.rslora
    if args.adapter_dir is not None and not use_lora:
        parser.error("--adapter-dir requires --lora or --rslora")
    if args.train_only and (args.adapter_dir is None or args.adapter_dir.exists()):
        parser.error("--train-only requires a new --adapter-dir")
    if args.init_adapter is not None and not args.paired:
        parser.error("--init-adapter requires --paired")
    if args.paired and args.init_adapter is None and not (args.adapter_dir and args.adapter_dir.is_dir()):
        parser.error("paired training requires --init-adapter")
    if args.out is None:
        suffix = "-rslora-paired" if args.paired else "-rslora" if args.rslora else "-lora" if args.lora else ""
        args.out = Path(f".annual-models/annual-intent-qwen{suffix}")
    if args.adapter_dir is not None:
        adapter_path, output_path = args.adapter_dir.resolve(), args.out.resolve()
        if adapter_path == output_path or adapter_path in output_path.parents or output_path in adapter_path.parents:
            parser.error("--adapter-dir and --out cannot overlap")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.out.exists() and not args.train_only:
        parser.error("output already exists; choose a new --out directory")
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(MODEL, revision=REVISION,
                          allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]))
    with (checkpoint / "model.safetensors").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != WEIGHTS_SHA256:
            raise ValueError("Qwen weights do not match the pinned checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(47)
    torch.set_num_threads(4)
    (_, training), (questions, validation), (calibration_questions, calibration) = build_datasets(
        tokenizer, rslora=args.rslora)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as folder:
        work = Path(folder)
        artifact = work / "artifact"
        artifact.mkdir()
        if args.adapter_dir is not None and args.adapter_dir.exists():
            backbone, head, training_details = load_lora(
                checkpoint, args.adapter_dir, rslora=args.rslora, paired=args.paired)
        elif use_lora:
            device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
                                  else args.device)
            paired_data = None
            if args.paired:
                _, paired_training = build_pair_dataset(
                    tokenizer, annual_intent_boundary_data.HARD_TRAIN_PAIRS, (2018, 2021, 2022, 2025))
                _, paired_selection = build_pair_dataset(
                    tokenizer, annual_intent_boundary_data.HARD_SELECTION_PAIRS, (2019, 2023), ("示例公司",))
                paired_data = paired_training, paired_selection
            encoder, head, training_details = train_lora(
                checkpoint, training, questions, validation, device, args.epochs, rslora=args.rslora,
                adapter_dir=args.adapter_dir, paired_data=paired_data, init_adapter=args.init_adapter)
            if args.train_only:
                print(json.dumps(training_details, indent=2))
                return
            encoder.to(device="cpu", dtype=torch.float32)
            backbone = encoder.merge_and_unload(safe_merge=True)
            backbone.config._attn_implementation = "eager"
            head = head.cpu()
            del encoder
        else:
            backbone = Qwen3Model.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32,
                                                  attn_implementation="eager")
            training_details = {"training_profile": "frozen_int8_backbone"}
        fp32 = work / "features-fp32.onnx"
        export_backbone(backbone, fp32)
        del backbone
        gc.collect()
        features_path = quantize_features(fp32, artifact)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        if not use_lora:
            session = ort.InferenceSession(str(features_path), options, providers=["CPUExecutionProvider"])
            train_x, train_y = encode(session, training, "training")
            del session

            head = nn.Linear(train_x.shape[1], len(LABELS))
            optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(head(train_x), train_y) + 0.0001 * head.weight.square().sum()
                loss.backward()
                return loss

            optimizer.step(closure)
        model_path = attach_head(features_path, head, normalize=not use_lora)
        print(f"Calibrating on {len(calibration)} single-question inputs", flush=True)
        scores = calibrate_artifact(model_path, calibration_questions, calibration, options,
                                    temperature_scaling=args.rslora)
        tokenizer.backend_tokenizer.no_padding()
        tokenizer.backend_tokenizer.no_truncation()
        tokenizer.backend_tokenizer.save(str(artifact / "tokenizer.json"))
        metadata = {"labels": list(LABELS), "base_model": MODEL, "revision": REVISION,
                    "training_questions": len(training), "selection_questions": len(validation),
                    **training_details, "validation_questions": len(calibration), **scores}
        (artifact / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        onnx.checker.check_model(str(model_path))
        artifact.rename(args.out)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
