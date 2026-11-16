"""Fine-tune and export an optional local annual-report intent classifier."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizerFast

from backend.annual_reports.intent import LABELS, MAX_TOKENS, guarded_metric_intent
from backend.annual_reports.workflow import METRICS
from scripts import annual_intent_v3_data

CHECKPOINTS = {
    "tinybert": ("huawei-noah/TinyBERT_4L_zh", "152bf15d86715b41dd89c1e03cf5664963d9b005",
                 "b4aec782e4a56727bb6ba1dcac8e6c3319460e58e11254f7e43a2dccee0c19c1"),
    "minirbt": ("hfl/minirbt-h256", "446e1b4012b10946f479563005700c54c9db3f7c",
                "6728ca573fd5197f28198eb7e6119a76e4d5dc626f953fca8c0dc5ba2798f9fc"),
    "mmbert": ("jhu-clsp/mmBERT-small", "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
               "be7c9ce2af947fb5bb17a3bc919552f898c5ef4164bf0a7190cf25a0f98f85b6"),
}
TRAIN_METRICS = ("营收", "营业收入", "经营现金流", "归母净利润", "经营性现金流", "归属于上市公司股东的净利润")

TRAIN = {
    "lookup": (
        "{year}年{metric}有多少", "想了解{year}年{metric}的大概金额", "帮我查下{year}年{metric}有多少钱",
        "{year}年{metric}的数值是什么", "{year}年{metric}到底有多少", "{year}年的{metric}是多少呀",
        "麻烦告诉我{year}年的{metric}金额", "{year}年{metric}总共有多少", "{year}年{metric}达到多少",
        "看看{year}年{metric}这个数字", "{year}年实际{metric}金额", "{year}年{metric}是多少钱",
    ),
    "yoy": (
        "{year}年{metric}相较于去年多了多少", "{year}年{metric}同比百分比", "{year}年{metric}和去年比有啥变化",
        "{year}年{metric}较上一年度降幅多少", "{year}年{metric}与上个年度相比差多少", "{year}年{metric}对比上年同期的变动",
        "{year}年的{metric}比上年高多少", "{year}年{metric}跟前一年度比减少了多少钱", "{year}年{metric}比去年少了多少",
        "{year}年{metric}比上一年增加几个百分点", "{year}年{metric}上年同期比较", "{year}年{metric}跟上一个财年比涨幅是多少",
    ),
    "open": (
        "{year}年{metric}为什么变化", "{year}年{metric}变动和供应链有什么关系", "{year}年海外业务{metric}是多少",
        "{year}年{metric}与去年的差异主要由什么造成", "{year}年{metric}增速是否可持续", "{year}年{metric}和现金回款是否一致",
        "{year}年{metric}环比多少", "{year}年{metric}占行业比重多少", "{year}年{metric}分地区披露", "总结年报中{metric}以及主要经营风险",
        "{year}年{metric}和另一家公司相差多少钱", "{year}年扣非归母净利润是多少", "明年的{metric}预计是多少", "帮我查询{year}年股价和宏观风险",
    ),
}

VALIDATION = {
    "lookup": ("{year}年{metric}大概在什么水平", "我想知道{year}年{metric}的数字", "{year}年度{metric}究竟为多少"),
    "yoy": ("{year}年{metric}对照前一个年度多了多少", "{year}年{metric}跟去年的差额有多少", "{year}年{metric}与上个财年比较涨了几个点"),
    "open": ("{year}年{metric}同比回落能归因于什么", "{year}年境外销售{metric}有多少", "{year}年{metric}相比两年前多了多少", "{year}年{metric}增长与利润含金量之间的关系"),
}

V2_TRAIN = {
    "lookup": (
        "能否告知{company}{year}年{metric}的金额", "想问{year}年度{metric}的实际数额",
        "麻烦说下{year}年{metric}究竟有多少", "{company}的{metric}在{year}年是多少钱",
        "{metric}于{year}年度大概是什么水平", "请给我{year}年{metric}的大概数字",
        "{company}{year}年{metric}的规模是什么水平", "能给我{year}年的{metric}金额吗",
        "{year}年{metric}合计有多少亿", "想知道{company}的{year}年度{metric}总数",
        "{year}年{metric}具体是多少亿元", "告诉我{company}{year}年{metric}大约有多少",
        "{metric}在{year}年总共多少", "帮我看看{year}年{metric}大约是多少钱",
        "{company}的{year}年{metric}到什么水平", "请说一下{year}年{metric}到底有多少亿",
        "{year}年度{metric}的数字有多少", "{metric}在{year}年度有多少金额",
    ),
    "yoy": (
        "和上一年度相比，{company}{year}年{metric}高了多少", "比上年同期，{year}年{metric}多出多少",
        "对照上一个财年，{year}年度{metric}低了多少", "{company}的{metric}在{year}年较上年差多少",
        "{year}年{metric}对照去年同期的差额", "{metric}于{year}年跟去年比少多少钱",
        "{year}年{metric}比上一年多出了多少亿", "较去年同期，{year}年{metric}的差额有多少",
        "{year}年{metric}较前一个年度的涨幅是多少", "想问{year}年{metric}和上年比变动有多大",
        "{company}{year}年{metric}跟去年的变动有多大", "{year}年度{metric}比上一财年高了几个点",
        "{year}年{metric}比去年多还是少", "{year}年{metric}同比差多少亿",
        "{metric}在{year}年跟上年同期相比降了多少", "能否告知{year}年{metric}比上一财年差多少",
        "{year}年的{metric}和去年差了多少钱", "与上年相比，{company}{year}年{metric}涨幅有多大",
    ),
    "open": (
        "{company}{year}年分产品的{metric}有多少", "{year}年出口业务的{metric}大概是多少钱",
        "{year}年母公司单体报表的{metric}有多少", "{metric}在{year}年上半年是什么水平",
        "{year}年扣除非经常性损益后的归母净利润有多少亿", "{company}{year}年扣非归母净利润比去年多了多少",
        "{year}年境外{metric}跟上年相比差多少", "{year}年经营现金流扣除资本开支还有多少",
        "比上年同期，{year}年{metric}多出多少以及为什么", "{year}年{metric}多出了多少是因为涨价",
        "和前两年相比，{year}年{metric}变化有多大", "{metric}于{year}年末季度和上季度比低了多少",
        "{company}{year}年{metric}和另一家公司相差多少", "{year}年{metric}实际增量是否可持续",
        "{year}年{metric}同比高了多少，顺便说下存货情况", "{year}年{metric}中并购贡献有多少",
        "{year}年{metric}扣掉汇率影响是多少", "{year}年{metric}达到多少才能覆盖债务",
    ),
}

V2_VALIDATION = {
    "lookup": ("能否告知{year}年{metric}到底是多少亿", "{company}的{metric}于{year}年有多少啊",
               "{year}年度{metric}总数有多少", "{metric}在{year}年具体是什么金额"),
    "yoy": ("与去年同期比较，{year}年{metric}多出几亿", "{company}的{metric}于{year}年度比上一年高多少",
            "{year}年{metric}对照前一个年度的差额为多少", "比上年，{year}年{metric}低了多少亿"),
    "open": ("{company}{year}年海外板块的{metric}有多少", "{year}年{metric}相比上年大概少多少以及主要原因",
             "{metric}于{year}年第二季度有多少", "{year}年归母净利润扣非前后分别为多少"),
}


def dataset(tokenizer, templates, years, companies=("示例公司",), *, normalize=None):
    rows = [(template.format(year=year, metric=metric, company=companies[(year + metric_index) % len(companies)]), index)
            for index, label in enumerate(LABELS) for template in templates[label]
            for year in years for metric_index, metric in enumerate(TRAIN_METRICS)]
    if normalize is not None:
        rows = [(normalize(text), index) for text, index in rows]
    rows = list(dict.fromkeys(rows))
    questions = [text for text, _ in rows]
    encoded = tokenizer(questions, padding=True, max_length=MAX_TOKENS,
                        truncation=True, return_tensors="pt")
    return questions, TensorDataset(encoded.input_ids, encoded.attention_mask, torch.tensor([index for _, index in rows]))


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.mean_pooling = getattr(encoder.config, "classifier_pooling", "cls") == "mean"
        self.head = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.mean_pooling:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return self.head((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
        return self.head(hidden[:, 0])


def calibrate(model, questions, validation):
    model.eval()
    with torch.inference_mode():
        input_ids, attention_mask, labels = validation.tensors
        return calibrate_logits(questions, labels, model(input_ids, attention_mask))


def calibrate_logits(questions, labels, logits, *, minimum_confidence=0.65):
    confidence, prediction = logits.softmax(dim=-1).max(dim=-1)
    eligible = [(float(score), proposed == expected)
                for question, score, proposed, expected in zip(questions, confidence.tolist(),
                                                               prediction.tolist(), labels.tolist(), strict=True)
                if guarded_metric_intent(question, {"示例公司"}, METRICS, lambda _, op=LABELS[proposed]: op)]
    errors = [score for score, correct in eligible if not correct]
    threshold = max(minimum_confidence, min(1.0, max(errors, default=0) + 0.0001))
    open_label = LABELS.index("open")
    open_as_fixed = (labels == open_label) & (prediction != open_label) & (confidence >= threshold)
    return {"threshold": threshold, "validation_accuracy": float((prediction == labels).float().mean()),
            "validation_correct_fast_routes": sum(correct and score >= threshold for score, correct in eligible),
            "validation_open_as_fixed": int(open_as_fixed.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="tinybert")
    parser.add_argument("--profile", choices=("baseline", "v2", "v3"), default="baseline")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the selected pinned checkpoint")
    args = parser.parse_args()
    domain_profile = args.profile != "baseline"
    use_v3 = args.profile == "v3"
    if domain_profile and args.model != "minirbt":
        parser.error("domain training profiles are for MiniRBT")
    base_model, revision, weights_sha256 = CHECKPOINTS[args.model]
    if args.out is None:
        suffix = "" if args.model == "tinybert" else f"-{args.model}"
        profile_suffix = f"-{args.profile}" if domain_profile else ""
        args.out = Path(f".annual-models/annual-intent{suffix}{profile_suffix}")
    if args.out.exists():
        parser.error("output already exists; choose a new --out directory")
    source = args.checkpoint
    if source is None:
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(base_model, revision=revision,
                                        allow_patterns=["*.json", "vocab.txt", "pytorch_model.bin"]))
    with (source / "pytorch_model.bin").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != weights_sha256:
            raise ValueError("Weights do not match the pinned upstream checkpoint")
    torch.manual_seed(47)
    torch.set_num_threads(4)
    if args.model == "mmbert":
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        encoder = AutoModel.from_pretrained(source, local_files_only=True, attn_implementation="eager",
                                            reference_compile=False)
        # Keep this experiment practical on CPU while adapting the last four encoder layers.
        encoder.requires_grad_(False)
        encoder.layers[-4:].requires_grad_(True)
        encoder.final_norm.requires_grad_(True)
    else:
        tokenizer = BertTokenizerFast(vocab_file=str(source / "vocab.txt"), do_lower_case=True)
        encoder = BertModel.from_pretrained(source, add_pooling_layer=False, attn_implementation="eager")
    model = Classifier(encoder)
    templates, dev_templates = TRAIN, VALIDATION
    companies = ("示例公司",)
    train_years, dev_years = (2021, 2022, 2025), (2020,)
    if domain_profile:
        templates = {label: TRAIN[label] + V2_TRAIN[label] for label in LABELS}
        dev_templates = V2_VALIDATION
        companies = ("华星科技", "江海实业", "星河能源")
    if use_v3:
        templates = {label: templates[label] + annual_intent_v3_data.TRAIN[label] for label in LABELS}
        dev_templates = {label: V2_VALIDATION[label] + annual_intent_v3_data.VALIDATION[label] for label in LABELS}
        companies += ("晨光材料", "青原电气", "山川设备", "东湾信息")
        train_years, dev_years = (2018, 2021, 2022, 2025), (2019, 2023)
    _, train = dataset(tokenizer, templates, train_years, companies)
    validation_questions, validation = dataset(tokenizer, dev_templates, dev_years)
    loader = DataLoader(train, batch_size=32 if use_v3 else 24, shuffle=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.00008)
    loss = nn.CrossEntropyLoss(label_smoothing=0.03 if use_v3 else 0)
    best, best_rank = None, (-1, 0, -1.0)
    epochs = {"baseline": 6, "v2": 5, "v3": 4}[args.profile]
    for epoch in range(epochs):
        model.train()
        for input_ids, attention_mask, labels in loader:
            if use_v3:
                # The BERT tokenizer right-pads; trim shared padding before the forward pass.
                length = int(attention_mask.sum(dim=1).max())
                input_ids, attention_mask = input_ids[:, :length], attention_mask[:, :length]
            optimizer.zero_grad()
            loss(model(input_ids, attention_mask), labels).backward()
            optimizer.step()
        if domain_profile:
            scores = calibrate(model, validation_questions, validation)
            open_errors = scores["validation_open_as_fixed"] if use_v3 else 0
            rank = (scores["validation_correct_fast_routes"], -open_errors, scores["validation_accuracy"])
            if rank > best_rank:
                best_rank = rank
                best = {name: value.detach().clone() for name, value in model.state_dict().items()}
                selected_epoch = epoch + 1
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)

    if best is not None:
        model.load_state_dict(best)
    metadata = {"labels": list(LABELS), "base_model": base_model, "revision": revision,
                "training_profile": args.profile, "selected_epoch": selected_epoch if best is not None else epochs,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "training_questions": len(train), "validation_questions": len(validation), "epochs_trained": epochs,
                **calibrate(model, validation_questions, validation)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        artifact = Path(temporary) / "artifact"
        artifact.mkdir()
        tokenizer.backend_tokenizer.no_padding()
        tokenizer.backend_tokenizer.no_truncation()
        tokenizer.backend_tokenizer.save(str(artifact / "tokenizer.json"))
        sample = tokenizer("2024年营业收入", return_tensors="pt")
        torch.onnx.export(model, (sample.input_ids, sample.attention_mask), str(artifact / "model.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {1: "tokens"}, "attention_mask": {1: "tokens"}},
                          opset_version=17, dynamo=False)
        (artifact / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        artifact.rename(args.out)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
