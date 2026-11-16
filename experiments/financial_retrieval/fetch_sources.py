"""Download pinned official datasets for local evaluation; no model API access."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCES = [
    ("finqa", "czyssrs/FinQA", "0f16e2867befa6840783e58be38c9efb9229d742",
     ["dataset/train.json", "dataset/dev.json", "dataset/test.json", "LICENSE"], "MIT"),
    ("multitat", "zhxlia/MULTITAT", "e35950f2d6301d51ebcdec68bbcdfe99174dbb55",
     ["dataset/dev.zh.json"], "No explicit repository license; local evaluation only, do not redistribute translations"),
    ("tatqa", "NExTplusplus/TAT-QA", "870accc41953dcde885aabeb963d94aabdc0fbc3",
     ["dataset_raw/tatqa_dataset_dev.json", "README.md"], "Dataset: CC-BY-4.0"),
]


def main():
    raw = HERE / "raw"; raw.mkdir(exist_ok=True)
    manifest = []
    for name, repo, revision, files, license in SOURCES:
        (raw / f"{name}_commit.txt").write_text(revision, encoding="utf-8")
        for relative in files:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{relative}"
            target = raw / (name + "_" + relative.replace("/", "_"))
            if not target.exists():
                request = urllib.request.Request(url, headers={"User-Agent": "FinSight-public-evaluation"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    target.write_bytes(response.read())
            manifest.append(dict(dataset=name, repository="https://github.com/" + repo, url=url,
                                 revision=revision, license=license, local_file=str(target.relative_to(HERE)),
                                 sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size))
            print(name, relative, target.stat().st_size, flush=True)
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
