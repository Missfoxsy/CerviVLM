#!/usr/bin/env python3
"""
Translate Chinese task2 (type2) descriptions to English, then compute SBERT.

Used for FO1 / cerviVLM style Chinese outputs that share a fixed pathology
template with the English evaluation set. Pipeline:

  1. Rule-based zh -> en translation (same schema as English GT)
  2. Prefer aligned English GT from a reference results file (by sample id)
  3. SBERT cosine similarity with paraphrase-multilingual-mpnet-base-v2

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/compute_task2_sbert_en.py \\
      --input eval_results/fo1_results.json \\
      --input eval_results/cerviVLM_results.json \\
      --name fo1_en --name cerviVLM_en
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Hide broken optional CUDA packages before importing transformers.
_orig_find_spec = importlib.util.find_spec


def _find_spec(name, package=None):  # type: ignore[no-untyped-def]
    if name in ("torchaudio", "flash_attn") or (
        isinstance(name, str)
        and (name.startswith("torchaudio.") or name.startswith("flash_attn."))
    ):
        return None
    return _orig_find_spec(name, package)


importlib.util.find_spec = _find_spec  # type: ignore[assignment]
try:
    import triton.ops  # noqa: F401
except Exception:
    _ops = types.ModuleType("triton.ops")
    _matmul = types.ModuleType("triton.ops.matmul_perf_model")
    _matmul.early_config_prune = lambda *a, **k: []
    _matmul.estimate_matmul_time = lambda *a, **k: 0.0
    _ops.matmul_perf_model = _matmul
    sys.modules["triton.ops"] = _ops
    sys.modules["triton.ops.matmul_perf_model"] = _matmul

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

VLM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_DEFAULT_EN_GT_CANDIDATES = (
    VLM_ROOT / "eval_results/en_gt_ref_stage2_cot_val.json",
    VLM_ROOT / "InternVL/eval_result/stage2_cot_internvl2_5-4b/results.json",
)
DEFAULT_EN_GT_REF = next(
    (p for p in _DEFAULT_EN_GT_CANDIDATES if p.exists()),
    _DEFAULT_EN_GT_CANDIDATES[0],
)
DEFAULT_SUMMARY = VLM_ROOT / "eval_results/task2_sbert.json"
STOP_TOKEN = re.compile(r"<\|im_end\|>")

ORIGIN_MAP = {
    "上皮成团": "epithelial cluster",
    "宫颈鳞状上皮中层": "cervical squamous mid layer",
    "宫颈鳞状上皮底层": "cervical squamous basal layer",
    "宫颈鳞状上皮表层": "cervical squamous superficial layer",
}
TYPE_MAP = {
    "EpitheliumNormal": "Normal",
    "ASCUS": "ASCUS",
    "ASC_H": "ASC_H",
    "LSIL": "LSIL",
    "HSIL": "HSIL",
    "Normal": "Normal",
}
FEAT_ADJ = {
    "核深染": "dark-stained",
    "核不规则": "irregular",
    "核增大": "enlarged",
    "核拥挤": "crowded",
}
ADJ_ORDER = ["enlarged", "irregular", "dark-stained", "crowded"]


def clean_text(text: Optional[str]) -> str:
    return STOP_TOKEN.sub("", text or "").strip()


def join_adjs(adjs: Sequence[str]) -> str:
    ordered = [a for a in ADJ_ORDER if a in adjs]
    for a in adjs:
        if a not in ordered:
            ordered.append(a)
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f" and {ordered[-1]}"


def _feature_body_en(feats: str) -> Optional[str]:
    """Translate Chinese feature list (e.g. '核深染，核增大') into English body."""
    parts = [p for p in feats.split("，") if p]
    adjs: List[str] = []
    high_nc = False
    koilo = False
    unknown: List[str] = []
    for p in parts:
        if p in FEAT_ADJ:
            adjs.append(FEAT_ADJ[p])
        elif p == "核浆比高":
            high_nc = True
        elif p == "挖空":
            koilo = True
        else:
            unknown.append(p)

    clauses: List[str] = []
    if adjs:
        clauses.append(f"{join_adjs(adjs)} nuclei")
    if koilo:
        clauses.append("with koilocytosis (perinuclear halo)")
    if high_nc:
        if clauses and not clauses[-1].startswith("with "):
            clauses[-1] = clauses[-1] + ", with a high ratio of nucleus to cytoplasm"
        else:
            clauses.append("with a high ratio of nucleus to cytoplasm")
    if unknown:
        clauses.append(" / ".join(unknown))
    if not clauses:
        return None

    body = clauses[0]
    for c in clauses[1:]:
        if c.startswith("with "):
            body = body + ", " + c
        else:
            body = body + " and " + c
    return body


def translate_zh_to_en(text: str) -> str:
    """Rule-based translation for Stage2 full template and Stage1 feature-only."""
    t = clean_text(text).rstrip("。")
    if not t:
        return ""

    # Stage1 align: normal / feature-only descriptions
    if t in ("无异常病理表现", "无明显异常", "无异常"):
        return "The cell shows no significant pathological abnormalities."
    m_feat = re.match(r"^表现出(.+)$", t)
    if m_feat:
        body = _feature_body_en(m_feat.group(1))
        if body is None:
            return "The cell shows no significant pathological abnormalities."
        return f"The cell shows {body}."

    m = re.match(
        r"^(阳性|阴性)宫颈细胞，细胞类型为([^，]+)，来源于([^，]+)(?:，表现出(.+))?$",
        t,
    )
    if not m:
        return f"[UNPARSED] {t}"

    pol, typ, ori, feats = m.group(1), m.group(2), m.group(3), m.group(4)
    pol_en = "Positive" if pol == "阳性" else "Negative"
    typ_en = TYPE_MAP.get(typ, typ)
    ori_en = ORIGIN_MAP.get(ori, ori)
    base = f"{pol_en} cervical cell. Type: {typ_en}. Origin: {ori_en}."

    if not feats:
        if pol == "阴性":
            return base + " The cell shows no significant pathological abnormalities."
        return base

    body = _feature_body_en(feats)
    if body is None:
        if pol == "阴性":
            return base + " The cell shows no significant pathological abnormalities."
        return base
    return base + f" The cell shows {body}."


def is_task2(row: Dict[str, Any]) -> bool:
    return (
        row.get("task_type") == "type2"
        or row.get("task_id") == 2
        or row.get("task_id_effective") == 2
    )


def get_pred_text(row: Dict[str, Any]) -> str:
    return row.get("pred_text") or row.get("prediction") or ""


def get_gt_text(row: Dict[str, Any]) -> str:
    return row.get("gt_text") or row.get("ground_truth") or ""


def load_rows(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Load json / jsonl. Returns (rows, format) where format in {json, jsonl}."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or (not text.lstrip().startswith("[") and "\n" in text.strip()):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows, "jsonl"
    return json.loads(text), "json"


def write_rows(path: Path, rows: List[Dict[str, Any]], fmt: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt == "jsonl":
        tmp.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    else:
        tmp.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    tmp.replace(path)


def load_en_gt(ref_path: Path) -> Dict[str, str]:
    rows, _ = load_rows(ref_path)
    return {
        r["id"]: r["ground_truth"]
        for r in rows
        if is_task2(r) and r.get("id") and r.get("ground_truth")
    }


def norm_for_compare(s: str) -> str:
    s = s.lower().replace(", with a high", " with a high")
    s = s.replace(" and ", " ").replace(", ", " ")
    return " ".join(sorted(s.replace(".", " ").split()))


def gt_fidelity(
    zh_gts: Iterable[str], en_gts: Iterable[str]
) -> Tuple[int, int, int, int]:
    exact = near = other = unparsed = 0
    for zh, en in zip(zh_gts, en_gts):
        tr = translate_zh_to_en(zh)
        if tr.startswith("[UNPARSED]"):
            unparsed += 1
        elif tr == en:
            exact += 1
        elif norm_for_compare(tr) == norm_for_compare(en):
            near += 1
        else:
            other += 1
    return exact, near, other, unparsed


class SBERTEncoder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        # Prefer local cache when Hub is unreachable (common on compute nodes).
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = F.normalize(emb, p=2, dim=1)
            vecs.append(emb.cpu())
        return torch.cat(vecs, 0)

    def cosine(self, cands: Sequence[str], refs: Sequence[str]) -> List[float]:
        return (self.encode(cands) * self.encode(refs)).sum(1).tolist()


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = float(sum(vals) / len(vals))
    if len(vals) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return mean, float(var**0.5)


def metrics_from_sbert_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Keys written into eval metrics.json for Type2 SBERT (English pipeline)."""
    return {
        "type2_mean_sbert": float(stats["sbert"]),
        "type2_mean_sbert_rule_gt": float(stats["sbert_rule_gt"]),
        "type2_std_sbert": float(stats.get("sbert_std", 0.0)),
        "type2_sbert_model": stats.get("model", DEFAULT_SBERT_MODEL),
        "type2_sbert_n": int(stats.get("n", 0)),
        "type2_sbert_pipeline": stats.get(
            "translation",
            "rule-based zh->en; aligned English GT preferred",
        ),
        "type2_sbert_gt_translate_exact": int(stats.get("gt_translate_exact", 0)),
        "type2_sbert_gt_translate_near": int(stats.get("gt_translate_near", 0)),
        "type2_sbert_pred_unparsed": int(stats.get("pred_unparsed", 0)),
    }


def augment_results_with_sbert(
    rows: List[Dict[str, Any]],
    *,
    en_gt: Optional[Dict[str, str]] = None,
    en_gt_ref: Optional[Path] = None,
    encoder: Optional[SBERTEncoder] = None,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    device: str = "cuda",
    batch_size: int = 64,
    language: str = "zh",
    require_aligned_gt: bool = False,
    log_prefix: str = "task2_sbert",
) -> Dict[str, Any]:
    """Mutate type2 rows with English-pipeline SBERT fields; return aggregate stats.

    zh: translate pred/GT via rules; prefer ``en_gt[id]`` as reference when present.
    en: score English pred vs English GT directly (still prefer aligned ref if given).
    """
    idxs = [i for i, r in enumerate(rows) if is_task2(r)]
    if not idxs:
        return {"n": 0, "sbert": 0.0, "sbert_rule_gt": 0.0, "skipped": True}

    if en_gt is None:
        en_gt = {}
        if en_gt_ref is not None and Path(en_gt_ref).exists():
            en_gt = load_en_gt(Path(en_gt_ref))

    missing = [rows[i].get("id") for i in idxs if rows[i].get("id") not in en_gt]
    if require_aligned_gt and missing:
        raise RuntimeError(
            f"{len(missing)} task2 ids missing in English GT ref "
            f"(e.g. {missing[:3]})"
        )
    if missing and en_gt:
        print(
            f"[{log_prefix}] WARNING: {len(missing)}/{len(idxs)} ids missing "
            f"aligned EN GT; falling back to rule/native GT",
            flush=True,
        )

    pred_en: List[str] = []
    gt_aligned: List[str] = []
    gt_rule: List[str] = []
    zh_gts_for_fidelity: List[str] = []
    en_gts_for_fidelity: List[str] = []

    for i in idxs:
        r = rows[i]
        sid = r.get("id")
        pred_raw = get_pred_text(r)
        gt_raw = get_gt_text(r)

        if language == "en":
            pe = clean_text(pred_raw)
            ge_rule = clean_text(gt_raw)
        else:
            pe = translate_zh_to_en(pred_raw)
            ge_rule = translate_zh_to_en(gt_raw)

        ge = en_gt.get(sid) if sid else None
        if not ge:
            ge = ge_rule

        r["prediction_en"] = pe
        r["ground_truth_en_rule"] = ge_rule
        r["ground_truth_en"] = ge
        pred_en.append(pe)
        gt_rule.append(ge_rule)
        gt_aligned.append(ge)

        if language != "en" and sid in en_gt:
            zh_gts_for_fidelity.append(gt_raw)
            en_gts_for_fidelity.append(en_gt[sid])

    exact = near = other = unparsed = 0
    if zh_gts_for_fidelity:
        exact, near, other, unparsed = gt_fidelity(zh_gts_for_fidelity, en_gts_for_fidelity)

    n_unparsed_pred = sum(1 for x in pred_en if x.startswith("[UNPARSED]"))
    print(
        f"[{log_prefix}] n={len(idxs)} language={language} "
        f"GT fidelity: exact={exact} near={near} other={other} unparsed={unparsed} "
        f"pred_unparsed={n_unparsed_pred}",
        flush=True,
    )

    if encoder is None:
        print(f"[{log_prefix}] loading SBERT model {sbert_model}", flush=True)
        encoder = SBERTEncoder(sbert_model, device=device)

    _orig_encode = encoder.encode

    def _encode(texts, batch_size=batch_size):  # noqa: B023
        return _orig_encode(texts, batch_size=batch_size)

    encoder.encode = _encode  # type: ignore[method-assign]

    print(f"[{log_prefix}] scoring SBERT", flush=True)
    sims_aligned = encoder.cosine(pred_en, gt_aligned)
    sims_rule = encoder.cosine(pred_en, gt_rule)
    for i, sa, sr in zip(idxs, sims_aligned, sims_rule):
        rows[i]["sbert_en"] = float(sa)
        rows[i]["sbert_en_rule_gt"] = float(sr)
        # Keep a short alias used by some older metrics dumps
        rows[i]["sbert"] = float(sa)

    mean_a, std_a = _mean_std(sims_aligned)
    mean_r, _ = _mean_std(sims_rule)
    print(
        f"[{log_prefix}] SBERT_en aligned={mean_a:.6f} rule_gt={mean_r:.6f}",
        flush=True,
    )

    return {
        "n": len(sims_aligned),
        "sbert": mean_a,
        "sbert_std": std_a,
        "sbert_rule_gt": mean_r,
        "model": sbert_model,
        "translation": (
            "english pred/gt"
            if language == "en"
            else "rule-based zh->en template; GT uses aligned English dataset GT when available"
        ),
        "gt_translate_exact": exact,
        "gt_translate_near": near,
        "gt_translate_other": other,
        "pred_unparsed": n_unparsed_pred,
        "skipped": False,
    }


def process_one(
    input_path: Path,
    name: str,
    en_gt: Dict[str, str],
    encoder: SBERTEncoder,
    summary: Dict[str, Any],
    sbert_model: str,
) -> Dict[str, Any]:
    print(f"loading {input_path}", flush=True)
    rows, fmt = load_rows(input_path)
    stats = augment_results_with_sbert(
        rows,
        en_gt=en_gt,
        encoder=encoder,
        sbert_model=sbert_model,
        language="zh",
        require_aligned_gt=True,
        log_prefix=name,
    )
    write_rows(input_path, rows, fmt)
    print(f"[{name}] wrote {input_path}", flush=True)
    stats["file"] = str(input_path.resolve())
    summary[name] = stats

    # If this looks like an eval_result dir, merge SBERT into metrics.json
    metrics_path = input_path.parent / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update(metrics_from_sbert_stats(stats))
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{name}] updated {metrics_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] WARNING: failed to update metrics.json: {exc}", flush=True)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Translate Chinese task2 outputs to English and compute SBERT."
    )
    p.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Results json/jsonl path (repeatable).",
    )
    p.add_argument(
        "--name",
        action="append",
        default=None,
        help="Summary key for each --input (same order). Default: stem + '_en'.",
    )
    p.add_argument(
        "--en-gt-ref",
        type=Path,
        default=DEFAULT_EN_GT_REF,
        help="English results file used as aligned GT by sample id.",
    )
    p.add_argument(
        "--sbert-model",
        default=DEFAULT_SBERT_MODEL,
        help="Sentence embedding model name or local path.",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Where to merge/write aggregate SBERT summary json.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inputs: List[Path] = [p.resolve() if p.is_absolute() else (VLM_ROOT / p).resolve() for p in args.input]
    if args.name is None:
        names = [p.stem + "_en" for p in inputs]
    else:
        names = list(args.name)
        if len(names) != len(inputs):
            raise SystemExit("--name count must match --input count")

    en_gt_ref = (
        args.en_gt_ref
        if args.en_gt_ref.is_absolute()
        else (VLM_ROOT / args.en_gt_ref)
    ).resolve()
    summary_path = (
        args.summary if args.summary.is_absolute() else (VLM_ROOT / args.summary)
    ).resolve()

    print(f"loading English GT from {en_gt_ref}", flush=True)
    en_gt = load_en_gt(en_gt_ref)
    print(f"English GT n={len(en_gt)}", flush=True)

    print(f"loading SBERT model {args.sbert_model}", flush=True)
    encoder = SBERTEncoder(args.sbert_model, device=args.device)
    # expose batch size without changing class signature much
    _orig_encode = encoder.encode

    def _encode(texts, batch_size=args.batch_size):
        return _orig_encode(texts, batch_size=batch_size)

    encoder.encode = _encode  # type: ignore[method-assign]

    summary: Dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("model", args.sbert_model)
    summary.setdefault("metric", "cosine")
    summary.setdefault("pooling", "mean")

    for path, name in zip(inputs, names):
        process_one(
            input_path=path,
            name=name,
            en_gt=en_gt,
            encoder=encoder,
            summary=summary,
            sbert_model=args.sbert_model,
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_path)
    print(f"wrote summary {summary_path}", flush=True)


if __name__ == "__main__":
    main()
