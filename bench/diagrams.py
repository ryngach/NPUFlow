#!/usr/bin/env python3
"""Mermaid component diagrams of every architecture variant in bench/variants.yaml.

Each diagram is built from the instantiated model, so channel counts and shapes are
the real ones. Nodes that differ from B0 are highlighted; components that a variant
removes are listed under its diagram. Measured device latency comes from
results/speed.jsonl.

    .venv/bin/python bench/diagrams.py            # writes docs/architecture.md
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import EdgeFlowNet, ModelConfig  # noqa: E402

VARIANTS = ROOT / "bench" / "variants.yaml"
SPEED = ROOT / "results" / "speed.jsonl"
OUT = ROOT / "docs" / "architecture.md"
ACT = {"leaky": "LeakyReLU", "relu": "ReLU"}


def latency():
    """variant -> {quant: median ms} from the last successful record of each pair."""
    out = {}
    if not SPEED.exists():
        return out
    for line in SPEED.read_text().splitlines():
        r = json.loads(line)
        ms = (r.get("device") or {}).get("inference_ms_median")
        if ms is not None and "error" not in r:
            out.setdefault(r["variant"], {})[r["quant"]] = ms
    return out


def shape(ch, h, w):
    return f"{ch}×{h}×{w}"


def nodes_and_edges(cfg: ModelConfig, model: EdgeFlowNet):
    """Return ({id: label}, [(src, dst, edge label)], {subgraph: [ids]})."""
    H, W = cfg.image_size
    C = cfg.base_channels
    enc = model.enc
    s4, s8, s16, s32 = enc.out_s4_ch, enc.out_s8_ch, enc.out_s16_ch, enc.out_s32_ch
    se = " + SE" if cfg.use_se else ""
    pe = "<br/>+ поз. кодування (y, x), 1×1" if cfg.use_pos_enc else ""
    n, e, g = {}, [], {}

    def add(group, key, label):
        n[key] = label
        g.setdefault(group, []).append(key)

    add("in", "I0", f"img0 {shape(3, H, W)}")
    add("in", "I1", f"img1 {shape(3, H, W)}")

    add("enc", "E4", f"DW s2 → DW s2 → DW<br/>s4: {shape(s4, H // 4, W // 4)}{pe}")
    add("enc", "E8", f"DW s2 → DW → DW{se}<br/>s8: {shape(s8, H // 8, W // 8)}{pe}")
    add("enc", "E16", f"DW s2 ⊕α maxpool{se}<br/>s16: {shape(s16, H // 16, W // 16)}{pe}")
    add("enc", "E32", f"DW s2 ⊕α maxpool{se}<br/>s32: {shape(s32, H // 32, W // 32)}{pe}")
    e += [("I0", "E4", ""), ("I1", "E4", ""), ("E4", "E8", ""), ("E8", "E16", ""), ("E16", "E32", "")]

    norm = "l2norm, " if cfg.use_l2norm else ""
    cv32 = H // 32 + W // 32
    add("s32", "C32", f"глобальна кореляція dual-1D<br/>{norm}MatMul → {cv32} кан.")
    add("s32", "U32", f"CoarseUpdate ×{cfg.iters32}<br/>вхід {model.update32.reduce[0].in_channels} → 128 → Δflow")
    e += [("E32", "C32", "f0, f1"), ("C32", "U32", ""), ("E32", "U32", "ctx")]

    if cfg.corr16 == "2d":
        c16 = f"локальна кореляція 2D r={cfg.r16}<br/>{norm}{model.corr16.out_channels} зсувів"
    else:
        c16 = f"локальна кореляція 1D r={cfg.r16}<br/>{norm}{model.corr16.out_channels} зсувів (гориз. + верт.)"
    add("s16", "C16", c16)
    add("s16", "UP32", f"bilinear ↑ cost32<br/>{cv32} кан.")
    add("s16", "U16", f"CoarseUpdate ×{cfg.iters16}<br/>вхід {model.update16.reduce[0].in_channels} → 128 → Δflow")
    add("s16", "R16", f"refine_s16<br/>вхід {model.refine_s16.reduce[0].in_channels} → 128 → Δflow")
    e += [("E16", "C16", "f0, f1"), ("C32", "UP32", ""), ("UP32", "U16", ""), ("C16", "U16", ""),
          ("U32", "U16", "flow ×2↑"), ("U16", "R16", "flow"), ("E16", "R16", "f0, f1, ctx")]

    r8_src = "R16"
    if cfg.match8 == "global1d":
        add("s8", "M8", f"глобальна кореляція dual-1D s8<br/>{norm}MatMul → {H // 8 + W // 8} кан.")
        e += [("E8", "M8", "f0, f1"), ("M8", "R8", "")]
        if cfg.match8_softargmax:
            add("s8", "SA8", "soft-argmax<br/>Softmax → початковий потік, 2 кан.")
            e += [("M8", "SA8", ""), ("SA8", "R8", "")]
        if cfg.match8_local_r > 0:
            add("s8", "L8", f"локальна кореляція 2D r={cfg.match8_local_r}<br/>{model.corr8_local.out_channels} зсувів")
            e += [("E8", "L8", "f0, f1"), ("L8", "R8", "")]
    elif cfg.match8 == "local1d":
        add("s8", "M8", f"локальна кореляція 1D r={cfg.r8}<br/>{model.corr8.out_channels} зсувів")
        e += [("E8", "M8", "f0, f1"), ("M8", "R8", "")]
    elif cfg.match8 == "dcv":
        dil = ", ".join(map(str, cfg.dcv_dilations))
        add("s8", "M8", f"розріджена кореляція 2D r={cfg.r8}<br/>дилатації {dil} → {model.corr8.out_channels} зсувів")
        e += [("E8", "M8", "f0, f1"), ("M8", "R8", "")]
    if cfg.warp8 == "kpn" and cfg.refine8_f1:
        add("s8", "K8", f"сепарабельне KPN-зміщення f1<br/>ядро {2 * cfg.kpn_r8 + 1}+{2 * cfg.kpn_r8 + 1}, без GridSample")
        e += [("E8", "K8", "f1"), ("R16", "K8", "flow, ctx"), ("K8", "R8", "f1 зміщений")]
        f_in = "f0, ctx"
    else:
        f_in = "f0, f1, ctx" if cfg.refine8_f1 else "f0, ctx"
    add("s8", "R8", f"refine_s8<br/>вхід {model.refine_s8.reduce[0].in_channels} → 128 → Δflow")
    e += [(r8_src, "R8", "flow ×2↑"), ("E8", "R8", f_in)]

    add("s4", "R4", f"refine_s4<br/>вхід {model.refine_s4.reduce[0].in_channels} → 64 → Δflow")
    e += [("R8", "R4", "flow ×2↑"), ("E4", "R4", "f0, f1, ctx")]

    add("full", "UPF", f"bilinear ×4<br/>{shape(2, H, W)}")
    e.append(("R4", "UPF", "flow"))
    if cfg.full_refine == "conv":
        add("full", "FR", f"full_refine на {H}×{W}<br/>DW + dilated DW + 1×1, вхід flow + img0")
        e += [("UPF", "FR", ""), ("I0", "FR", "img0"), ("FR", "OUT", "")]
    elif cfg.full_refine == "pixelshuffle":
        add("full", "FR", "up_head на s4: DW → 1×1 → 32 кан.<br/>PixelShuffle ×4 → Δflow")
        e += [("R4", "FR", "flow"), ("E4", "FR", "f0"), ("UPF", "OUT", ""), ("FR", "OUT", "+ α·Δ")]
    else:
        e.append(("UPF", "OUT", ""))
    add("out", "OUT", f"потік {shape(2, H, W)}")
    return n, e, g


GROUPS = [("in", None), ("enc", "Енкодер (спільні ваги для обох кадрів) · {act}"), ("s32", "Рівень 1/32"),
          ("s16", "Рівень 1/16"), ("s8", "Рівень 1/8"), ("s4", "Рівень 1/4"), ("full", "Повна роздільність"),
          ("out", None)]


def mermaid(cfg, nodes, edges, groups, changed):
    lines = ["```mermaid", "flowchart TB"]
    for key, title in GROUPS:
        ids = groups.get(key, [])
        if not ids:
            continue
        if title:
            lines.append(f'  subgraph G_{key}["{title.format(act=ACT[cfg.act])}"]')
            lines.append("    direction TB")
        for i in ids:
            lines.append(f'    {i}["{nodes[i]}"]')
        if title:
            lines.append("  end")
    for a, b, lab in edges:
        lines.append(f"  {a} -->|{lab}| {b}" if lab else f"  {a} --> {b}")
    lines.append("  classDef changed fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000")
    if changed:
        lines.append(f"  class {','.join(changed)} changed")
    lines.append("```")
    return "\n".join(lines)


def plain(label):
    return label.replace("<br/>", ", ")


def main():
    variants = yaml.safe_load(VARIANTS.read_text())
    speed = latency()
    base_cfg = ModelConfig()
    base_nodes, _, _ = nodes_and_edges(base_cfg, EdgeFlowNet(base_cfg))

    rows, sections = [], []
    for name, overrides in variants.items():
        overrides = overrides or {}
        cfg = ModelConfig(**overrides)
        model = EdgeFlowNet(cfg)
        nodes, edges, groups = nodes_and_edges(cfg, model)
        changed = [k for k, v in nodes.items() if base_nodes.get(k) != v]
        removed = [plain(base_nodes[k]) for k in base_nodes if k not in nodes]
        params = sum(p.numel() for p in model.parameters()) / 1e6
        ms = speed.get(name, {})
        int8 = f"{ms['int8']:.2f}" if "int8" in ms else "—"
        mixed = f"{ms['mixed']:.2f}" if "mixed" in ms else "—"
        diff = ", ".join(f"`{k}: {v}`" for k, v in overrides.items()) or "базова мережа"
        rows.append(f"| [{name}](#{name.lower()}) | {diff} | {params:.3f} | {int8} | {mixed} |")
        body = [f"## {name}", "",
                f"Зміни відносно B0: {diff}. Параметрів: {params:.3f} M. "
                f"Час на MaixCAM: INT8 {int8} мс, mixed {mixed} мс.", ""]
        if cfg.act != base_cfg.act:
            body += [f"Активація {ACT[cfg.act]} в усіх блоках (у B0 — {ACT[base_cfg.act]}); вузли через це не підсвічено.", ""]
        body.append(mermaid(cfg, nodes, edges, groups, changed))
        if removed:
            body += ["", "Прибрано відносно B0: " + "; ".join(removed) + "."]
        sections.append("\n".join(body))

    header = f"""# Архітектура NPUFlow: діаграми компонентів варіантів

Згенеровано `bench/diagrams.py` з `bench/variants.yaml` і `model.py`; не редагувати вручну.
Канали й розміри взято з побудованої моделі для входу {base_cfg.image_size[1]}×{base_cfg.image_size[0]}
(ширина × висота); тензори в діаграмах — канали × висота × ширина. Час — медіана
`model_runner --enable-timer` на MaixCAM (CV181x) з `results/speed.jsonl`, «—» — не заміряно.

Позначення: DW — depthwise-separable блок (DW 3×3 → 1×1 → BN → активація), s2 — крок 2,
⊕α — зважена сума з навчуваним α, SE — squeeze-and-excitation, ctx — контекст із f0 через 1×1 conv,
Δflow — приріст потоку, що додається до піднятого потоку попереднього рівня
(на рівнях refine — з навчуваним коефіцієнтом α). Помаранчевим — вузли, що відрізняються від B0.

| Варіант | Зміни відносно B0 | Параметри, M | INT8, мс | mixed, мс |
|---|---|---|---|---|
""" + "\n".join(rows) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + "\n" + "\n\n".join(sections) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(sections)} variants")


if __name__ == "__main__":
    main()
