#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hashlib
import json
from dataclasses import replace
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageDraw
from ultralytics import YOLO

import costing as cs

WEIGHTS = Path("weights/best.pt")
RESULTS = Path("results")
needed = [WEIGHTS] + [RESULTS / f for f in (
    "image_scores.csv", "params.json", "operating_point.json", "metrics.json",
    "dataset_stats.json", "train_settings.json")]
missing = [str(p) for p in needed if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing " + ", ".join(missing) + ". Run `uv run python main.py train` first, "
        "or commit the weights/ and results/ it produced.")

model = YOLO(str(WEIGHTS))  # the committed weights, never whatever runs/ holds
PARAMS = cs.CostParams.from_dict(json.load(open(RESULTS / "params.json")))
OPERATING_POINT = json.load(open(RESULTS / "operating_point.json"))
METRICS = json.load(open(RESULTS / "metrics.json"))
DATASET = json.load(open(RESULTS / "dataset_stats.json"))
SETTINGS = json.load(open(RESULTS / "train_settings.json"))
IMGSZ = SETTINGS["imgsz"]
POSITIVE_NAMES = {"fire", "smoke"}

scores = pd.read_csv(RESULTS / "image_scores.csv")
scores = scores[scores.system == "finetuned"]
POPULATIONS = {
    split: (np.sort(d[d.truth == 1].score.to_numpy()), np.sort(d[d.truth == 0].score.to_numpy()))
    for split, d in scores.groupby("split")
}
LOW0, HIGH0 = OPERATING_POINT["low"], OPERATING_POINT["high"]
RATE0 = PARAMS.base_rate * 1e6  # fires per million photos

if SETTINGS.get("fast_mode"):
    print("WARNING: these weights come from a FAST smoke-test run, not a real training run.")
print(f"loaded {WEIGHTS} | splits {sorted(POPULATIONS)} | chosen band low={LOW0:.3f} "
      f"high={'never' if HIGH0 > 1 else f'{HIGH0:.3f}'}")


# In[2]:


_cache = {}


def detect(image):
    key = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    if key not in _cache:
        res = model.predict(image.convert("RGB"), conf=0.001, imgsz=IMGSZ, device="cpu",
                            verbose=False)[0]
        _cache[key] = [
            {"name": model.names[int(c)], "conf": float(s), "box": [float(v) for v in b]}
            for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                               res.boxes.xyxy.tolist())
        ]
        if len(_cache) > 16:
            _cache.pop(next(iter(_cache)))
    return _cache[key]


COLORS = {"fire": (220, 38, 38), "smoke": (71, 85, 105)}


def decide(score, low, high):
    if score < low:
        return "PASS", "no alarm"
    if score >= high:
        return "FAIL", "call the emergency service now, nobody looks first"
    return "REVIEW", "send the photo to an operator"


def predict_photo(image, low, high):
    if image is None:
        return None, "Upload a photo to see the decision.", pd.DataFrame()
    high = max(high, low)
    dets = detect(image)
    fire_smoke = [d for d in dets if d["name"].lower() in POSITIVE_NAMES]
    score = max((d["conf"] for d in fire_smoke), default=0.0)

    floor = max(low, 0.05)  # draw what clears the alarm edge; hide the 0.1% clutter
    out = image.convert("RGB").copy()
    pen = ImageDraw.Draw(out)
    width = max(2, out.width // 200)
    for d in fire_smoke:
        if d["conf"] >= floor:
            x1, y1, x2, y2 = d["box"]
            color = COLORS.get(d["name"].lower(), (0, 0, 0))
            pen.rectangle([x1, y1, x2, y2], outline=color, width=width)
            pen.text((x1 + 3, y1 + 2), f"{d['name']} {d['conf']:.2f}", fill=color)

    verdict, action = decide(score, low, high)
    top = "never" if high > 1 else f"{high:.3f}"
    summary = (f"### {verdict}: {action}\n\n"
               f"score **{score:.3f}**  (pass below {low:.3f}, fail from {top}). "
               f"Boxes under {floor:.2f} are hidden.")
    table = pd.DataFrame([{"class": d["name"], "confidence": round(d["conf"], 3)}
                          for d in sorted(dets, key=lambda d: -d["conf"])
                          if d["conf"] >= 0.05][:15])
    return out, summary, table


# In[3]:


def heatmap(z, xs, ys, title):
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, text=z, texttemplate="%{text}",
                               colorscale="Blues", showscale=False, hoverinfo="skip"))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def band_view(split, low, high, c_fn, rate_per_million):
    pos, neg = POPULATIONS[split]
    high = max(high, low)
    params = replace(PARAMS, c_fn=float(c_fn))
    rate = rate_per_million / 1e6
    here = cs.evaluate(pos, neg, low, high, params, base_rate=rate)
    chosen = cs.evaluate(pos, neg, LOW0, HIGH0, params, base_rate=rate)
    nothing = cs.evaluate(pos, neg, cs.NEVER, cs.NEVER, params, base_rate=rate)
    everyone = cs.evaluate(pos, neg, 0.0, cs.NEVER, params, base_rate=rate)

    # 1) the confusion matrix: plain counts
    tpos, tneg = cs.three_way_table(pos, neg, low, high)
    matrix = heatmap([list(tpos), list(tneg)], ["pass", "review", "fail (dispatch)"],
                     [f"fire present ({len(pos)})", f"no fire ({len(neg)})"],
                     f"Photos in {split}, by truth and decision")

    # 2) the headline numbers
    over = here["operators_needed"] > PARAMS.max_operators
    top = "never" if high > 1 else f"{high:.3f}"
    summary = (
        f"### EUR {here['cost_per_1000']:,.3f} per 1,000 photos\n"
        f"band: pass below **{low:.3f}**, review up to **{top}**, fail from there  \n"
        f"at **{rate_per_million:.3g} fires per million photos**, a miss costing "
        f"**EUR {c_fn:,.0f}**\n\n"
        f"- **sent to review:** {here['review_share']:.3%} of photos = "
        f"{here['reviews_per_day']:,.0f} a day = {here['review_hours_per_day']:,.1f} h of work "
        f"= **{here['operators_needed']:.2f} operators** "
        f"({'OVER the ' if over else 'within the '}{PARAMS.max_operators:g} FindFire can staff)\n"
        f"- **dispatched without a person:** {here['fail_share']:.3%} of photos, about "
        f"{here['false_dispatches_per_day']:,.2f} false dispatches a day\n"
        f"- **fires missed:** about {here['fires_missed_per_year']:.2f} a year across "
        f"{PARAMS.cameras:,} cameras\n"
        f"- **fleet cost:** EUR {here['cost_per_year']:,.0f} a year\n\n"
        f"*{split} holds only {len(pos)} photos with fire and {len(neg)} without. Rates are "
        f"smoothed (half an observation per cell), so a zero count still costs something.*"
    )

    # 3) the four designs
    rows = {"A. do nothing": nothing, "B. a person checks every photo": everyone,
            "your band (sliders)": here, "chosen band (from training)": chosen}
    designs = pd.DataFrame([{
        "design": name,
        "EUR per 1,000": round(r["cost_per_1000"], 3),
        "EUR per year": round(r["cost_per_year"]),
        "review share": f"{r['review_share']:.3%}",
        "operators": round(r["operators_needed"], 2),
        "false dispatches/day": round(r["false_dispatches_per_day"], 2),
        "fires missed/year": round(r["fires_missed_per_year"], 2),
    } for name, r in rows.items()])

    # 4) the cost against threshold curve
    sweep = cs.sweep(pos, neg, params, mode="operator", base_rate=rate)
    xs = [r["low"] for r in sweep if r["low"] <= 1]
    ys = [r["cost_per_1000"] for r in sweep if r["low"] <= 1]
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="every photo above t goes to a person"))
    curve.add_hline(y=nothing["cost_per_1000"], line_dash="dot", line_color="grey",
                    annotation_text="do nothing")
    t_star = OPERATING_POINT["binary_threshold"]
    if t_star <= 1:
        star = cs.evaluate(pos, neg, t_star, cs.NEVER, params, base_rate=rate)
        curve.add_trace(go.Scatter(x=[t_star], y=[star["cost_per_1000"]], mode="markers",
                                   marker=dict(symbol="star", size=16, color="red"),
                                   name=f"chosen threshold t* = {t_star:.3f}"))
    else:
        curve.add_annotation(text="chosen: never alarm (t* above 1)", x=0.5, y=1, xref="x",
                             yref="paper", showarrow=False)
    curve.add_vline(x=low, line_dash="dash", line_color="green", annotation_text="your low")
    if high <= 1:
        curve.add_vline(x=high, line_dash="dash", line_color="orange", annotation_text="your high")
    curve.update_layout(title=f"Cost against threshold ({split}, real base rate)",
                        xaxis_title="confidence threshold t", yaxis_title="EUR per 1,000 photos",
                        yaxis_type="log", height=420, margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", y=-0.25))
    return matrix, summary, designs, curve


# In[4]:


def per_class_tables(split):
    d = METRICS[split]
    rows = [{"class": c, "precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
             "AP50": round(v["ap50"], 3), "mAP50-95": round(v["map50_95"], 3)}
            for c, v in d["per_class"].items()]
    rows.append({"class": "fire + smoke (headline)", "precision": None, "recall": None,
                 "AP50": round(d["map50_fire_smoke"], 3),
                 "mAP50-95": round(d["map50_95_fire_smoke"], 3)})
    return pd.DataFrame(rows)


def confusion_figure(split):
    d = METRICS[split]
    labels = d["confusion_labels"]
    return heatmap(d["confusion"], labels, labels,
                   f"Box-level confusion, {split} at conf {d.get('confusion_conf', 0.25)} "
                   "(rows = predicted, columns = true)")


baselines = pd.DataFrame(METRICS["image_level_comparison"]).round(3)
baselines.insert(0, "box-level AP50 (fire+smoke)", [
    round(METRICS["test"]["map50_fire_smoke"], 3), 0.0, 0.0])

dataset_rows = []
for split, d in DATASET.items():
    row = {"split": split, "photos": d["images"], "with fire/smoke": d["positive_images"],
           "without": d["negative_images"]}
    row.update({f"{c} boxes": n for c, n in d["instances"].items()})
    row.update({f"median {c} area": round(a, 4) for c, a in d["median_box_area_share"].items()})
    dataset_rows.append(row)
dataset_table = pd.DataFrame(dataset_rows)

settings_md = ("**Trained with:** " + ", ".join(
    f"{k} = {SETTINGS[k]}" for k in ("base_model", "epochs_requested", "imgsz", "batch",
                                    "device", "seed", "ultralytics", "training_seconds")))


# In[5]:


with gr.Blocks(title="FindFire") as demo:
    gr.Markdown("# FindFire: fire and smoke screening\n"
                "A photo scores between 0 and 1. Below **low** it passes, from **high** up the "
                "emergency service is called, in between an operator looks.")
    with gr.Row():
        low = gr.Slider(0.0, 1.0, value=min(LOW0, 1.0), step=0.005, label="low (pass below)")
        high = gr.Slider(0.0, 1.01, value=HIGH0, step=0.005,
                         label="high (fail from here; 1.01 = never dispatch automatically)")

    with gr.Tab("Try a photo"):
        with gr.Row():
            photo = gr.Image(type="pil", label="Upload a photo")
            shown = gr.Image(label="Fire and smoke boxes above the low edge")
        decision = gr.Markdown()
        boxes = gr.Dataframe(label="Boxes the model found (confidence 0.05 and up)",
                             interactive=False)
        for trigger in (photo.change, low.change, high.change):
            trigger(predict_photo, [photo, low, high], [shown, decision, boxes])

    with gr.Tab("Thresholds and cost"):
        with gr.Row():
            split = gr.Dropdown([s for s in ("test", "valid") if s in POPULATIONS],
                                value="test" if "test" in POPULATIONS else "valid",
                                label="photos to evaluate on")
            c_fn = gr.Number(value=PARAMS.c_fn, label="cost of a missed fire (EUR)")
            rate = gr.Number(value=round(RATE0, 4),
                             label="fires per million photos (the assumed base rate)")
        with gr.Row():
            matrix = gr.Plot(label="confusion matrix")
            summary = gr.Markdown()
        curve = gr.Plot(label="cost against threshold")
        designs = gr.Dataframe(label="Priced against the alternatives", interactive=False)
        inputs = [split, low, high, c_fn, rate]
        outputs = [matrix, summary, designs, curve]
        for control in (split, low, high, c_fn, rate):
            control.change(band_view, inputs, outputs)
        demo.load(band_view, inputs, outputs)

    with gr.Tab("Per-class results"):
        gr.Markdown(settings_md)
        gr.Markdown("### Test split, per class (box level)")
        gr.Dataframe(per_class_tables("test"), interactive=False)
        gr.Markdown("### Validation split, per class (box level)")
        gr.Dataframe(per_class_tables("valid"), interactive=False)
        gr.Markdown("### Against the baselines, per photo (cut 0.25)\n"
                    "The COCO model has no fire or smoke class, so it scores 0 by construction.")
        gr.Dataframe(baselines, interactive=False)
        gr.Plot(confusion_figure("test"))
        gr.Markdown("### The dataset (train is augmented 3x; valid and test are not)")
        gr.Dataframe(dataset_table, interactive=False)


# In[6]:


demo.launch()

