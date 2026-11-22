"""Durable, CPU-only provenance and component diagnostics for staged rewards.

The MAD normalization below is a diagnostic of saved scalar rewards, not an
additional transformation of the rewards passed to ART covariance.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .reward_diagnostics import atomic_json, normalize_records


def json_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collection_context(c, request, bundle):
    output = Path(c["runtime"]["output_dir"]).resolve()
    update, rollout = int(request.get("update", 0)), int(request.get("rollout", 0))
    step = output/"steps"/("step_%08d" % update)
    commit = Path(bundle)/"COMMITTED"
    return dict(version=1, run_output_dir=str(output), step_dir=str(step), update=update,
                rollout=rollout, replay_bundle=str(Path(bundle).resolve()),
                config_fingerprint=request.get("config_fingerprint"),
                config_sha256=json_sha256(c), source_commit=commit.read_text().strip() if commit.exists() else None)


def new_command_attempt(context):
    root = Path(context["step_dir"])/"reward_attempts"
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000000):
        attempt = root/("attempt_%04d" % number)
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            continue
    raise RuntimeError("Too many reward attempts in one collection")


def log_tail(path, limit=8000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flatten_numeric(value, prefix, out, depth=0):
    if _finite_number(value):
        out[prefix] = float(value)
    elif isinstance(value, dict) and depth < 6:
        for key, child in value.items():
            clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
            _flatten_numeric(child, prefix+"/"+clean, out, depth+1)


def component_values(detail):
    """Summarize declared score maps only, never coordinates or token arrays."""
    if not isinstance(detail, dict):
        return {}
    values = {}
    for key in ("raw_scores", "normalized_scores", "contributions", "weighted_scores", "components", "model_details"):
        if isinstance(detail.get(key), dict):
            _flatten_numeric(detail[key], key, values)
    for key in ("reward", "aesthetic_reward", "fusion_z", "artifact_penalty",
                "weighted_artifact_penalty", "artifact_count", "ignored_misalignment_count"):
        if _finite_number(detail.get(key)):
            values[key] = float(detail[key])
    return values


def _artifact_flag(detail):
    if not isinstance(detail, dict):
        return None
    if _finite_number(detail.get("artifact_count")):
        return detail["artifact_count"] > 0
    if isinstance(detail.get("sdg"), dict):
        nested = _artifact_flag(detail["sdg"])
        if nested is not None:
            return nested
    for key in ("sdg_boxes", "boxes"):
        if isinstance(detail.get(key), list):
            return any(isinstance(box, dict) and box.get("label") == "artifact" for box in detail[key])
    return None


def analyze_rewards(c, request, result):
    samples, scores = request["items"], result["scores"]
    if len(samples) != len(scores):
        raise ValueError("Reward audit requires one score per original sample")
    details = result.get("details")
    if isinstance(details, list) and len(details) != len(samples):
        raise ValueError("Reward details must preserve the original sample order and count")
    per_image = details if isinstance(details, list) else [None]*len(samples)
    norm_samples, labels, identities = [], {}, []
    for i, row in enumerate(samples):
        if row.get("index", i) != i:
            raise ValueError("Reward audit requires saved sample indices in original order")
        # Prompt group is collection-local, not the dataset sample ID: repeated
        # selections of the same dataset record may be independent groups.
        label = ("prompt_index:"+str(row["prompt_index"]) if "prompt_index" in row else
                 "sample_id:"+str(row["sample_id"]) if "sample_id" in row else "index:"+str(i))
        labels.setdefault(label, len(labels))
        norm_samples.append(dict(index=i, sample_id=label, seed=row.get("seed")))
        identities.append(dict(index=i, group_id=labels[label], group_key=label,
                               **{k: row[k] for k in ("sample_id", "prompt_index", "rollout_id", "seed", "image", "sampling_method") if k in row}))
    k = int(c["sampling"]["group_size"])
    analysis = normalize_records(norm_samples, scores, k)
    for row, identity, detail in zip(analysis["rows"], identities, per_image):
        row.pop("sample_id", None)
        row.update(identity, reward_detail_index=identity["index"], components=component_values(detail))
        # This is the exact scalar saved for ART, including its FP32 cast,
        # while detail.reward may retain the fusion worker's original precision.
        row["components"]["final_reward"] = row["raw_reward"]
    for group in analysis["groups"]:
        label = group.pop("sample_id")
        first = next(x for x in identities if x["group_key"] == label)
        group.update(group_id=first["group_id"], group_key=label,
                     **{k: first[k] for k in ("sample_id", "prompt_index") if k in first})
    components = {}
    for row in analysis["rows"]:
        for key, value in row["components"].items():
            components.setdefault(key, []).append((row["group_id"], value))
    analysis["component_summary"], analysis["component_group_statistics"] = {}, {}
    for key, pairs in components.items():
        values = [value for _,value in pairs]
        grouped = {}
        for group_id, value in pairs:
            grouped.setdefault(group_id, []).append(value)
        summaries, pooled = [], []
        for group_id, numbers in grouped.items():
            mean = float(np.mean(numbers))
            centered = np.asarray(numbers, np.float64)-mean
            summaries.append(dict(group_id=group_id, count=len(numbers), complete=len(numbers)==k,
                mean=mean, std=float(np.std(numbers)), mad=float(np.mean(np.abs(centered))),
                min=float(min(numbers)), max=float(max(numbers)), constant=max(numbers)==min(numbers)))
            pooled.extend(centered.tolist())
        complete = [row for row in summaries if row["complete"]]
        summary = dict(count=len(values), mean=float(np.mean(values)), std=float(np.std(values)),
                       min=float(min(values)), max=float(max(values)), observed_groups=len(summaries), complete_groups=len(complete))
        if complete:
            summary.update(group_std_mean=float(np.mean([row["std"] for row in complete])),
                group_mad_mean=float(np.mean([row["mad"] for row in complete])),
                group_mad_min=float(min(row["mad"] for row in complete)),
                group_mad_max=float(max(row["mad"] for row in complete)),
                constant_group_fraction=float(np.mean([row["constant"] for row in complete])))
        summary["pooled_within_prompt_std"] = float(np.sqrt(np.mean(np.square(pooled))))
        analysis["component_summary"][key] = summary
        analysis["component_group_statistics"][key] = summaries
    flags = [flag for flag in (_artifact_flag(x) for x in per_image) if flag is not None]
    analysis["sdg_summary"] = dict(artifact_observed_images=len(flags),
        artifact_observed_fraction=len(flags)/len(samples) if samples else 0.)
    if flags:
        analysis["sdg_summary"]["artifact_fraction"] = sum(flags)/len(flags)
    analysis.update(version=1, rollout=request.get("rollout"), update=request.get("update"),
        height=c["sampling"].get("height"), width=c["sampling"].get("width"),
        config_fingerprint=request.get("config_fingerprint"), config_sha256=json_sha256(c),
        evaluation_sha256=json_sha256(result), samples_sha256=json_sha256(samples),
        feedback=c.get("art", {}).get("feedback", "scalar"),
        formula="diagnostic_MAD_normalized_reward = (raw_reward - per_prompt_mean) / per_prompt_MAD; constant groups become zero",
        note="Analysis only. Saved raw scalar scores are passed unchanged to ART covariance; no row directions or activations are stored.")
    sampling, training, runtime = c.get("sampling", {}), c.get("training", {}), c.get("runtime", {})
    timesteps = sampling.get("train_timesteps")
    point_count = len(timesteps) if isinstance(timesteps, (list, tuple)) else training.get("training_points_per_trajectory")
    windows = training.get("updates_per_rollout")
    total_points = len(samples)*point_count if isinstance(point_count, int) else None
    per_window = total_points/windows if total_points is not None and isinstance(windows, int) and windows > 0 else None
    if per_window is not None and per_window.is_integer():
        per_window = int(per_window)
    analysis["collection"] = dict(source_update=request.get("update"), rollout_start=request.get("rollout"),
        prompts_per_collection=len(labels), trajectories=len(samples),
        training_points_per_trajectory=point_count, training_points_per_collection=total_points,
        update_windows=windows, points_per_update_window=per_window,
        microbatch_per_device=training.get("microbatch_size"), compute_devices=len(runtime.get("devices", [])))
    analysis["batch_semantics"] = dict(
        prompts_per_collection="Observed complete prompt groups in this saved collection, not nominal configuration alone",
        trajectories="Saved generated images/trajectories; prompts times rollout images per prompt",
        training_points_per_collection="Trajectories times cached training point count; final output is not an extra training point",
        points_per_update_window="Logical global point count per ART update window; not the per-device microbatch size",
        compute_devices="Configured physical compute device count, not an inferred tensor/data parallel factor")
    analysis["tensorboard_axes"] = dict(
        reward_components="collection.source_update: the trainer update counter of the model that collected this batch",
        reward_sdg="collection.source_update, identical to reward_components",
        collection="source_update on x axis; source_update and rollout_start are also recorded as scalar values",
        legacy_rollout_tags="Unchanged: rollout_start, not a training update counter",
        example="With two ART updates per collection, reward at source_update=0 is evaluated before train updates 1 and 2. Read collection.rollout_start for the actual run mapping; do not assume it equals the training update counter.")
    return analysis


def persist_reward_evaluation(c, request, bundle, result):
    context = collection_context(c, request, bundle)
    step = Path(context["step_dir"])
    step.mkdir(parents=True, exist_ok=True)
    analysis = analyze_rewards(c, request, result)
    analysis["context"] = context
    analysis["reward_evaluation"] = str(step/"reward_evaluation.json")
    atomic_json(step/"reward_evaluation.json", result)
    atomic_json(step/"reward_analysis.json", analysis)
    # Add the audit files to the original inference manifest when available.
    manifest = step/"manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text())
        value["reward_evaluation"] = dict(path=str(step/"reward_evaluation.json"),
            analysis=str(step/"reward_analysis.json"), source_sha256=analysis["evaluation_sha256"],
            config_sha256=analysis["config_sha256"])
        value["files"] = list(dict.fromkeys(value.get("files", [])+["reward_evaluation.json", "reward_analysis.json"]))
        atomic_json(manifest, value)
    return analysis


def log_reward_components(logger, rollout, analysis):
    # New reward tags share the source model's training-update axis. The old
    # rollout/reward_norm namespaces are intentionally left untouched elsewhere.
    source_update = analysis.get("update")
    source_update = int(rollout if source_update is None else source_update)
    rollout_start = int(analysis.get("rollout") if analysis.get("rollout") is not None else rollout)
    logger.write("collection", source_update, dict(source_update=source_update, rollout_start=rollout_start))
    batch = {key: value for key, value in analysis.get("collection", {}).items()
             if key not in ("source_update", "rollout_start") and _finite_number(value)}
    logger.write("batch", source_update, batch)
    for key, values in analysis.get("component_summary", {}).items():
        logger.write("reward_components/"+key, source_update, values)
        if analysis.get("height") and analysis.get("width"):
            logger.write("reward_components/%sx%s/%s" % (analysis["height"],analysis["width"],key), source_update, values)
    logger.write("reward_sdg", source_update, analysis.get("sdg_summary", {}))
    if getattr(logger, "writer", None) is None:
        return
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
    for key in analysis.get("component_summary", {}):
        values = np.asarray([row["components"][key] for row in analysis["rows"] if key in row["components"]], np.float64)
        counts, edges = np.histogram(values, bins=24)
        hist = HistogramProto(min=float(values.min()), max=float(values.max()), num=len(values),
            sum=float(values.sum()), sum_squares=float(np.sum(values**2)),
            bucket_limit=edges[1:].tolist(), bucket=counts.astype(float).tolist())
        logger.writer.write(Event(wall_time=time.time(),step=source_update,summary=Summary(value=[
            Summary.Value(tag="reward_components/"+key+"/distribution",histo=hist)])).SerializeToString())
    logger.writer.flush()
