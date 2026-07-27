#!/usr/bin/env python3
"""Audit and compare MoE route predictors and per-layer VRAM allocations.

The tool consumes llama.cpp-style MoE routing JSONL traces containing ``weight``
and ``route`` records. It deliberately treats every conclusion as experimental:
all prediction metrics are prequential (predict -> score -> update), static cache
plans train only on the configured warmup section, and output records assumptions.

Major functions:
  * verifies predictor state is physically independent for every target layer;
  * verifies (history-distance, expert-id) feature indexing is injective;
  * evaluates popularity, affinity, perceptron, fixed hybrid, and tournament
    predictors at real prefetch horizons;
  * reports tagged tournament-table collision/aliasing statistics;
  * compares equal per-layer cache allocation with a held-out marginal-value
    allocator under the same byte budget.

This is an offline experiment tool. It does not establish runtime throughput or
transfer correctness by itself.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required: python -m pip install numpy") from exc

LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
)
EXPERT_MARKERS = (
    ".ffn_gate_exps",
    ".ffn_up_exps",
    ".ffn_down_exps",
    ".mlp.experts",
)
CALL_ID_FIELDS = ("call", "call_id", "scheduler_call", "step", "decode_step", "token_step")


@dataclass
class LayerInfo:
    key: str
    numeric_id: int | None = None
    n_expert: int = 0
    tensor_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def bundle_bytes(self) -> int:
        return sum(self.tensor_bytes.values())


@dataclass(frozen=True)
class RouteEvent:
    order: int
    layer: str
    experts: tuple[int, ...]
    batch_tokens: int
    call_id: str | int | None


@dataclass
class TokenStep:
    index: int
    routes: dict[str, tuple[int, ...]]


@dataclass
class Metric:
    events: int = 0
    actual: int = 0
    predicted: int = 0
    true_positive: int = 0
    exact_sets: int = 0
    missed_cost: float = 0.0
    false_cost: float = 0.0

    def add(
        self,
        prediction: Sequence[int],
        actual: Sequence[int],
        miss_cost: float = 1.0,
        false_cost: float = 0.15,
    ) -> float:
        p = set(prediction)
        a = set(actual)
        tp = len(p & a)
        missed = len(a - p)
        false = len(p - a)
        self.events += 1
        self.actual += len(a)
        self.predicted += len(p)
        self.true_positive += tp
        self.exact_sets += int(p == a)
        self.missed_cost += missed * miss_cost
        self.false_cost += false * false_cost
        return missed * miss_cost + false * false_cost

    def as_dict(self) -> dict[str, float | int]:
        recall = self.true_positive / self.actual if self.actual else 0.0
        precision = self.true_positive / self.predicted if self.predicted else 0.0
        return {
            "events": self.events,
            "recall": recall,
            "precision": precision,
            "exact_set_rate": self.exact_sets / self.events if self.events else 0.0,
            "mean_cost": (self.missed_cost + self.false_cost) / self.events if self.events else 0.0,
            "missed_cost": self.missed_cost,
            "false_cost": self.false_cost,
        }


def parse_layer(name: str) -> tuple[str, int | None]:
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            number = int(match.group(1))
            return f"layer-{number:03d}", number
    for marker in EXPERT_MARKERS:
        index = name.lower().find(marker)
        if index >= 0:
            return name[:index].rstrip("."), None
    return name or "unknown-layer", None


def is_expert_tensor(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in EXPERT_MARKERS)


def row_call_id(row: Mapping[str, object]) -> str | int | None:
    for key in CALL_ID_FIELDS:
        value = row.get(key)
        if isinstance(value, (str, int)):
            return value
    return None


def load_trace(path: pathlib.Path) -> tuple[dict[str, LayerInfo], list[RouteEvent]]:
    layers: dict[str, LayerInfo] = {}
    events: list[RouteEvent] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                continue
            kind = str(row.get("type", row.get("event", "")))
            tensor = str(row.get("tensor", row.get("tensor_name", "")))
            layer_key_value = row.get("layer")
            if isinstance(layer_key_value, int):
                layer_key = f"layer-{layer_key_value:03d}"
                numeric_id = layer_key_value
            elif isinstance(layer_key_value, str) and layer_key_value:
                layer_key, numeric_id = parse_layer(layer_key_value)
            else:
                layer_key, numeric_id = parse_layer(tensor)
            info = layers.setdefault(layer_key, LayerInfo(layer_key, numeric_id=numeric_id))
            if info.numeric_id is None and numeric_id is not None:
                info.numeric_id = numeric_id

            if kind == "weight" and is_expert_tensor(tensor):
                info.n_expert = max(info.n_expert, int(row.get("n_expert", 0)))
                expert_bytes = int(row.get("expert_bytes", row.get("bytes", 0)))
                if tensor:
                    info.tensor_bytes[tensor] = max(info.tensor_bytes.get(tensor, 0), expert_bytes)
                continue

            if kind not in {"route", "expert-route", "route_decision"}:
                continue
            info.n_expert = max(info.n_expert, int(row.get("n_expert", 0)))
            ids_obj = row.get("ids", row.get("experts", row.get("actual_experts", [])))
            if not isinstance(ids_obj, list) or not ids_obj:
                continue
            if isinstance(ids_obj[0], int):
                token_routes = [ids_obj]
            else:
                token_routes = [x for x in ids_obj if isinstance(x, list)]
            batch_tokens = int(row.get("tokens", row.get("batch_tokens", len(token_routes))))
            for experts_obj in token_routes:
                experts = tuple(dict.fromkeys(int(x) for x in experts_obj))
                if not experts:
                    continue
                if info.n_expert <= 0:
                    info.n_expert = max(experts) + 1
                events.append(
                    RouteEvent(
                        order=len(events),
                        layer=layer_key,
                        experts=experts,
                        batch_tokens=batch_tokens,
                        call_id=row_call_id(row),
                    )
                )
    if not events:
        raise RuntimeError("trace contains no route records")
    return layers, events


def layer_sort_key(info: LayerInfo) -> tuple[int, str]:
    return (info.numeric_id if info.numeric_id is not None else 10**9, info.key)


def group_decode_steps(
    layers: Mapping[str, LayerInfo], events: Sequence[RouteEvent]
) -> tuple[list[str], list[TokenStep], dict[str, object]]:
    decode = [event for event in events if event.batch_tokens == 1]
    if not decode:
        raise RuntimeError("trace contains no decode routes (batch_tokens == 1)")
    ordered_layers = [info.key for info in sorted(layers.values(), key=layer_sort_key)]
    routed = [layer for layer in ordered_layers if any(event.layer == layer for event in decode)]
    ordinal = {layer: index for index, layer in enumerate(routed)}

    all_have_call_id = all(event.call_id is not None for event in decode)
    steps: list[TokenStep] = []
    if all_have_call_id:
        grouped: collections.OrderedDict[str | int, dict[str, tuple[int, ...]]] = collections.OrderedDict()
        for event in decode:
            grouped.setdefault(event.call_id, {})[event.layer] = event.experts
        steps = [TokenStep(index, routes) for index, routes in enumerate(grouped.values())]
        method = "call_id"
    else:
        current: dict[str, tuple[int, ...]] = {}
        previous_ordinal = -1
        for event in decode:
            current_ordinal = ordinal.get(event.layer, previous_ordinal + 1)
            if current and (current_ordinal <= previous_ordinal or event.layer in current):
                steps.append(TokenStep(len(steps), current))
                current = {}
            current[event.layer] = event.experts
            previous_ordinal = current_ordinal
        if current:
            steps.append(TokenStep(len(steps), current))
        method = "layer_reset_heuristic"

    complete = [step for step in steps if all(layer in step.routes for layer in routed)]
    diagnostics = {
        "grouping_method": method,
        "decode_route_events": len(decode),
        "steps_total": len(steps),
        "steps_complete": len(complete),
        "routed_layers": len(routed),
        "dropped_incomplete_steps": len(steps) - len(complete),
    }
    if len(complete) < 2:
        raise RuntimeError(f"fewer than two complete decode steps: {diagnostics}")
    for index, step in enumerate(complete):
        step.index = index
    return routed, complete, diagnostics


def stable_history_hash(history: Sequence[Sequence[int]], bits: int = 64) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for distance, experts in enumerate(history):
        digest.update(distance.to_bytes(2, "little"))
        for expert in sorted(experts):
            digest.update(int(expert).to_bytes(4, "little", signed=False))
        digest.update(b"|")
    value = int.from_bytes(digest.digest(), "little")
    return value & ((1 << bits) - 1)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float64, copy=False)
    mean = float(scores.mean())
    std = float(scores.std())
    if std < 1e-9:
        return scores - mean
    return (scores - mean) / std


def top_indices(scores: np.ndarray, count: int) -> tuple[int, ...]:
    count = min(max(0, count), int(scores.size))
    if count == 0:
        return ()
    if count == scores.size:
        order = np.argsort(-scores, kind="stable")
    else:
        selected = np.argpartition(-scores, count - 1)[:count]
        order = selected[np.argsort(-scores[selected], kind="stable")]
    return tuple(int(x) for x in order)


class PerLayerPopularity:
    def __init__(self, n_experts: Mapping[str, int], decay: float = 1.0):
        self.counts = {layer: np.zeros(count, dtype=np.float64) for layer, count in n_experts.items()}
        self.decay = decay

    def score(self, layer: str, _history: Sequence[Sequence[int]]) -> np.ndarray:
        return self.counts[layer].copy()

    def update(self, layer: str, actual: Sequence[int], _history: Sequence[Sequence[int]]) -> None:
        if self.decay < 1.0:
            self.counts[layer] *= self.decay
        self.counts[layer][list(actual)] += 1.0


class PerLayerAffinity:
    """Independent per-target-layer, per-history-position conditional tables."""

    def __init__(self, n_experts: Mapping[str, int], history_len: int, decay: float = 1.0):
        self.history_len = history_len
        self.decay = decay
        self.popularity = PerLayerPopularity(n_experts, decay)
        self.tables: dict[str, list[np.ndarray]] = {
            layer: [np.zeros((count, count), dtype=np.float64) for _ in range(history_len)]
            for layer, count in n_experts.items()
        }

    def score(self, layer: str, history: Sequence[Sequence[int]]) -> np.ndarray:
        scores = self.popularity.score(layer, history)
        for distance, experts in enumerate(history[: self.history_len]):
            for expert in experts:
                if expert < self.tables[layer][distance].shape[0]:
                    scores += self.tables[layer][distance][expert]
        return scores

    def update(self, layer: str, actual: Sequence[int], history: Sequence[Sequence[int]]) -> None:
        self.popularity.update(layer, actual, history)
        for distance, experts in enumerate(history[: self.history_len]):
            table = self.tables[layer][distance]
            if self.decay < 1.0:
                table *= self.decay
            for source in experts:
                if source < table.shape[0]:
                    table[source, list(actual)] += 1.0


class PerLayerPerceptron:
    """Dense independent multi-label ranking perceptron heads."""

    def __init__(
        self,
        n_experts: Mapping[str, int],
        history_len: int,
        minimum_weight: int = -31,
        maximum_weight: int = 31,
    ):
        self.history_len = history_len
        self.minimum_weight = minimum_weight
        self.maximum_weight = maximum_weight
        self.weights: dict[str, np.ndarray] = {
            layer: np.zeros((count, 1 + history_len * count), dtype=np.int16)
            for layer, count in n_experts.items()
        }

    def active_features(self, layer: str, history: Sequence[Sequence[int]]) -> np.ndarray:
        n_experts = self.weights[layer].shape[0]
        active = [0]
        for distance, experts in enumerate(history[: self.history_len]):
            for expert in experts:
                if 0 <= expert < n_experts:
                    active.append(1 + distance * n_experts + expert)
        return np.array(sorted(set(active)), dtype=np.int64)

    def score(self, layer: str, history: Sequence[Sequence[int]]) -> np.ndarray:
        active = self.active_features(layer, history)
        return self.weights[layer][:, active].sum(axis=1, dtype=np.int32).astype(np.float64)

    def update(
        self,
        layer: str,
        actual: Sequence[int],
        history: Sequence[Sequence[int]],
        predicted: Sequence[int],
        update_count: int,
    ) -> int:
        actual_set = set(actual)
        predicted_set = set(predicted)
        positives = [expert for expert in actual if expert not in predicted_set]
        negatives = [expert for expert in predicted if expert not in actual_set]
        pair_count = min(len(positives), len(negatives), max(0, update_count))
        if not pair_count:
            return 0
        active = self.active_features(layer, history)
        weights = self.weights[layer]
        for positive, negative in zip(positives[:pair_count], negatives[:pair_count]):
            weights[positive, active] = np.minimum(weights[positive, active] + 1, self.maximum_weight)
            weights[negative, active] = np.maximum(weights[negative, active] - 1, self.minimum_weight)
        return pair_count


@dataclass
class ChooserEntry:
    tag: int = -1
    counter: int = 0
    useful: int = 0


class TaggedTournamentChooser:
    """Small tagged chooser selecting base or corrected predictor."""

    def __init__(self, entries: int, tag_bits: int):
        if entries <= 0:
            raise ValueError("chooser entries must be positive")
        self.entries = [ChooserEntry() for _ in range(entries)]
        self.tag_mask = (1 << max(1, tag_bits)) - 1
        self.lookups = 0
        self.hits = 0
        self.collisions = 0
        self.allocations = 0
        self.disagreements = 0
        self.correct_choices = 0

    def location(self, layer_index: int, horizon: int, history: Sequence[Sequence[int]]) -> tuple[int, int]:
        history_hash = stable_history_hash(history)
        mixed = history_hash ^ (layer_index * 0x9E3779B185EBCA87) ^ (horizon * 0xD6E8FEB86659FD93)
        index = mixed % len(self.entries)
        tag = ((mixed >> max(1, int(math.log2(len(self.entries))))) ^ history_hash) & self.tag_mask
        return int(index), int(tag)

    def choose(self, layer_index: int, horizon: int, history: Sequence[Sequence[int]]) -> tuple[bool, int, int]:
        index, tag = self.location(layer_index, horizon, history)
        entry = self.entries[index]
        self.lookups += 1
        if entry.tag == tag:
            self.hits += 1
            return entry.counter >= 0, index, tag
        if entry.tag != -1:
            self.collisions += 1
        return False, index, tag

    def update(self, index: int, tag: int, corrected_better: bool, disagreed: bool, selected_correctly: bool) -> None:
        entry = self.entries[index]
        if entry.tag != tag:
            if entry.tag == -1 or entry.useful <= 0:
                entry.tag = tag
                entry.counter = 0
                entry.useful = 0
                self.allocations += 1
            else:
                entry.useful -= 1
                return
        if disagreed:
            self.disagreements += 1
            self.correct_choices += int(selected_correctly)
            delta = 1 if corrected_better else -1
            entry.counter = min(3, max(-4, entry.counter + delta))
            entry.useful = min(3, entry.useful + 1)

    def stats(self) -> dict[str, float | int]:
        return {
            "lookups": self.lookups,
            "tag_hits": self.hits,
            "collisions": self.collisions,
            "collision_rate": self.collisions / self.lookups if self.lookups else 0.0,
            "allocations": self.allocations,
            "disagreements": self.disagreements,
            "choice_accuracy_on_disagreements": self.correct_choices / self.disagreements if self.disagreements else 0.0,
        }


def histories_for_target(
    step: TokenStep,
    routed_layers: Sequence[str],
    target_index: int,
    horizon: int,
    history_len: int,
) -> list[tuple[int, ...]] | None:
    latest_available = target_index - horizon
    earliest = latest_available - history_len + 1
    if earliest < 0:
        return None
    return [step.routes[routed_layers[index]] for index in range(latest_available, earliest - 1, -1)]


def audit_predictor_independence(
    routed_layers: Sequence[str],
    n_experts: Mapping[str, int],
    history_len: int,
) -> dict[str, object]:
    predictor = PerLayerPerceptron(n_experts, history_len)
    layer_ids = {layer: id(weights) for layer, weights in predictor.weights.items()}
    unique_objects = len(set(layer_ids.values())) == len(layer_ids)
    shared_memory: list[tuple[str, str]] = []
    for index, left in enumerate(routed_layers):
        for right in routed_layers[index + 1 :]:
            if np.shares_memory(predictor.weights[left], predictor.weights[right]):
                shared_memory.append((left, right))

    collisions: dict[str, list[tuple[tuple[int, int], tuple[int, int], int]]] = {}
    for layer in routed_layers:
        seen: dict[int, tuple[int, int]] = {}
        layer_collisions = []
        count = n_experts[layer]
        for distance in range(history_len):
            for expert in range(count):
                feature = 1 + distance * count + expert
                pair = (distance, expert)
                if feature in seen:
                    layer_collisions.append((seen[feature], pair, feature))
                seen[feature] = pair
        if layer_collisions:
            collisions[layer] = layer_collisions

    mutated_layer = routed_layers[len(routed_layers) // 2]
    snapshots = {layer: weights.copy() for layer, weights in predictor.weights.items()}
    history = [tuple(range(min(2, n_experts[mutated_layer]))) for _ in range(history_len)]
    actual = (0,)
    predicted = (1,) if n_experts[mutated_layer] > 1 else ()
    predictor.update(mutated_layer, actual, history, predicted, 1)
    changed = [layer for layer in routed_layers if not np.array_equal(snapshots[layer], predictor.weights[layer])]
    mutation_isolated = changed == [mutated_layer]

    numeric_layers = [int(layer.rsplit("-", 1)[-1]) for layer in routed_layers if layer.rsplit("-", 1)[-1].isdigit()]
    mapping_unique = len(numeric_layers) == len(set(numeric_layers))
    return {
        "passed": unique_objects and not shared_memory and not collisions and mutation_isolated and mapping_unique,
        "object_ids_unique": unique_objects,
        "shared_memory_pairs": shared_memory,
        "feature_index_collisions": collisions,
        "mutation_isolated": mutation_isolated,
        "mutation_details": {"mutated_layer": mutated_layer, "changed_layers": changed},
        "layer_mapping_unique": mapping_unique,
        "important_runtime_unknowns": [
            "Whether live history buffers are per sequence/request rather than global",
            "Whether +1/+2/+3 horizons use separate heads or an explicit horizon feature",
            "Whether tagged/Markov tables include target layer and horizon in their key",
            "Whether predictor updates are synchronized safely when multiple sequences decode concurrently",
        ],
    }


def prediction_cost(
    prediction: Sequence[int], actual: Sequence[int], miss_cost: float, false_cost: float
) -> float:
    p, a = set(prediction), set(actual)
    return len(a - p) * miss_cost + len(p - a) * false_cost


def evaluate_predictors(
    routed_layers: Sequence[str],
    steps: Sequence[TokenStep],
    n_experts: Mapping[str, int],
    candidate_count: int,
    history_len: int,
    horizon: int,
    eval_start: int,
    hybrid_alpha: float,
    perceptron_update_count: int,
    miss_cost: float,
    false_cost: float,
    chooser_entries: int,
    chooser_tag_bits: int,
) -> dict[str, object]:
    popularity = PerLayerPopularity(n_experts, decay=1.0)
    affinity = PerLayerAffinity(n_experts, history_len, decay=1.0)
    perceptron = PerLayerPerceptron(n_experts, history_len)
    chooser = TaggedTournamentChooser(chooser_entries, chooser_tag_bits)
    layer_index = {layer: index for index, layer in enumerate(routed_layers)}
    metrics = {name: Metric() for name in ("popularity", "affinity", "perceptron", "hybrid", "tournament", "oracle_pair")}
    layer_metrics: dict[str, dict[str, Metric]] = {
        layer: {name: Metric() for name in metrics} for layer in routed_layers
    }
    pair_updates = 0
    eligible_events = 0
    trained_events = 0
    evaluated_events = 0
    oracle_corrected_wins = 0
    oracle_base_wins = 0
    oracle_ties = 0

    for step_index, step in enumerate(steps):
        evaluation = step_index >= eval_start
        for target_index, layer in enumerate(routed_layers):
            history = histories_for_target(step, routed_layers, target_index, horizon, history_len)
            if history is None:
                continue
            eligible_events += 1
            actual = step.routes[layer]
            pop_scores = popularity.score(layer, history)
            affinity_scores = affinity.score(layer, history)
            perceptron_scores = perceptron.score(layer, history)
            hybrid_scores = normalize_scores(affinity_scores) + hybrid_alpha * normalize_scores(perceptron_scores)
            predictions = {
                "popularity": top_indices(pop_scores, candidate_count),
                "affinity": top_indices(affinity_scores, candidate_count),
                "perceptron": top_indices(perceptron_scores, candidate_count),
                "hybrid": top_indices(hybrid_scores, candidate_count),
            }
            base_prediction = predictions["affinity"]
            corrected_prediction = predictions["hybrid"]
            choose_corrected, chooser_index, chooser_tag = chooser.choose(
                layer_index[layer], horizon, history
            )
            tournament_prediction = corrected_prediction if choose_corrected else base_prediction
            predictions["tournament"] = tournament_prediction
            base_cost = prediction_cost(base_prediction, actual, miss_cost, false_cost)
            corrected_cost = prediction_cost(corrected_prediction, actual, miss_cost, false_cost)
            if corrected_cost < base_cost:
                oracle_prediction = corrected_prediction
                oracle_corrected_wins += int(evaluation)
            elif base_cost < corrected_cost:
                oracle_prediction = base_prediction
                oracle_base_wins += int(evaluation)
            else:
                oracle_prediction = base_prediction
                oracle_ties += int(evaluation)
            predictions["oracle_pair"] = oracle_prediction

            if evaluation:
                evaluated_events += 1
                for name, prediction in predictions.items():
                    metrics[name].add(prediction, actual, miss_cost, false_cost)
                    layer_metrics[layer][name].add(prediction, actual, miss_cost, false_cost)
                disagreed = base_prediction != corrected_prediction
                selected_cost = corrected_cost if choose_corrected else base_cost
                best_cost = min(base_cost, corrected_cost)
                chooser.update(
                    chooser_index,
                    chooser_tag,
                    corrected_better=corrected_cost < base_cost,
                    disagreed=disagreed,
                    selected_correctly=selected_cost == best_cost,
                )
            else:
                trained_events += 1

            popularity.update(layer, actual, history)
            affinity.update(layer, actual, history)
            update_count = perceptron_update_count or len(actual)
            pair_updates += perceptron.update(
                layer, actual, history, predictions["perceptron"], update_count
            )

    serializable_layer_metrics = {
        layer: {name: metric.as_dict() for name, metric in values.items()}
        for layer, values in layer_metrics.items()
        if any(metric.events for metric in values.values())
    }
    return {
        "horizon": horizon,
        "history_len": history_len,
        "candidate_count": candidate_count,
        "eval_start_step": eval_start,
        "eligible_events": eligible_events,
        "training_events": trained_events,
        "evaluated_events": evaluated_events,
        "pair_updates": pair_updates,
        "metrics": {name: metric.as_dict() for name, metric in metrics.items()},
        "oracle_selection": {
            "corrected_wins": oracle_corrected_wins,
            "base_wins": oracle_base_wins,
            "ties": oracle_ties,
        },
        "chooser": chooser.stats(),
        "per_layer": serializable_layer_metrics,
        "guardrails": [
            "Tournament state in this simulator is shared across layers but every key includes target layer and horizon.",
            "Production should begin with separate small chooser counters per (layer, horizon) before a tagged table.",
            "Oracle pair is an event-wise upper bound and cannot be implemented without future knowledge.",
            "Recall and the configured cost proxy do not include deadline, eviction, or graph topology costs.",
        ],
    }


def expert_rankings(
    routed_layers: Sequence[str],
    steps: Sequence[TokenStep],
    n_experts: Mapping[str, int],
    train_count: int,
) -> dict[str, list[int]]:
    counts = {layer: np.zeros(n_experts[layer], dtype=np.int64) for layer in routed_layers}
    for step in steps[:train_count]:
        for layer in routed_layers:
            counts[layer][list(step.routes[layer])] += 1
    return {
        layer: [int(x) for x in np.argsort(-values, kind="stable")]
        for layer, values in counts.items()
    }


def layer_value_curves(
    routed_layers: Sequence[str],
    layers: Mapping[str, LayerInfo],
    evaluation: Sequence[TokenStep],
    rankings: Mapping[str, Sequence[int]],
    objective: str,
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for layer in routed_layers:
        n = max(1, layers[layer].n_expert)
        ranking = list(rankings[layer])[:n]
        rank_of = {expert: rank for rank, expert in enumerate(ranking)}
        difference = np.zeros(n + 2, dtype=np.float64)
        bundle_bytes = max(1, layers[layer].bundle_bytes)
        for step in evaluation:
            actual = step.routes[layer]
            ranks = [rank_of.get(expert, n) for expert in actual]
            if objective in {"route_hits", "saved_bytes"}:
                scale = bundle_bytes if objective == "saved_bytes" else 1.0
                for rank in ranks:
                    threshold = rank + 1
                    if threshold <= n:
                        difference[threshold] += scale
            elif objective == "complete_layers":
                threshold = max(ranks, default=n) + 1
                if threshold <= n:
                    difference[threshold] += 1.0
            else:
                raise ValueError(f"unknown objective: {objective}")
        values = np.cumsum(difference[: n + 1])
        curves[layer] = values
    return curves


def allocation_value(allocation: Mapping[str, int], curves: Mapping[str, np.ndarray]) -> float:
    return float(sum(curves[layer][min(capacity, len(curves[layer]) - 1)] for layer, capacity in allocation.items()))


def allocation_bytes(allocation: Mapping[str, int], layers: Mapping[str, LayerInfo]) -> int:
    return sum(max(0, count) * max(1, layers[layer].bundle_bytes) for layer, count in allocation.items())


def equal_allocation(
    routed_layers: Sequence[str], layers: Mapping[str, LayerInfo], budget: int, minimum: int
) -> dict[str, int]:
    allocation = {layer: 0 for layer in routed_layers}
    for _ in range(max(0, minimum)):
        progressed = False
        for layer in routed_layers:
            size = max(1, layers[layer].bundle_bytes)
            if allocation[layer] < layers[layer].n_expert and budget >= size:
                allocation[layer] += 1
                budget -= size
                progressed = True
        if not progressed:
            break
    while True:
        choices = [
            layer
            for layer in routed_layers
            if allocation[layer] < layers[layer].n_expert and budget >= max(1, layers[layer].bundle_bytes)
        ]
        if not choices:
            break
        layer = min(choices, key=lambda key: (allocation[key], max(1, layers[key].bundle_bytes), key))
        allocation[layer] += 1
        budget -= max(1, layers[layer].bundle_bytes)
    return allocation


def marginal_allocation(
    routed_layers: Sequence[str],
    layers: Mapping[str, LayerInfo],
    curves: Mapping[str, np.ndarray],
    budget: int,
    minimum: int,
    max_chunk: int,
) -> dict[str, int]:
    """Cost-aware allocation with chunk lookahead and one-slot exchange repair.

    Chunk lookahead can cross zero-marginal plateaus in complete-layer curves.
    This is not claimed to be an exact multiple-choice knapsack solver.
    """
    allocation = {layer: 0 for layer in routed_layers}
    for _ in range(max(0, minimum)):
        for layer in routed_layers:
            size = max(1, layers[layer].bundle_bytes)
            if allocation[layer] < layers[layer].n_expert and budget >= size:
                allocation[layer] += 1
                budget -= size

    while True:
        best: tuple[float, float, str, int, int] | None = None
        for layer in routed_layers:
            current = allocation[layer]
            limit = min(layers[layer].n_expert, current + max(1, max_chunk))
            for target in range(current + 1, limit + 1):
                slots = target - current
                cost = slots * max(1, layers[layer].bundle_bytes)
                if cost > budget:
                    break
                gain = curves[layer][target] - curves[layer][current]
                density = gain / cost
                candidate = (density, gain, layer, slots, cost)
                if gain > 0 and (best is None or candidate[:2] > best[:2]):
                    best = candidate
        if best is None:
            break
        _, _, layer, slots, cost = best
        allocation[layer] += slots
        budget -= cost

    improved = True
    while improved:
        improved = False
        current_value = allocation_value(allocation, curves)
        for donor in routed_layers:
            if allocation[donor] <= minimum:
                continue
            donor_size = max(1, layers[donor].bundle_bytes)
            trial_budget = budget + donor_size
            for receiver in routed_layers:
                if receiver == donor or allocation[receiver] >= layers[receiver].n_expert:
                    continue
                receiver_size = max(1, layers[receiver].bundle_bytes)
                if receiver_size > trial_budget:
                    continue
                trial = dict(allocation)
                trial[donor] -= 1
                trial[receiver] += 1
                trial_value = allocation_value(trial, curves)
                if trial_value > current_value + 1e-12:
                    allocation = trial
                    budget = trial_budget - receiver_size
                    improved = True
                    break
            if improved:
                break
    return allocation


def evaluate_allocation(
    allocation: Mapping[str, int],
    routed_layers: Sequence[str],
    layers: Mapping[str, LayerInfo],
    evaluation: Sequence[TokenStep],
    rankings: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    resident = {
        layer: set(list(rankings[layer])[: min(allocation[layer], layers[layer].n_expert)])
        for layer in routed_layers
    }
    hits = 0
    accesses = 0
    complete = 0
    layer_events = 0
    saved_bytes = 0
    for step in evaluation:
        for layer in routed_layers:
            actual = step.routes[layer]
            actual_set = set(actual)
            layer_hits = len(actual_set & resident[layer])
            hits += layer_hits
            accesses += len(actual_set)
            complete += int(actual_set <= resident[layer])
            layer_events += 1
            saved_bytes += layer_hits * max(1, layers[layer].bundle_bytes)
    return {
        "allocated_bytes": allocation_bytes(allocation, layers),
        "total_slots": sum(allocation.values()),
        "route_hit_rate": hits / accesses if accesses else 0.0,
        "complete_layer_rate": complete / layer_events if layer_events else 0.0,
        "saved_bytes_proxy": saved_bytes,
    }


def compare_allocations(
    routed_layers: Sequence[str],
    layers: Mapping[str, LayerInfo],
    steps: Sequence[TokenStep],
    budget: int,
    train_fraction: float,
    minimum: int,
    objective: str,
    max_chunk: int,
) -> dict[str, object]:
    train_count = min(max(1, round(len(steps) * train_fraction)), len(steps) - 1)
    evaluation = steps[train_count:]
    rankings = expert_rankings(routed_layers, steps, {key: layers[key].n_expert for key in routed_layers}, train_count)
    curves = layer_value_curves(routed_layers, layers, evaluation, rankings, objective)
    equal = equal_allocation(routed_layers, layers, budget, minimum)
    optimized = marginal_allocation(routed_layers, layers, curves, budget, minimum, max_chunk)
    equal_metrics = evaluate_allocation(equal, routed_layers, layers, evaluation, rankings)
    optimized_metrics = evaluate_allocation(optimized, routed_layers, layers, evaluation, rankings)
    per_layer = []
    for layer in routed_layers:
        per_layer.append(
            {
                "layer": layer,
                "bundle_bytes": layers[layer].bundle_bytes,
                "equal_slots": equal[layer],
                "optimized_slots": optimized[layer],
                "slot_delta": optimized[layer] - equal[layer],
                "equal_objective": float(curves[layer][equal[layer]]),
                "optimized_objective": float(curves[layer][optimized[layer]]),
            }
        )
    per_layer.sort(key=lambda row: (abs(int(row["slot_delta"])), str(row["layer"])), reverse=True)
    return {
        "config": {
            "budget_bytes": budget,
            "train_fraction": train_fraction,
            "train_steps": train_count,
            "evaluation_steps": len(evaluation),
            "minimum_slots": minimum,
            "objective": objective,
            "max_chunk_lookahead": max_chunk,
        },
        "equal": {"metrics": equal_metrics, "allocation": equal},
        "optimized": {"metrics": optimized_metrics, "allocation": optimized},
        "objective_value_equal": allocation_value(equal, curves),
        "objective_value_optimized": allocation_value(optimized, curves),
        "per_layer_largest_changes": per_layer,
        "guardrails": [
            "The optimized allocation is held-out for this trace, not proof of cross-prompt generalization.",
            "The saved-bytes proxy is not measured latency and ignores graph/synchronization effects.",
            "Complete-layer objectives are non-concave; the chunk-lookahead allocator is approximate.",
            "Production allocation should use several prompts and measured CPU/GPU latency value per layer.",
        ],
    }


def human_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--horizons", default="1,2,3", help="comma-separated routed-layer lookahead horizons")
    parser.add_argument("--eval-start-fraction", type=float, default=0.67)
    parser.add_argument("--hybrid-alpha", type=float, default=0.75)
    parser.add_argument("--perceptron-update-count", type=int, default=0, help="0 uses actual route width (normally top-8)")
    parser.add_argument("--miss-cost", type=float, default=1.0)
    parser.add_argument("--false-prefetch-cost", type=float, default=0.15)
    parser.add_argument("--chooser-entries", type=int, default=256)
    parser.add_argument("--chooser-tag-bits", type=int, default=12)
    parser.add_argument("--vram-gib", type=float, default=None)
    parser.add_argument("--minimum-slots", type=int, default=0)
    parser.add_argument("--allocator-train-fraction", type=float, default=0.67)
    parser.add_argument("--allocator-objective", choices=("route_hits", "complete_layers", "saved_bytes"), default="complete_layers")
    parser.add_argument("--allocator-max-chunk", type=int, default=16)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.candidate_count <= 0:
        parser.error("--candidate-count must be positive")
    if args.history_len <= 0:
        parser.error("--history-len must be positive")
    if not 0 <= args.eval_start_fraction < 1:
        parser.error("--eval-start-fraction must be in [0, 1)")
    if not 0 < args.allocator_train_fraction < 1:
        parser.error("--allocator-train-fraction must be in (0, 1)")
    if args.vram_gib is not None and args.vram_gib < 0:
        parser.error("--vram-gib must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        layers, events = load_trace(args.trace)
        routed_layers, steps, grouping = group_decode_steps(layers, events)
        n_experts = {layer: layers[layer].n_expert for layer in routed_layers}
        if any(count <= 0 for count in n_experts.values()):
            raise RuntimeError(f"invalid expert counts: {n_experts}")
        horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise RuntimeError("horizons must be positive")
        eval_start = min(max(1, round(len(steps) * args.eval_start_fraction)), len(steps) - 1)
        independence = audit_predictor_independence(routed_layers, n_experts, args.history_len)
        predictor_results = {
            str(horizon): evaluate_predictors(
                routed_layers,
                steps,
                n_experts,
                args.candidate_count,
                args.history_len,
                horizon,
                eval_start,
                args.hybrid_alpha,
                args.perceptron_update_count,
                args.miss_cost,
                args.false_prefetch_cost,
                args.chooser_entries,
                args.chooser_tag_bits,
            )
            for horizon in horizons
        }
        budget = int(args.vram_gib * 1024**3) if args.vram_gib is not None else 0
        allocation = None
        if budget:
            missing_sizes = [layer for layer in routed_layers if layers[layer].bundle_bytes <= 0]
            if missing_sizes:
                raise RuntimeError(f"allocation requested but expert bundle bytes are missing for: {missing_sizes}")
            allocation = compare_allocations(
                routed_layers,
                layers,
                steps,
                budget,
                args.allocator_train_fraction,
                args.minimum_slots,
                args.allocator_objective,
                args.allocator_max_chunk,
            )
        report = {
            "trace": str(args.trace),
            "grouping": grouping,
            "predictor_independence_audit": independence,
            "predictors_by_horizon": predictor_results,
            "allocation": allocation,
        }
        print("MoE predictor and allocation audit")
        print(f"Complete decode steps: {len(steps)}  Routed layers: {len(routed_layers)}")
        print(f"Grouping: {grouping['grouping_method']}")
        print(f"Per-layer independence audit: {'PASS' if independence['passed'] else 'FAIL'}")
        for horizon in horizons:
            result = predictor_results[str(horizon)]
            print(f"\nHorizon +{horizon} routed layers, top-{args.candidate_count}:")
            for name, metric in result["metrics"].items():
                print(
                    f"  {name:11s} recall={metric['recall'] * 100:6.2f}% "
                    f"precision={metric['precision'] * 100:6.2f}% mean_cost={metric['mean_cost']:.4f}"
                )
            chooser = result["chooser"]
            print(
                f"  chooser collisions={chooser['collisions']} ({chooser['collision_rate'] * 100:.3f}%), "
                f"accuracy-on-disagreements={chooser['choice_accuracy_on_disagreements'] * 100:.2f}%"
            )
        if allocation:
            equal = allocation["equal"]["metrics"]
            optimized = allocation["optimized"]["metrics"]
            print(f"\nVRAM allocation budget: {human_bytes(budget)}")
            print(
                f"  equal:     slots={equal['total_slots']} route={equal['route_hit_rate'] * 100:.2f}% "
                f"complete={equal['complete_layer_rate'] * 100:.2f}%"
            )
            print(
                f"  optimized: slots={optimized['total_slots']} route={optimized['route_hit_rate'] * 100:.2f}% "
                f"complete={optimized['complete_layer_rate'] * 100:.2f}%"
            )
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"\nJSON: {args.json}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
