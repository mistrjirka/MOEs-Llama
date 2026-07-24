#!/usr/bin/env python3
"""Analyze ggml MoE routing JSONL and simulate persistent expert caches."""
from __future__ import annotations

import argparse, collections, json, pathlib, re, sys
from dataclasses import dataclass, field

LAYER_RE = (re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)"),
            re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)"),
            re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"))


def layer_key(name: str) -> str:
    for pattern in LAYER_RE:
        if match := pattern.search(name):
            return f"layer-{int(match.group(1)):03d}"
    for marker in (".ffn_gate_exps", ".ffn_up_exps", ".ffn_down_exps", ".mlp.experts"):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def expert_tensor(name: str) -> bool:
    name = name.lower()
    return any(x in name for x in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", ".mlp.experts"))


@dataclass
class Layer:
    n: int = 0
    parts: dict[str, int] = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        return sum(self.parts.values())


@dataclass(frozen=True)
class Route:
    layer: str
    ids: tuple[int, ...]
    batch: int


class LRU:
    def __init__(self, cap): self.cap, self.d = max(0, cap), collections.OrderedDict()
    def access(self, e):
        if e in self.d:
            self.d.move_to_end(e); return True
        if self.cap:
            if len(self.d) >= self.cap: self.d.popitem(last=False)
            self.d[e] = None
        return False


class LFU:
    def __init__(self, cap): self.cap, self.clock, self.d = max(0, cap), 0, {}
    def access(self, e):
        self.clock += 1
        if e in self.d:
            f, _ = self.d[e]; self.d[e] = (f + 1, self.clock); return True
        if self.cap:
            if len(self.d) >= self.cap: del self.d[min(self.d, key=lambda x: self.d[x])]
            self.d[e] = (1, self.clock)
        return False


class SLRU:
    def __init__(self, cap, second_touch):
        self.cap = max(0, cap); self.second = second_touch
        self.protected_cap = min(self.cap, max(0, round(self.cap * .8)))
        self.probation_cap = self.cap - self.protected_cap
        if self.cap and not self.probation_cap: self.probation_cap, self.protected_cap = 1, self.cap - 1
        self.history, self.probation, self.protected = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()

    def _probation(self, e):
        if not self.probation_cap: return self._protected(e)
        if len(self.probation) >= self.probation_cap: self.probation.popitem(last=False)
        self.probation[e] = None

    def _protected(self, e):
        if not self.protected_cap: return self._probation(e)
        if len(self.protected) >= self.protected_cap:
            old, _ = self.protected.popitem(last=False); self._probation(old)
        self.protected[e] = None

    def access(self, e):
        if e in self.protected: self.protected.move_to_end(e); return True
        if e in self.probation:
            del self.probation[e]; self._protected(e); return True
        if not self.cap: return False
        if self.second and e not in self.history:
            self.history[e] = None
            while len(self.history) > max(32, self.cap * 4): self.history.popitem(last=False)
            return False
        self.history.pop(e, None); self._probation(e); return False


class Static:
    def __init__(self, ids): self.ids = frozenset(ids)
    def access(self, e): return e in self.ids


def load(path):
    layers, routes = {}, []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: raise RuntimeError(f"{path}:{lineno}: {exc}") from exc
            name, kind = str(row.get("tensor", "")), row.get("type")
            key = layer_key(name)
            if kind == "weight" and expert_tensor(name):
                layer = layers.setdefault(key, Layer()); layer.n = max(layer.n, int(row.get("n_expert", 0)))
                layer.parts[name] = int(row.get("expert_bytes", 0))
            elif kind == "route":
                layer = layers.setdefault(key, Layer()); layer.n = max(layer.n, int(row.get("n_expert", 0)))
                ids = row.get("ids", [])
                if ids and isinstance(ids[0], int): ids = [ids]
                batch = int(row.get("tokens", len(ids)))
                routes += [Route(key, tuple(dict.fromkeys(map(int, experts))), batch) for experts in ids]
    if not routes: raise RuntimeError("trace contains no route records")
    return layers, routes


def slots_from(value, layers):
    if "=" not in value: return {k: max(0, int(value)) for k in layers}
    result = {k: 0 for k in layers}
    for item in value.split(","):
        key, count = item.split("=", 1)
        if key not in layers: raise RuntimeError(f"unknown layer in --slots: {key}")
        result[key] = max(0, int(count))
    return result


def allocate(layers, budget, minimum):
    slots = {k: 0 for k in layers}; ordered = [k for k in sorted(layers) if layers[k].bytes]
    while True:
        changed = False
        for k in ordered:
            if slots[k] < minimum and budget >= layers[k].bytes:
                slots[k] += 1; budget -= layers[k].bytes; changed = True
        if not changed: break
    while True:
        choices = [k for k in ordered if slots[k] < layers[k].n and layers[k].bytes <= budget]
        if not choices: return slots
        k = min(choices, key=lambda x: (slots[x], layers[x].bytes, x))
        slots[k] += 1; budget -= layers[k].bytes


def in_scope(route, scope):
    return scope == "all" or (scope == "decode" and route.batch == 1) or (scope == "prefill" and route.batch > 1)


def static_sets(routes, slots, warmup, scope):
    chosen = [r for r in routes if in_scope(r, scope)]
    counts = collections.defaultdict(collections.Counter)
    for r in chosen[:max(1, round(len(chosen) * warmup))]: counts[r.layer].update(r.ids)
    return {k: {e for e, _ in counts[k].most_common(slots.get(k, 0))} for k in slots}


def simulate(layers, routes, slots, policy, scope, warmup):
    hot = static_sets(routes, slots, warmup, scope) if policy == "static" else {}
    caches = {}
    for k in layers:
        cap = slots.get(k, 0)
        caches[k] = {"lru": LRU(cap), "lfu": LFU(cap), "slru": SLRU(cap, False),
                     "slru-2touch": SLRU(cap, True), "static": Static(hot.get(k, set()))}[policy]
    rows = {k: {"accesses": 0, "hits": 0, "transferred_bytes": 0} for k in layers}
    previous, overlap, exact, comparisons = {}, 0.0, 0, 0
    for route in routes:
        if not in_scope(route, scope) or not layers[route.layer].bytes: continue
        if route.layer in previous:
            comparisons += 1; overlap += len(set(previous[route.layer]) & set(route.ids)) / max(1, len(route.ids))
            exact += previous[route.layer] == route.ids
        previous[route.layer] = route.ids
        row = rows[route.layer]
        for e in route.ids:
            hit = caches[route.layer].access(e); row["accesses"] += 1; row["hits"] += int(hit)
            if not hit: row["transferred_bytes"] += layers[route.layer].bytes
    layer_rows, total = [], {"accesses": 0, "hits": 0, "transferred_bytes": 0, "baseline_bytes": 0}
    for k in sorted(layers):
        r = rows[k]; r["baseline_bytes"] = r["accesses"] * layers[k].bytes
        layer_rows.append({"layer": k, "experts": layers[k].n, "bundle_bytes": layers[k].bytes,
                           "slots": min(layers[k].n, slots.get(k, 0)), **r,
                           "hit_rate": r["hits"] / r["accesses"] if r["accesses"] else 0.0})
        for name in total: total[name] += r[name]
    total["hit_rate"] = total["hits"] / total["accesses"] if total["accesses"] else 0.0
    total["saved_bytes"] = total["baseline_bytes"] - total["transferred_bytes"]
    total["cache_vram_bytes"] = sum(min(layers[k].n, slots.get(k, 0)) * layers[k].bytes for k in layers)
    locality = {"comparisons": comparisons, "mean_previous_token_recall": overlap / comparisons if comparisons else 0.0,
                "exact_route_repeat_rate": exact / comparisons if comparisons else 0.0}
    return {"policy": policy, "mode": scope, "warmup_fraction": warmup, "layers": layer_rows,
            "total": total, "locality": locality,
            "static_plan": {k: sorted(v) for k, v in hot.items()} if policy == "static" else None}


def human_bytes(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB": return f"{value:.2f} {unit}"
        value /= 1024


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("trace", type=pathlib.Path)
    p.add_argument("--policy", choices=("lru", "lfu", "slru", "slru-2touch", "static"), default="slru-2touch")
    cap = p.add_mutually_exclusive_group(required=True); cap.add_argument("--slots"); cap.add_argument("--vram-gib", type=float)
    p.add_argument("--minimum-slots", type=int, default=0); p.add_argument("--scope", choices=("all", "decode", "prefill"), default="decode")
    p.add_argument("--warmup-fraction", type=float, default=.25); p.add_argument("--json", type=pathlib.Path); p.add_argument("--write-plan", type=pathlib.Path)
    args = p.parse_args(argv)
    if not 0 <= args.warmup_fraction < 1: p.error("--warmup-fraction must be in [0, 1)")
    if args.vram_gib is not None and args.vram_gib < 0: p.error("--vram-gib must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    try:
        layers, routes = load(args.trace)
        slots = slots_from(args.slots, layers) if args.slots is not None else allocate(layers, int(args.vram_gib * 1024**3), max(0, args.minimum_slots))
        data = simulate(layers, routes, slots, args.policy, args.scope, args.warmup_fraction)
        total = data["total"]
        print(f"Policy: {args.policy}   Scope: {args.scope}")
        print(f"Cache capacity: {human_bytes(total['cache_vram_bytes'])}")
        print(f"Hit rate: {total['hit_rate'] * 100:.2f}% ({total['hits']:,}/{total['accesses']:,})")
        print(f"RAM→VRAM bytes: {human_bytes(total['transferred_bytes'])} / {human_bytes(total['baseline_bytes'])} baseline")
        print(f"Previous-token route recall: {data['locality']['mean_previous_token_recall'] * 100:.2f}%")
        if args.json: args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if args.write_plan:
            static = simulate(layers, routes, slots, "static", args.scope, args.warmup_fraction)
            plan = {r["layer"]: {"slots": r["slots"], "expert_bundle_bytes": r["bundle_bytes"],
                                  "experts": static["static_plan"].get(r["layer"], [])} for r in static["layers"]}
            args.write_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
