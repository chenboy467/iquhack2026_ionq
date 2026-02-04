
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple
import argparse
import ast
import csv
import heapq
import json
import math
import random
import time

@dataclass(frozen=True)
class Plan:
    start: str
    nodes: Set[str]                 # vertices included in the connected subgraph
    edges_in_order: List[Tuple[str, str]]  # claim actions (u -> v), in executable order
    total_utility: float
    total_bonus: float
    total_edge_cost: float
    final_budget: float

def _coerce_float(x: object, default: float = 0.0) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if s == "":
        return default
    return float(s)

def read_nodes_csv(
    path: str,
    *,
    node_col: str = "node_id",
    utility_col: str = "utility_qubits",
    bonus_col: str = "bonus_bell_pairs",
) -> Tuple[Set[str], Dict[str, float], Dict[str, float]]:
    """
    Returns:
      all_nodes, utility, bonus
    """
    all_nodes: Set[str] = set()
    utility: Dict[str, float] = {}
    bonus: Dict[str, float] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in (node_col, utility_col, bonus_col) if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(f"Nodes CSV missing required columns: {missing}. Found: {rdr.fieldnames}")

        for row in rdr:
            node = str(row[node_col]).strip()
            if not node:
                continue
            all_nodes.add(node)
            utility[node] = _coerce_float(row.get(utility_col), 0.0)
            bonus[node] = _coerce_float(row.get(bonus_col), 0.0)

    return all_nodes, utility, bonus

def _parse_edge_endpoints(row: dict) -> Tuple[str, str]:
    """
    Supports:
      - edge_id column that looks like "['A','B']"
      - separate columns like (u,v) or (source,target) etc.
    """
    if "edge_id" in row and row["edge_id"] is not None and str(row["edge_id"]).strip() != "":
        try:
            u, v = ast.literal_eval(row["edge_id"])
            return str(u).strip(), str(v).strip()
        except Exception as e:
            raise ValueError(f"Failed parsing edge_id={row['edge_id']!r} as python literal list: {e}") from e

    pairs = [
        ("u", "v"),
        ("src", "dst"),
        ("source", "target"),
        ("from", "to"),
        ("node_u", "node_v"),
    ]
    for a, b in pairs:
        if a in row and b in row and str(row[a]).strip() != "" and str(row[b]).strip() != "":
            return str(row[a]).strip(), str(row[b]).strip()

    raise ValueError("Edges CSV has no supported endpoint columns. Expected 'edge_id' or (u,v)/(source,target)/etc.")

def parse_difficulty_map(s: Optional[str]) -> Optional[Dict[int, float]]:
    """
    Parse a string like: "1:1,2:2,3:5,4:8,5:13" -> {1:1.0,2:2.0,...}
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: Dict[int, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad difficulty-map token {part!r}. Expected 'k:v'.")
        k, v = part.split(":", 1)
        out[int(k.strip())] = float(v.strip())
    return out



def read_edges_csv(
    path: str,
    *,
    all_nodes: Set[str],
    difficulty_col: str = "difficulty_rating",
    difficulty_map: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Builds an undirected cost dict-of-dicts:
      cost[u][v] = bell_pair_cost_to_claim_edge(u,v)

    If difficulty_map is None: cost = float(row[difficulty_col])
    Else: cost = difficulty_map[int(difficulty)]
    """
    cost: Dict[str, Dict[str, float]] = {n: {} for n in all_nodes}
    threshold: Dict[str, Dict[str, float]] = {n: {} for n in all_nodes}

    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if difficulty_col not in (rdr.fieldnames or []):
            raise ValueError(f"Edges CSV missing required column '{difficulty_col}'. Found: {rdr.fieldnames}")

        for row in rdr:
            u, v = _parse_edge_endpoints(row)
            if u not in all_nodes or v not in all_nodes:
                raise ValueError(f"Edge references unknown node(s): ({u!r}, {v!r}). Check nodes CSV.")

            diff = _coerce_float(row.get(difficulty_col), 0.0)
            if difficulty_map is None:
                c = float(diff)
            else:
                c = float(difficulty_map[int(round(diff))])

            if c < 0:
                raise ValueError(f"Negative edge cost is not allowed for Dijkstra: edge ({u},{v}) cost={c}")

            prev_uv = cost[u].get(v)
            if prev_uv is None or c < prev_uv:
                cost[u][v] = c
                cost[v][u] = c
                threshold[u][v] = _coerce_float(row.get("base_threshold"))
                threshold[v][u] = _coerce_float(row.get("base_threshold"))

    return cost, threshold

def build_adjacency_from_cost(
    cost: Dict[str, Dict[str, float]],
    all_nodes: Iterable[str],
) -> Dict[str, List[Tuple[str, float]]]:
    adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in all_nodes}
    for u, nbrs in cost.items():
        adj.setdefault(u, [])
        for v, c in nbrs.items():
            adj.setdefault(v, [])
            adj[u].append((v, float(c)))
    return adj

def multisource_dijkstra_no_reenter_tree(
    adj: Dict[str, List[Tuple[str, float]]],
    tree: Set[str],
    weight_fn: Callable[[str, str, float], float],
    rng: random.Random,
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Multi-source Dijkstra from all nodes in `tree` with distance 0.
    Once a path leaves `tree`, it is not allowed to re-enter `tree`.
    Also, we ignore movement within `tree` since you can start from any tree node for free.
    """
    dist: Dict[str, float] = {}
    prev: Dict[str, Optional[str]] = {}
    pq: List[Tuple[float, float, str]] = []

    for s in tree:
        dist[s] = 0.0
        prev[s] = None
        heapq.heappush(pq, (0.0, rng.random(), s))  # random tie-break

    while pq:
        d, _, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue

        u_in_tree = (u in tree)
        for v, base_cost in adj.get(u, []):
            v_in_tree = (v in tree)

            # Don't traverse edges entirely within tree (free "teleport" between tree nodes)
            if u_in_tree and v_in_tree:
                continue
            # Don't re-enter the tree after leaving it
            if (not u_in_tree) and v_in_tree:
                continue

            w = weight_fn(u, v, base_cost)
            if w < 0:
                raise ValueError("Negative weights are not allowed for Dijkstra.")
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, rng.random(), v))

    return dist, prev


def reconstruct_path(prev: Dict[str, Optional[str]], target: str) -> Optional[List[str]]:
    """Reconstruct path using predecessor map. Returns None if target not reachable."""
    if target not in prev:
        return None
    path: List[str] = []
    cur: Optional[str] = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def path_metrics(
    path: List[str],
    tree: Set[str],
    cost: Dict[str, Dict[str, float]],
    utility: Dict[str, float],
    bonus: Dict[str, float],
    kappa: float,
) -> Tuple[float, float, float, float, float, List[Tuple[str, str]]]:
    """
    Compute:
      required_budget: minimum starting budget needed to traverse path without going negative
      delta_utility: utility gained by newly added nodes on this path
      delta_bonus: bonus gained by newly added nodes on this path
      delta_cost: sum of edge costs along path
      delta_value: delta_utility + kappa * delta_bonus
      edges: edges along the path in order
    """
    cash = 0.0
    min_cash = 0.0
    delta_cost = 0.0
    delta_bonus = 0.0
    delta_utility = 0.0
    edges: List[Tuple[str, str]] = []

    for i in range(1, len(path)):
        a = path[i - 1]
        b = path[i]
        c = float(cost[a][b])
        edges.append((a, b))

        # Pay edge cost first
        cash -= c
        delta_cost += c
        min_cash = min(min_cash, cash)

        # If b is new, collect its rewards
        if b not in tree:
            bb = float(bonus.get(b, 0.0))
            uu = float(utility.get(b, 0.0))
            cash += bb
            delta_bonus += bb
            delta_utility += uu

    required_budget = -min_cash
    delta_value = delta_utility + kappa * delta_bonus
    return required_budget, delta_utility, delta_bonus, delta_cost, delta_value, edges


def weighted_choice_top_k(
    candidates: List[Tuple[float, Tuple[List[str], List[Tuple[str, str]], float, float, float]]],
    k: int,
    rng: random.Random,
) -> Tuple[List[str], List[Tuple[str, str]], float, float, float]:
    """
    Pick among top-k candidates with probability proportional to score.
    candidates: list of (score, payload), assumed sorted descending by score.
    """
    k = max(1, min(k, len(candidates)))
    top = candidates[:k]
    weights = [max(0.0, s) for s, _ in top]
    total = sum(weights)
    if total <= 0:
        # fall back to uniform
        return top[rng.randrange(k)][1]
    r = rng.random() * total
    acc = 0.0
    for w, (_, payload) in zip(weights, top):
        acc += w
        if acc >= r:
            return payload
    return top[-1][1]


def greedy_connected_plan(
    adj: Dict[str, List[Tuple[str, float]]],
    cost: Dict[str, Dict[str, float]],
    utility: Dict[str, float],
    bonus: Dict[str, float],
    start: str,
    initial_budget: float,
    *,
    kappa: float = 0.5,
    betas: Sequence[float] = (0.0,),
    top_k: int = 1,
    max_steps: Optional[int] = None,
    rng: Optional[random.Random] = None,
    include_start_bonus: bool = True,
) -> Plan:
    """
    Build a connected subgraph plan from `start` using greedy path additions.

    - kappa: value weight for bonus (bonus treated as future purchasing power)
    - betas: candidate-generation weights; beta>0 biases paths toward bonus-rich nodes
    - top_k: randomized choice among top-k scored paths (GRASP diversification)
    """
    if rng is None:
        rng = random.Random(0)

    tree: Set[str] = {start}
    edges_in_order: List[Tuple[str, str]] = []

    B = float(initial_budget) + (float(bonus.get(start, 0.0)) if include_start_bonus else 0.0)
    total_u = float(utility.get(start, 0.0))
    total_b = float(bonus.get(start, 0.0)) if include_start_bonus else 0.0
    total_c = 0.0

    steps = 0
    all_nodes = list(adj.keys())

    # Dijkstra weight function family (candidate generator)
    eps = 1e-6

    while True:
        if max_steps is not None and steps >= max_steps:
            break

        best_candidates: List[Tuple[float, Tuple[List[str], List[Tuple[str, str]], float, float, float]]] = []
        # Each payload: (path_nodes, path_edges, delta_cost, delta_bonus, delta_utility)

        for beta in betas:
            def weight_fn(u: str, v: str, base_cost: float, beta=beta) -> float:
                # Bias toward grabbing bonus earlier (heuristic). Keep nonnegative.
                return max(eps, float(base_cost) - beta * float(bonus.get(v, 0.0)))

            _, prev = multisource_dijkstra_no_reenter_tree(adj, tree, weight_fn, rng)

            for v in all_nodes:
                if v in tree:
                    continue
                path = reconstruct_path(prev, v)
                if path is None:
                    continue

                req, dU, dB, dC, dV, path_edges = path_metrics(path, tree, cost, utility, bonus, kappa)

                if req <= B and dV > 0:
                    # Use up-front required budget in denominator; this enforces sequential feasibility.
                    denom = max(req, 1.0)   
                    score = dV / denom
                    payload = (path, path_edges, dC, dB, dU)
                    best_candidates.append((score, payload))

        if not best_candidates:
            break

        best_candidates.sort(key=lambda x: x[0], reverse=True)

        # Choose candidate (deterministic best or randomized among top-k)
        if top_k <= 1:
            _, (path, path_edges, dC, dB, dU) = best_candidates[0]
        else:
            payload = weighted_choice_top_k(best_candidates, top_k, rng)
            path, path_edges, dC, dB, dU = payload

        # Execute chosen path step-by-step to maintain true sequential budget.
        for (a, b) in path_edges:
            if b in tree:
                # Shouldn't happen due to "no re-enter tree" rule, but safe-guard anyway.
                continue
            c_ab = float(cost[a][b])
            if c_ab > B + 1e-9:
                # Should not happen if req<=B; but guard against numeric quirks.
                break

            # Pay to claim edge
            B -= c_ab
            total_c += c_ab
            edges_in_order.append((a, b))

            # Arrive at b, collect rewards
            tree.add(b)
            uu = float(utility.get(b, 0.0))
            bb = float(bonus.get(b, 0.0))
            total_u += uu
            total_b += bb
            B += bb

        steps += 1

    return Plan(
        start=start,
        nodes=tree,
        edges_in_order=edges_in_order,
        total_utility=total_u,
        total_bonus=total_b,
        total_edge_cost=total_c,
        final_budget=B,
    )


def choose_best_start_and_subgraph(
    adj: Dict[str, List[Tuple[str, float]]],
    cost: Dict[str, Dict[str, float]],
    utility: Dict[str, float],
    bonus: Dict[str, float],
    starts: Sequence[str],
    initial_budget: float,
    *,
    kappa: float = 0.5,
    betas: Sequence[float] = (0.0, 0.25, 0.5),
    top_k: int = 3,
    runs_per_start: int = 200,
    time_limit_seconds: Optional[float] = None,
    seed: int = 0,
    include_start_bonus: bool = True,
) -> Plan:
    """
    GRASP-style planner:
      - For each candidate start, run greedy_connected_plan multiple times with randomness
      - Keep the plan with maximum total_utility (tie-break by final_budget)
    """

    best_plan: Optional[Plan] = None
    t0 = time.time()
    rng_master = random.Random(seed)

    for si, s in enumerate(starts):
        for r in range(runs_per_start):
            if time_limit_seconds is not None and (time.time() - t0) >= time_limit_seconds:
                break

            # Derive a reproducible per-run RNG
            run_seed = rng_master.randrange(1_000_000_000) ^ (si * 1_000_003 + r * 97)
            rng = random.Random(run_seed)

            plan = greedy_connected_plan(
                adj=adj,
                cost=cost,
                utility=utility,
                bonus=bonus,
                start=s,
                initial_budget=initial_budget,
                kappa=kappa,
                betas=betas,
                top_k=top_k,
                rng=rng,
                include_start_bonus=include_start_bonus,
            )

            if best_plan is None:
                best_plan = plan
            else:
                # Primary: maximize utility; secondary: maximize final budget; tertiary: minimize spent cost
                if (plan.total_utility > best_plan.total_utility + 1e-9 or
                    (abs(plan.total_utility - best_plan.total_utility) <= 1e-9 and plan.final_budget > best_plan.final_budget + 1e-9) or
                    (abs(plan.total_utility - best_plan.total_utility) <= 1e-9 and abs(plan.final_budget - best_plan.final_budget) <= 1e-9 and plan.total_edge_cost < best_plan.total_edge_cost - 1e-9)):
                    best_plan = plan

        if time_limit_seconds is not None and (time.time() - t0) >= time_limit_seconds:
            break

    if best_plan is None:
        raise ValueError("No plan found. Check graph connectivity and input dictionaries.")

    assert verify_plan_sequential_feasibility(best_plan, cost, bonus, initial_budget, include_start_bonus=include_start_bonus)
    return best_plan


def verify_plan_sequential_feasibility(
    plan: Plan,
    cost: Dict[str, Dict[str, float]],
    bonus: Dict[str, float],
    initial_budget: float,
    *,
    include_start_bonus: bool = True
) -> bool:
    """
    Verify that executing edges_in_order never makes budget negative and always expands from owned nodes.
    """
    owned: Set[str] = {plan.start}
    B = float(initial_budget) + (float(bonus.get(plan.start, 0.0)) if include_start_bonus else 0.0)

    for (u, v) in plan.edges_in_order:
        if u not in owned:
            return False
        c_uv = float(cost[u][v])
        if c_uv > B + 1e-9:
            return False
        B -= c_uv
        if v not in owned:
            owned.add(v)
            B += float(bonus.get(v, 0.0))

        if B < -1e-9:
            return False

    return True



def simulate_plan_steps(
    plan: Plan,
    cost: Dict[str, Dict[str, float]],
    threshold: Dict[str, Dict[str, float]],
    utility: Dict[str, float],
    bonus: Dict[str, float],
    initial_budget: float,
    *,
    include_start_bonus: bool = True,
) -> List[dict]:
    owned: Set[str] = {plan.start}
    B = float(initial_budget) + (float(bonus.get(plan.start, 0.0)) if include_start_bonus else 0.0)

    steps: List[dict] = []
    for i, (u, v) in enumerate(plan.edges_in_order, start=1):
        c = float(cost[u][v])
        t = float(threshold[u][v])
        if u not in owned:
            raise RuntimeError(f"Invalid plan: step {i} uses from-node not owned: {u!r}")
        if c > B + 1e-9:
            raise RuntimeError(f"Invalid plan: step {i} insufficient budget. Need {c}, have {B}.")

        B -= c
        gained_u = 0.0
        gained_b = 0.0
        if v not in owned:
            owned.add(v)
            gained_u = float(utility.get(v, 0.0))
            gained_b = float(bonus.get(v, 0.0))
            B += gained_b

        steps.append(
            {
                "step": i,
                "from": u,
                "to": v,
                "edge_cost": c,
                "utility_gained": gained_u,
                "bonus_gained": gained_b,
                "budget_after": B,
                "threshold": t
            }
        )
    return steps

def write_edges_csv(path: str, steps: List[dict]) -> None:
    fields = ["step", "from", "to", "edge_cost", "utility_gained", "bonus_gained", "budget_after"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in steps:
            w.writerow({k: row[k] for k in fields})

def start_heuristic_scores(
    adj: Dict[str, List[Tuple[str, float]]],
    utility: Dict[str, float],
    bonus: Dict[str, float],
    *,
    kappa: float,
    neighbor_weight: float = 0.5,
) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for v, nbrs in adj.items():
        base = float(utility.get(v, 0.0)) + kappa * float(bonus.get(v, 0.0))
        neigh = 0.0
        for w, c in nbrs:
            val = float(utility.get(w, 0.0)) + kappa * float(bonus.get(w, 0.0))
            neigh += val / max(1.0, float(c))
        out.append((base + neighbor_weight * neigh, v))
    out.sort(reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Classical planner for the quantum network game.")
    ap.add_argument("--nodes", required=True, help="Path to nodes CSV (data1.csv).")
    ap.add_argument("--edges", required=True, help="Path to edges CSV (data2.csv).")
    ap.add_argument("--budget", type=float, default=75.0, help="Initial bell-pair budget (default: 75).")

    ap.add_argument(
        "--starts",
        nargs="*",
        default=None,
        help="Optional list of candidate start node IDs. If omitted, the script prescreens nodes and chooses among top starts.",
    )
    ap.add_argument(
        "--preselect",
        type=int,
        default=20,
        help="If --starts is omitted, consider only the top-N heuristic start nodes (default: 20).",
    )

    ap.add_argument("--kappa", type=float, default=0.6, help="Weight for bonus bell pairs in planning value (default: 0.6).")
    ap.add_argument(
        "--betas",
        type=str,
        default="0.0,0.25,0.5",
        help="Comma-separated list of beta values for bonus-biased candidate path generation (default: 0.0,0.25,0.5).",
    )
    ap.add_argument("--top-k", type=int, default=5, help="Randomized choice among top-K candidate paths (default: 5).")

    ap.add_argument("--runs-per-start", type=int, default=500, help="Random restarts per start node (default: 500).")
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional global wall-clock time limit in seconds (e.g., 7200 for 2 hours).",
    )
    ap.add_argument("--seed", type=int, default=123, help="Random seed base (default: 123).")

    ap.add_argument(
        "--difficulty-col",
        type=str,
        default="difficulty_rating",
        help="Column in edges CSV containing difficulty rating (default: difficulty_rating).",
    )
    ap.add_argument(
        "--difficulty-map",
        type=str,
        default=None,
        help="Optional mapping from difficulty->cost, e.g. '1:2,2:3,3:5,4:8,5:13'. If omitted, cost=difficulty.",
    )

    ap.add_argument("--out-json", type=str, default=None, help="Optional output JSON path for the best plan.")
    ap.add_argument("--out-edges", type=str, default=None, help="Optional output CSV path for step-by-step edge claims.")

    args = ap.parse_args()

    # Load inputs
    all_nodes, utility, bonus = read_nodes_csv(args.nodes)
    diff_map = parse_difficulty_map(args.difficulty_map)
    cost, threshold = read_edges_csv(args.edges, all_nodes=all_nodes, difficulty_col=args.difficulty_col, difficulty_map=diff_map)
    adj = build_adjacency_from_cost(cost, all_nodes)

    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip() != ""]

    # Choose starts
    if args.starts is not None and len(args.starts) > 0:
        starts = [str(s).strip() for s in args.starts]
        unknown = [s for s in starts if s not in all_nodes]
        if unknown:
            raise ValueError(f"Unknown start node(s) not found in nodes CSV: {unknown}")
    else:
        hs = start_heuristic_scores(adj, utility, bonus, kappa=args.kappa)
        starts = [v for _, v in hs[: max(1, min(args.preselect, len(hs)))]]

    
    # Solve
    t0 = time.time()

    plan = choose_best_start_and_subgraph(
        adj=adj,
        cost=cost,
        utility=utility,
        bonus=bonus,
        starts=starts,
        initial_budget=args.budget,
        kappa=args.kappa,
        betas=betas,
        top_k=args.top_k,
        runs_per_start=args.runs_per_start,
        time_limit_seconds=args.time_limit,
        seed=args.seed,
        include_start_bonus=True,
    )
    elapsed = time.time() - t0

    # Trace steps (for outputs and validation)
    steps = simulate_plan_steps(plan, cost, threshold, utility, bonus, args.budget, include_start_bonus=True)

    # Console summary
    print("\n=== BEST PLAN ===")
    print(f"Start node:           {plan.start}")
    print(f"Nodes in subgraph:    {len(plan.nodes)}")
    print(f"Edges to claim:       {len(plan.edges_in_order)}")
    print(f"Total utility:        {plan.total_utility:.2f}")
    print(f"Total bonus:          {plan.total_bonus:.2f}")
    print(f"Total edge cost:      {plan.total_edge_cost:.2f}")
    print(f"Initial budget:       {args.budget:.2f}")
    print(f"Final budget:         {plan.final_budget:.2f}")
    print(f"Elapsed (solver):     {elapsed:.2f} s")
    if args.time_limit is not None:
        print(f"Time limit (global):  {args.time_limit:.2f} s")
    print(f"Starts evaluated:     {len(starts)}")

    # Outputs
    if args.out_edges:
        write_edges_csv(args.out_edges, steps)
        print(f"\nWrote edge-claim steps CSV: {args.out_edges}")

    if args.out_json:
        out = {
            "start": plan.start,
            "initial_budget": args.budget,
            "final_budget": plan.final_budget,
            "total_utility": plan.total_utility,
            "total_bonus": plan.total_bonus,
            "total_edge_cost": plan.total_edge_cost,
            "num_nodes": len(plan.nodes),
            "num_edges": len(plan.edges_in_order),
            "nodes": sorted(plan.nodes),
            "edges_in_order": steps,
            "params": {
                "kappa": args.kappa,
                "betas": betas,
                "top_k": args.top_k,
                "runs_per_start": args.runs_per_start,
                "seed": args.seed,
                "difficulty_col": args.difficulty_col,
                "difficulty_map": diff_map,
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote plan JSON: {args.out_json}")


if __name__ == "__main__":
    main()
