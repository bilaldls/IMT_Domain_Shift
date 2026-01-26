import os
import re
import glob
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Utils: load embeddings (.npy per file)
# -----------------------------

def list_npy_files(dir_path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(dir_path, "*.npy")))
    if not files:
        raise FileNotFoundError(f"Aucun .npy trouvé dans: {dir_path}")
    return files


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def stream_mean(dir_path: str, normalize: bool = True, dtype=np.float32) -> np.ndarray:
    """Moyenne (barycentre) en streaming sans charger tout en RAM."""
    files = list_npy_files(dir_path)
    mu = None
    n = 0
    for fp in files:
        v = np.load(fp).astype(dtype, copy=False)
        if normalize:
            v = l2_normalize(v)
        if mu is None:
            mu = np.zeros_like(v, dtype=np.float64)
        mu += v
        n += 1
    mu = (mu / max(n, 1)).astype(np.float32)
    return mu


def stream_rms_radius(dir_path: str, mu: np.ndarray, normalize: bool = True, dtype=np.float32) -> float:
    """RMS radius = sqrt(mean(||x - mu||^2)) en streaming."""
    files = list_npy_files(dir_path)
    sse = 0.0
    n = 0
    mu64 = mu.astype(np.float64)
    for fp in files:
        v = np.load(fp).astype(dtype, copy=False)
        if normalize:
            v = l2_normalize(v)
        d = (v.astype(np.float64) - mu64)
        sse += float(np.dot(d, d))
        n += 1
    return math.sqrt(sse / max(n, 1))


def load_all_embeddings(dir_path: str, normalize: bool = True, dtype=np.float32) -> np.ndarray:
    """Charge tous les embeddings d’un dossier (utile pour NN interne few-shot, qui est petit)."""
    files = list_npy_files(dir_path)
    X = np.stack([np.load(fp).astype(dtype, copy=False) for fp in files], axis=0)
    if normalize:
        X = l2_normalize(X)
    return X


# -----------------------------
# Parsing folders: BASE + SHOT + REPLICATE
# -----------------------------

# On accepte des noms comme:
# DIOR_1SHOT_1, Dior_5shot_07, DOTA-10SHOT-3, etc.
FEWSHOT_RE = re.compile(
    r"^(?P<base>.+?)[_-](?P<shot>\d+)\s*SHOT(?:[_-](?P<rep>\d+))?$",
    re.IGNORECASE
)

def parse_fewshot_name(folder_name: str) -> Optional[Tuple[str, int, Optional[int]]]:
    m = FEWSHOT_RE.match(folder_name)
    if not m:
        return None
    base = m.group("base")
    shot = int(m.group("shot"))
    rep = m.group("rep")
    rep_i = int(rep) if rep is not None else None
    # Normalisation base: DIOR, DOTA, etc. (on garde la casse telle quelle pour retrouver le dossier)
    return base, shot, rep_i


def find_full_dataset_dir(results_root: str, base: str) -> Optional[str]:
    """
    Trouve le dossier du dataset complet (ex: results/DIOR) même si la casse diffère (Dior/DIOR).
    """
    candidates = glob.glob(os.path.join(results_root, "*"))
    base_lower = base.lower()
    for c in candidates:
        if os.path.isdir(c) and os.path.basename(c).lower() == base_lower:
            return c
    return None


# -----------------------------
# Metrics
# -----------------------------

@dataclass
class FewshotMetrics:
    base: str
    shot: int
    replicate: str
    n_few: int
    n_full: int
    bias_mean: float          # ||mu_few - mu_full||
    compact_rms: float        # RMS radius around mu_few
    nn_mean: float            # mean nearest-neighbor distance within few-shot
    coverage_mean: float      # mean distance full -> nearest few-shot


def compute_nn_mean_within(X_few: np.ndarray, metric: str = "euclidean") -> float:
    """
    Distance moyenne au plus proche voisin *dans* le few-shot.
    On utilise k=2 car le plus proche voisin de chaque point est lui-même.
    """
    n = X_few.shape[0]
    if n < 2:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    nn.fit(X_few)
    dists, _ = nn.kneighbors(X_few, return_distance=True)
    # dists[:,0] = 0 (self), dists[:,1] = NN
    return float(np.mean(dists[:, 1]))


def compute_coverage_full_to_few(
    full_dir: str,
    X_few: np.ndarray,
    metric: str = "euclidean",
    normalize: bool = True,
    dtype=np.float32,
) -> Tuple[float, int]:
    """
    Coverage = moyenne sur X_full de dist(x_full, nearest(X_few)).
    On fit NN sur few-shot (petit) et on query le full en streaming (pas besoin de charger tout).
    """
    nn = NearestNeighbors(n_neighbors=1, metric=metric)
    nn.fit(X_few)

    files = list_npy_files(full_dir)
    total = 0.0
    n = 0
    for fp in files:
        v = np.load(fp).astype(dtype, copy=False)
        if normalize:
            v = l2_normalize(v)
        d, _ = nn.kneighbors(v.reshape(1, -1), return_distance=True)
        total += float(d[0, 0])
        n += 1
    return total / max(n, 1), n


def compute_metrics_for_pair(
    base: str,
    full_dir: str,
    few_dir: str,
    shot: int,
    replicate: str,
    metric: str = "euclidean",
    normalize: bool = True,
) -> FewshotMetrics:
    # barycentres
    mu_full = stream_mean(full_dir, normalize=normalize)
    mu_few = stream_mean(few_dir, normalize=normalize)

    bias_mean = float(np.linalg.norm(mu_few - mu_full))

    # compacité few-shot
    compact = stream_rms_radius(few_dir, mu_few, normalize=normalize)

    # NN interne (few-shot -> on charge tout car petit)
    X_few = load_all_embeddings(few_dir, normalize=normalize)
    nn_mean = compute_nn_mean_within(X_few, metric=metric)

    # coverage full -> few (streaming sur full)
    coverage_mean, n_full = compute_coverage_full_to_few(
        full_dir=full_dir,
        X_few=X_few,
        metric=metric,
        normalize=normalize,
    )

    n_few = X_few.shape[0]
    # n_full déjà obtenu
    return FewshotMetrics(
        base=base,
        shot=shot,
        replicate=replicate,
        n_few=n_few,
        n_full=n_full,
        bias_mean=bias_mean,
        compact_rms=float(compact),
        nn_mean=float(nn_mean),
        coverage_mean=float(coverage_mean),
    )


# -----------------------------
# Aggregation across replicates
# -----------------------------

def summarize(metrics: List[FewshotMetrics]) -> Dict[str, float]:
    """Retourne mean/std par métrique pour une liste de réplicats (ex: 10 runs)."""
    def arr(name: str) -> np.ndarray:
        return np.array([getattr(m, name) for m in metrics], dtype=np.float64)

    out = {
        "n_runs": len(metrics),
        "n_few_mean": float(np.mean(arr("n_few"))),
        "bias_mean_mean": float(np.mean(arr("bias_mean"))),
        "bias_mean_std": float(np.std(arr("bias_mean"), ddof=0)),
        "compact_rms_mean": float(np.mean(arr("compact_rms"))),
        "compact_rms_std": float(np.std(arr("compact_rms"), ddof=0)),
        "nn_mean_mean": float(np.nanmean(arr("nn_mean"))),
        "nn_mean_std": float(np.nanstd(arr("nn_mean"), ddof=0)),
        "coverage_mean_mean": float(np.mean(arr("coverage_mean"))),
        "coverage_mean_std": float(np.std(arr("coverage_mean"), ddof=0)),
    }
    return out


def analyze_results_tree(
    results_root: str = "results",
    only_bases: Optional[List[str]] = None,
    metric: str = "euclidean",
    normalize: bool = True,
    save_csv: Optional[str] = "fewshot_internal_metrics.csv",
) -> None:
    """
    - Scanne results/
    - détecte les dossiers few-shot BASE_1SHOT_*, BASE_5SHOT_*, BASE_10SHOT_*
    - trouve le dossier full results/BASE
    - calcule métriques par réplicat + résumé par shot
    """
    results_root = os.path.abspath(results_root)
    folders = [f for f in glob.glob(os.path.join(results_root, "*")) if os.path.isdir(f)]

    # collecte few-shot dirs
    by_base_shot: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
    for d in folders:
        name = os.path.basename(d)
        parsed = parse_fewshot_name(name)
        if parsed is None:
            continue
        base, shot, rep = parsed
        if only_bases and base.lower() not in {b.lower() for b in only_bases}:
            continue
        rep_str = str(rep) if rep is not None else "NA"
        by_base_shot.setdefault((base, shot), []).append((rep_str, d))

    if not by_base_shot:
        raise RuntimeError(
            f"Aucun dossier few-shot détecté dans {results_root}.\n"
            f"Attendu: BASE_1SHOT_1, BASE_5SHOT_3, BASE_10SHOT_10, etc."
        )

    rows = []
    print(f"== Analyse dans {results_root} | metric={metric} | normalize={normalize} ==")

    for (base, shot), rep_dirs in sorted(by_base_shot.items(), key=lambda x: (x[0][0].lower(), x[0][1])):
        full_dir = find_full_dataset_dir(results_root, base)
        if full_dir is None:
            print(f"[WARN] Dataset complet introuvable pour base={base} (attendu results/{base}). Skip.")
            continue

        print(f"\n--- BASE={base} | SHOT={shot} | full={os.path.basename(full_dir)} | runs={len(rep_dirs)} ---")

        run_metrics: List[FewshotMetrics] = []
        for rep_str, few_dir in sorted(rep_dirs, key=lambda t: int(t[0]) if t[0].isdigit() else 10**9):
            m = compute_metrics_for_pair(
                base=base,
                full_dir=full_dir,
                few_dir=few_dir,
                shot=shot,
                replicate=rep_str,
                metric=metric,
                normalize=normalize,
            )
            run_metrics.append(m)

            d = asdict(m)
            d["level"] = "run"
            rows.append(d)
            print(f"  run={rep_str:>2} | n_few={m.n_few:<4} | bias={m.bias_mean:.4f} | compact={m.compact_rms:.4f} | nn={m.nn_mean:.4f} | cov={m.coverage_mean:.4f}")

        # summary over runs
        s = summarize(run_metrics)
        summary_row = {
            "level": "summary",
            "base": base,
            "shot": shot,
            "replicate": "ALL",
            "n_few": s["n_few_mean"],
            "n_full": run_metrics[0].n_full if run_metrics else None,
            "bias_mean": s["bias_mean_mean"],
            "bias_mean_std": s["bias_mean_std"],
            "compact_rms": s["compact_rms_mean"],
            "compact_rms_std": s["compact_rms_std"],
            "nn_mean": s["nn_mean_mean"],
            "nn_mean_std": s["nn_mean_std"],
            "coverage_mean": s["coverage_mean_mean"],
            "coverage_mean_std": s["coverage_mean_std"],
            "n_runs": s["n_runs"],
        }
        rows.append(summary_row)

        print(f"  ==> SUMMARY | bias={s['bias_mean_mean']:.4f}±{s['bias_mean_std']:.4f} | "
              f"compact={s['compact_rms_mean']:.4f}±{s['compact_rms_std']:.4f} | "
              f"nn={s['nn_mean_mean']:.4f}±{s['nn_mean_std']:.4f} | "
              f"cov={s['coverage_mean_mean']:.4f}±{s['coverage_mean_std']:.4f}")

    # Save CSV (optionnel)
    if save_csv:
        import csv
        out_path = os.path.abspath(os.path.join("Stable", save_csv))
        # union des clés
        keys = set()
        for r in rows:
            keys.update(r.keys())
        keys = sorted(keys)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"\n[OK] CSV écrit: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results")
    p.add_argument("--bases", nargs="*", default=None, help="Limiter à certaines bases (ex: DIOR DOTA)")
    p.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--csv", type=str, default="fewshot_internal_metrics.csv")
    args = p.parse_args()

    analyze_results_tree(
        results_root=args.results,
        only_bases=args.bases,
        metric=args.metric,
        normalize=not args.no_normalize,
        save_csv=args.csv,
    )