#!/usr/bin/env python3
"""
HYDRAS Source Seeking — Explainability della policy PPO.

Tre analisi, in ordine di forza probatoria:

1) ABLATION CAUSALE DI GRUPPO — quali blocchi di feature SERVONO al task.
   Azzera un blocco nello spazio normalizzato (con VecNormalize azzerare equivale
   a sostituire il blocco con la media vista in training: "sensore che riporta il
   valore tipico") e rimisura success rate, step al successo, distanza finale.
   È un intervento vero sul simulatore, non un surrogato dell'importanza: nel
   supervised learning i metodi post-hoc servono perché non puoi rifare il mondo
   senza una feature, qui invece puoi.
   Design APPAIATO: stesso seed di spawn per tutte le condizioni, così la
   differenza baseline-vs-ablazione non è confusa dalla varianza degli spawn.

2) VALORI DI SHAPLEY ESATTI — quali blocchi la policy GUARDA.
   Con 12 gruppi le coalizioni sono 2^12 = 4096: enumerabili, quindi i valori di
   Shapley si calcolano ESATTAMENTE e non serve alcuno stimatore (KernelSHAP,
   DeepSHAP). Payoff v(S) = pi(a*|x_S), con a* l'azione scelta a osservazione
   piena e i gruppi fuori da S posti alla baseline. Per efficienza vale
   sum_i phi_i = pi(a*|x_pieno) - pi(a*|baseline) — verificato a runtime.

3) ANALISI COMPORTAMENTALE — QUALE STRATEGIA ha imparato.
   - heatmap sensore-massimo vs direzione scelta: gradient ascent locale o no?
   - response curve sul vento: anemotassi vera o vento decorativo?
   - V(s) vs distanza dalla sorgente: il critic ha imparato una stima di distanza?
   - entropia della policy vs distanza: esplora quando perde il plume?

Nota su masking: MaskablePPO mette -inf sui logit delle azioni che finirebbero a
terra o fuori dominio. Le attribuzioni e le sonde comportamentali usano i logit
PRE-MASK, altrimenti si misura la geometria della costa invece della policy.
L'ablation invece gira con il masking attivo, perché lì interessa il task reale.

Uso:
    python src/explainability.py                    # tutto, modello dual-ring v_max=2
    python src/explainability.py --stages behavior  # solo una parte
    python src/explainability.py --n-sources 3 --n-episodes 1   # pilota veloce
"""

import argparse
import json
import sys
from collections import OrderedDict
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from inference import (
    MASKABLE_PPO_AVAILABLE,
    build_env,
    get_inner_env,
    load_config,
    load_model,
    make_env_config,
)
from utils.data_loader import DataManager


# ─── Palette (light mode: le figure finiscono nel PDF di tesi) ───────────────
INK          = "#0b0b0b"
INK_SECOND   = "#52514e"
INK_MUTED    = "#898781"
GRID         = "#e1e0d9"
BASELINE     = "#c3c2b7"
SURFACE      = "#fcfcfb"
SERIES_BLUE  = "#2a78d6"
DIVERGE_RED  = "#d03b3b"
NEUTRAL_MID  = "#f0efec"

# Rampa sequenziale a una sola tinta (blu 100→700), non un rainbow.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# Diverging blu↔rosso con midpoint grigio neutro.
DIV_BR = LinearSegmentedColormap.from_list(
    "div_br", ["#0d366b", "#3987e5", "#cde2fb", NEUTRAL_MID, "#f3c0c0", "#e07575", DIVERGE_RED]
)


def apply_style() -> None:
    """Chrome recessivo: griglia hairline, assi tenui, niente cornice."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECOND,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
    })


# ─── Gruppi semantici dell'osservazione ──────────────────────────────────────
# Le 196 feature del dual-ring sono blocchi fortemente collineari (72 valori sono
# la storia della corona 1). "Quanto conta la feature 87" non ha risposta sensata:
# è il sensore NE di 5 step fa, ridondante con una dozzina di altri. L'unità di
# analisi corretta — e l'unità di cui si parla in tesi — è il blocco.

GROUP_LABELS = {
    "conc_now":    "Concentrazione (istante)",
    "conc_hist":   "Concentrazione (storia)",
    "displ_hist":  "Spostamenti (storia)",
    "ring1_now":   "Corona 20 m (istante)",
    "ring2_now":   "Corona 50 m (istante)",
    "ring1_hist":  "Corona 20 m (storia)",
    "ring2_hist":  "Corona 50 m (storia)",
    "wind":        "Vento (u, v)",
    "current":     "Corrente (u, v)",
    "maxconc_vec": "Vettore verso max conc.",
    "maxconc_val": "Valore max conc.",
    "plume_age":   "Età ultimo contatto",
}


def obs_groups(cfg) -> "OrderedDict[str, np.ndarray]":
    """Indici delle feature per gruppo semantico, derivati dal layout di
    SourceSeekingEnv._get_observation (single-ring 116 o dual-ring 196).
    """
    m = int(cfg.memory_length)
    dual = cfg.sensor_range_2 is not None
    g: "OrderedDict[str, np.ndarray]" = OrderedDict()
    i = 0

    def take(name: str, n: int) -> None:
        nonlocal i
        g[name] = np.arange(i, i + n)
        i += n

    take("conc_now", 1)
    take("conc_hist", m)
    take("displ_hist", 2 * m)
    take("ring1_now", 8)
    if dual:
        take("ring2_now", 8)
    take("ring1_hist", 8 * m)
    if dual:
        take("ring2_hist", 8 * m)
    take("wind", 2)
    take("current", 2)
    take("maxconc_vec", 2)
    take("maxconc_val", 1)
    take("plume_age", 1)

    expected = (1 + m + 2 * m + 8 + (8 if dual else 0)
                + 8 * m + (8 * m if dual else 0) + 2 + 2 + 4)
    assert i == expected, f"layout gruppi incoerente: {i} != {expected}"
    return g


DIR_NAMES = ["N", "S", "E", "O", "NE", "SE", "NO", "SO"]
# Angoli delle 8 direzioni di _ACTION_MAP, in radianti (0 = Est, antiorario).
DIR_ANGLES = np.array([
    np.pi / 2,        # 0 N
    -np.pi / 2,       # 1 S
    0.0,              # 2 E
    np.pi,            # 3 O
    np.pi / 4,        # 4 NE
    -np.pi / 4,       # 5 SE
    3 * np.pi / 4,    # 6 NO
    -3 * np.pi / 4,   # 7 SO
])


# ─── Accesso alla policy (pre-mask) ──────────────────────────────────────────

def policy_forward(model, obs_np: np.ndarray, batch: int = 4096):
    """Distribuzione PRE-MASK e valore del critic per un batch di osservazioni
    già normalizzate. Ritorna (probs (N,A), values (N,), entropy (N,)).
    """
    probs_all, vals_all, ent_all = [], [], []
    with torch.no_grad():
        for s in range(0, len(obs_np), batch):
            obs_t = torch.as_tensor(obs_np[s:s + batch], dtype=torch.float32)
            dist = model.policy.get_distribution(obs_t)      # action_masks=None → nessun mask
            probs_all.append(dist.distribution.probs.cpu().numpy())
            ent_all.append(dist.entropy().cpu().numpy())
            vals_all.append(model.policy.predict_values(obs_t).cpu().numpy().ravel())
    return (np.concatenate(probs_all), np.concatenate(vals_all), np.concatenate(ent_all))


def policy_probs_only(model, obs_np: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Solo le probabilità pre-mask (hot path dello Shapley: evita il critic)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(obs_np), batch):
            obs_t = torch.as_tensor(obs_np[s:s + batch], dtype=torch.float32)
            out.append(model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy())
    return np.concatenate(out)


# ─── Scenari di valutazione (stesso protocollo held-out dell'inferenza) ──────

def build_scenarios(data_manager: DataManager, n_sources: int) -> List[tuple]:
    """(source_id, version) held-out: SRC107-SRC132, come run_inference."""
    all_sources = data_manager.get_discovered_sources()
    held_out = sorted(s for s in all_sources if int(s[3:]) > 106)
    if n_sources > 0:
        # Sottocampionamento uniforme sull'intervallo, non i primi N: copre
        # tutta la geometria delle sorgenti held-out.
        idx = np.linspace(0, len(held_out) - 1, min(n_sources, len(held_out)))
        held_out = [held_out[int(round(k))] for k in idx]
    return [(s, v) for s in held_out for v in ("V0", "V1", "V2", "V3")]


WIND_MAPPING = {
    "_V0": "CI_WIND_faseII_V0.txt",
    "_V1": "CI_WIND_faseII_V1.txt",
    "_V2": "CI_WIND_faseII_V2.txt",
    "_V3": "CI_WIND_faseII_V3.txt",
}
CURRENT_MAPPING = {
    "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
    "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
    "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
    "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
}


def load_field(data_manager: DataManager, source_id: str, version: str):
    files = [f for f in data_manager._nc_files if version in f.name and source_id in f.name]
    if not files:
        return None
    field = data_manager._nc_loader.load(str(files[0]),
                                         concentration_var="Concentration - component 1")
    if field is None:
        return None
    coords = data_manager.get_source_coordinates(source_id)
    if coords:
        field.source_position = coords
    field.run_id = f"{source_id}_{version}"
    return field


# ─── Rollout con ablazione ───────────────────────────────────────────────────

def run_episode_ablated(model, vec_env, ablate_idx: Optional[np.ndarray], seed: int,
                        collect: bool = False) -> dict:
    """Un episodio con un blocco di feature azzerato DOPO VecNormalize.

    Il seed rende lo spawn riproducibile: chiamando questa funzione con lo stesso
    seed per condizioni diverse si ottiene un confronto APPAIATO (stesso punto di
    partenza, stesso campo), che è ciò che rende leggibile un delta di pochi punti
    di success rate.

    Con collect=True registra anche il buffer di stati per le analisi 2 e 3.
    """
    vec_env.seed(seed)
    obs = vec_env.reset()
    if ablate_idx is not None:
        obs[:, ablate_idx] = 0.0

    inner = get_inner_env(vec_env)
    spawn = inner.state.position.copy()
    initial_dist = float(np.linalg.norm(spawn - inner.source_position))

    buf_norm, buf_raw, buf_act, buf_dist = [], [], [], []
    done = False
    last_info: dict = {}

    while not done:
        if collect:
            buf_norm.append(obs[0].copy())
            buf_raw.append(vec_env.get_original_obs()[0].copy())

        if MASKABLE_PPO_AVAILABLE:
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=vec_env.env_method("action_masks")[0])
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, dones, infos = vec_env.step(action)
        if ablate_idx is not None:
            obs[:, ablate_idx] = 0.0
        done = bool(dones[0])
        last_info = infos[0]

        if collect:
            buf_act.append(int(np.asarray(action).flatten()[0]))
            buf_dist.append(float(last_info.get("distance_to_source", np.nan)))

    termination = last_info.get("termination_reason")
    if termination not in {"success", "boundary", "land", "timeout"}:
        if last_info.get("source_reached", False):
            termination = "success"
        elif last_info.get("out_of_bounds", False):
            termination = "boundary"
        elif last_info.get("on_land", False):
            termination = "land"
        else:
            termination = "timeout"

    out = {
        "success": termination == "success",
        "termination": termination,
        "steps": int(last_info.get("steps", 0)),
        "initial_distance": initial_dist,
        "final_distance": float(last_info.get("distance_to_source", np.nan)),
    }
    if collect:
        out["obs_norm"] = np.array(buf_norm, dtype=np.float32)
        out["obs_raw"] = np.array(buf_raw, dtype=np.float32)
        out["actions"] = np.array(buf_act, dtype=np.int64)
        out["distances"] = np.array(buf_dist, dtype=np.float32)
    return out


# ─── 1) Ablation causale di gruppo ───────────────────────────────────────────

def bootstrap_ci(stat_fn, n: int, reps: int = 2000, seed: int = 0) -> tuple:
    """IC 95% bootstrap sugli SPAWN (l'unità di ricampionamento è lo spawn, non
    l'episodio: baseline e ablazione sullo stesso spawn sono appaiate e vanno
    ricampionate insieme, altrimenti l'IC ignora la correlazione e si allarga)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        v = stat_fn(rng.integers(0, n, n))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def stage_ablation(model, config, data_manager, groups, scenarios, chunk_ids,
                   n_episodes, vec_norm_path, out_dir) -> dict:
    conditions: List[tuple] = [("baseline", None)]
    conditions += [(name, idx) for name, idx in groups.items()]

    # Metriche per spawn e per condizione, allineate per indice: il confronto è
    # appaiato, quindi va tenuta la corrispondenza spawn-per-spawn.
    acc = {name: {"success": [], "steps": [], "final": []} for name, _ in conditions}
    n_done = 0

    for source_id, version in scenarios:
        field = load_field(data_manager, source_id, version)
        if field is None:
            print(f"  [SKIP] {source_id}_{version}: campo non caricabile")
            continue

        for chunk_id in chunk_ids:
            env_cfg = make_env_config(config, chunk_id=chunk_id)
            vec_env = build_env(env_cfg, field, vec_norm_path,
                                use_masking=MASKABLE_PPO_AVAILABLE,
                                data_manager=data_manager,
                                wind_data=None, current_data=None,
                                wind_mapping=WIND_MAPPING, current_mapping=CURRENT_MAPPING)
            try:
                for ep in range(n_episodes):
                    # Lo stesso seed per tutte le condizioni: confronto appaiato.
                    seed = abs(hash((source_id, version, chunk_id, ep))) % (2**31)
                    for name, idx in conditions:
                        r = run_episode_ablated(model, vec_env, idx, seed=seed)
                        acc[name]["success"].append(1.0 if r["success"] else 0.0)
                        acc[name]["steps"].append(float(r["steps"]))
                        acc[name]["final"].append(r["final_distance"])
                    n_done += 1
                    if n_done % 25 == 0:
                        sr_b = 100 * float(np.mean(acc["baseline"]["success"]))
                        print(f"  [{n_done} spawn appaiati]  SR baseline = {sr_b:.1f}%")
            finally:
                vec_env.close()

    A = {name: {k: np.asarray(v) for k, v in d.items()} for name, d in acc.items()}
    base = A["baseline"]
    n_spawns = len(base["success"])
    base_sr = 100 * float(base["success"].mean())

    rows = []
    for name, _ in conditions:
        cur = A[name]
        both = (base["success"] == 1) & (cur["success"] == 1)

        def d_sr(idx, cur=cur):
            return 100 * (cur["success"][idx].mean() - base["success"][idx].mean())

        def d_steps(idx, cur=cur, both=both):
            m = both[idx]
            if m.sum() < 10:
                return None
            b = base["steps"][idx][m]
            return 100 * (cur["steps"][idx][m] - b).mean() / b.mean()

        delta_sr = 100 * float(cur["success"].mean() - base["success"].mean())
        dsp = (100 * float((cur["steps"][both] - base["steps"][both]).mean())
               / float(base["steps"][both].mean())) if both.sum() >= 10 else float("nan")

        sr_lo, sr_hi = bootstrap_ci(d_sr, n_spawns)
        st_lo, st_hi = bootstrap_ci(d_steps, n_spawns)

        rows.append({
            "group": name,
            "label": "— nessuna ablazione —" if name == "baseline" else GROUP_LABELS[name],
            "success_rate": 100 * float(cur["success"].mean()),
            "delta_sr": delta_sr,
            "delta_sr_ci": [sr_lo, sr_hi],
            "delta_steps_pct": dsp,
            "delta_steps_ci": [st_lo, st_hi],
            "mean_steps_success": float(cur["steps"][cur["success"] == 1].mean())
                                  if (cur["success"] == 1).any() else float("nan"),
            "mean_final_distance": float(np.nanmean(cur["final"])),
            "n_paired_both_success": int(both.sum()),
        })

    res = {"baseline_sr": base_sr, "rows": rows, "n_paired_spawns": n_spawns,
           "baseline_mean_steps": float(base["steps"][base["success"] == 1].mean())}
    plot_ablation(res, out_dir / "fig_ablation_gruppi.png")
    return res


def _evidence_color(lo: float, hi: float, worse_if_positive: bool) -> str:
    """Colore per classe di evidenza, non per magnitudine: la magnitudine è già
    sull'asse, ri-codificarla nel colore sbiadisce le barre senza aggiungere
    informazione. Qui il colore dice se l'IC esclude lo zero, cioè se il gruppo è
    distinguibile dalla baseline — che è la domanda che il lettore si pone.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return BASELINE
    if lo > 0:
        return DIVERGE_RED if worse_if_positive else SERIES_BLUE
    if hi < 0:
        return SERIES_BLUE if worse_if_positive else DIVERGE_RED
    return BASELINE            # IC attraversa lo zero: nessuna evidenza


def plot_ablation(res: dict, path: Path) -> None:
    """Due metriche, due pannelli (mai due assi sullo stesso plot).

    L'ordinamento segue l'efficienza (step al successo), la metrica con più
    risoluzione: il success rate parte da 99.8% e ha poco margine per peggiorare,
    quindi da solo separerebbe male i gruppi minori.
    """
    rows = [r for r in res["rows"] if r["group"] != "baseline"]
    rows.sort(key=lambda r: (r["delta_steps_pct"] if np.isfinite(r["delta_steps_pct"])
                             else -1e9))
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), sharey=True)

    # Pannello 1 — Δ step al successo (%): positivo = più lento = peggio
    ds = np.array([r["delta_steps_pct"] for r in rows])
    lo = np.array([r["delta_steps_ci"][0] for r in rows])
    hi = np.array([r["delta_steps_ci"][1] for r in rows])
    ax = axes[0]
    ax.barh(y, ds, height=0.62,
            color=[_evidence_color(a, b, True) for a, b in zip(lo, hi)])
    ax.errorbar(ds, y, xerr=[ds - lo, hi - ds], fmt="none", ecolor=INK_SECOND,
                elinewidth=1.1, capsize=2.5)
    ax.axvline(0, color=INK_SECOND, lw=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ step al successo (%, appaiato)")
    ax.set_title(f"Costo in efficienza — baseline {res['baseline_mean_steps']:.0f} step",
                 loc="left")
    ax.grid(axis="y", visible=False)

    # Pannello 2 — ΔSR (punti percentuali): negativo = peggio
    dsr = np.array([r["delta_sr"] for r in rows])
    slo = np.array([r["delta_sr_ci"][0] for r in rows])
    shi = np.array([r["delta_sr_ci"][1] for r in rows])
    ax = axes[1]
    ax.barh(y, dsr, height=0.62,
            color=[_evidence_color(a, b, False) for a, b in zip(slo, shi)])
    ax.errorbar(dsr, y, xerr=[dsr - slo, shi - dsr], fmt="none", ecolor=INK_SECOND,
                elinewidth=1.1, capsize=2.5)
    ax.axvline(0, color=INK_SECOND, lw=1.0)
    ax.set_xlabel("Δ success rate (punti percentuali)")
    ax.set_title(f"Costo in riuscita — baseline {res['baseline_sr']:.1f}%", loc="left")
    ax.grid(axis="y", visible=False)

    # La legenda spiega il colore (classe di evidenza), che non è auto-esplicativo.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DIVERGE_RED),
        plt.Rectangle((0, 0), 1, 1, color=BASELINE),
    ]
    axes[1].legend(handles, ["IC 95% esclude lo zero", "IC 95% contiene lo zero"],
                   loc="lower right", fontsize=8)

    fig.suptitle(f"Ablation causale di gruppo — {res['n_paired_spawns']} spawn appaiati, "
                 f"IC 95% bootstrap", x=0.012, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)


# ─── 2) Valori di Shapley esatti sui gruppi ──────────────────────────────────

def stage_shapley(model, groups, obs_norm: np.ndarray, out_dir: Path) -> dict:
    """Shapley ESATTO su n gruppi: 2^n coalizioni enumerate, nessuno stimatore.

    v(S) = pi(a*|x_S), a* = argmax pi(.|x_pieno), gruppi fuori da S alla baseline
    (zero = media di training di VecNormalize).
    """
    names = list(groups.keys())
    n = len(names)
    n_states = len(obs_norm)
    print(f"  {n} gruppi → {2**n} coalizioni × {n_states} stati (Shapley esatto)")

    a_star = policy_probs_only(model, obs_norm).argmax(axis=1)
    rows = np.arange(n_states)

    # v[S] per ogni coalizione: probabilità dell'azione a* con solo i gruppi in S.
    v = np.zeros((2 ** n, n_states), dtype=np.float32)
    for S in range(2 ** n):
        x = np.zeros_like(obs_norm)
        for i in range(n):
            if S & (1 << i):
                x[:, groups[names[i]]] = obs_norm[:, groups[names[i]]]
        v[S] = policy_probs_only(model, x)[rows, a_star]
        if S % 512 == 0:
            print(f"    coalizione {S}/{2**n}")

    # phi_i = sum_{S non contiene i} |S|!(n-|S|-1)!/n! * [v(S+i) - v(S)]
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    popcount = np.array([bin(S).count("1") for S in range(2 ** n)])
    phi = np.zeros((n_states, n), dtype=np.float64)
    for i in range(n):
        bit = 1 << i
        without = np.where((np.arange(2 ** n) & bit) == 0)[0]
        contrib = (v[without | bit] - v[without])           # (n_subsets, n_states)
        phi[:, i] = (w[popcount[without]][:, None] * contrib).sum(axis=0)

    # Efficienza: sum_i phi_i == v(pieno) - v(vuoto). Se salta, il calcolo è rotto.
    eff_lhs = phi.sum(axis=1)
    eff_rhs = v[2 ** n - 1] - v[0]
    eff_err = float(np.abs(eff_lhs - eff_rhs).max())
    print(f"  errore massimo di efficienza: {eff_err:.2e}  (deve essere ~0)")

    res = {
        "groups": names,
        "mean_abs_phi": {names[i]: float(np.abs(phi[:, i]).mean()) for i in range(n)},
        "mean_phi": {names[i]: float(phi[:, i].mean()) for i in range(n)},
        "efficiency_max_error": eff_err,
        "n_states": n_states,
        "mean_prob_full": float(v[2 ** n - 1].mean()),
        "mean_prob_baseline": float(v[0].mean()),
    }
    plot_shapley(res, out_dir / "fig_shapley_gruppi.png")
    return res


def plot_shapley(res: dict, path: Path) -> None:
    """Importanza globale = media di |phi|. Magnitudine sull'asse, non nel colore:
    serie unica → una sola tinta, nessuna legenda."""
    items = sorted(res["mean_abs_phi"].items(), key=lambda kv: kv[1])
    labels = [GROUP_LABELS[k] for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(items))
    ax.barh(y, vals, color=SERIES_BLUE, height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"|$\phi$| medio  —  contributo a $\pi(a^*|x)$")
    ax.set_title(f"Valori di Shapley esatti sui gruppi — {res['n_states']} stati, "
                 f"{2**len(res['groups'])} coalizioni", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.14)      # spazio per le etichette a fine barra
    for yi, val in zip(y, vals):
        ax.text(val + max(vals) * 0.012, yi, f"{val:.3f}", va="center",
                fontsize=8, color=INK_SECOND)
    fig.savefig(path)
    plt.close(fig)


# ─── 3) Analisi comportamentale ──────────────────────────────────────────────

def stage_behavior(model, cfg, groups, buf: dict, vec_env_for_norm, out_dir: Path) -> dict:
    K = int(cfg.n_velocity_levels)
    obs_norm, obs_raw = buf["obs_norm"], buf["obs_raw"]
    actions, dists = buf["actions"], buf["distances"]
    dir_idx = actions // K

    res: dict = {}

    # (a) sensore direzionale massimo vs direzione scelta.
    # L'argmax va preso sui sensori GREZZI: VecNormalize normalizza per-feature con
    # media/std diverse per sensore, quindi l'argmax del vettore normalizzato non è
    # l'argmax fisico.
    ring1 = obs_raw[:, groups["ring1_now"]]
    in_plume = ring1.max(axis=1) > 0.5          # stessa soglia dello spawn
    counts = np.zeros((8, 8))
    if in_plume.sum() > 0:
        smax = ring1[in_plume].argmax(axis=1)
        dsel = dir_idx[in_plume]
        for s, d in zip(smax, dsel):
            counts[s, d] += 1
    res["sensor_action"] = {
        "counts": counts.tolist(),               # grezzi: servono per il marginale
        "frac_states_in_plume": float(in_plume.mean()),
        "n_states": int(in_plume.sum()),
        **sensor_action_stats(counts),
    }
    plot_sensor_action(counts, out_dir / "fig_sensore_azione.png")

    # (b) response curve sul vento: ruoto il vento a 360° tenendo fermo il resto.
    # Sonda CONTROFATTUALE: dice se l'output della policy dipende dall'input vento,
    # non cosa succederebbe in un mondo con vento diverso (vento e forma del plume
    # sono fisicamente accoppiati, qui li disaccoppio di proposito).
    n_probe = min(400, len(obs_raw))
    sel = np.linspace(0, len(obs_raw) - 1, n_probe).astype(int)
    probe_raw = obs_raw[sel].copy()
    w_idx = groups["wind"]
    wu, wv = probe_raw[:, w_idx[0]].copy(), probe_raw[:, w_idx[1]].copy()
    w_mag = np.sqrt(wu**2 + wv**2)
    w_mag[w_mag < 1e-6] = float(np.median(w_mag[w_mag > 1e-6])) if (w_mag > 1e-6).any() else 1.0

    thetas = np.deg2rad(np.arange(0, 360, 10))
    resp = np.zeros((len(thetas), 8))
    for t, th in enumerate(thetas):
        x = probe_raw.copy()
        x[:, w_idx[0]] = w_mag * np.cos(th)
        x[:, w_idx[1]] = w_mag * np.sin(th)
        p = policy_probs_only(model, vec_env_for_norm.normalize_obs(x))
        p_dir = p.reshape(len(x), 8, K).sum(axis=2)      # marginalizza la velocità
        resp[t] = p_dir.mean(axis=0)

    # Quanto la direzione preferita insegue il "controvento" (θ+180)?
    pref = np.array([DIR_ANGLES[resp[t].argmax()] for t in range(len(thetas))])
    upwind = np.angle(np.exp(1j * (thetas + np.pi)))
    align = float(np.mean(np.cos(pref - upwind)))
    res["wind_response"] = {
        "matrix": resp.tolist(),
        "thetas_deg": np.rad2deg(thetas).tolist(),
        "upwind_alignment": align,
        "n_probe_states": int(n_probe),
    }
    plot_wind_response(resp, np.rad2deg(thetas), align, out_dir / "fig_response_vento.png")

    # (c)+(d) V(s) e entropia (pre-mask) vs distanza dalla sorgente.
    _, values, entropy = policy_forward(model, obs_norm)
    ok = np.isfinite(dists) & np.isfinite(values) & np.isfinite(entropy)
    bins = np.linspace(0, min(3000, np.nanpercentile(dists[ok], 99)), 26)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(dists[ok], bins) - 1
    v_ok, e_ok = values[ok], entropy[ok]

    def binned(arr):
        med, lo, hi = [], [], []
        for b in range(len(ctr)):
            s = arr[which == b]
            if len(s) < 5:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                med.append(np.median(s)); lo.append(np.percentile(s, 25)); hi.append(np.percentile(s, 75))
        return np.array(med), np.array(lo), np.array(hi)

    vm, vlo, vhi = binned(v_ok)
    em, elo, ehi = binned(e_ok)
    corr = float(np.corrcoef(dists[ok], v_ok)[0, 1])
    # Salvo anche i quartili, non solo la mediana: senza di essi la figura non è
    # ricostruibile dal JSON (--replot).
    res["value_vs_distance"] = {"pearson_r": corr, "bin_centers": ctr.tolist(),
                                "median": vm.tolist(), "q25": vlo.tolist(),
                                "q75": vhi.tolist()}
    res["entropy_vs_distance"] = {"bin_centers": ctr.tolist(), "median": em.tolist(),
                                  "q25": elo.tolist(), "q75": ehi.tolist()}
    plot_value_entropy(ctr, (vm, vlo, vhi), (em, elo, ehi), corr,
                       out_dir / "fig_valore_entropia.png")
    return res


def sensor_action_stats(counts: np.ndarray) -> dict:
    """Statistiche della relazione sensore-massimo → azione.

    La massa diagonale grezza va confrontata con quella ATTESA SOTTO INDIPENDENZA,
    non con 1/8: gli spawn stanno sul plume, che è avvettato a valle della sorgente,
    quindi la sorgente cade quasi sempre in una direzione sistematica e il marginale
    P(azione) è fortemente sbilanciato. Una diagonale sopra 1/8 può essere solo quel
    bias geografico, non una risposta al sensore.
    """
    tot = counts.sum()
    if tot == 0:
        return {"diagonal_mass": float("nan"), "diagonal_expected": float("nan"),
                "diagonal_lift": float("nan")}
    p_joint = counts / tot
    p_s = p_joint.sum(axis=1)            # marginale del sensore massimo
    p_a = p_joint.sum(axis=0)            # marginale dell'azione (il bias geografico)
    diag = float(np.trace(p_joint))
    diag_exp = float((p_s * p_a).sum())  # atteso se azione e sensore fossero indipendenti
    return {"diagonal_mass": diag,
            "diagonal_expected": diag_exp,
            "diagonal_lift": float(diag / diag_exp) if diag_exp > 0 else float("nan")}


def plot_sensor_action(counts: np.ndarray, path: Path) -> None:
    """Due pannelli: la condizionale (leggibile) e il lift (deconfondato).

    Il pannello di sinistra risponde a "dove va l'agente quando il sensore X è il più
    intenso", ma è dominato dal bias direzionale degli scenari. Quello di destra divide
    per il marginale P(azione) e isola quindi il solo contributo del sensore: 1 = il
    sensore non sposta nulla, >1 = lo attrae, <1 = lo respinge.
    """
    st = sensor_action_stats(counts)
    tot = counts.sum()
    p_joint = counts / max(tot, 1)
    p_a = p_joint.sum(axis=0)
    row = counts.sum(axis=1, keepdims=True)
    cond = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.divide(cond, p_a[None, :], out=np.ones_like(cond), where=p_a[None, :] > 0)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))

    ax = axes[0]
    im = ax.imshow(cond, cmap=SEQ_BLUE, vmin=0, vmax=max(0.35, cond.max()), aspect="equal")
    ax.set_title(f"P(azione | sensore massimo)\nmassa diagonale {st['diagonal_mass']:.2f}",
                 loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(azione | sensore)", fontsize=8, color=INK_SECOND)
    cb.outline.set_visible(False)
    for i in range(8):
        for j in range(8):
            if cond[i, j] >= 0.12:
                ax.text(j, i, f"{cond[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color=SURFACE if cond[i, j] > 0.22 else INK_SECOND)

    ax = axes[1]
    # Lift: diverging attorno a 1 (nessun effetto) → scala simmetrica in log2.
    # Due accortezze: le celle con pochi campioni sono rumore e vanno mascherate,
    # e la scala va limitata a ±2 (0.25×–4×), altrimenti le celle mai scelte
    # (lift→0, log2→-10) schiacciano tutto il resto sul grigio.
    MIN_N = 20
    lg = np.log2(np.clip(lift, 1e-3, None))
    lg = np.ma.masked_where(counts < MIN_N, lg)
    cmap = DIV_BR.copy()
    cmap.set_bad(GRID)
    im2 = ax.imshow(lg, cmap=cmap, vmin=-2, vmax=2, aspect="equal")
    ax.set_title(f"Lift  P(a|s) / P(a)\ndiagonale {st['diagonal_lift']:.2f}× "
                 f"dell'atteso ({st['diagonal_expected']:.2f})", loc="left")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, extend="both")
    cb2.set_label("log$_2$ lift  (0 = nessun effetto)", fontsize=8, color=INK_SECOND)
    cb2.outline.set_visible(False)

    for ax in axes:
        ax.set_xticks(range(8), DIR_NAMES)
        ax.set_yticks(range(8), DIR_NAMES)
        ax.set_xlabel("Direzione scelta dall'agente")
        ax.grid(visible=False)
    axes[0].set_ylabel("Direzione del sensore più intenso (corona 20 m)")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.015, f"Celle grigie nel pannello destro: meno di {MIN_N} campioni. "
                         f"Lift saturato a 0.25×–4×.",
             ha="center", fontsize=7.5, color=INK_MUTED)
    fig.savefig(path)
    plt.close(fig)


def plot_wind_response(resp: np.ndarray, thetas_deg: np.ndarray, align: float,
                       path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(resp.T, cmap=SEQ_BLUE, aspect="auto", origin="lower",
                   extent=[thetas_deg[0], thetas_deg[-1], -0.5, 7.5])
    ax.set_yticks(range(8), DIR_NAMES)
    ax.set_xlabel("Direzione del vento imposta (gradi, 0 = Est)")
    ax.set_ylabel("Direzione dell'azione")
    ax.set_title(f"Response curve sul vento — allineamento controvento {align:+.2f}",
                 loc="left")
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(direzione)", fontsize=8, color=INK_SECOND)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def plot_value_entropy(ctr, vtriple, etriple, corr, path: Path) -> None:
    """Due misure di scala diversa → due pannelli, mai due assi y sullo stesso plot."""
    vm, vlo, vhi = vtriple
    em, elo, ehi = etriple
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    axes[0].fill_between(ctr, vlo, vhi, color=SERIES_BLUE, alpha=0.16, lw=0)
    axes[0].plot(ctr, vm, color=SERIES_BLUE, lw=2)
    axes[0].set_xlabel("Distanza dalla sorgente (m)")
    axes[0].set_ylabel("V(s) stimato dal critic")
    axes[0].set_title(f"Il critic stima la distanza (r = {corr:+.2f})", loc="left")

    axes[1].fill_between(ctr, elo, ehi, color=SERIES_BLUE, alpha=0.16, lw=0)
    axes[1].plot(ctr, em, color=SERIES_BLUE, lw=2)
    axes[1].set_xlabel("Distanza dalla sorgente (m)")
    axes[1].set_ylabel("Entropia della policy (nats, pre-mask)")
    axes[1].set_title("Esplorazione vs sfruttamento", loc="left")

    fig.savefig(path)
    plt.close(fig)


# ─── Raccolta del buffer di stati ────────────────────────────────────────────

MAX_BUFFER = 40000


def collect_states(model, config, data_manager, scenarios, chunk_ids, vec_norm_path,
                   n_episodes: int = 1) -> tuple:
    """Rollout con policy piena per raccogliere gli stati su cui girano le analisi
    2 e 3. Ritorna (buffer, vec_env aperto per normalize_obs).

    Attraversa TUTTI gli scenari: un buffer riempito dai primi due scenari
    descriverebbe due plume, non la policy.
    """
    parts = {"obs_norm": [], "obs_raw": [], "actions": [], "distances": []}
    total = 0
    keep_env = None
    n_scen = 0

    for source_id, version in scenarios:
        field = load_field(data_manager, source_id, version)
        if field is None:
            continue
        for chunk_id in chunk_ids:
            env_cfg = make_env_config(config, chunk_id=chunk_id)
            vec_env = build_env(env_cfg, field, vec_norm_path,
                                use_masking=MASKABLE_PPO_AVAILABLE,
                                data_manager=data_manager,
                                wind_data=None, current_data=None,
                                wind_mapping=WIND_MAPPING, current_mapping=CURRENT_MAPPING)
            for ep in range(n_episodes):
                seed = abs(hash(("collect", source_id, version, chunk_id, ep))) % (2**31)
                r = run_episode_ablated(model, vec_env, None, seed=seed, collect=True)
                n = len(r["actions"])
                if n == 0:
                    continue
                for k in parts:
                    parts[k].append(r[k][:n])
                total += n
                n_scen += 1
            if keep_env is None:
                keep_env = vec_env          # serve VecNormalize.normalize_obs più tardi
            else:
                vec_env.close()
        if total >= MAX_BUFFER:
            print(f"  cap del buffer raggiunto ({MAX_BUFFER} stati)")
            break

    buf = {k: np.concatenate(v) for k, v in parts.items()}
    print(f"  buffer: {len(buf['actions'])} stati da {n_scen} episodi")
    return buf, keep_env


def subsample(buf: dict, n: int) -> dict:
    """Sottocampionamento uniforme lungo il buffer: preserva la copertura degli
    scenari (gli episodi sono concatenati in ordine) senza far dominare quelli
    lunghi, cioè le fasi di stallo."""
    if len(buf["actions"]) <= n:
        return buf
    sel = np.linspace(0, len(buf["actions"]) - 1, n).astype(int)
    return {k: v[sel] for k, v in buf.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

def replot(out_dir: Path) -> None:
    """Rigenera le figure dall'explainability.json già prodotto.

    Ritoccare una figura non deve costare un'ora di rollout: tutto ciò che serve a
    disegnarle è già serializzato.
    """
    p = out_dir / "explainability.json"
    if not p.exists():
        print(f"[ERRORE] {p} non trovato: serve un run completo prima di --replot.")
        sys.exit(1)
    with open(p) as f:
        res = json.load(f)

    done = []
    if "ablation" in res:
        plot_ablation(res["ablation"], out_dir / "fig_ablation_gruppi.png")
        done.append("fig_ablation_gruppi.png")
    if "shapley" in res:
        plot_shapley(res["shapley"], out_dir / "fig_shapley_gruppi.png")
        done.append("fig_shapley_gruppi.png")

    b = res.get("behavior", {})
    if "counts" in b.get("sensor_action", {}):
        plot_sensor_action(np.array(b["sensor_action"]["counts"]),
                           out_dir / "fig_sensore_azione.png")
        done.append("fig_sensore_azione.png")
    elif "sensor_action" in b:
        print("  [skip] fig_sensore_azione: il JSON contiene solo la condizionale "
              "normalizzata, non i conteggi grezzi necessari al marginale P(azione); "
              "serve rieseguire --stages behavior.")
    if "wind_response" in b:
        plot_wind_response(np.array(b["wind_response"]["matrix"]),
                           np.array(b["wind_response"]["thetas_deg"]),
                           b["wind_response"]["upwind_alignment"],
                           out_dir / "fig_response_vento.png")
        done.append("fig_response_vento.png")
    v, e = b.get("value_vs_distance", {}), b.get("entropy_vs_distance", {})
    if "q25" in v and "q25" in e:
        plot_value_entropy(np.array(v["bin_centers"]),
                           (np.array(v["median"]), np.array(v["q25"]), np.array(v["q75"])),
                           (np.array(e["median"]), np.array(e["q25"]), np.array(e["q75"])),
                           v["pearson_r"], out_dir / "fig_valore_entropia.png")
        done.append("fig_valore_entropia.png")
    elif v:
        print("  [skip] fig_valore_entropia: il JSON non contiene i quartili "
              "(prodotto da una versione precedente); serve rieseguire --stages behavior.")

    for d in done:
        print(f"  rigenerata {d}")
    print(f"\nFigure → {out_dir}")


def find_dualcorona_run(trained_dir: Path, vmax: float, K: int = 5) -> Optional[Path]:
    best = None
    for d in sorted(trained_dir.glob("ppo_*")):
        p = d / "config.yaml"
        if not p.exists():
            continue
        try:
            ag = load_config(str(p)).get("agent", {})
        except Exception:
            continue
        if (int(ag.get("n_velocity_levels", 1)) == K
                and abs(float(ag.get("max_velocity", 1.0)) - vmax) < 1e-6
                and ag.get("sensor_range_2") is not None):
            best = d
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Explainability della policy PPO HYDRAS")
    ap.add_argument("--vmax", type=float, default=2.0)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--n-sources", type=int, default=6,
                    help="sorgenti held-out da usare (0 = tutte e 26)")
    ap.add_argument("--n-episodes", type=int, default=2, help="episodi per scenario")
    ap.add_argument("--n-states", type=int, default=1500, help="stati nel buffer")
    ap.add_argument("--chunk-ids", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stages", type=str, default="ablation,shapley,behavior")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--replot", action="store_true",
                    help="rigenera le figure dall'explainability.json esistente, "
                         "senza rieseguire l'analisi")
    args = ap.parse_args()

    apply_style()
    root = Path(__file__).resolve().parent.parent
    trained = root / "trained_models"

    run_dir = Path(args.run_dir) if args.run_dir else find_dualcorona_run(trained, args.vmax)
    if run_dir is None:
        print(f"[ERRORE] Nessun modello doppia corona v_max={args.vmax} trovato.")
        sys.exit(1)
    model_path = run_dir / "models" / "final_model.zip"
    vec_norm_path = run_dir / "models" / "vec_normalize.pkl"
    if not model_path.exists():
        print(f"[ERRORE] final_model.zip mancante in {run_dir / 'models'}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else (
        root / "evaluations" / "evaluations_RL" / "evaluations_RL_adaptive"
        / f"explainability_dualcorona_v{int(args.vmax)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        replot(out_dir)
        return

    config = load_config(str(run_dir / "config.yaml"))
    model = load_model(str(model_path))
    cfg = make_env_config(config, chunk_id=0)
    groups = obs_groups(cfg)

    print(f"\n{'#'*100}")
    print(f"#  EXPLAINABILITY — doppia corona v_max={args.vmax}")
    print(f"#  modello : {model_path}")
    print(f"#  obs     : {sum(len(v) for v in groups.values())} feature in {len(groups)} gruppi")
    print(f"#  azioni  : {cfg.n_discrete_actions} direzioni × {cfg.n_velocity_levels} velocità")
    print(f"#  output  : {out_dir}")
    print(f"{'#'*100}\n")

    data_manager = DataManager(data_dir=str(root / "data"), preload_all=False,
                               sources_csv="Coordinate_Sorgenti_FaseII.csv")
    scenarios = build_scenarios(data_manager, args.n_sources)
    stages = {s.strip() for s in args.stages.split(",")}
    results: Dict[str, dict] = {
        "model": str(model_path),
        "groups": {k: [int(i) for i in v] for k, v in groups.items()},
    }

    if "ablation" in stages:
        print(f"\n[1/3] Ablation causale di gruppo — {len(scenarios)} scenari × "
              f"{len(args.chunk_ids)} chunk × {args.n_episodes} ep × {len(groups)+1} condizioni")
        results["ablation"] = stage_ablation(
            model, config, data_manager, groups, scenarios, args.chunk_ids,
            args.n_episodes, vec_norm_path, out_dir)

    if {"shapley", "behavior"} & stages:
        print("\n[buffer] raccolta stati con policy piena")
        buf, env_for_norm = collect_states(model, config, data_manager, scenarios,
                                           args.chunk_ids, vec_norm_path)
        try:
            if "shapley" in stages:
                # Solo lo Shapley paga 2^12 forward per stato: gli basta un
                # sottocampione. Le sonde comportamentali usano il buffer pieno.
                print("\n[2/3] Valori di Shapley esatti sui gruppi")
                results["shapley"] = stage_shapley(
                    model, groups, subsample(buf, args.n_states)["obs_norm"], out_dir)
            if "behavior" in stages:
                print("\n[3/3] Analisi comportamentale")
                results["behavior"] = stage_behavior(model, cfg, groups, buf,
                                                     env_for_norm, out_dir)
        finally:
            if env_for_norm is not None:
                env_for_norm.close()

    # Fusione con il JSON esistente: rieseguire un solo stage non deve cancellare
    # gli altri (l'ablation costa ~45 min di rollout).
    json_path = out_dir / "explainability.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                merged = json.load(f)
            merged.update(results)
            results = merged
        except Exception as e:
            print(f"  [warn] {json_path} illeggibile ({e}): lo riscrivo da zero.")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*100}")
    if "ablation" in results:
        ab = results["ablation"]
        print(f"Baseline: SR {ab['baseline_sr']:.1f}%, {ab['baseline_mean_steps']:.0f} step "
              f"al successo  ({ab['n_paired_spawns']} spawn appaiati)\n")
        rows = sorted((r for r in ab["rows"] if r["group"] != "baseline"),
                      key=lambda r: -(r["delta_steps_pct"] if np.isfinite(r["delta_steps_pct"])
                                      else -1e9))
        print(f"{'gruppo ablato':<28s} {'Δstep %':>9s} {'IC 95%':>18s} "
              f"{'ΔSR':>7s} {'IC 95%':>16s}")
        for r in rows:
            sci = f"[{r['delta_steps_ci'][0]:+.1f},{r['delta_steps_ci'][1]:+.1f}]"
            rci = f"[{r['delta_sr_ci'][0]:+.1f},{r['delta_sr_ci'][1]:+.1f}]"
            print(f"{r['label']:<28s} {r['delta_steps_pct']:>+9.1f} {sci:>18s} "
                  f"{r['delta_sr']:>+7.1f} {rci:>16s}")
    if "shapley" in results:
        print("\nShapley |phi| medio:")
        for k, v in sorted(results["shapley"]["mean_abs_phi"].items(),
                           key=lambda kv: -kv[1]):
            print(f"  {GROUP_LABELS[k]:<28s} {v:.4f}")
    print(f"\nRisultati e figure → {out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
