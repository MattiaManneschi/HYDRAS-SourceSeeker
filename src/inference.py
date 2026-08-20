#!/usr/bin/env python3
"""
HYDRAS Source Seeking - Inference Script
Valutazione completa del modello addestrato su tutti gli scenari.

Output:
  - Success rate per sorgente e scenario
  - Distanza finale dalla sorgente (media, min, max)
  - Numero di step per raggiungere la sorgente (episodi di successo)
  - Motivo di terminazione (successo / timeout)
  - Traiettorie visualizzate per ogni episodio
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_generator import generate_showcase_videos

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    ActionMasker = None  # Fallback per ActionMasker
    MASKABLE_PPO_AVAILABLE = False
    print("WARNING: sb3_contrib non disponibile, uso PPO standard")

from utils.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from utils.data_loader import DataManager


def plot_trajectory(trajectory: np.ndarray, field, ax=None, title: str = "",
                    show_arrows: bool = True, arrow_freq: int = 10):
    """Plot della traiettoria su campo di concentrazione."""
    from matplotlib.colors import ListedColormap
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Usa il frame corrente del field (impostato dall'env)
    conc = field.get_current_field()  # [y, x]
    # Le coordinate NetCDF sono centri cella; imshow interpreta extent come bordi.
    # Correzione di mezzo passo (half-pixel) per allineare correttamente i pixel
    # ai centri cella: la cornice del plot coincide col dominio MIKE21 e il
    # dato all'ultima riga/colonna non viene tagliato.
    dx = float(field.x_coords[1] - field.x_coords[0]) if len(field.x_coords) > 1 else 10.0
    dy = float(field.y_coords[1] - field.y_coords[0]) if len(field.y_coords) > 1 else 10.0
    extent = [
        float(field.x_coords[0])  - dx / 2, float(field.x_coords[-1]) + dx / 2,
        float(field.y_coords[0])  - dy / 2, float(field.y_coords[-1]) + dy / 2,
    ]
    
    # 1) Sfondo: azzurro uniforme su tutto il plot
    ax.set_facecolor('#87CEEB')
    
    # 2) Terra bianca
    if field.land_mask is not None:
        white_cmap = ListedColormap(['#FFFFFF'])  # Bianco puro
        land_display = np.ma.masked_where(~field.land_mask, np.ones_like(conc))
        ax.imshow(land_display, origin='lower', extent=extent, cmap=white_cmap, alpha=1.0, zorder=1)
    
    # 3) Plume di concentrazione (mascherato su terra e dove conc ~ 0)
    plume_threshold = 0.01  # Mostra solo dove c'è concentrazione significativa
    if field.land_mask is not None:
        mask = field.land_mask | (conc < plume_threshold)
    else:
        mask = conc < plume_threshold
    conc_masked = np.ma.masked_where(mask, conc)
    im = ax.imshow(conc_masked, origin='lower', extent=extent, cmap='YlOrRd', alpha=0.9, 
                   vmin=0, vmax=max(conc.max(), 0.1), zorder=2)
    plt.colorbar(im, ax=ax, label='Concentrazione')
    
    # Traiettoria
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1.5, alpha=0.8, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='darkred', s=100, marker='X', zorder=5, label='End')
    
    # Frecce direzione
    if show_arrows and len(trajectory) > arrow_freq:
        for i in range(0, len(trajectory) - 1, arrow_freq):
            dx = trajectory[i + 1, 0] - trajectory[i, 0]
            dy = trajectory[i + 1, 1] - trajectory[i, 1]
            if np.sqrt(dx**2 + dy**2) > 1:
                ax.arrow(trajectory[i, 0], trajectory[i, 1], dx * 0.8, dy * 0.8,
                         head_width=20, head_length=10, fc='red', ec='red', alpha=0.6)
    
    # Sorgente
    if field.source_position is not None:
        ax.scatter(field.source_position[0], field.source_position[1], c='yellow', s=200, 
                   marker='*', edgecolors='black', zorder=6, label='Source')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    # Ripristina i limiti al dominio esatto (i centri cella, non i bordi pixel)
    ax.set_xlim(float(field.x_coords[0]), float(field.x_coords[-1]))
    ax.set_ylim(float(field.y_coords[0]), float(field.y_coords[-1]))

    return ax


# ─── Dataclasses per i risultati ────────────────────────────────────────────

@dataclass
class EpisodeResult:
    scenario: str          # es. "SRC001"
    source_id: str         # "SRC001" ... "SRC132"
    episode: int
    success: bool
    termination: str       # "success" / "timeout" / "boundary" / "land"
    initial_distance: float  # m - distanza spawn-sorgente
    final_distance: float  # m
    steps: int
    trajectory: np.ndarray
    start_frame: int = 0   # Frame iniziale (dipende da chunk_id)
    end_frame: int = 0     # Frame finale (al quale plottare)
    spawn_x: float = 0.0   # Posizione iniziale agente
    spawn_y: float = 0.0
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))    # m/s scelta per step
    dist_history: np.ndarray = field(default_factory=lambda: np.array([]))  # m, distanza dalla sorgente per step


@dataclass
class ScenarioStats:
    scenario: str          # es. "SRC001_Q1/4"
    source_id: str         # es. "SRC001"
    n_episodes: int
    success_rate: float
    mean_final_dist: float
    min_final_dist: float
    max_final_dist: float
    mean_steps_success: Optional[float]   # None se nessun successo
    mean_initial_dist: float               # distanza media di partenza dalla sorgente
    termination_counts: Dict[str, int]


# ─── Utility ────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def write_inference_log(
    output_path: Path,
    model_path: str,
    dt_seconds: float,
    global_success_rate: Optional[float],
    version_success_rates: Dict[str, Optional[float]],
    chunk_success_rates: Dict[str, Optional[float]],
    mean_initial_distance: Optional[float],
    mean_success_steps: Optional[float],
    total_scenarios: int,
    total_episodes: int,
) -> Path:
    """Scrive un log di sintesi in output_path/log.txt."""

    def fmt_percent(value: Optional[float]) -> str:
        return "n/d" if value is None else f"{value * 100:.1f}%"

    def fmt_distance(value: Optional[float]) -> str:
        return "n/d" if value is None else f"{value:.1f} m"

    def fmt_steps(value: Optional[float]) -> str:
        return "n/d" if value is None else f"{value:.1f}"

    mean_success_minutes = None
    if mean_success_steps is not None:
        mean_success_minutes = (mean_success_steps * dt_seconds) / 60.0

    lines = [
        "HYDRAS Inference Summary",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_path}",
        "",
        "Metrics:",
        f"- Global Success Rate: {fmt_percent(global_success_rate)}",
        f"- Mean Initial Distance: {fmt_distance(mean_initial_distance)}",
        (
            f"- Mean Success Steps: {fmt_steps(mean_success_steps)} "
            f"(~{'n/d' if mean_success_minutes is None else f'{mean_success_minutes:.1f} min'})"
        ),
        "",
        "Success Rate by Wind:",
        f"- V0: {fmt_percent(version_success_rates.get('V0'))}",
        f"- V1: {fmt_percent(version_success_rates.get('V1'))}",
        f"- V2: {fmt_percent(version_success_rates.get('V2'))}",
        f"- V3: {fmt_percent(version_success_rates.get('V3'))}",
        "",
        "Success Rate by Frame:",
        f"- Q1/4: {fmt_percent(chunk_success_rates.get('Q1/4'))}",
        f"- Q1/2: {fmt_percent(chunk_success_rates.get('Q1/2'))}",
        f"- Q3/4: {fmt_percent(chunk_success_rates.get('Q3/4'))}",
        "",
        f"Total Scenarios Evaluated: {total_scenarios}",
        f"Total Episodes Evaluated: {total_episodes}",
    ]

    log_path = output_path / "log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    return log_path





def load_model(model_path: str):
    """Carica MaskablePPO o PPO dal path."""
    try:
        model = MaskablePPO.load(model_path, device='cpu')
        print(f"  Modello caricato: {model_path}")
        print(f"  Device: cpu (forzato per CUDA compatibility)")
    except Exception as e:
        raise RuntimeError(f"Impossibile caricare il modello: {e}")
    return model


def make_env_config(config: dict, chunk_id: int = 0) -> SourceSeekingConfig:
    return SourceSeekingConfig.from_config(config, chunk_id=chunk_id)


def mask_fn(env) -> np.ndarray:
    """Estrae la maschera azioni attraversando i wrapper."""
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner.action_masks()


def build_env(env_cfg, field, vec_norm_path, use_masking,
              data_manager: Optional[DataManager] = None,
              wind_data = None,
              current_data = None,
              wind_mapping: Optional[Dict[str, str]] = None,
              current_mapping: Optional[Dict[str, str]] = None):
    """Costruisce e wrappa l'environment per l'inferenza.
    
    Args:
        data_manager: DataManager per accesso ai dati
        wind_data: Dati di vento (caricati da DataManager)
        current_data: Dati di corrente (caricati da DataManager)
        wind_mapping: Mapping versione -> wind file (es. {'_V0': '...', '_V1': '...', ...})
                     Se passato, l'env caricherà il vento dinamicamente per versione.
        current_mapping: Mapping versione -> current file (es. {'_V0': '...', '_V1': '...', ...})
                        Se passato, l'env caricherà la corrente dinamicamente per versione.
    """
    raw_env = SourceSeekingEnv(
        config=env_cfg,
        concentration_field=field,
        wind_data=wind_data,
        current_data=current_data,
        data_manager=data_manager,
        wind_mapping=wind_mapping,
        current_mapping=current_mapping,
    )

    if use_masking and MASKABLE_PPO_AVAILABLE and ActionMasker is not None:
        raw_env = ActionMasker(raw_env, mask_fn)

    vec_env = DummyVecEnv([lambda e=raw_env: e])

    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    return vec_env


def get_inner_env(vec_env) -> SourceSeekingEnv:
    """Estrae il SourceSeekingEnv dal fondo dello stack di wrapper."""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env


def run_episode(model, vec_env, deterministic=True) -> EpisodeResult:
    """Esegue un singolo episodio e ritorna il risultato."""
    obs = vec_env.reset()
    inner = get_inner_env(vec_env)

    # Leggi il frame iniziale dall'environment (salvato durante reset)
    start_frame = inner.info_reset.get('start_time_idx', 0) if hasattr(inner, 'info_reset') else 0

    # Salva posizione iniziale e calcola distanza dalla sorgente
    spawn_pos = inner.state.position.copy()
    source_pos = inner.source_position
    initial_dist = float(np.linalg.norm(spawn_pos - source_pos))

    trajectory = [spawn_pos]
    velocities = []        # andamento della velocità: m/s scelta dall'agente a ogni step
    dist_hist = []         # distanza dalla sorgente a ogni step (post-step)
    done = False
    last_info = {}

    while not done:
        if MASKABLE_PPO_AVAILABLE:
            action, _ = model.predict(obs, deterministic=deterministic,
                                      action_masks=vec_env.env_method('action_masks')[0])
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        # Velocità scelta: decodifica l'azione (dir_idx, velocity). Indipendente
        # dallo stato → valida anche se l'env si è già auto-resettato a done=True.
        try:
            a_int = int(np.asarray(action).flatten()[0])
            _, vel = inner._decode_action(a_int)
            velocities.append(float(vel))
        except Exception:
            pass

        obs, _, dones, infos = vec_env.step(action)
        done = dones[0]
        last_info = infos[0]

        d = last_info.get('distance_to_source')
        if d is not None:
            dist_hist.append(float(d))

        # Usa posizione dall'info (inner è già resettato quando done=True)
        pos = last_info.get('position', inner.state.position.tolist())
        trajectory.append(np.array(pos))

    # Determina terminazione: usa la reason esplicita dall'ambiente quando disponibile.
    termination = last_info.get('termination_reason')
    if termination not in {'success', 'boundary', 'land', 'timeout'}:
        if last_info.get('source_reached', False):
            termination = 'success'
        elif last_info.get('out_of_bounds', False):
            termination = 'boundary'
        elif last_info.get('on_land', False):
            termination = 'land'
        else:
            n_steps_info = int(last_info.get('steps', 0))
            max_steps = int(getattr(inner.config, 'max_steps', 0)) if hasattr(inner, 'config') else 0
            termination = 'timeout' if max_steps > 0 and n_steps_info >= max_steps else 'timeout'

    # Usa i valori dall'info dict
    final_dist = last_info.get('distance_to_source', 0.0)
    n_steps = last_info.get('steps', len(trajectory) - 1)
    
    # Leggi il frame finale dall'info dict dell'ultimo step (prima del reset automatico)
    end_frame = last_info.get('end_time_idx', start_frame)

    return EpisodeResult(
        scenario="",           # impostato dal chiamante
        source_id="",          # impostato dal chiamante
        episode=0,             # impostato dal chiamante
        success=termination == 'success',
        termination=termination,
        initial_distance=initial_dist,
        final_distance=final_dist,
        steps=n_steps,
        trajectory=np.array(trajectory),
        start_frame=start_frame,
        end_frame=end_frame,
        spawn_x=float(spawn_pos[0]),
        spawn_y=float(spawn_pos[1]),
        velocities=np.array(velocities),
        dist_history=np.array(dist_hist),
    )


# ─── Funzioni di analisi ─────────────────────────────────────────────────────

def compute_scenario_stats(results: List[EpisodeResult], scenario: str, source_id: str) -> ScenarioStats:
    n = len(results)
    successes = [r for r in results if r.success]
    final_dists = [r.final_distance for r in results]
    initial_dists = [r.initial_distance for r in results]
    termination_counts = {}
    for r in results:
        termination_counts[r.termination] = termination_counts.get(r.termination, 0) + 1

    return ScenarioStats(
        scenario=scenario,
        source_id=source_id,
        n_episodes=n,
        success_rate=len(successes) / n,
        mean_final_dist=float(np.mean(final_dists)),
        min_final_dist=float(np.min(final_dists)),
        max_final_dist=float(np.max(final_dists)),
        mean_steps_success=float(np.mean([r.steps for r in successes])) if successes else None,
        mean_initial_dist=float(np.mean(initial_dists)),
        termination_counts=termination_counts,
    )


def print_scenario_stats(stats: ScenarioStats):
    sr = f"{stats.success_rate * 100:.0f}%"
    dist = f"{stats.mean_final_dist:.0f}m (min={stats.min_final_dist:.0f}m, max={stats.max_final_dist:.0f}m)"
    steps = f"{stats.mean_steps_success:.0f}" if stats.mean_steps_success else "—"
    term = ", ".join(f"{k}={v}" for k, v in stats.termination_counts.items())
    print(f"  {stats.scenario:8s}  SR={sr:5s}  dist={dist}  steps_success={steps}  [{term}]")


def print_source_summary(all_stats: List[ScenarioStats], source_id: str):
    source_stats = [s for s in all_stats if s.source_id == source_id]
    if not source_stats:
        return
    mean_sr = np.mean([s.success_rate for s in source_stats])
    mean_dist = np.mean([s.mean_final_dist for s in source_stats])
    success_steps = [s.mean_steps_success for s in source_stats if s.mean_steps_success]
    mean_steps = np.mean(success_steps) if success_steps else None

    print(f"\n  {source_id} SUMMARY: SR={mean_sr*100:.0f}%  "
          f"mean_dist={mean_dist:.0f}m  "
          f"mean_steps_success={'—' if mean_steps is None else f'{mean_steps:.0f}'}")


# ─── Plotting ────────────────────────────────────────────────────────────────

def save_trajectory_plot(result: EpisodeResult, field, output_path: Path, threshold: float = 100):
    """Salva il plot della traiettoria per un singolo episodio.
    
    Mostra il campo di concentrazione al frame finale dell'episodio.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Usa il frame finale salvato dall'episodio
    display_frame = result.end_frame
    
    # Aggiorna il field al frame appropriato
    if hasattr(field, 'set_time'):
        field.set_time(display_frame)

    status = "SUCCESS ✓" if result.success else f"FAILED [{result.termination}]"
    title = (f"{result.scenario} — Ep {result.episode+1} — {status}\n"
             f"dist={result.final_distance:.0f}m  steps={result.steps}  (frame={display_frame})")

    plot_trajectory(result.trajectory, field, ax=ax, title=title,
                    show_arrows=True, arrow_freq=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_velocity_mean_over_time(results, output_path, max_velocity: float,
                                 n_levels: int, dt_seconds: float = 10.0):
    """Andamento MEDIO della velocità nel tempo per un dato v_max.

    Prende l'andamento della velocità di ogni run (episodio) registrato durante
    l'inferenza e ne calcola la media (± dev.std.) fra tutte le run, per indice di
    step. Le linee tratteggiate indicano i K livelli di velocità ammessi in (0, v_max].
    Salva il grafico in  output_path/analysis/velocity_mean_over_time.png .
    """
    series = [r.velocities for r in results
              if getattr(r, 'velocities', None) is not None and len(r.velocities) > 0]
    if not series:
        print("[velocity] nessun andamento di velocità registrato: grafico saltato.")
        return

    n_levels = max(1, int(n_levels))
    levels = [(j + 1) / n_levels * max_velocity for j in range(n_levels)]

    # Media/std per indice di step su andamenti di lunghezza diversa (ragged)
    max_len = max(len(v) for v in series)
    mean_t, std_t, n_t = [], [], []
    for i in range(max_len):
        col = [v[i] for v in series if len(v) > i]
        n_t.append(len(col)); mean_t.append(float(np.mean(col))); std_t.append(float(np.std(col)))
    mean_t, std_t, n_t = np.array(mean_t), np.array(std_t), np.array(n_t)
    t_min = np.arange(max_len) * dt_seconds / 60.0
    keep = n_t >= 3          # taglia la coda con pochissimi episodi (rumorosa)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_min[keep], mean_t[keep], color='tab:blue', lw=2, label='mean velocity')
    # La velocità è fisicamente limitata a [0, v_max] (livelli discreti (vel_idx+1)/K·v_max):
    # la media è sempre <= v_max, ma la banda ±std, essendo simmetrica su una quantità
    # bounded, sfora v_max quando la media è vicina al tetto. Si clippa la banda ai limiti
    # fisici (la MEDIA non è toccata).
    lo = np.maximum(mean_t - std_t, 0.0)
    hi = np.minimum(mean_t + std_t, max_velocity)
    ax.fill_between(t_min[keep], lo[keep], hi[keep],
                    color='tab:blue', alpha=0.2, label='±1 std')
    for lv in levels:
        ax.axhline(lv, ls='--', color='gray', lw=0.7, alpha=0.6)
    ax.set_xlabel('time [min]'); ax.set_ylabel('velocity [m/s]')
    ax.set_ylim(0, max_velocity * 1.08); ax.grid(alpha=0.3); ax.legend(loc='upper right')

    overall = float(np.mean(np.concatenate(series)))
    ax.set_title(f'Mean velocity over time — v_max={max_velocity:g} m/s, K={n_levels}\n'
                 f'mean over {len(series)} runs  (global mean velocity {overall:.2f} m/s)', fontsize=12)
    fig.tight_layout()

    analysis_dir = Path(output_path) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = analysis_dir / "velocity_mean_over_time.png"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Andamento medio velocità salvato in: {out}")


def plot_velocity_vs_distance(results, output_path, max_velocity: float,
                              n_levels: int, success_radius: float = 50.0):
    """Velocità media in funzione della distanza dalla sorgente (binned su tutte le run).

    È il grafico diagnostico per la MODULAZIONE: se l'agente rallenta avvicinandosi,
    la curva scende verso destra (vicino alla sorgente). Asse x invertito: sinistra =
    lontano, destra = vicino. La linea verticale segna il raggio di successo.
    """
    dv = [(r.dist_history, r.velocities) for r in results
          if getattr(r, 'dist_history', None) is not None and len(getattr(r, 'dist_history', [])) > 0
          and len(r.velocities) > 0]
    if not dv:
        print("[velocity-dist] nessun dato distanza/velocità: grafico saltato.")
        return

    n_levels = max(1, int(n_levels))
    levels = [(j + 1) / n_levels * max_velocity for j in range(n_levels)]
    dists = np.concatenate([d[:min(len(d), len(v))] for d, v in dv])
    vels  = np.concatenate([v[:min(len(d), len(v))] for d, v in dv])

    nb = 30
    edges = np.linspace(0, np.percentile(dists, 99), nb + 1)
    idx = np.clip(np.digitize(dists, edges) - 1, 0, nb - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bmean = np.array([vels[idx == b].mean() if np.any(idx == b) else np.nan for b in range(nb)])
    bstd  = np.array([vels[idx == b].std()  if np.any(idx == b) else np.nan for b in range(nb)])
    ok = ~np.isnan(bmean)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(centers[ok], bmean[ok], color='tab:green', lw=2, marker='o', ms=3, label='mean velocity')
    lo = np.maximum(bmean - bstd, 0.0); hi = np.minimum(bmean + bstd, max_velocity)   # banda in [0, v_max]
    ax.fill_between(centers[ok], lo[ok], hi[ok], color='tab:green', alpha=0.15, label='±1 std')
    for lv in levels:
        ax.axhline(lv, ls='--', color='gray', lw=0.7, alpha=0.6)
    ax.axvline(success_radius, ls=':', color='red', lw=1.2, label=f'success radius {success_radius:g} m')
    ax.set_xlabel('distance from source [m]'); ax.set_ylabel('mean velocity [m/s]')
    ax.set_ylim(0, max_velocity * 1.08); ax.grid(alpha=0.3); ax.legend(loc='upper left')
    ax.invert_xaxis()    # left = far, right = near
    ax.set_title(f'Mean velocity vs distance from source — v_max={max_velocity:g} m/s, K={n_levels}\n'
                 f'(curve dropping to the right = slows down on approach)', fontsize=12)
    fig.tight_layout()

    analysis_dir = Path(output_path) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = analysis_dir / "velocity_vs_distance.png"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Velocità vs distanza salvato in: {out}")


# ─── Main ────────────────────────────────────────────────────────────────────


def run_inference(
    model_path: str,
    config_path: str,
    data_dir: str,
    output_dir: str,
    n_episodes: int = 5,
    deterministic: bool = True,
    sources_csv: str = "Coordinate_Sorgenti_FaseII.csv",
    chunk_ids: List[int] = None,
    save_videos: bool = True,
    config_override: Optional[dict] = None,
    save_plots: bool = True,
    seed: Optional[int] = None,
    plot_velocity: bool = False,
):
    """
    Esegue l'inferenza completa su 26 sorgenti held-out (SRC107-SRC132, 20% del totale 132) con chunk multipli per fonte.

    Args:
        model_path:   Path al modello (.zip)
        config_path:  Path al config YAML
        data_dir:     Directory con i file NC (Output_HD_FaseII_CL2_V1)
        output_dir:   Directory di output per plot e risultati
        n_episodes:   Episodi per sorgente e chunk
        deterministic: Policy deterministica o stocastica
        sources_csv:  File CSV con coordinate delle sorgenti
        chunk_ids:    Lista di chunk_id da testare (default [0, 1, 2] = Q1/4, Q1/2, Q3/4)
                     0 = spawn @1/4, 1 = spawn @1/2, 2 = spawn @3/4
    """
    if chunk_ids is None:
        chunk_ids = [0, 1, 2]  # Default: Q1/4, Q1/2, Q3/4
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    config = config_override if config_override is not None else load_config(config_path)
    model = load_model(model_path)
    vec_norm_path = Path(model_path).parent / "vec_normalize.pkl"
    success_threshold = config.get('environment', {}).get('reward', {}).get('distance_threshold', 50)
    
    # Inizializza DataManager con auto-discovery di 132 sorgenti
    data_manager = DataManager(
        data_dir=data_dir,
        preload_all=False,
        sources_csv=sources_csv
    )
    
    print(f"\nInference data: {len(data_manager._nc_files)} files (tutte le versioni V0+V1+V2+V3)")
    
    # Usa le sorgenti escluse dal training (SRC107-SRC132) per valutazione
    all_sources = data_manager.get_discovered_sources()
    inference_sources = [s for s in all_sources if int(s[3:]) > 106]  # SRC107-SRC132 (26 file, ~20%)
    
    chunk_labels = {0: "Q1/4", 1: "Q1/2", 2: "Q3/4"}
    chunk_descriptions = ", ".join([
        f"{chunk_labels[cid]} (chunk_id={cid})" for cid in chunk_ids
    ])
    
    print(f"\n{'='*100}")
    print(f"HYDRAS Inference — {len(inference_sources)} sorgenti × 4 scenari vento × {len(chunk_ids)} chunk × {n_episodes} episodi")
    print(f"  = {len(inference_sources)*4*len(chunk_ids)*n_episodes} episodi totali")
    print(f"Chunk testati: {chunk_descriptions}")
    print(f"Modello: {model_path}")
    print(f"Dati: {data_dir}")
    print(f"Sorgenti training: SRC001-SRC106 (80%)")
    print(f"Sorgenti inference (20%): SRC107-SRC132 ({len(inference_sources)} sorgenti)")
    print(f"{'='*100}\n")
    
    # Wind mapping per caricamento dinamico vento per versione
    # (come nel training, per coerenza tra Conc_Vx e Wind_Vx)
    wind_mapping = {
        "_V0": "CI_WIND_faseII_V0.txt",
        "_V1": "CI_WIND_faseII_V1.txt",
        "_V2": "CI_WIND_faseII_V2.txt",
        "_V3": "CI_WIND_faseII_V3.txt",
    }
    print(f"Wind mapping (versions V0-V3): {len(wind_mapping)} file")
    
    # Current mapping per caricamento dinamico corrente per versione
    # (come nel training, per coerenza tra Conc_Vx e Current_Vx)
    current_mapping = {
        "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
        "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
        "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
        "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
    }
    print(f"Current mapping (versions V0-V3): {len(current_mapping)} file")
    
    dt_seconds = config.get('environment', {}).get('dt', 10)
    
    all_stats: List[ScenarioStats] = []
    episode_success_all: List[float] = []
    episode_success_by_version: Dict[str, List[float]] = {'V0': [], 'V1': [], 'V2': [], 'V3': []}
    episode_success_by_chunk: Dict[str, List[float]] = {'Q1/4': [], 'Q1/2': [], 'Q3/4': []}
    initial_distances_all: List[float] = []
    success_steps_all: List[float] = []
    all_results_global: List[EpisodeResult] = []
    episodes_data: List[dict] = []

    for src_idx, source_id in enumerate(inference_sources, 1):
        if src_idx % 5 == 0:
            print(f"\n[Progress: {src_idx}/{len(inference_sources)} sources]\n")

        for version in ['V0', 'V1', 'V2', 'V3']:
            version_files = [f for f in data_manager._nc_files
                             if version in f.name and source_id in f.name]
            if not version_files:
                continue

            version_dir = output_path / source_id / version
            version_dir.mkdir(parents=True, exist_ok=True)

            try:
                field = data_manager._nc_loader.load(
                    str(version_files[0]),
                    concentration_var="Concentration - component 1"
                )
                if field is None:
                    continue
                coords = data_manager.get_source_coordinates(source_id)
                if coords:
                    field.source_position = coords
                field.run_id = f"{source_id}_{version}"
            except Exception as e:
                print(f"  [SKIP] {version}_{source_id}: {e}")
                continue

            for chunk_id in chunk_ids:
                chunk_label = chunk_labels[chunk_id]
                scenario_label = f"{version}_{source_id}_{chunk_label}"
                episode_results: List[EpisodeResult] = []

                env_cfg = make_env_config(config, chunk_id=chunk_id)
                vec_env = build_env(env_cfg, field, vec_norm_path,
                                   use_masking=MASKABLE_PPO_AVAILABLE,
                                   data_manager=data_manager,
                                   wind_data=None, current_data=None,
                                   wind_mapping=wind_mapping,
                                   current_mapping=current_mapping)

                for ep in range(n_episodes):
                    result = run_episode(model, vec_env, deterministic=deterministic)
                    result.scenario = scenario_label
                    result.source_id = source_id
                    result.episode = ep

                    sv = 1.0 if result.success else 0.0
                    episode_success_all.append(sv)
                    episode_success_by_version[version].append(sv)
                    episode_success_by_chunk[chunk_label].append(sv)
                    initial_distances_all.append(result.initial_distance)
                    if result.success:
                        success_steps_all.append(float(result.steps))
                    episode_results.append(result)
                    all_results_global.append(result)

                    src_x, src_y = field.source_position
                    dist_history = [
                        float(np.sqrt((pos[0]-src_x)**2 + (pos[1]-src_y)**2))
                        for pos in result.trajectory
                    ]
                    episodes_data.append({
                        "source_id": source_id, "version": version,
                        "chunk": chunk_label, "chunk_id": chunk_id,
                        "episode": ep + 1, "success": result.success,
                        "termination": result.termination,
                        "initial_distance": result.initial_distance,
                        "final_distance": result.final_distance,
                        "steps": result.steps, "distance_history": dist_history,
                    })

                    if save_plots:
                        plot_path = version_dir / f"ep{ep+1:02d}_chunk{chunk_id}_trajectory.png"
                        save_trajectory_plot(result, field, plot_path, success_threshold)

                    init_dist = f"{result.initial_distance:.0f}m"
                    if result.success:
                        time_mins = (result.steps * dt_seconds) / 60
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → SUCCESS in {result.steps:3d} steps ({time_mins:5.1f}m)")
                    else:
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → {result.termination.upper()} at {result.steps:4d} steps (final_dist={result.final_distance:6.1f}m)")

                vec_env.close()

                if episode_results:
                    all_stats.append(compute_scenario_stats(episode_results, scenario_label, source_id))

    def safe_mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    global_sr = safe_mean(episode_success_all)
    mean_initial_dist = safe_mean(initial_distances_all)
    mean_success_steps = safe_mean(success_steps_all)

    version_sr = {
        version: safe_mean(episode_success_by_version.get(version, []))
        for version in ['V0', 'V1', 'V2', 'V3']
    }
    chunk_sr = {
        chunk: safe_mean(episode_success_by_chunk.get(chunk, []))
        for chunk in ['Q1/4', 'Q1/2', 'Q3/4']
    }

    # Riepilogo globale
    if all_stats:
        print(f"\n{'='*80}")
        print(f"RESULTS BY CHUNK (Time Instant)")
        print(f"{'='*80}")
        for chunk_label in ['Q1/4', 'Q1/2', 'Q3/4']:
            sr = chunk_sr.get(chunk_label)
            n_eps = len(episode_success_by_chunk.get(chunk_label, []))
            if sr is None:
                print(f"{chunk_label}: (no data)")
            else:
                print(f"{chunk_label}: {sr*100:6.1f}% ({n_eps} episodes)")
        
        print(f"\n{'='*80}")
        print(f"RESULTS BY WIND SCENARIO (across all chunks)")
        print(f"{'='*80}")
        
        for version in ['V0', 'V1', 'V2', 'V3']:
            sr = version_sr.get(version)
            n_eps = len(episode_success_by_version.get(version, []))
            if sr is not None:
                print(f"{version}: {sr*100:6.1f}% ({n_eps} episodes)")
            else:
                print(f"{version}: (no data)")
        
        print(f"\n{'='*80}")
        print(f"Global Success Rate: {('n/d' if global_sr is None else f'{global_sr*100:.1f}%')}")
        print(f"Mean initial distance: {('n/d' if mean_initial_dist is None else f'{mean_initial_dist:.0f}m')}")
        if mean_success_steps is not None:
            mean_success_minutes = (mean_success_steps * dt_seconds) / 60.0
            print(f"Mean success steps: {mean_success_steps:.1f} (~{mean_success_minutes:.1f} min)")
        else:
            print("Mean success steps: n/d (nessun episodio di successo)")
        print(f"Total scenarios: {len(all_stats)}")
        print(f"Total episodes: {len(episode_success_all)}")
        print(f"{'='*80}\n")

    # Scrive sempre il log di riepilogo nello stesso output_dir delle valutazioni
    log_path = write_inference_log(
        output_path=output_path,
        model_path=model_path,
        dt_seconds=dt_seconds,
        global_success_rate=global_sr,
        version_success_rates=version_sr,
        chunk_success_rates=chunk_sr,
        mean_initial_distance=mean_initial_dist,
        mean_success_steps=mean_success_steps,
        total_scenarios=len(all_stats),
        total_episodes=len(episode_success_all),
    )
    print(f"Summary log salvato in: {log_path}")

    # Salva dati per-episodio per analisi quantitativa
    import json
    episodes_json_path = output_path / "episodes_data.json"
    with open(episodes_json_path, "w") as f:
        json.dump(episodes_data, f)
    print(f"Dati per-episodio salvati in: {episodes_json_path}")

    if save_videos:
        generate_showcase_videos(all_results_global, data_manager, output_path)

    generate_analysis_plots(episodes_data, output_path, dt_seconds=dt_seconds,
                            max_steps=config.get('environment', {}).get('max_episode_steps', 1080))

    if plot_velocity:
        agent_cfg = config.get('agent', {})
        mv = float(agent_cfg.get('max_velocity', 1.0))
        kl = int(agent_cfg.get('n_velocity_levels', 1))
        plot_velocity_mean_over_time(
            all_results_global, output_path,
            max_velocity=mv, n_levels=kl, dt_seconds=dt_seconds,
        )
        plot_velocity_vs_distance(
            all_results_global, output_path,
            max_velocity=mv, n_levels=kl, success_radius=float(success_threshold),
        )

    return all_stats


def find_model_for_sensor_range(trained_dir: Path, sensor_range: float) -> Optional[Path]:
    """Trova il modello più recente addestrato con il dato sensor_range."""
    candidates = []
    for run_dir in sorted(trained_dir.glob("ppo_*")):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            cfg = load_config(str(cfg_path))
            if cfg.get('agent', {}).get('sensor_range') == sensor_range:
                model_zip = run_dir / "models" / "final_model.zip"
                if model_zip.exists():
                    candidates.append(model_zip)
        except Exception:
            continue
    return candidates[-1] if candidates else None


# ─── FCM: Field Climbing Method ───────────────────────────────────────────────

FCM_STEP_MAX = 50.0    # cap superiore passo adattivo [m]

class AdamFCMAgent:
    """
    Agente Field Climbing Method con ottimizzatore Adam.

    Stima il gradiente locale del campo di concentrazione tramite regressione
    ai minimi quadrati (Taylor al primo ordine), poi applica Adam per calcolare
    il vettore di passo adattivo:

        step = lr * m̂ / (√v̂ + ε)

    Il momentum (β1) mantiene la direzione coerente su più timestep; la
    normalizzazione per la varianza storica (β2) riduce il passo dove il
    gradiente oscilla molto.
    """

    _DIAG = 1.0 / np.sqrt(2.0)
    _DIRECTIONS = np.array([
        [0.0,    1.0   ],   # 0: Nord
        [0.0,   -1.0   ],   # 1: Sud
        [1.0,    0.0   ],   # 2: Est
        [-1.0,   0.0   ],   # 3: Ovest
        [_DIAG,  _DIAG ],   # 4: NordEst
        [_DIAG, -_DIAG ],   # 5: SudEst
        [-_DIAG, _DIAG ],   # 6: NordOvest
        [-_DIAG,-_DIAG ],   # 7: SudOvest
    ], dtype=float)

    def __init__(self, sensor_range: float = 50.0, lr: float = 50.0,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, step_max: float = FCM_STEP_MAX):
        self.sensor_range = sensor_range
        self.lr       = lr
        self.beta1    = beta1
        self.beta2    = beta2
        self.eps      = eps
        self.step_max = step_max
        self._last_step: float = lr

    def reset(self):
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        self._last_step = self.lr

    def _estimate_gradient_ls(self, obs: np.ndarray) -> np.ndarray:
        """Stima ∇C tramite minimi quadrati (Taylor 1° ordine)."""
        center_conc = float(obs[0])
        sensors = obs[28:36].astype(float)
        b = (sensors - center_conc) / max(self.sensor_range, 1.0)
        gradient, _, _, _ = np.linalg.lstsq(self._DIRECTIONS, b, rcond=None)
        return gradient

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, None]:
        """Seleziona l'azione tramite Adam sul gradiente stimato."""
        flat_obs = obs[0] if obs.ndim == 2 else obs
        g = self._estimate_gradient_ls(flat_obs)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        step   = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        norm = float(np.linalg.norm(step))
        self._last_step = min(norm, self.step_max) if norm > 1e-12 else self.lr

        if norm < 1e-12:
            if action_masks is not None:
                valid = np.where(action_masks)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            else:
                action = int(np.random.randint(0, 8))
        else:
            scores = self._DIRECTIONS @ step
            if action_masks is not None:
                scores[~action_masks.astype(bool)] = -np.inf
            action = int(np.argmax(scores))

        return np.array([action]), None


def build_env_fcm(
    env_cfg,
    field,
    use_masking: bool,
    data_manager: Optional[DataManager] = None,
    wind_data=None,
    current_data=None,
    wind_mapping: Optional[Dict[str, str]] = None,
    current_mapping: Optional[Dict[str, str]] = None,
):
    """Costruisce l'environment per FCM (senza VecNormalize)."""
    raw_env = SourceSeekingEnv(
        config=env_cfg,
        concentration_field=field,
        wind_data=wind_data,
        current_data=current_data,
        data_manager=data_manager,
        wind_mapping=wind_mapping,
        current_mapping=current_mapping,
    )
    if use_masking and MASKABLE_PPO_AVAILABLE and ActionMasker is not None:
        raw_env = ActionMasker(raw_env, mask_fn)
    return DummyVecEnv([lambda e=raw_env: e])


def run_episode_fcm(fcm_agent, vec_env, deterministic: bool = True) -> EpisodeResult:
    """Esegue un singolo episodio FCM con passo adattivo step = min(K/||∇C||, step_max)."""
    fcm_agent.reset()
    inner = get_inner_env(vec_env)
    obs = vec_env.reset()

    start_frame = inner.info_reset.get('start_time_idx', 0) if hasattr(inner, 'info_reset') else 0
    spawn_pos   = inner.state.position.copy()
    initial_dist = float(np.linalg.norm(spawn_pos - inner.source_position))
    dt           = inner.config.dt
    default_vel  = inner.config.max_velocity

    trajectory = [spawn_pos]
    done = False
    last_info: dict = {}

    while not done:
        if MASKABLE_PPO_AVAILABLE:
            action, _ = fcm_agent.predict(obs, deterministic=deterministic,
                                          action_masks=vec_env.env_method('action_masks')[0])
        else:
            action, _ = fcm_agent.predict(obs, deterministic=deterministic)

        inner.config.max_velocity = fcm_agent._last_step / dt
        obs, _, dones, infos = vec_env.step(action)
        done = dones[0]
        last_info = infos[0]
        trajectory.append(np.array(last_info.get('position', inner.state.position.tolist())))

    inner.config.max_velocity = default_vel

    termination = last_info.get('termination_reason')
    if termination not in {'success', 'boundary', 'land', 'timeout'}:
        if last_info.get('source_reached', False):       termination = 'success'
        elif last_info.get('out_of_bounds', False):      termination = 'boundary'
        elif last_info.get('on_land', False):            termination = 'land'
        else:                                            termination = 'timeout'

    return EpisodeResult(
        scenario="", source_id="", episode=0,
        success=termination == 'success',
        termination=termination,
        initial_distance=initial_dist,
        final_distance=last_info.get('distance_to_source', 0.0),
        steps=last_info.get('steps', len(trajectory) - 1),
        trajectory=np.array(trajectory),
        start_frame=start_frame,
        end_frame=last_info.get('end_time_idx', start_frame),
        spawn_x=float(spawn_pos[0]),
        spawn_y=float(spawn_pos[1]),
    )


def run_inference_fcm(
    config_path: str,
    data_dir: str,
    output_dir: str,
    n_episodes: int = 5,
    sources_csv: str = "Coordinate_Sorgenti_FaseII.csv",
    chunk_ids: Optional[List[int]] = None,
    sensor_range: Optional[float] = None,
    lr: float = 50.0,
    save_plots: bool = True,
    save_videos: bool = True,
    seed: Optional[int] = None,
    config_override: Optional[dict] = None,
) -> List[ScenarioStats]:
    """Inferenza completa con AdamFCMAgent.

    Stessa struttura e stessi output di run_inference() ma con AdamFCMAgent
    al posto del modello RL.

    Args:
        sensor_range: Range sensori in metri (default: da config).
        lr: Learning rate Adam (corrisponde al passo base in metri).
    """
    if chunk_ids is None:
        chunk_ids = [0, 1, 2]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    config = config_override if config_override is not None else load_config(config_path)
    dt_seconds      = config.get('environment', {}).get('dt', 10)
    max_steps_cfg   = config.get('environment', {}).get('max_episode_steps', 1080)
    success_threshold = config.get('environment', {}).get('reward', {}).get('distance_threshold', 50)

    if sensor_range is None:
        sensor_range = float(config.get('agent', {}).get('sensor_range', 20.0))

    fcm_agent   = AdamFCMAgent(sensor_range=sensor_range, lr=lr)
    agent_label = f"FCM Adam (lr={lr}m, sensor_range={sensor_range}m)"

    data_manager = DataManager(
        data_dir=data_dir,
        preload_all=False,
        sources_csv=sources_csv,
    )

    all_sources       = data_manager.get_discovered_sources()
    inference_sources = [s for s in all_sources if int(s[3:]) > 106]

    chunk_labels = {0: "Q1/4", 1: "Q1/2", 2: "Q3/4"}
    chunk_descriptions = ", ".join([
        f"{chunk_labels[cid]} (chunk_id={cid})" for cid in chunk_ids
    ])

    wind_mapping = {
        "_V0": "CI_WIND_faseII_V0.txt",
        "_V1": "CI_WIND_faseII_V1.txt",
        "_V2": "CI_WIND_faseII_V2.txt",
        "_V3": "CI_WIND_faseII_V3.txt",
    }
    current_mapping = {
        "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
        "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
        "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
        "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
    }

    print(f"\n{'='*100}")
    print(f"HYDRAS Inference FCM — {len(inference_sources)} sorgenti × 4 versioni × "
          f"{len(chunk_ids)} chunk × {n_episodes} ep.")
    print(f"  = {len(inference_sources)*4*len(chunk_ids)*n_episodes} episodi totali")
    print(f"Variante FCM: {agent_label}")
    print(f"Chunk testati: {chunk_descriptions}")
    print(f"Dati: {data_dir}")
    print(f"Sorgenti inference (20%): SRC107–SRC132 ({len(inference_sources)} sorgenti)")
    print(f"{'='*100}\n")

    all_stats: List[ScenarioStats]          = []
    episode_success_all: List[float]        = []
    episode_success_by_version: Dict[str, List[float]] = {'V0': [], 'V1': [], 'V2': [], 'V3': []}
    episode_success_by_chunk: Dict[str, List[float]]   = {'Q1/4': [], 'Q1/2': [], 'Q3/4': []}
    initial_distances_all: List[float]      = []
    success_steps_all: List[float]          = []
    all_results_global: List[EpisodeResult] = []
    episodes_data: List[dict]               = []

    for src_idx, source_id in enumerate(inference_sources, 1):
        if src_idx % 5 == 0:
            print(f"\n[Progress: {src_idx}/{len(inference_sources)} sources]\n")

        for version in ['V0', 'V1', 'V2', 'V3']:
            version_files = [f for f in data_manager._nc_files
                             if version in f.name and source_id in f.name]
            if not version_files:
                continue

            version_dir = output_path / source_id / version
            version_dir.mkdir(parents=True, exist_ok=True)

            try:
                field = data_manager._nc_loader.load(
                    str(version_files[0]),
                    concentration_var="Concentration - component 1",
                )
                if field is None:
                    continue
                coords = data_manager.get_source_coordinates(source_id)
                if coords:
                    field.source_position = coords
                field.run_id = f"{source_id}_{version}"
            except Exception as e:
                print(f"  [SKIP] {version}_{source_id}: {e}")
                continue

            for chunk_id in chunk_ids:
                chunk_label    = chunk_labels[chunk_id]
                scenario_label = f"{version}_{source_id}_{chunk_label}"
                episode_results: List[EpisodeResult] = []

                env_cfg = make_env_config(config, chunk_id=chunk_id)
                env_cfg.sensor_range = sensor_range  # FCM usa sensor_range diverso dall'RL
                vec_env = build_env_fcm(
                    env_cfg, field,
                    use_masking=MASKABLE_PPO_AVAILABLE,
                    data_manager=data_manager,
                    wind_data=None, current_data=None,
                    wind_mapping=wind_mapping,
                    current_mapping=current_mapping,
                )

                for ep in range(n_episodes):
                    result = run_episode_fcm(fcm_agent, vec_env, deterministic=True)
                    result.scenario  = scenario_label
                    result.source_id = source_id
                    result.episode   = ep

                    sv = 1.0 if result.success else 0.0
                    episode_success_all.append(sv)
                    episode_success_by_version[version].append(sv)
                    episode_success_by_chunk[chunk_label].append(sv)
                    initial_distances_all.append(result.initial_distance)
                    if result.success:
                        success_steps_all.append(float(result.steps))
                    episode_results.append(result)
                    all_results_global.append(result)

                    src_x, src_y = field.source_position
                    dist_history = [
                        float(np.sqrt((pos[0] - src_x)**2 + (pos[1] - src_y)**2))
                        for pos in result.trajectory
                    ]
                    episodes_data.append({
                        "source_id": source_id, "version": version,
                        "chunk": chunk_label, "chunk_id": chunk_id,
                        "episode": ep + 1, "success": result.success,
                        "termination": result.termination,
                        "initial_distance": result.initial_distance,
                        "final_distance": result.final_distance,
                        "steps": result.steps, "distance_history": dist_history,
                    })

                    if save_plots:
                        plot_path = version_dir / f"ep{ep+1:02d}_chunk{chunk_id}_trajectory.png"
                        save_trajectory_plot(result, field, plot_path, success_threshold)

                    init_dist = f"{result.initial_distance:.0f}m"
                    if result.success:
                        time_mins = (result.steps * dt_seconds) / 60
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → SUCCESS in {result.steps:3d} steps ({time_mins:5.1f}m)")
                    else:
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → {result.termination.upper()} at {result.steps:4d} steps (final_dist={result.final_distance:6.1f}m)")

                vec_env.close()

                if episode_results:
                    all_stats.append(compute_scenario_stats(episode_results, scenario_label, source_id))

    def safe_mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    global_sr       = safe_mean(episode_success_all)
    mean_initial_dist = safe_mean(initial_distances_all)
    mean_success_steps = safe_mean(success_steps_all)

    version_sr = {v: safe_mean(episode_success_by_version.get(v, [])) for v in ['V0', 'V1', 'V2', 'V3']}
    chunk_sr   = {c: safe_mean(episode_success_by_chunk.get(c, []))   for c in ['Q1/4', 'Q1/2', 'Q3/4']}

    if all_stats:
        print(f"\n{'='*80}")
        print(f"FCM — RESULTS BY CHUNK")
        print(f"{'='*80}")
        for chunk_label in ['Q1/4', 'Q1/2', 'Q3/4']:
            sr  = chunk_sr.get(chunk_label)
            n_e = len(episode_success_by_chunk.get(chunk_label, []))
            print(f"{chunk_label}: {('n/d' if sr is None else f'{sr*100:6.1f}%')} ({n_e} episodes)")
        print(f"\n{'='*80}")
        print(f"FCM — RESULTS BY WIND SCENARIO")
        print(f"{'='*80}")
        for v in ['V0', 'V1', 'V2', 'V3']:
            sr  = version_sr.get(v)
            n_e = len(episode_success_by_version.get(v, []))
            print(f"{v}: {('n/d' if sr is None else f'{sr*100:6.1f}%')} ({n_e} episodes)")
        print(f"\n{'='*80}")
        print(f"Global Success Rate: {('n/d' if global_sr is None else f'{global_sr*100:.1f}%')}")
        print(f"Mean initial distance: {('n/d' if mean_initial_dist is None else f'{mean_initial_dist:.0f}m')}")
        if mean_success_steps is not None:
            print(f"Mean success steps: {mean_success_steps:.1f} (~{(mean_success_steps * dt_seconds)/60:.1f} min)")
        print(f"Total scenarios: {len(all_stats)}")
        print(f"Total episodes: {len(episode_success_all)}")
        print(f"{'='*80}\n")

    log_path = write_inference_log(
        output_path=output_path,
        model_path=agent_label,
        dt_seconds=dt_seconds,
        global_success_rate=global_sr,
        version_success_rates=version_sr,
        chunk_success_rates=chunk_sr,
        mean_initial_distance=mean_initial_dist,
        mean_success_steps=mean_success_steps,
        total_scenarios=len(all_stats),
        total_episodes=len(episode_success_all),
    )
    print(f"Summary log salvato in: {log_path}")

    import json
    episodes_json_path = output_path / "episodes_data.json"
    with open(episodes_json_path, "w") as f:
        json.dump(episodes_data, f)
    print(f"Dati per-episodio salvati in: {episodes_json_path}")

    if save_videos:
        generate_showcase_videos(all_results_global, data_manager, output_path)

    generate_analysis_plots(episodes_data, output_path, dt_seconds=dt_seconds,
                            max_steps=max_steps_cfg)

    return all_stats


def main_fcm_adam_sweep():
    """Sweep su lr=[10,20,30,40,50] per FCM Adam con sensor_range=50m.

    Output: evaluations/evaluations_FCM/fcm_adaptive/lr_{lr}/
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = str(PROJECT_ROOT / "data")
    CONFIG_PATH  = str(PROJECT_ROOT / "utils" / "config" / "config_base_no_wind_reward.yaml")

    for lr in [10, 20, 30, 40, 50]:
        print(f"\n{'='*80}")
        print(f"FCM Adam sweep — lr={lr}m")
        print(f"{'='*80}")
        output_dir = str(PROJECT_ROOT / "thesis" / "evaluations" / "evaluations_FCM"
                         / "fcm_adaptive" / f"lr_{lr}")
        run_inference_fcm(
            config_path=CONFIG_PATH, data_dir=DATA_DIR, output_dir=output_dir,
            n_episodes=2, sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2], sensor_range=50.0, lr=float(lr),
        )


# ─── Analysis Plots ──────────────────────────────────────────────────────────

def generate_analysis_plots(
    episodes_data: List[dict],
    output_path: Path,
    dt_seconds: float = 10.0,
    max_steps: int = 1080,
) -> dict:
    """Genera i 4 plot di analisi quantitativa nella sottocartella analysis/.

    Returns:
        Dict path_name → Path dei PNG generati.
    """
    import matplotlib
    matplotlib.use('Agg')

    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    episodes = episodes_data
    success_eps = [e for e in episodes if e['success']]
    fail_eps    = [e for e in episodes if not e['success']]
    plot_paths: dict = {}

    # ── Plot 1: Distribuzione tempi di successo ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    if success_eps:
        steps_arr   = np.array([e['steps'] for e in success_eps])
        minutes_arr = steps_arr * dt_seconds / 60.0
        p25 = np.percentile(minutes_arr, 25)
        p50 = np.percentile(minutes_arr, 50)
        p75 = np.percentile(minutes_arr, 75)
        p95 = np.percentile(minutes_arr, 95)
        mean_val = float(minutes_arr.mean())
        std_val  = float(minutes_arr.std())
        n_outliers = int(np.sum(minutes_arr > p95))
        clipped = minutes_arr[minutes_arr <= p95]
        n_bins = min(40, max(15, int(len(clipped) ** 0.5)))
        ax.hist(clipped, bins=n_bins, color='#2196F3', edgecolor='white',
                linewidth=0.5, alpha=0.8)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_val:.1f} min')
        ax.axvline(p50, color='orange', linestyle=':', linewidth=1.8,
                   label=f'Median: {p50:.1f} min')
        ax.set_xlim(0, p95)
        ax.set_xlabel('Time to reach the source (simulated minutes)', fontsize=11)
        ax.set_ylabel('Number of episodes', fontsize=11)
        ax.set_title('Distribution of Success Times', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.35)
        stats_txt = (
            f"N successi = {len(minutes_arr)}\n"
            f"Dev.std = {std_val:.1f} min\n"
            f"P25–P75 = [{p25:.1f}, {p75:.1f}] min\n"
            f"Outliers (>{p95:.0f} min) = {n_outliers}"
        )
        ax.text(0.97, 0.03, stats_txt, transform=ax.transAxes,
                fontsize=8.5, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.9))
    else:
        ax.text(0.5, 0.5, 'No successful episodes', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    fig.tight_layout()
    p1 = analysis_dir / "plot_success_time_dist.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['time_dist'] = p1

    # ── Plot 2: Heatmap SR versione×chunk + SR per distanza iniziale ─────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    versions = ['V0', 'V1', 'V2', 'V3']
    chunks   = ['Q1/4', 'Q1/2', 'Q3/4']
    matrix   = np.full((len(versions), len(chunks)), np.nan)
    for vi, v in enumerate(versions):
        for ci, c in enumerate(chunks):
            sub = [e for e in episodes if e['version'] == v and e['chunk'] == c]
            if sub:
                matrix[vi, ci] = float(np.mean([e['success'] for e in sub])) * 100

    ax = axes[0]
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(chunks)));  ax.set_xticklabels(chunks, fontsize=10)
    ax.set_yticks(range(len(versions))); ax.set_yticklabels(versions, fontsize=10)
    ax.set_title('Success Rate (%) — Version × Chunk', fontsize=11, fontweight='bold')
    for vi in range(len(versions)):
        for ci in range(len(chunks)):
            val = matrix[vi, ci]
            if not np.isnan(val):
                ax.text(ci, vi, f'{val:.0f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if val < 50 else 'black')
    fig.colorbar(im, ax=ax, label='SR (%)')

    ax2 = axes[1]
    bins_edges = [0, 500, 1000, 1500, 2000, 2500]
    bin_labels  = ['0–500', '500–1000', '1000–1500', '1500–2000', '2000+']
    bin_sr, bin_n = [], []
    for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
        sub = [e for e in episodes if lo <= e['initial_distance'] < hi]
        bin_sr.append(float(np.mean([e['success'] for e in sub])) * 100 if sub else 0.0)
        bin_n.append(len(sub))
    bar_colors = ['#4CAF50' if s >= 80 else '#FF9800' if s >= 50 else '#F44336'
                  for s in bin_sr]
    bars = ax2.bar(bin_labels, bin_sr, color=bar_colors, edgecolor='white', linewidth=0.5)
    for bar, n in zip(bars, bin_n):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'n={n}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel('Initial distance from source (m)', fontsize=10)
    ax2.set_ylabel('Success Rate (%)', fontsize=10)
    ax2.set_title('SR vs Initial Distance', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=9)
    ax2.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    p2 = analysis_dir / "plot_sr_analysis.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['sr_analysis'] = p2

    # ── Plot 3: Distanza media dalla sorgente nel tempo ──────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    def _pad_and_stack(ep_list, max_len):
        padded = []
        for e in ep_list:
            h = e['distance_history']
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h[:max_len])
        return np.array(padded) if padded else None

    max_len    = max_steps + 1
    steps_axis = np.arange(max_len) * dt_seconds / 60.0
    all_mat = _pad_and_stack(episodes, max_len)
    suc_mat = _pad_and_stack(success_eps, max_len)
    fai_mat = _pad_and_stack(fail_eps, max_len)
    if all_mat is not None:
        ax.plot(steps_axis, all_mat.mean(axis=0), color='steelblue',
                linewidth=1.8, label=f'All ({len(episodes)} ep.)')
    if suc_mat is not None:
        ax.plot(steps_axis, suc_mat.mean(axis=0), color='green',
                linewidth=1.8, linestyle='--', label=f'Successes ({len(success_eps)} ep.)')
    if fai_mat is not None:
        ax.plot(steps_axis, fai_mat.mean(axis=0), color='crimson',
                linewidth=1.8, linestyle=':', label=f'Failures ({len(fail_eps)} ep.)')
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Mean distance from source (m)', fontsize=11)
    ax.set_title('Distance from Source over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.4)
    ax.set_xlim(0, max_steps * dt_seconds / 60.0)
    fig.tight_layout()
    p3 = analysis_dir / "plot_distance_over_time.png"
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['distance_time'] = p3

    # ── Plot 4: Distribuzione distanza iniziale di spawn ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    init_dists      = np.array([e['initial_distance'] for e in episodes])
    init_dists_suc  = np.array([e['initial_distance'] for e in success_eps]) if success_eps else np.array([])
    init_dists_fail = np.array([e['initial_distance'] for e in fail_eps])    if fail_eps    else np.array([])

    ax = axes[0]
    bin_edges   = np.arange(0, float(init_dists.max()) + 200, 200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_w = 160
    counts_suc  = np.histogram(init_dists_suc,  bins=bin_edges)[0] if len(init_dists_suc)  else np.zeros(len(bin_centers), int)
    counts_fail = np.histogram(init_dists_fail, bins=bin_edges)[0] if len(init_dists_fail) else np.zeros(len(bin_centers), int)
    ax.bar(bin_centers, counts_suc,  width=bar_w, color='#4CAF50', edgecolor='white',
           linewidth=0.4, label=f'Successes ({len(success_eps)})', alpha=0.9)
    ax.bar(bin_centers, counts_fail, width=bar_w, bottom=counts_suc, color='#F44336',
           edgecolor='white', linewidth=0.4, label=f'Failures ({len(fail_eps)})', alpha=0.9)
    ax.axvline(float(np.mean(init_dists)), color='navy', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(init_dists):.0f} m')
    ax.axvline(float(np.median(init_dists)), color='darkorange', linestyle=':', linewidth=1.8,
               label=f'Median: {np.median(init_dists):.0f} m')
    ax.set_xlabel('Initial distance from source (m)', fontsize=11)
    ax.set_ylabel('Number of episodes', fontsize=11)
    ax.set_title('Distribution of Spawn Distances', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35)
    stats_txt = (
        f"N tot = {len(init_dists)}\n"
        f"Min = {init_dists.min():.0f} m\n"
        f"Max = {init_dists.max():.0f} m\n"
        f"Std = {init_dists.std():.0f} m\n"
        f"P25 = {np.percentile(init_dists, 25):.0f} m\n"
        f"P75 = {np.percentile(init_dists, 75):.0f} m"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.9))

    ax2 = axes[1]
    fine_edges   = np.arange(0, float(init_dists.max()) + 250, 250)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    fine_sr, fine_n = [], []
    for lo, hi in zip(fine_edges[:-1], fine_edges[1:]):
        sub = [e for e in episodes if lo <= e['initial_distance'] < hi]
        fine_sr.append(float(np.mean([e['success'] for e in sub])) * 100 if sub else np.nan)
        fine_n.append(len(sub))
    valid = [(c, s, n) for c, s, n in zip(fine_centers, fine_sr, fine_n) if not np.isnan(s)]
    if valid:
        vcs, vsr, vns = zip(*valid)
        bar_colors2 = ['#4CAF50' if s >= 80 else '#FF9800' if s >= 50 else '#F44336'
                       for s in vsr]
        bars2 = ax2.bar(vcs, vsr, width=220, color=bar_colors2, edgecolor='white', linewidth=0.4)
        for bar, n in zip(bars2, vns):
            if n > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1.5, f'n={n}',
                         ha='center', va='bottom', fontsize=7.5)
    global_sr_val = round(float(np.mean([e['success'] for e in episodes])) * 100)
    ax2.axhline(global_sr_val, color='navy', linestyle='--', linewidth=1.2,
                label=f'SR globale {global_sr_val}%')
    ax2.set_ylim(0, 115)
    ax2.set_xlabel('Initial distance from source (m)', fontsize=11)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('SR by Initial Distance Band (250 m bins)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.35)
    fig.tight_layout()
    p4 = analysis_dir / "plot_initial_dist.png"
    fig.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['initial_dist'] = p4

    print(f"  Analisi plots salvati in: {analysis_dir}")
    return plot_paths


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR    = str(PROJECT_ROOT / "data")
    CONFIG_PATH = str(PROJECT_ROOT / "utils" / "config" / "config.yaml")
    trained_dir = PROJECT_ROOT / "trained_models"

    config = load_config(CONFIG_PATH)
    sweep = config.get('training', {}).get('sensor_range_sweep', [])

    if sweep:
        BASE_VERSION = 8  # evaluations_v8 → v8+len(sweep)-1
        print(f"\nInference sweep: sensor_range {sweep} m → evaluations_v{BASE_VERSION}–v{BASE_VERSION+len(sweep)-1}\n")

        for i, sr in enumerate(sweep):
            model_path = find_model_for_sensor_range(trained_dir, sr)
            if model_path is None:
                print(f"[SKIP] Nessun modello trovato per sensor_range={sr}m in {trained_dir}")
                continue

            output_dir = str(PROJECT_ROOT / "thesis" / "evaluations" / "evaluations_RL" / f"evaluations_v{BASE_VERSION + i}")

            cfg_override = load_config(CONFIG_PATH)
            cfg_override['agent']['sensor_range'] = sr

            print(f"\n{'='*70}")
            print(f"Inference  sensor_range={sr}m  →  evaluations_v{BASE_VERSION + i}")
            print(f"Modello: {model_path}")
            print(f"{'='*70}\n")

            run_inference(
                model_path=str(model_path),
                config_path=CONFIG_PATH,
                config_override=cfg_override,
                data_dir=DATA_DIR,
                output_dir=output_dir,
                n_episodes=5,
                deterministic=True,
                sources_csv="Coordinate_Sorgenti_FaseII.csv",
                chunk_ids=[0, 1, 2],
            )

        print(f"\nInference sweep completato. Output: evaluations_v{BASE_VERSION}–v{BASE_VERSION+len(sweep)-1}")

    else:
        # Fallback: usa l'ultimo modello disponibile
        run_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_")])
        if not run_dirs:
            print("ERRORE: Nessuna directory di training trovata in trained_models/")
            sys.exit(1)

        latest_run = run_dirs[-1]
        model_path = latest_run / "models" / "final_model.zip"
        if not model_path.exists():
            model_path = latest_run / "models" / "best" / "best_model.zip"
        if not model_path.exists():
            print(f"ERRORE: Nessun modello trovato in {latest_run}/models/")
            sys.exit(1)

        # Leggi sensor_range e sensor_range_2 dal config salvato nel run
        cfg_override = load_config(CONFIG_PATH)
        run_cfg_path = latest_run / "config.yaml"
        if run_cfg_path.exists():
            run_cfg = load_config(str(run_cfg_path))
            sr = run_cfg.get('agent', {}).get('sensor_range', cfg_override['agent'].get('sensor_range', 20))
            sr2 = run_cfg.get('agent', {}).get('sensor_range_2', cfg_override['agent'].get('sensor_range_2', 50))
            cfg_override['agent']['sensor_range'] = sr
            cfg_override['agent']['sensor_range_2'] = sr2
            print(f"sensor_range dal modello: {sr}m  |  sensor_range_2: {sr2}m")

        output_dir = str(PROJECT_ROOT / "thesis" / "evaluations" / "evaluations_RL" / "evaluations_v13")

        print(f"Modello selezionato: {model_path}")
        print(f"Output valutazioni: {output_dir}")

        run_inference(
            model_path=str(model_path),
            config_path=CONFIG_PATH,
            config_override=cfg_override,
            data_dir=DATA_DIR,
            output_dir=output_dir,
            n_episodes=5,
            deterministic=True,
            sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2],
        )

def main_spawn_map(mode: str = 'ppo'):
    """
    Genera griglie di spawn map (una per sorgente, subplot per combo vento/chunk):
    - sfondo: plume della sorgente al frame del chunk;
    - disco virtuale di spawn: 2 circonferenze tratteggiate a d_min e d_max (centro =
      sorgente), clippate al dominio;
    - marker: spawn → verde=successo, X rossa=fallimento secondo il "coloratore".

    mode='fcm' → coloratore FCM Adam (sensor 50, lr 40); output spawn_maps_fcm/
    mode='ppo' → coloratore PPO v1 warm-start;             output spawn_maps_ppo/
    Lo spawn (punti + disco) è la NUOVA metrica e identico nei due (dipende dall'env).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap as _LCmap
    import json as _json
    import csv
    from collections import defaultdict
    import copy

    N_SOURCES  = 5    # sorgenti con più fallimenti da selezionare
    N_EPISODES = 10   # episodi per terna (sorgente, versione, chunk)
    mode = mode.lower()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = str(PROJECT_ROOT / "data")
    SPAWN_ROOT   = PROJECT_ROOT / "thesis" / "evaluations" / "spawn_maps"
    chunk_labels = {0: 'Q1/4', 1: 'Q1/2', 2: 'Q3/4'}

    data_manager = DataManager(data_dir=DATA_DIR, preload_all=False,
                               sources_csv="Coordinate_Sorgenti_FaseII.csv")
    wind_mapping    = {"_V0": "CI_WIND_faseII_V0.txt", "_V1": "CI_WIND_faseII_V1.txt",
                       "_V2": "CI_WIND_faseII_V2.txt", "_V3": "CI_WIND_faseII_V3.txt"}
    current_mapping = {"_V0": "CL02_V0_SRC000_U_V_10mGrid.nc", "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
                       "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc", "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc"}

    # Coloratore: FCM Adam (config migliore) oppure v1 PPO warm-start. Lo SPAWN (punti +
    # disco) è identico nei due — dipende solo dalla logica di spawn dell'env, non dall'agente.
    model = None; fcm_agent = None; vec_norm_path = None
    if mode == 'fcm':
        OUTPUT_DIR = SPAWN_ROOT / "spawn_maps_fcm"
        config = load_config(str(PROJECT_ROOT / "utils" / "config" / "config_base_no_wind_reward.yaml"))
        config['agent']['sensor_range'] = 50          # FCM ottimale
        config['agent'].pop('sensor_range_2', None)
        config['agent']['n_velocity_levels'] = 1      # Discrete(8): FCM sceglie solo direzione
        fcm_agent = AdamFCMAgent(sensor_range=50.0, lr=40.0)   # migliore dallo sweep (63.5%)
        print(f"Spawn map — coloratore FCM Adam (sensor 50 m, lr 40 m)  →  {OUTPUT_DIR}")
    elif mode == 'ppo_double':  # doppia corona (sensor_range_2=50), warm-start da v1
        OUTPUT_DIR = SPAWN_ROOT / "spawn_maps_ppo" / "spawn_maps_ppo_double"
        run_dir    = PROJECT_ROOT / "trained_models" / "ppo_20260630_145300"
        config     = load_config(str(run_dir / "config.yaml"))
        MODEL_PATH = str(run_dir / "models" / "final_model.zip")
        model = load_model(MODEL_PATH)
        vec_norm_path = Path(MODEL_PATH).parent / "vec_normalize.pkl"
        print(f"Spawn map — coloratore PPO doppia corona  →  {OUTPUT_DIR}")
    else:  # ppo: v1 warm-start (passo variabile, di fatto 1 m/s)
        OUTPUT_DIR = SPAWN_ROOT / "spawn_maps_ppo"
        run_dir    = PROJECT_ROOT / "trained_models" / "ppo_20260628_192155"
        config     = load_config(str(run_dir / "config.yaml"))
        MODEL_PATH = str(run_dir / "models" / "final_model.zip")
        model = load_model(MODEL_PATH)
        vec_norm_path = Path(MODEL_PATH).parent / "vec_normalize.pkl"
        print(f"Spawn map — coloratore PPO v1 warm-start  →  {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inference_sources = [s for s in data_manager.get_discovered_sources() if int(s[3:]) > 106]

    # ── 1. Combinazioni con SR < 100% dai dati esistenti ──────────────────────
    eval_path = (PROJECT_ROOT / "thesis" / "evaluations" / "evaluations_RL"
                 / "evaluations_minimal_reward" / "episodes_data.json")
    eps_data = _json.loads(eval_path.read_text())

    # failures per (source, version, chunk_id)
    failures_by_src_combo = defaultdict(int)
    combo_outcomes        = defaultdict(list)
    for e in eps_data:
        key = (e['version'], e['chunk_id'])
        combo_outcomes[key].append(e['success'])
        if not e['success']:
            failures_by_src_combo[(e['source_id'], e['version'], e['chunk_id'])] += 1

    active_combos = {k for k, v in combo_outcomes.items() if not all(v)}
    print(f"Combinazioni attive (SR<100%): {sorted(active_combos)}")

    # ── 2. Top-N sorgenti con più fallimenti sulle combo attive ───────────────
    src_fail_count = defaultdict(int)
    for (src, v, c), n in failures_by_src_combo.items():
        if (v, c) in active_combos:
            src_fail_count[src] += n
    top_sources = sorted(src_fail_count, key=src_fail_count.get, reverse=True)[:N_SOURCES]
    print(f"\nTop {N_SOURCES} sorgenti per fallimenti: "
          + ", ".join(f"{s}({src_fail_count[s]})" for s in top_sources))

    # ── 3. Inferenza: N_EPISODES per (sorgente, combo) ────────────────────────
    # results[(src, version, chunk_id)] = [(spawn_x, spawn_y, success), ...]
    results = {}
    active_combos_sorted = sorted(active_combos)
    n_total = len(top_sources) * len(active_combos_sorted) * N_EPISODES
    print(f"\nEpisodi totali: {len(top_sources)} sorgenti × "
          f"{len(active_combos_sorted)} combo × {N_EPISODES} ep = {n_total}\n")

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from utils.data_loader import NetCDFLoader as _NCLoader

    for src_idx, source_id in enumerate(top_sources, 1):
        print(f"  [{src_idx}/{len(top_sources)}] {source_id}")
        for version, chunk_id in active_combos_sorted:
            nc_files = [f for f in data_manager._nc_files
                        if f'_{version}_' in f.name and source_id in f.name and 'Conc' in f.name]
            if not nc_files:
                continue
            try:
                field = _NCLoader(DATA_DIR).load(str(nc_files[0]),
                                                  concentration_var="Concentration - component 1")
                if field is None:
                    continue
                coords = data_manager.get_source_coordinates(source_id)
                if coords:
                    field.source_position = coords
                field.run_id = f"{source_id}_{version}"
            except Exception as ex:
                print(f"    [SKIP] {source_id}_{version}: {ex}")
                continue

            env_cfg = SourceSeekingConfig.from_config(config, chunk_id=chunk_id)
            key = (source_id, version, chunk_id)
            results[key] = []

            for _ in range(N_EPISODES):
                if mode == 'fcm':
                    vec_env = build_env_fcm(
                        SourceSeekingConfig.from_config(config, chunk_id=chunk_id),
                        copy.deepcopy(field), use_masking=MASKABLE_PPO_AVAILABLE,
                        data_manager=data_manager, wind_mapping=wind_mapping,
                        current_mapping=current_mapping)
                    r = run_episode_fcm(fcm_agent, vec_env, deterministic=True)
                else:
                    def _make(f=field, cfg=env_cfg):
                        env = SourceSeekingEnv(config=cfg,
                                               concentration_field=copy.deepcopy(f),
                                               data_manager=data_manager,
                                               wind_mapping=wind_mapping,
                                               current_mapping=current_mapping)
                        from stable_baselines3.common.monitor import Monitor as _Mon
                        env = _Mon(env)
                        if MASKABLE_PPO_AVAILABLE:
                            from sb3_contrib.common.wrappers import ActionMasker
                            env = ActionMasker(env, mask_fn)
                        return env

                    vec_env = DummyVecEnv([_make])
                    if vec_norm_path is not None and vec_norm_path.exists():
                        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
                        vec_env.training = False
                    r = run_episode(model, vec_env, deterministic=True)
                vec_env.close()
                results[key].append((r.spawn_x, r.spawn_y, r.success))

    # ── 4. Plot: by_combo e by_source ─────────────────────────────────────────
    def _draw_subplot(ax, source_id, version, chunk_id):
        key = (source_id, version, chunk_id)
        _draw_extent = [None]   # extent del dominio, per fissare i limiti (clip cerchi)
        nc_files = [f for f in data_manager._nc_files
                    if f'_{version}_' in f.name and source_id in f.name and 'Conc' in f.name]
        bg_field = None
        if nc_files:
            try:
                bg_field = _NCLoader(DATA_DIR).load(str(nc_files[0]),
                                                     concentration_var="Concentration - component 1")
                coords = data_manager.get_source_coordinates(source_id)
                if coords and bg_field:
                    bg_field.source_position = coords
            except Exception:
                pass

        ax.set_facecolor('#87CEEB')
        if bg_field is not None:
            nt = bg_field.n_timesteps
            frame = {0: nt // 4, 1: nt // 2, 2: (nt * 3) // 4}[chunk_id]
            bg_field.set_time(min(frame, nt - 1))
            conc = bg_field.get_current_field()
            ddx = float(bg_field.x_coords[1] - bg_field.x_coords[0]) if len(bg_field.x_coords) > 1 else 10.0
            ddy = float(bg_field.y_coords[1] - bg_field.y_coords[0]) if len(bg_field.y_coords) > 1 else 10.0
            extent = [float(bg_field.x_coords[0]) - ddx/2, float(bg_field.x_coords[-1]) + ddx/2,
                      float(bg_field.y_coords[0]) - ddy/2, float(bg_field.y_coords[-1]) + ddy/2]
            if bg_field.land_mask is not None:
                land = np.ma.masked_where(~bg_field.land_mask, np.ones_like(conc))
                ax.imshow(land, origin='lower', extent=extent,
                          cmap=_LCmap(['#FFFFFF']), alpha=1.0, zorder=1, aspect='auto')
            mask = (bg_field.land_mask | (conc < 0.01)) if bg_field.land_mask is not None else (conc < 0.01)
            conc_m = np.ma.masked_where(mask, conc)
            ax.imshow(conc_m, origin='lower', extent=extent, cmap='YlOrRd',
                      alpha=0.9, vmin=0, vmax=max(float(conc.max()), 0.1),
                      zorder=2, aspect='auto')
            _draw_extent[0] = extent   # per fissare i limiti (clip dei cerchi al dominio)

            # Disco virtuale di spawn: 2 circonferenze nere tratteggiate a d_min e d_max,
            # centro = sorgente. Clippate al dominio (appaiono come sezione di disco).
            try:
                disk_env = SourceSeekingEnv(
                    config=SourceSeekingConfig.from_config(config, chunk_id=chunk_id),
                    concentration_field=bg_field, data_manager=data_manager,
                    wind_mapping=wind_mapping, current_mapping=current_mapping)
                disk_env.field = bg_field
                if bg_field.source_position is not None:
                    disk_env.source_position = np.array(bg_field.source_position, dtype=float)
                d_min, d_max = disk_env.spawn_disk_bounds()
                sx0, sy0 = bg_field.source_position
                from matplotlib.patches import Circle as _Circle
                for dd in (d_min, d_max):
                    ax.add_patch(_Circle((sx0, sy0), dd, fill=False, ls='--',
                                         ec='black', lw=1.3, zorder=4))
                ax.plot([], [], ls='--', color='black', lw=1.3,
                        label=f'spawn disk ({d_min:.0f}–{d_max:.0f} m)')
            except Exception as _e:
                print(f"    [disk] {source_id} {version} Q{chunk_id}: {_e}")

        pts = results.get(key, [])
        xs_ok   = [p[0] for p in pts if p[2]]
        ys_ok   = [p[1] for p in pts if p[2]]
        xs_fail = [p[0] for p in pts if not p[2]]
        ys_fail = [p[1] for p in pts if not p[2]]
        if xs_ok:
            ax.scatter(xs_ok, ys_ok, c='green', s=60, zorder=5,
                       label=f'✓ {len(xs_ok)}', edgecolors='white', linewidths=0.5)
        if xs_fail:
            ax.scatter(xs_fail, ys_fail, marker='X', c='darkred', s=80, zorder=6,
                       edgecolors='white', linewidths=0.5, label=f'✗ {len(xs_fail)}')
        if bg_field is not None and bg_field.source_position is not None:
            sx, sy = bg_field.source_position
            ax.scatter(sx, sy, c='yellow', s=200, marker='*',
                       edgecolors='black', linewidths=0.8, zorder=7, label='Source')
        ax.set_title(source_id, fontsize=13, fontweight='bold')
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, loc='upper right')
        # Fissa i limiti al dominio: i cerchi del disco di spawn (anche > dominio)
        # restano clippati → appaiono come sezione di disco.
        if _draw_extent[0] is not None:
            ax.set_xlim(_draw_extent[0][0], _draw_extent[0][1])
            ax.set_ylim(_draw_extent[0][2], _draw_extent[0][3])

    import math

    def _make_grid_fig(items, draw_fn, suptitle, n_cols=2, cell_size=6):
        """Crea una figura a griglia con max n_cols colonne."""
        n = len(items)
        n_cols = min(n, n_cols)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(cell_size * n_cols, cell_size * n_rows),
                                 squeeze=False)
        fig.suptitle(suptitle, fontsize=15, fontweight='bold')
        for idx, item in enumerate(items):
            ax = axes[idx // n_cols][idx % n_cols]
            draw_fn(ax, item)
        # Nascondi assi vuoti
        for idx in range(len(items), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    saved = []

    for source_id in top_sources:
        def _draw_combo(ax, vc, src=source_id):
            v, c = vc
            _draw_subplot(ax, src, v, c)
            ax.set_title(f'{v} / {chunk_labels[c]}', fontsize=13, fontweight='bold')

        title = f'Spawn map  {source_id}'
        fig = _make_grid_fig(active_combos_sorted, _draw_combo, title)
        fname = f"spawn_map_{source_id}.png"
        out_path = OUTPUT_DIR / fname
        fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)
        print(f"  Salvato: {fname}")

    print(f"\nTotale file salvati: {len(saved)} in {OUTPUT_DIR}")


def _find_velocity_run(trained_dir: Path, vmax: int, K: int) -> Optional[Path]:
    """Run dir più recente addestrato con n_velocity_levels==K e max_velocity==vmax.

    I modelli della catena a velocità adattiva si distinguono da quelli a passo
    fisso (n_velocity_levels==1) proprio per K=5. A parità di v_max prende il più
    recente (le directory ppo_<timestamp> sono ordinabili cronologicamente).
    """
    best = None
    for d in sorted(trained_dir.glob("ppo_*")):
        cfg_p = d / "config.yaml"
        if not cfg_p.exists():
            continue
        try:
            c = load_config(str(cfg_p))
        except Exception:
            continue
        ag = c.get('agent', {})
        thr = c.get('environment', {}).get('reward', {}).get('distance_threshold', 50)
        # Esclude esperimenti non-standard dalla catena normale: raggio ≠ 50 (r10),
        # doppia corona (sensor_range_2 presente) e agente senza formazione
        # (mask_formation) → non sovrascrivono i vmax_X.
        if (abs(float(thr) - 50.0) > 1e-6 or ag.get('sensor_range_2') is not None
                or ag.get('mask_formation', False)):
            continue
        if (int(ag.get('n_velocity_levels', 1)) == K
                and abs(float(ag.get('max_velocity', 1.0)) - float(vmax)) < 1e-6):
            best = d
    return best


def _read_global_sr(output_dir: Path) -> float:
    """Legge il Global Success Rate (%) dal log.txt prodotto dall'inferenza."""
    import re
    log = Path(output_dir) / "log.txt"
    if log.exists():
        for ln in log.read_text(errors="ignore").splitlines():
            if "Global Success Rate" in ln:
                m = re.search(r"([\d.,]+)\s*%", ln)
                if m:
                    return float(m.group(1).replace(",", "."))
    return float("nan")


def main_velocity_inference(vmax_list=(1, 2, 3, 4, 5)):
    """Inferenza multi-modello sulla catena a velocità adattiva (v_max da `vmax_list`).

    Per ogni v_max in `vmax_list`: 5 episodi per combinazione (sorgente × scenario
    vento × chunk) → 1560 episodi/modello, policy deterministica, niente video. I
    modelli sono individuati automaticamente in trained_models/ (K=5, max_velocity
    corrispondente), e l'ambiente è ricostruito dal config.yaml salvato nel run
    (n_velocity_levels, max_velocity, sensori, spawn).

    Output: thesis/evaluations/evaluations_RL/evaluations_RL_adaptive/vmax_{v}/
    (v_max float → es. vmax_1.2, che NON sovrascrive vmax_1 della catena intera).
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = str(PROJECT_ROOT / "data")
    trained_dir  = PROJECT_ROOT / "trained_models"
    out_root     = (PROJECT_ROOT / "thesis" / "evaluations"
                    / "evaluations_RL" / "evaluations_RL_adaptive")
    out_root.mkdir(parents=True, exist_ok=True)

    K          = 5
    VMAX_LIST  = list(vmax_list)
    N_EPISODES = 5          # episodi per combinazione (sorgente × vento × chunk) → 1560/modello

    print(f"\n{'#'*100}")
    print(f"#  INFERENZA MULTI-MODELLO — catena velocità adattiva  (v_max = {VMAX_LIST} m/s, K={K})")
    print(f"#  {N_EPISODES} episodi/combinazione (= 1560/modello), deterministica, no video")
    print(f"{'#'*100}")

    results = []
    for vmax in VMAX_LIST:
        run_dir = _find_velocity_run(trained_dir, vmax, K)
        if run_dir is None:
            print(f"\n[SKIP] Nessun modello con K={K}, v_max={vmax} m/s in {trained_dir}")
            results.append((vmax, None, float("nan")))
            continue
        model_path = run_dir / "models" / "final_model.zip"
        if not model_path.exists():
            print(f"\n[SKIP] final_model.zip mancante in {run_dir / 'models'}")
            results.append((vmax, None, float("nan")))
            continue

        cfg_override = load_config(str(run_dir / "config.yaml"))
        out = out_root / f"vmax_{vmax}"
        print(f"\n{'='*100}")
        print(f"  v_max = {vmax} m/s   →   {out.name}")
        print(f"  modello: {model_path}")
        print(f"{'='*100}")

        run_inference(
            model_path=str(model_path),
            config_path=str(run_dir / "config.yaml"),
            config_override=cfg_override,
            data_dir=DATA_DIR,
            output_dir=str(out),
            n_episodes=N_EPISODES,
            deterministic=True,
            sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2],
            save_videos=False,
            plot_velocity=True,
        )
        results.append((vmax, model_path, _read_global_sr(out)))

    # --- Tabella riepilogo SR vs v_max ---
    print(f"\n{'#'*100}")
    print("#  RIEPILOGO — Global Success Rate per v_max")
    print(f"{'#'*100}")
    print(f"  {'v_max':>8}{'SR':>10}   modello")
    print(f"  {'-'*88}")
    for vmax, mp, sr in results:
        sr_s = f"{sr:.1f}%" if sr == sr else "n/d"
        print(f"  {str(vmax)+' m/s':>8}{sr_s:>10}   {mp if mp else '— (assente)'}")
    print(f"  {'-'*88}\n")
    return results


def _find_radius10_run(trained_dir: Path, K: int = 5):
    """Run più recente dell'esperimento raggio 10 m (K=5, max_velocity=5, distance_threshold==10)."""
    best = None
    for d in sorted(trained_dir.glob("ppo_*")):
        cfg_p = d / "config.yaml"
        if not cfg_p.exists():
            continue
        try:
            c = load_config(str(cfg_p))
        except Exception:
            continue
        ag = c.get('agent', {})
        thr = c.get('environment', {}).get('reward', {}).get('distance_threshold', 50)
        if (int(ag.get('n_velocity_levels', 1)) == K
                and abs(float(ag.get('max_velocity', 1.0)) - 5.0) < 1e-6
                and abs(float(thr) - 10.0) < 1e-6):
            best = d
    return best


def main_radius10_inference():
    """Inferenza dell'esperimento 'raggio di successo 10 m' (v_max=5, modulazione forzata).

    Output dedicato (vmax_5_r10/) per non sovrascrivere la catena. Genera entrambi i
    grafici velocità: nel tempo e — soprattutto — VS DISTANZA (il diagnostico per la
    modulazione: se rallenta vicino alla sorgente la curva scende verso destra).
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = str(PROJECT_ROOT / "data")
    trained_dir  = PROJECT_ROOT / "trained_models"
    out = (PROJECT_ROOT / "thesis" / "evaluations"
           / "evaluations_RL" / "evaluations_RL_adaptive" / "r10")

    run_dir = _find_radius10_run(trained_dir)
    if run_dir is None:
        print("[ERRORE] Nessun modello raggio-10 trovato (K=5, v_max=5, distance_threshold=10).")
        return
    model_path = run_dir / "models" / "final_model.zip"
    if not model_path.exists():
        print(f"[ERRORE] final_model.zip mancante in {run_dir / 'models'}")
        return

    cfg_override = load_config(str(run_dir / "config.yaml"))
    print(f"\n{'#'*100}")
    print(f"#  INFERENZA ESPERIMENTO RAGGIO 10 m — v_max=5, modulazione forzata")
    print(f"#  modello: {model_path}")
    print(f"{'#'*100}")
    run_inference(
        model_path=str(model_path),
        config_path=str(run_dir / "config.yaml"),
        config_override=cfg_override,
        data_dir=DATA_DIR,
        output_dir=str(out),
        n_episodes=5,
        deterministic=True,
        sources_csv="Coordinate_Sorgenti_FaseII.csv",
        chunk_ids=[0, 1, 2],
        save_videos=False,
        plot_velocity=True,
    )
    print(f"\nSR globale: {_read_global_sr(out):.1f}%   →  vedi {out}/analysis/velocity_vs_distance.png")


def _find_dualcorona_run(trained_dir: Path, vmax: float = 1.0, K: int = 5):
    """Run più recente della doppia corona per il dato v_max: K livelli, sensor_range_2 presente."""
    best = None
    for d in sorted(trained_dir.glob("ppo_*")):
        cfg_p = d / "config.yaml"
        if not cfg_p.exists():
            continue
        try:
            ag = load_config(str(cfg_p)).get('agent', {})
        except Exception:
            continue
        if (int(ag.get('n_velocity_levels', 1)) == K
                and abs(float(ag.get('max_velocity', 1.0)) - float(vmax)) < 1e-6
                and ag.get('sensor_range_2') is not None):
            best = d
    return best


def main_dualcorona_inference(vmax: int = 1):
    """Inferenza della doppia corona a un dato v_max — confronto SR + step-to-success.

    Output dedicato (dualcorona_v{vmax}/). Stesso protocollo held-out della catena, così i
    numeri sono confrontabili con vmax_{vmax} (singola corona). Ritorna la cartella di output
    (o None se il modello non c'è).
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = str(PROJECT_ROOT / "data")
    trained_dir  = PROJECT_ROOT / "trained_models"
    out = (PROJECT_ROOT / "thesis" / "evaluations"
           / "evaluations_RL" / "evaluations_RL_adaptive" / f"dualcorona_v{vmax}")

    run_dir = _find_dualcorona_run(trained_dir, vmax=vmax)
    if run_dir is None:
        print(f"[ERRORE] Nessun modello doppia corona v_max={vmax} trovato "
              f"(K=5, sensor_range_2 presente).")
        return None
    model_path = run_dir / "models" / "final_model.zip"
    if not model_path.exists():
        print(f"[ERRORE] final_model.zip mancante in {run_dir / 'models'}")
        return None

    cfg_override = load_config(str(run_dir / "config.yaml"))
    print(f"\n{'#'*100}")
    print(f"#  INFERENZA DOPPIA CORONA v_max={vmax} — confronto vs singola corona (vmax_{vmax})")
    print(f"#  modello: {model_path}")
    print(f"{'#'*100}")
    run_inference(
        model_path=str(model_path),
        config_path=str(run_dir / "config.yaml"),
        config_override=cfg_override,
        data_dir=DATA_DIR,
        output_dir=str(out),
        n_episodes=5,
        deterministic=True,
        sources_csv="Coordinate_Sorgenti_FaseII.csv",
        chunk_ids=[0, 1, 2],
        save_videos=False,
    )
    print(f"\nSR globale doppia corona v_max={vmax}: {_read_global_sr(out):.1f}%   →  {out}/log.txt")
    return out


def _dualcorona_step_table(vmax_list):
    """Stampa la tabella riassuntiva doppia vs singola corona: SR e step medi al successo
    per ogni v_max, con il guadagno percentuale di step (obiettivo dello sweep opzione 2)."""
    import json as _json
    base = (Path(__file__).resolve().parent.parent / "thesis" / "evaluations"
            / "evaluations_RL" / "evaluations_RL_adaptive")

    def _stats(p: Path):
        d = _json.loads(p.read_text())
        sr = float(np.mean([e['success'] for e in d]) * 100.0)
        st = [e['steps'] for e in d if e['success']]
        return sr, (float(np.mean(st)) if st else float('nan')), len(d)

    print(f"\n{'='*84}")
    print("  DOPPIA vs SINGOLA CORONA — step medi al successo per v_max (guadagno %)")
    print(f"{'='*84}")
    print(f"  {'v_max':>6} | {'SR sing.':>9} {'SR dopp.':>9} | "
          f"{'step sing.':>10} {'step dopp.':>10} | {'Δstep':>8} {'Δstep%':>8}")
    print(f"  {'-'*80}")
    for vmax in vmax_list:
        sp = base / f"vmax_{vmax}" / "episodes_data.json"
        dp = base / f"dualcorona_v{vmax}" / "episodes_data.json"
        if not (sp.exists() and dp.exists()):
            miss = ([] if sp.exists() else ["singola"]) + ([] if dp.exists() else ["doppia"])
            print(f"  {vmax:>6} | (dati mancanti: {', '.join(miss)})")
            continue
        ssr, sst, _ = _stats(sp)
        dsr, dst, _ = _stats(dp)
        dstep = dst - sst
        dstep_pct = 100.0 * dstep / sst if sst == sst and sst > 0 else float('nan')
        print(f"  {vmax:>6} | {ssr:>8.1f}% {dsr:>8.1f}% | "
              f"{sst:>10.1f} {dst:>10.1f} | {dstep:>+8.1f} {dstep_pct:>+7.1f}%")
    print(f"  {'-'*80}")
    print("  (Δstep% < 0 = la doppia corona è più veloce; obiettivo: vedere se il -18% di v1 regge)")


def main_dualcorona_sweep_inference(vmax_list=(1, 2, 3, 4, 5)):
    """Inferenza della doppia corona a TUTTI i v_max (opzione 2) + tabella Δstep vs singola.

    Cerca ed esegue i modelli dualcorona_v{X} presenti; alla fine stampa il confronto degli
    step medi al successo contro la catena singola (vmax_X). I livelli senza modello vengono
    saltati (utile per lanciarla mentre i training procedono).
    """
    print(f"\n{'#'*100}")
    print(f"#  SWEEP INFERENZA DOPPIA CORONA — v_max = {list(vmax_list)}")
    print(f"{'#'*100}")
    done = []
    for vmax in vmax_list:
        if main_dualcorona_inference(vmax=vmax) is not None:
            done.append(vmax)
    if done:
        _dualcorona_step_table(done)
    else:
        print("\n[nota] Nessun modello doppia corona trovato per i v_max richiesti.")


def main_agent_heatmap(scenarios=None, n_episodes: int = 10,
                       ppo_single_run: str = "ppo_20260629_065041",
                       ppo_double_run: str = "ppo_20260704_015003",
                       fcm_sensor: float = 50.0, fcm_lr: float = 50.0):
    """Heatmap di occupazione dell'agente (densità di presenza) per scenario.

    Per ogni scenario (source_id, versione, chunk_id) esegue `n_episodes` episodi con tre
    agenti alla stessa velocità massima --- FCM (sensor/lr indicati, $\\sim$5 m/s), PPO singola
    corona (v5) e PPO doppia corona (v5) --- accumula le posizioni in una densità di presenza
    (istogramma fine + smoothing gaussiano) e disegna 3 pannelli affiancati (mare/costa/
    sorgente + heatmap) su SCALA COMUNE, fissata sui due modelli PPO; l'FCM, che oscilla a
    lungo negli stessi punti, ha picchi più alti e satura.
    Output: thesis/evaluations/agent_heatmaps/heatmap_<i>_<src>_<ver>_Q<chunk>.png
    """
    import copy
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import ListedColormap
    from utils.data_loader import NetCDFLoader as _NCLoader

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = str(PROJECT_ROOT / "data")
    OUT = PROJECT_ROOT / "thesis" / "evaluations" / "agent_heatmaps"
    OUT.mkdir(parents=True, exist_ok=True)
    if scenarios is None:   # default: 5 scenari difficili (V1/V2 × Q1/2,Q3/4)
        scenarios = [('SRC114', 'V2', 2), ('SRC120', 'V2', 1), ('SRC117', 'V2', 2),
                     ('SRC109', 'V1', 2), ('SRC130', 'V2', 1)]
    chunk_lbl = {0: 'Q1/4', 1: 'Q1/2', 2: 'Q3/4'}

    dm = DataManager(data_dir=DATA_DIR, preload_all=False, sources_csv="Coordinate_Sorgenti_FaseII.csv")
    wind_mapping = {"_V0": "CI_WIND_faseII_V0.txt", "_V1": "CI_WIND_faseII_V1.txt",
                    "_V2": "CI_WIND_faseII_V2.txt", "_V3": "CI_WIND_faseII_V3.txt"}
    current_mapping = {"_V0": "CL02_V0_SRC000_U_V_10mGrid.nc", "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
                       "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc", "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc"}

    # --- agenti: FCM Adam + PPO singola (v5) + PPO doppia (v5) ---
    fcm_cfg = load_config(str(PROJECT_ROOT / "utils" / "config" / "config_base_no_wind_reward.yaml"))
    fcm_cfg['agent']['sensor_range'] = fcm_sensor
    fcm_cfg['agent'].pop('sensor_range_2', None)
    fcm_cfg['agent']['n_velocity_levels'] = 1
    fcm_agent = AdamFCMAgent(sensor_range=fcm_sensor, lr=fcm_lr)
    ps_dir = PROJECT_ROOT / "trained_models" / ppo_single_run
    pd_dir = PROJECT_ROOT / "trained_models" / ppo_double_run
    ps_cfg = load_config(str(ps_dir / "config.yaml")); ps_model = load_model(str(ps_dir / "models" / "final_model.zip")); ps_vn = ps_dir / "models" / "vec_normalize.pkl"
    pd_cfg = load_config(str(pd_dir / "config.yaml")); pd_model = load_model(str(pd_dir / "models" / "final_model.zip")); pd_vn = pd_dir / "models" / "vec_normalize.pkl"

    def _load_field(src, ver):
        files = [f for f in dm._nc_files if f'_{ver}_' in f.name and src in f.name and 'Conc' in f.name]
        if not files:
            return None
        fld = _NCLoader(DATA_DIR).load(str(files[0]), concentration_var="Concentration - component 1")
        if fld is None:
            return None
        c = dm.get_source_coordinates(src)
        if c:
            fld.source_position = c
        fld.run_id = f'{src}_{ver}'
        return fld

    def _collect(kind, ch, field):
        env_cfg = SourceSeekingConfig.from_config(fcm_cfg if kind == 'fcm' else (ps_cfg if kind == 'ps' else pd_cfg), chunk_id=ch)
        pts = []; succ = 0
        for _ in range(n_episodes):
            if kind == 'fcm':
                ve = build_env_fcm(SourceSeekingConfig.from_config(fcm_cfg, chunk_id=ch), copy.deepcopy(field),
                                   use_masking=MASKABLE_PPO_AVAILABLE, data_manager=dm,
                                   wind_mapping=wind_mapping, current_mapping=current_mapping)
                r = run_episode_fcm(fcm_agent, ve, deterministic=True)
            else:
                model = ps_model if kind == 'ps' else pd_model
                vn = ps_vn if kind == 'ps' else pd_vn
                ve = build_env(env_cfg, copy.deepcopy(field), vn, use_masking=MASKABLE_PPO_AVAILABLE,
                               data_manager=dm, wind_mapping=wind_mapping, current_mapping=current_mapping)
                r = run_episode(model, ve, deterministic=False)
            ve.close()
            pts.append(np.asarray(r.trajectory)); succ += int(r.success)
        return np.concatenate(pts, axis=0), succ

    def _density(pts, field):
        xc, yc = np.asarray(field.x_coords), np.asarray(field.y_coords)
        dx = float(xc[1]-xc[0]) if len(xc) > 1 else 10.0
        dy = float(yc[1]-yc[0]) if len(yc) > 1 else 10.0
        x0, x1 = float(xc[0])-dx/2, float(xc[-1])+dx/2
        y0, y1 = float(yc[0])-dy/2, float(yc[-1])+dy/2
        xe = np.linspace(x0, x1, 181); ye = np.linspace(y0, y1, 151)   # ~17 m/bin
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[xe, ye])
        H = gaussian_filter(H.T, sigma=3.2)
        s = H.sum()
        if s > 0:
            H = H / s
        return H, [x0, x1, y0, y1]

    def _draw(ax, H, extent, field, src_pos, title, vmax):
        ax.set_facecolor('#87CEEB')                                    # mare
        if field.land_mask is not None:                                # terra bianca
            land = np.ma.masked_where(~field.land_mask, np.ones(field.land_mask.shape))
            ax.imshow(land, origin='lower', extent=extent, cmap=ListedColormap(['#FFFFFF']), zorder=1, aspect='auto')
        Hm = np.ma.masked_where(H < vmax * 0.03, H)                    # heatmap (scala comune)
        im = ax.imshow(Hm, origin='lower', extent=extent, cmap='hot', alpha=0.85, zorder=3,
                       aspect='auto', interpolation='bilinear', vmin=0.0, vmax=vmax)
        ax.scatter(*src_pos, marker='*', s=260, c='lime', edgecolors='k', linewidths=0.8, zorder=6)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        return im

    print(f"\n{'#'*92}")
    print(f"#  HEATMAP DI OCCUPAZIONE — {len(scenarios)} scenari, {n_episodes} episodi/modello")
    print(f"{'#'*92}")
    for si, (src, ver, ch) in enumerate(scenarios, 1):
        fld = _load_field(src, ver)
        if fld is None:
            print(f"  [SKIP] {src} {ver}: campo non trovato"); continue
        sp = np.array(dm.get_source_coordinates(src), dtype=float)
        panels = []
        for kind, name in zip(['fcm', 'ps', 'pd'], ['FCM (5 m/s)', 'PPO single v5', 'PPO double v5']):
            pts, succ = _collect(kind, ch, fld)
            H, extent = _density(pts, fld)
            panels.append((H, extent, succ, name))
            print(f"  {src} {ver} {chunk_lbl[ch]} [{name}] {len(pts)} punti, {succ}/{n_episodes}")
        vmax = max(panels[1][0].max(), panels[2][0].max())             # scala comune fissata sui PPO
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
        im = None
        for ax, (H, extent, succ, name) in zip(axes, panels):
            im = _draw(ax, H, extent, fld, sp, f'{name}  —  {succ}/{n_episodes} successes', vmax)
        cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.015)
        cb.set_label('presence density (norm., common scale)', fontsize=8); cb.ax.tick_params(labelsize=7)
        fig.suptitle(f'Occupancy heatmap — {src}, {ver}, {chunk_lbl[ch]}', fontsize=13)
        out = OUT / f'heatmap_{si}_{src}_{ver}_Q{ch}.png'
        fig.savefig(str(out), dpi=160, bbox_inches='tight'); plt.close(fig)
        print(f"  salvato: {out.name}")
    print(f"\nTotale in {OUT}")


if __name__ == "__main__":
    import sys as _sys
    arg = _sys.argv[1] if len(_sys.argv) > 1 else ""
    if arg == "r10":
        main_radius10_inference()
    elif arg == "dc":
        _vmax = int(_sys.argv[2]) if len(_sys.argv) > 2 else 1
        main_dualcorona_inference(vmax=_vmax)
    elif arg == "dc-sweep":
        if len(_sys.argv) > 2:
            main_dualcorona_sweep_inference(vmax_list=[
                (int(float(x)) if float(x).is_integer() else float(x)) for x in _sys.argv[2:]])
        else:
            main_dualcorona_sweep_inference()
    elif arg == "dc-mid":
        # Doppia corona v_max INTERMEDI (dopo STAGE=dc_mid).
        main_dualcorona_sweep_inference(vmax_list=[1.2, 1.5, 1.7, 1.9])
    elif arg == "dc-low":
        # Doppia corona v_max BASSI (dopo STAGE=dc_low).
        main_dualcorona_sweep_inference(vmax_list=[0.7, 0.4, 0.1])
    elif arg == "spawn-fcm":
        main_spawn_map(mode='fcm')
    elif arg == "spawn-ppo":
        main_spawn_map(mode='ppo')
    elif arg == "spawn-ppo-double":
        main_spawn_map(mode='ppo_double')
    elif arg == "spawn":
        main_spawn_map(mode='ppo')
    elif arg == "heatmaps":
        _n = int(_sys.argv[2]) if len(_sys.argv) > 2 else 10
        main_agent_heatmap(n_episodes=_n)
    elif arg in ("full", "chain", "1-5"):
        # Catena INTERA v_max 1→5 m/s (5 run × 1560 ep).
        main_velocity_inference(vmax_list=[1, 2, 3, 4, 5])
    elif arg in ("low", "0.1-0.7"):
        # Basse velocità 0.7/0.4/0.1 m/s (3 run × 1560 ep) — dopo STAGE=ppo_low.
        main_velocity_inference(vmax_list=[0.7, 0.4, 0.1])
    elif arg in ("prof-all", "all-new"):
        # Inferenza held-out (1560 ep/modello) su TUTTI i modelli del prof, in sequenza:
        #   1) SINGOLA corona 0.7/0.4/0.1   2) DOPPIA corona 0.7/0.4/0.1
        #   3) SINGOLA corona 1.2/1.5/1.7/1.9 (GIA' valutata -> saltata; "prof-all full" per rifarla)
        #   4) DOPPIA  corona 1.2/1.5/1.7/1.9
        _redo_single_mid = len(_sys.argv) > 2 and _sys.argv[2] == "full"
        print("\n########## PROF-ALL 1/4 — SINGOLA corona 0.7/0.4/0.1 ##########", flush=True)
        main_velocity_inference(vmax_list=[0.7, 0.4, 0.1])
        print("\n########## PROF-ALL 2/4 — DOPPIA corona 0.7/0.4/0.1 ##########", flush=True)
        main_dualcorona_sweep_inference(vmax_list=[0.7, 0.4, 0.1])
        if _redo_single_mid:
            print("\n########## PROF-ALL 3/4 — SINGOLA corona 1.2/1.5/1.7/1.9 (re-run) ##########", flush=True)
            main_velocity_inference(vmax_list=[1.2, 1.5, 1.7, 1.9])
        else:
            print("\n########## PROF-ALL 3/4 — SINGOLA corona 1.2/1.5/1.7/1.9: GIA' PRESENTE, saltata "
                  "(usa 'prof-all full' per rifarla) ##########", flush=True)
        print("\n########## PROF-ALL 4/4 — DOPPIA corona 1.2/1.5/1.7/1.9 ##########", flush=True)
        main_dualcorona_sweep_inference(vmax_list=[1.2, 1.5, 1.7, 1.9])
        print("\n########## PROF-ALL COMPLETATO ##########", flush=True)
    else:
        # DEFAULT: catena v_max INTERMEDI 1.2/1.5/1.7/1.9 (4 run × 1560 ep).
        main_velocity_inference(vmax_list=[1.2, 1.5, 1.7, 1.9])