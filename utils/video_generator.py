"""
Generazione video showcase per inferenza HYDRAS.

Seleziona automaticamente 8 successi (2 per versione vento V0-V3:
step più vicino alla mediana + step più vicino al 25° percentile)
+ 2 fallimenti (distanza finale minima e massima tra i peggiori),
poi genera video MP4 animati ricaricando i field dal disco.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import ListedColormap


def select_showcase_episodes(all_results: list) -> list:
    """
    Seleziona 8 successi (2 per V0/V1/V2/V3: step più vicino alla mediana
    e step più vicino al 25° percentile) e 2 fallimenti (distanza finale
    minima e massima tra i fallimenti con dist > mediana fallimenti).

    Il formato di result.scenario è "V1_SRC109_Q1/4".
    """
    successes_by_version = defaultdict(list)
    failures = []

    for result in all_results:
        version = result.scenario.split('_')[0]
        if result.success and version in ('V0', 'V1', 'V2', 'V3'):
            successes_by_version[version].append(result)
        elif not result.success:
            failures.append(result)

    selected = []
    for version in ['V0', 'V1', 'V2', 'V3']:
        eps = successes_by_version.get(version, [])
        if not eps:
            continue
        steps_list = [e.steps for e in eps]
        median_steps = float(np.median(steps_list))
        p25_steps = float(np.percentile(steps_list, 25))

        # Episodio più vicino alla mediana (tipico)
        ep_median = min(eps, key=lambda e: abs(e.steps - median_steps))
        selected.append(ep_median)

        # Episodio più vicino al 25° percentile (veloce), diverso dal primo
        remaining = [e for e in eps if e is not ep_median]
        if remaining:
            ep_fast = min(remaining, key=lambda e: abs(e.steps - p25_steps))
            selected.append(ep_fast)

    # 2 fallimenti: quello con distanza finale minima (quasi-successo)
    # e quello con distanza finale massima (fallimento totale)
    if failures:
        fail_sorted = sorted(failures, key=lambda e: e.final_distance)
        selected.append(fail_sorted[0])   # più vicino alla sorgente
        if len(fail_sorted) > 1:
            selected.append(fail_sorted[-1])  # più lontano dalla sorgente

    return selected


def _make_video(result, field, output_path: Path,
                fps: int = 15, target_duration_s: int = 30) -> Path:
    """Genera e salva un video animato per un singolo episodio."""
    traj = result.trajectory          # (N, 2)
    N = len(traj)
    stride = max(1, N // (fps * target_duration_s))
    n_anim_frames = (N + stride - 1) // stride

    extent = [float(field.x_coords.min()), float(field.x_coords.max()),
              float(field.y_coords.min()), float(field.y_coords.max())]
    plume_threshold = 0.01

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#87CEEB')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Terra (fissa)
    if field.land_mask is not None:
        white_cmap = ListedColormap(['#FFFFFF'])
        land_display = np.ma.masked_where(~field.land_mask, np.ones(field.land_mask.shape))
        ax.imshow(land_display, origin='lower', extent=extent,
                  cmap=white_cmap, alpha=1.0, zorder=1)

    # Campo concentrazione al frame iniziale
    field.set_time(result.start_frame)
    conc0 = field.get_current_field()
    vmax = max(float(conc0.max()), 0.1)
    mask0 = (field.land_mask | (conc0 < plume_threshold)) if field.land_mask is not None \
             else (conc0 < plume_threshold)
    im = ax.imshow(np.ma.masked_where(mask0, conc0), origin='lower', extent=extent,
                   cmap='YlOrRd', alpha=0.9, vmin=0, vmax=vmax, zorder=2)
    plt.colorbar(im, ax=ax, label='Concentrazione')

    # Sorgente e punto di partenza (fissi)
    if field.source_position is not None:
        ax.scatter(field.source_position[0], field.source_position[1],
                   c='yellow', s=200, marker='*', edgecolors='black', zorder=6, label='Source')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')

    # Artisti animati
    traj_line, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.85, label='Traiettoria')
    agent_dot  = ax.scatter([], [], c='darkred', s=140, marker='o', zorder=7, label='Agente')

    status_str = "SUCCESS ✓" if result.success else f"FAILED [{result.termination}]"
    base_title = f"{result.scenario} — Ep {result.episode + 1} — {status_str}"
    title_obj = ax.set_title(base_title, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    n_frames_field = result.end_frame - result.start_frame

    def _update(anim_idx):
        traj_idx = min(anim_idx * stride, N - 1)
        t = (result.start_frame + n_frames_field * traj_idx / max(N - 1, 1)) \
            if n_frames_field > 0 else float(result.start_frame)
        field.set_time(t)
        conc = field.get_current_field()
        mask = (field.land_mask | (conc < plume_threshold)) if field.land_mask is not None \
               else (conc < plume_threshold)
        im.set_data(np.ma.masked_where(mask, conc))
        traj_line.set_data(traj[:traj_idx + 1, 0], traj[:traj_idx + 1, 1])
        agent_dot.set_offsets([[traj[traj_idx, 0], traj[traj_idx, 1]]])
        title_obj.set_text(f"{base_title}\nStep {traj_idx}/{N - 1}")
        return im, traj_line, agent_dot, title_obj

    ani = FuncAnimation(fig, _update, frames=n_anim_frames, blit=True, interval=1000 // fps)

    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, metadata=dict(title=base_title), bitrate=2000)
        video_path = output_path.with_suffix('.mp4')
        ani.save(str(video_path), writer=writer, dpi=120)
    else:
        writer = PillowWriter(fps=fps)
        video_path = output_path.with_suffix('.gif')
        ani.save(str(video_path), writer=writer, dpi=100)

    plt.close(fig)
    return video_path


def generate_showcase_videos(
    all_results: list,
    data_manager,
    output_dir: Path,
    fps: int = 15,
):
    """
    Seleziona i 10 episodi showcase tra tutti i risultati dell'inferenza
    (8 successi: 2 per versione V0-V3; 2 fallimenti: distanza minima e massima)
    e genera i video MP4 corrispondenti ricaricando i field dal disco.

    Args:
        all_results:       lista di EpisodeResult accumulati durante run_inference
        data_manager:      DataManager per ricaricare i field
        output_dir:        directory radice dell'inferenza (videos/ creata al suo interno)
        fps:               fotogrammi al secondo
    """
    selected = select_showcase_episodes(all_results)
    if not selected:
        print("[Video] Nessun episodio idoneo trovato.")
        return

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generazione video showcase — {len(selected)} episodi selezionati (target: 10)")
    print(f"{'='*60}")
    for r in selected:
        label = "SUCCESSO" if r.success else "FALLIMENTO"
        print(f"  • {r.scenario}  Ep{r.episode + 1}  [{label}]  steps={r.steps}  "
              f"final_dist={r.final_distance:.0f}m")
    print()

    for result in selected:
        # scenario format: "V1_SRC109_Q1/4"
        parts = result.scenario.split('_')
        version   = parts[0]
        source_id = parts[1] if len(parts) > 1 else "SRC000"

        print(f"  Generando: {result.scenario}  Ep{result.episode + 1} ...", flush=True)
        try:
            matching = [
                f for f in data_manager._nc_files
                if source_id in f.name and version in f.name and 'Conc' in f.name
            ]
            if not matching:
                raise FileNotFoundError(
                    f"Nessun file NC per {source_id}/{version} in {data_manager.data_dir}"
                )
            field = data_manager._nc_loader.load(str(matching[0]))
            coords = data_manager.get_source_coordinates(source_id)
            if coords:
                field.source_position = coords
        except Exception as exc:
            print(f"  [SKIP] Impossibile caricare il field: {exc}")
            continue

        tag = "success" if result.success else "failure"
        safe_scenario = result.scenario.replace('/', '-')
        video_stem = videos_dir / f"{safe_scenario}_ep{result.episode + 1:02d}_{tag}"

        saved = _make_video(result, field, video_stem, fps=fps)
        print(f"  ✓ {saved.name}")

    print(f"\nVideo salvati in: {videos_dir}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Video di CONFRONTO (FCM vs PPO) a v_max = 1.2 m/s (FCM lr = 12) — STANDALONE
#  Eseguibile senza script esterni:
#     python utils/video_generator.py batch  all          # 4 gruppo + 4 singoli per V0/V1/V2
#     python utils/video_generator.py batch  v2
#     python utils/video_generator.py group  SRC128 V2 1   # un video di gruppo (chunk 1 = Q1/2)
#     python utils/video_generator.py single SRC128 V2 1   # 4 video singoli
#  I video finiscono in  <root>/videos/<v0|v1|v2>/ .  Gli import di src.inference
#  sono lazy (dentro le funzioni) per evitare import circolari con inference.py.
# ═══════════════════════════════════════════════════════════════════════════

import copy as _copy

_SINGLE_RUN = "ppo_20260731_092012"   # PPO singola corona, v_max 1.2
_DOUBLE_RUN = "ppo_20260805_103656"   # PPO doppia corona,  v_max 1.2
_NOFORM_RUN = "ppo_20260819_213118"   # PPO central-agent-only (no formation), v_max 1.2
_FCM_SENSOR, _FCM_LR = 50.0, 12.0     # FCM Adam corrispondente a v_max 1.2
_WIND = {"_V0": "CI_WIND_faseII_V0.txt", "_V1": "CI_WIND_faseII_V1.txt",
         "_V2": "CI_WIND_faseII_V2.txt", "_V3": "CI_WIND_faseII_V3.txt"}
_CURR = {"_V0": "CL02_V0_SRC000_U_V_10mGrid.nc", "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
         "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc", "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc"}
_C_FCM, _C_NF, _C_S, _C_D = "#7f7f7f", "#2ca02c", "#1f5fb4", "#e07b28"
_CHUNK_LBL = {0: "Q1-4", 1: "Q1-2", 2: "Q3-4"}
_METHODS = [('fcm', 'FCM', _C_FCM), ('n', 'PPO, central-agent-only', _C_NF),
            ('s', 'PPO single ring', _C_S), ('d', 'PPO double ring', _C_D)]
_MLAB = {k: (lab, col) for k, lab, col in _METHODS}
_SUGGESTED = {'v0': ['SRC115', 'SRC116'], 'v1': ['SRC117', 'SRC125'], 'v2': ['SRC122', 'SRC109']}
_CTX = None


def _root():
    import sys
    r = Path(__file__).resolve().parent.parent
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))
    return r


def _ctx():
    """Carica (una volta sola) modelli, config, DataManager e agente FCM."""
    global _CTX
    if _CTX is not None:
        return _CTX
    root = _root()
    from src.inference import load_config, load_model, AdamFCMAgent
    from utils.data_loader import DataManager
    data = str(root / "data"); tm = root / "trained_models"

    def _load(run):
        d = tm / run
        return dict(cfg=load_config(str(d / "config.yaml")),
                    model=load_model(str(d / "models" / "final_model.zip")),
                    vn=d / "models" / "vec_normalize.pkl")

    _CTX = dict(
        root=root, data=data,
        dm=DataManager(data_dir=data, preload_all=False,
                       sources_csv="Coordinate_Sorgenti_FaseII.csv"),
        s=_load(_SINGLE_RUN), d=_load(_DOUBLE_RUN), n=_load(_NOFORM_RUN),
        f_cfg=load_config(str(root / "utils" / "config" / "config_base_no_wind_reward.yaml")),
        fcm=AdamFCMAgent(sensor_range=_FCM_SENSOR, lr=_FCM_LR),
    )
    return _CTX


def _cmp_load_field(src, ver):
    from utils.data_loader import NetCDFLoader
    c = _ctx(); dm = c['dm']
    files = [f for f in dm._nc_files if f'_{ver}_' in f.name and src in f.name and 'Conc' in f.name]
    if not files:
        return None
    fld = NetCDFLoader(c['data']).load(str(files[0]), concentration_var="Concentration - component 1")
    if fld is None:
        return None
    coords = dm.get_source_coordinates(src)
    if coords:
        fld.source_position = coords
    fld.run_id = f'{src}_{ver}'
    return fld


def _cmp_natural_start(field, ch, seed):
    from src.inference import build_env, get_inner_env, MASKABLE_PPO_AVAILABLE
    from utils.source_seeking_env import SourceSeekingConfig
    c = _ctx()
    cfg = SourceSeekingConfig.from_config(c['s']['cfg'], chunk_id=ch)
    ve = build_env(cfg, _copy.deepcopy(field), c['s']['vn'], use_masking=MASKABLE_PPO_AVAILABLE,
                   data_manager=c['dm'], wind_mapping=_WIND, current_mapping=_CURR)
    inner = get_inner_env(ve); inner.reset(seed=seed)
    pos = inner.state.position.copy(); ve.close()
    return (float(pos[0]), float(pos[1]))


def _cmp_rollout(kind, ch, field, start, seed=42):
    from src.inference import (build_env, build_env_fcm, run_episode, run_episode_fcm,
                               get_inner_env, make_env_config, MASKABLE_PPO_AVAILABLE)
    from utils.source_seeking_env import SourceSeekingConfig
    c = _ctx()
    if kind == 'fcm':
        cfg = make_env_config(c['f_cfg'], chunk_id=ch); cfg.sensor_range = _FCM_SENSOR
        ve = build_env_fcm(cfg, _copy.deepcopy(field), use_masking=MASKABLE_PPO_AVAILABLE,
                           data_manager=c['dm'], wind_mapping=_WIND, current_mapping=_CURR)
        model = None
    else:
        mc = c[kind]                                   # 's' | 'd' | 'n'
        cfg = SourceSeekingConfig.from_config(mc['cfg'], chunk_id=ch)
        ve = build_env(cfg, _copy.deepcopy(field), mc['vn'], use_masking=MASKABLE_PPO_AVAILABLE,
                       data_manager=c['dm'], wind_mapping=_WIND, current_mapping=_CURR)
        model = mc['model']
    inner = get_inner_env(ve)
    inner._spawn_on_plume = lambda s=start: (s[0], s[1])   # START COMUNE forzato
    inner.reset(seed=seed)
    r = run_episode_fcm(c['fcm'], ve, deterministic=True) if kind == 'fcm' \
        else run_episode(model, ve, deterministic=True)
    ve.close()
    return r


def _animate_multi(field, src_pos, start, agents, title, out_path, fps=15, target_s=25):
    """Anima N agenti (1 = video singolo, 4 = video di gruppo) sul campo che evolve.
    Stile canonico: mare azzurro, terra bianca, plume YlOrRd. Durata naturale."""
    nmax = max(len(a['traj']) for a in agents)
    stride = max(1, nmax // (fps * target_s))
    nfr = (nmax + stride - 1) // stride
    xc, yc = field.x_coords, field.y_coords
    extent = [float(xc.min()), float(xc.max()), float(yc.min()), float(yc.max())]
    thr = 0.01
    ref = min(agents, key=lambda a: len(a['traj']))
    sf = float(getattr(agents[0]['res'], 'start_frame', 0) or 0)
    ef = float(getattr(ref['res'], 'end_frame', sf) or sf)
    rate = (ef - sf) / max(len(ref['traj']) - 1, 1)
    nT = getattr(field, 'n_timesteps', 1)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor('#87CEEB'); ax.set_aspect('equal')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    if field.land_mask is not None:
        land = np.ma.masked_where(~field.land_mask, np.ones(field.land_mask.shape))
        ax.imshow(land, origin='lower', extent=extent, cmap=ListedColormap(['#FFFFFF']),
                  alpha=1.0, zorder=1)
    field.set_time(sf)
    c0 = field.get_current_field(); vmax = max(float(c0.max()), 0.1)
    m0 = (field.land_mask | (c0 < thr)) if field.land_mask is not None else (c0 < thr)
    im = ax.imshow(np.ma.masked_where(m0, c0), origin='lower', extent=extent, cmap='YlOrRd',
                   alpha=0.9, vmin=0, vmax=vmax, zorder=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label('Concentration')
    ax.scatter(src_pos[0], src_pos[1], c='yellow', s=240, marker='*', edgecolors='black',
               zorder=8, label='Source')
    ax.scatter(start[0], start[1], c='white', s=90, marker='o', edgecolors='black',
               linewidths=1.2, zorder=7, label='Start')
    lines, dots = [], []
    for a in agents:
        ln, = ax.plot([], [], '-', color=a['color'], lw=2.0, alpha=0.9, zorder=4, label=a['label'])
        dt = ax.scatter([], [], color=a['color'], s=95, marker='o', edgecolors='k',
                        linewidths=0.6, zorder=6)
        lines.append(ln); dots.append(dt)
    ax.set_xlim(float(xc.min()), float(xc.max())); ax.set_ylim(float(yc.min()), float(yc.max()))
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.93)
    ttl = ax.set_title(f"{title}\nstep 0", fontsize=11)

    def _upd(i):
        k = min(i * stride, nmax - 1)
        field.set_time(min(sf + k * rate, nT - 1))
        conc = field.get_current_field()
        mask = (field.land_mask | (conc < thr)) if field.land_mask is not None else (conc < thr)
        im.set_data(np.ma.masked_where(mask, conc))
        for a, ln, dt in zip(agents, lines, dots):
            j = min(k, len(a['traj']) - 1); tr = a['traj']
            ln.set_data(tr[:j + 1, 0], tr[:j + 1, 1]); dt.set_offsets([[tr[j, 0], tr[j, 1]]])
        ttl.set_text(f"{title}\nstep {k}")
        return [im, ttl] + lines + dots

    ani = FuncAnimation(fig, _upd, frames=nfr, blit=False, interval=1000 // fps)
    out_path = out_path.with_suffix('.mp4')
    ani.save(str(out_path), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=110)
    plt.close(fig)
    print("  ->", out_path.relative_to(_ctx()['root'] / "videos"), flush=True)


def _oc(r):
    return "success" if r.success else "fail"


def _agent(traj, kind, r):
    lab, col = _MLAB[kind]
    return dict(traj=np.asarray(traj), color=col, label=f"{lab} ({_oc(r)}, {r.steps} steps)", res=r)


def make_group_video(src, ver, ch, seed=None, sub=None):
    """Un video di gruppo (4 agenti) per lo scenario dato."""
    c = _ctx(); sub = sub or ver.lower()
    field = _cmp_load_field(src, ver)
    if field is None:
        print(f"  [SKIP] {src} {ver}: campo non trovato"); return
    sp = np.array(c['dm'].get_source_coordinates(src), float)
    if seed is None:
        seed = _pick_seed(field, ch) or 0
    start = _cmp_natural_start(field, ch, seed)
    res = {k: _cmp_rollout(k, ch, field, start, seed=seed) for k in ('fcm', 'n', 's', 'd')}
    agents = [_agent(res[k].trajectory, k, res[k]) for k in ('fcm', 'n', 's', 'd')]
    title = f"$v_{{max}}$=1.2 m/s  {src} {ver} {_CHUNK_LBL[ch].replace('-','/')}  (group)"
    d = c['root'] / "videos" / sub; d.mkdir(parents=True, exist_ok=True)
    _animate_multi(field, sp, start, agents, title, d / f"group_{src}_{ver}_{_CHUNK_LBL[ch]}")


def make_single_video(src, ver, ch, method, seed=None, sub=None):
    """Un video singolo modello (`method` in {'fcm','n','s','d'})."""
    c = _ctx(); sub = sub or ver.lower()
    field = _cmp_load_field(src, ver)
    if field is None:
        print(f"  [SKIP] {src} {ver}: campo non trovato"); return
    sp = np.array(c['dm'].get_source_coordinates(src), float)
    if seed is None:
        seed = _pick_seed(field, ch) or 0
    start = _cmp_natural_start(field, ch, seed)
    r = _cmp_rollout(method, ch, field, start, seed=seed)
    lab = _MLAB[method][0]
    title = f"$v_{{max}}$=1.2 m/s  {src} {ver} {_CHUNK_LBL[ch].replace('-','/')}  ({lab})"
    d = c['root'] / "videos" / sub; d.mkdir(parents=True, exist_ok=True)
    _animate_multi(field, sp, start, [_agent(r.trajectory, method, r)], title,
                   d / f"single_{method}_{src}_{ver}_{_CHUNK_LBL[ch]}")


def _pick_seed(field, ch, dmin=450.0):
    """Primo seed che dà uno start non banale (distanza sorgente >= dmin)."""
    sp = np.array(field.source_position, float)
    for seed in [0, 1, 7, 42, 3, 5, 9, 11, 13]:
        st = _cmp_natural_start(field, ch, seed)
        if np.hypot(st[0] - sp[0], st[1] - sp[1]) >= dmin:
            return seed
    return None


def batch_videos(cat, n_group=4, n_single=4, keep_existing=True):
    """Genera n_group video di gruppo + n_single singoli per la categoria (v0|v1|v2).
    Sorgenti: le suggerite + altre casuali (mix). start non banale (d0>=450m)."""
    import random
    ver = {'v0': 'V0', 'v1': 'V1', 'v2': 'V2'}[cat]
    random.seed({'v0': 11, 'v1': 22, 'v2': 33}[cat])
    all_srcs = [f"SRC{i}" for i in range(107, 133)]
    others = [s for s in all_srcs if s not in _SUGGESTED[cat]]
    random.shuffle(others)
    pool = _SUGGESTED[cat] + others
    c = _ctx(); vdir = c['root'] / "videos" / cat
    used = set()
    if keep_existing and vdir.exists():
        for p in vdir.glob("group_*.mp4"):        # non rigenerare gruppi gia' presenti
            used.add(p.stem.split('_')[1])
    print(f"\n##### BATCH {cat} ({ver}) — {n_group} gruppo + {n_single} singoli #####", flush=True)
    gi = 0
    for src in pool:
        if gi >= n_group:
            break
        if src in used:
            continue
        ch = random.choice([1, 2]); field = _cmp_load_field(src, ver)
        if field is None:
            continue
        seed = _pick_seed(field, ch)
        if seed is None:
            continue
        print(f"[group {gi+1}/{n_group}] {src} {ver} Q{ch} seed{seed}", flush=True)
        make_group_video(src, ver, ch, seed, cat); used.add(src); gi += 1
    methods = ['fcm', 'n', 's', 'd']; si = 0
    for src in pool + all_srcs:
        if si >= n_single:
            break
        if src in used:
            continue
        ch = random.choice([1, 2]); field = _cmp_load_field(src, ver)
        if field is None:
            continue
        seed = _pick_seed(field, ch)
        if seed is None:
            continue
        m = methods[si % 4]
        print(f"[single {si+1}/{n_single}] {src} {ver} Q{ch} seed{seed} [{m}]", flush=True)
        make_single_video(src, ver, ch, m, seed, cat); used.add(src); si += 1


def _cli():
    import sys
    _root()
    if len(sys.argv) < 2:
        print(__doc__); return
    mode = sys.argv[1]
    if mode == "batch":
        cats = ['v0', 'v1', 'v2'] if (len(sys.argv) < 3 or sys.argv[2] == "all") else [sys.argv[2]]
        for cat in cats:
            batch_videos(cat)
    elif mode == "group":
        src, ver, ch = sys.argv[2], sys.argv[3], int(sys.argv[4])
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
        make_group_video(src, ver, ch, seed)
    elif mode == "single":
        src, ver, ch = sys.argv[2], sys.argv[3], int(sys.argv[4])
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
        for m in ('fcm', 'n', 's', 'd'):
            make_single_video(src, ver, ch, m, seed)
    else:
        print(f"modo sconosciuto: {mode}\n{__doc__}")
    print("done", flush=True)


if __name__ == "__main__":
    _cli()
