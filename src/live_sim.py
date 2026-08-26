#!/usr/bin/env python3
"""
HYDRAS Source Seeking — LIVE simulation with a control panel.

Small GUI (Tkinter): the user picks the scenario and the agent from drop-down
menus, presses "Start" and watches ONE episode play out in real time in the same
window. When the episode ends the program resets and stays ready for a new run.

Drop-down menus:
  - Technology  : PPO  or  FCM Adam
  - Wind        : V0 / V1 / V2 / V3
  - Time Chunk  : Q1/4 / Q1/2 / Q3/4
  - Max Speed   : 0.1..5              (PPO only)
  - Formation   : Single / Double ring   (PPO only)

Once Wind and Time Chunk are set, a random source is picked for that scenario. FCM
uses Adam with a selectable step (lr, in metres; default 40) and sensor_range=50 m.

The code has two parts:
  - the ENGINE (make_agent_env / step_once / draw_scene): builds agent+env, steps
    once and draws. No Tkinter dependency, testable headless.
  - the GUI (App): the widgets and the non-blocking loop via root.after().

Usage:
    python src/live_sim.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# Gli import "pesanti" (numpy, matplotlib, inference, utils) NON stanno in cima di
# proposito: questo stesso file fa da bootstrapper su una macchina fresca, dove
# quei moduli ancora non esistono (vengono installati/scaricati dalle fasi 1-2 di
# main()). Sono caricati in modo lazy da _import_heavy() ed esposti come globali.
# 'from __future__ import annotations' evita che le annotazioni (np.ndarray,
# DataManager, …) vengano valutate all'import, quando quei nomi non esistono ancora.

_HEAVY_LOADED = False


def _import_heavy() -> None:
    """Carica numpy, matplotlib e il codice del progetto (inference, utils) e li
    espone come globali del modulo. Idempotente. Da chiamare DOPO che requisiti e
    script sono stati installati/scaricati (fasi 1-2 di main())."""
    global _HEAVY_LOADED
    if _HEAVY_LOADED:
        return
    here = Path(__file__).resolve()
    for p in (str(here.parent), str(here.parent.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.invalidate_caches()
    import numpy as _np
    from matplotlib.colors import ListedColormap as _LCM
    from matplotlib.figure import Figure as _Figure
    from matplotlib.patches import Circle as _Circle
    from inference import (
        MASKABLE_PPO_AVAILABLE as _MPA, AdamFCMAgent as _AFA,
        _find_dualcorona_run as _fdr, _find_velocity_run as _fvr,
        build_env as _be, build_env_fcm as _bef, get_inner_env as _gie,
        load_config as _lc, load_model as _lm, make_env_config as _mec)
    from utils.data_loader import DataManager as _DM
    globals().update(
        np=_np, ListedColormap=_LCM, Figure=_Figure, Circle=_Circle,
        MASKABLE_PPO_AVAILABLE=_MPA, AdamFCMAgent=_AFA,
        _find_dualcorona_run=_fdr, _find_velocity_run=_fvr,
        build_env=_be, build_env_fcm=_bef, get_inner_env=_gie,
        load_config=_lc, load_model=_lm, make_env_config=_mec, DataManager=_DM)
    _HEAVY_LOADED = True


# ─── Costanti ────────────────────────────────────────────────────────────────
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
CHUNK_BY_LABEL = {"Q1/4": 0, "Q1/2": 1, "Q3/4": 2}
VERSIONS = ["V0", "V1", "V2", "V3"]
FCM_LR = 40.0            # miglior configurazione dello sweep (Cap. 3)
FCM_SENSOR_RANGE = 50.0

# Modelli confrontabili in gruppo: (key, etichetta GUI, colore traiettoria).
# I colori coincidono con quelli dei plot/video del progetto.
MODEL_DEFS = [
    ("fcm",    "FCM",             "#7f7f7f"),
    ("noform", "PPO no ring",     "#2ca02c"),
    ("single", "PPO single ring", "#1f5fb4"),
    ("double", "PPO double ring", "#e07b28"),
]
MODEL_COLOR = {k: c for k, _, c in MODEL_DEFS}
MODEL_LABEL = {k: lab for k, lab, _ in MODEL_DEFS}
DEFAULT_MODELS = ("fcm", "single", "double")   # selezione iniziale (no-ring escluso: solo v_max<=2)

# Default dei menu a tendina (ripristinati a fine episodio).
DEFAULTS = {"version": "V0", "chunk": "Q1/4", "vmax": "1.2"}


# ─── Requisiti / dati / modelli / script: check e download ───────────────────
REPO_SLUG = "MattiaManneschi/HYDRAS-SourceSeeker"
BRANCH = "master"
TARBALL_URL = f"https://codeload.github.com/{REPO_SLUG}/tar.gz/refs/heads/{BRANCH}"
MODELS_TARBALL_URL = TARBALL_URL   # stesso tarball: contiene sia script sia modelli
REQUIREMENTS_URL = f"https://raw.githubusercontent.com/{REPO_SLUG}/{BRANCH}/requirements.txt"
# Radice di installazione: la cartella del progetto (parent di src/), o override.
INSTALL_DIR = Path(os.environ.get("HYDRAS_HOME", Path(__file__).resolve().parent.parent))
# Dati .nc (subset live, ~7,4 GB): su una GitHub Release (niente Google Drive, che
# blocca i download pubblici in massa con la sua quota). Gli asset 'hydras_data.part-*'
# della release, concatenati in ordine, formano un tar del subset. Vedi
# make_data_release.sh per crearli. Download HTTPS diretto, senza gdown.
DATA_RELEASE_TAG = "data-v1"
DATA_ASSET_PREFIX = "hydras_data.part-"

REQUIRED_PACKAGES = ["torch", "stable_baselines3", "sb3_contrib", "gymnasium",
                     "numpy", "scipy", "matplotlib", "netCDF4", "yaml"]


def missing_packages() -> list:
    """Pacchetti richiesti non importabili nell'ambiente corrente."""
    import importlib
    out = []
    for m in REQUIRED_PACKAGES:
        try:
            importlib.import_module(m)
        except Exception:
            out.append(m)
    return out


def install_requirements(root_dir: Path, progress_cb=None) -> None:
    """pip install -r requirements.txt nell'interprete corrente.

    Se `progress_cb` è dato, riceve una stringa di stato ("Downloading torch…",
    "Building X…", "Installing packages…") per aggiornare la didascalia; le righe di
    pip sono comunque ri-emesse su stdout, così restano visibili nel terminale."""
    import subprocess, re
    cmd = [sys.executable, "-m", "pip", "install", "-r",
           str(root_dir / "requirements.txt")]
    if progress_cb is None:
        subprocess.run(cmd, check=True)
        return
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)                        # tee: resta visibile nel terminale
        s = line.strip()
        status = None
        if s.startswith("Collecting "):
            name = re.split(r"[<>=!~;\[\s]", s[11:].strip(), maxsplit=1)[0]
            status = f"Downloading {name}…"
        elif s.startswith("Downloading "):
            tok = s[12:].strip().split()[0]           # nome file del wheel/sdist
            m = re.match(r"(.+?)-\d", tok)             # 'torch-2.5.0-...' -> 'torch'
            status = f"Downloading {m.group(1) if m else tok}…"
        elif s.startswith("Building wheel for "):
            status = f"Building {s[19:].split()[0]}…"
        elif s.startswith("Installing collected packages"):
            status = "Installing packages…"           # fase finale: niente più download
        if status:
            progress_cb(status)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def data_present(root_dir: Path) -> bool:
    """Verifica che i file .nc necessari (concentrazione + 4 corrente + 4 vento)
    siano presenti in data/."""
    d = root_dir / "data"
    conc = list(d.glob("**/*Conc_10mGrid.nc")) + list(d.glob("*Conc_10mGrid.nc"))
    cur = list(d.glob("**/*SRC000_U_V_10mGrid.nc")) + list(d.glob("*SRC000_U_V_10mGrid.nc"))
    wind = list(d.glob("**/CI_WIND_faseII_V*.txt")) + list(d.glob("CI_WIND_faseII_V*.txt"))
    return len(conc) > 0 and len(cur) >= 4 and len(wind) >= 4


# v_max coperti dai modelli PPO: singola e doppia corona coprono ora l'intero
# range (basse 0.1/0.4/0.7, intermedie 1.2–1.9 e interi 1–5).
PPO_VMAX = (0.1, 0.4, 0.7, 1, 1.2, 1.5, 1.7, 1.9, 2, 3, 4, 5)


def missing_models(root_dir: Path) -> list:
    """Combinazioni (formazione, v_max) per cui manca il modello PPO."""
    trained = root_dir / "trained_models"
    miss = []
    for vmax in PPO_VMAX:
        s = _find_velocity_run(trained, float(vmax), 5)
        if s is None or not (s / "models" / "final_model.zip").exists():
            miss.append(f"singola v{vmax:g}")
        d = _find_dualcorona_run(trained, float(vmax), 5)
        if d is None or not (d / "models" / "final_model.zip").exists():
            miss.append(f"doppia v{vmax:g}")
    return miss


def _ssl_context():
    """Contesto SSL con il bundle CA di certifi: il Python di python.org su macOS
    non usa il cert store di sistema e fallirebbe la verifica (CERTIFICATE_VERIFY_
    FAILED). Fallback al contesto di default se certifi non è disponibile."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_file(url: str, dest: Path, progress_cb=None) -> None:
    """Scarica url in dest in streaming; progress_cb(frazione 0..1) se la
    dimensione è nota. Usa un contesto SSL con certifi."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "HYDRAS-live-sim"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if progress_cb and total > 0:
                progress_cb(min(1.0, read / total))


def extract_models_tarball(tar_path: Path, root_dir: Path) -> None:
    """Estrae solo trained_models/** dal tarball del repo (entry del tipo
    'HYDRAS-SourceSeeker-master/trained_models/…') in root_dir/trained_models."""
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("trained_models/"):
                m.name = parts[1]                 # rimuove il prefisso top-level
                tf.extract(m, path=str(root_dir))


def scripts_present(root_dir: Path) -> bool:
    """Dipendenze-codice necessarie a questo file (che le importa in modo lazy):
    inference.py e il pacchetto utils. Questo stesso file NON è nel check (esiste
    già: è quello in esecuzione)."""
    return (root_dir / "src" / "inference.py").exists() and \
           (root_dir / "utils" / "data_loader.py").exists()


def _extract_prefixes(tar_path: Path, root_dir: Path, prefixes: tuple) -> int:
    """Estrae dal tarball del repo le voci il cui path (tolto il prefisso
    top-level 'HYDRAS-SourceSeeker-<branch>/') inizia con uno dei prefissi dati."""
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if any(rel.startswith(p) for p in prefixes):
                m.name = rel
                tf.extract(m, path=str(root_dir))
                n += 1
    return n


def _get_tarball(cache: dict, cb=None) -> Path:
    """Scarica il tarball del repo una sola volta (riusato da script e modelli)."""
    if cache.get("path") is None:
        tmp = Path(tempfile.mkdtemp()) / "hydras.tar.gz"
        download_file(TARBALL_URL, tmp, cb)
        cache["path"] = tmp
    return cache["path"]


def _download_stream(url: str, fileobj, base: int = 0, total: int = 0,
                     progress_cb=None) -> None:
    """Scarica url scrivendo i byte in `fileobj` (aperto in append), riportando il
    progresso in byte CUMULATIVI: progress_cb(base + letti_finora, total)."""
    req = urllib.request.Request(url, headers={"User-Agent": "HYDRAS-live-sim"})
    with urllib.request.urlopen(req, context=_ssl_context()) as resp:
        read = 0
        while True:
            b = resp.read(1 << 16)
            if not b:
                break
            fileobj.write(b)
            read += len(b)
            if progress_cb and total > 0:
                progress_cb(base + read, total)


def download_data_from_release(root_dir: Path, tag: str = DATA_RELEASE_TAG,
                               progress_cb=None) -> None:
    """Scarica i dati (subset live) da una GitHub Release ed estraeli in data/.

    La release `tag` contiene gli asset 'hydras_data.part-*' che, concatenati in
    ordine alfabetico, formano un tar del subset (creato da make_data_release.sh).
    Download HTTPS diretto: niente gdown, niente quota di Google Drive.
    progress_cb(byte_scaricati, byte_totali) consente una barra in GB.
    """
    api = f"https://api.github.com/repos/{REPO_SLUG}/releases/tags/{tag}"
    req = urllib.request.Request(api, headers={
        "User-Agent": "HYDRAS-live-sim", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, context=_ssl_context()) as r:
        rel = json.load(r)
    assets = sorted((a for a in rel.get("assets", [])
                     if a.get("name", "").startswith(DATA_ASSET_PREFIX)),
                    key=lambda a: a["name"])
    if not assets:
        raise RuntimeError(f"No '{DATA_ASSET_PREFIX}*' asset in release "
                           f"'{tag}' of {REPO_SLUG}.")
    total = sum(int(a.get("size", 0)) for a in assets)

    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp())
    try:
        tar_path = tmpdir / "hydras_data.tar"
        done = 0
        with open(tar_path, "wb") as tar_out:      # ricompone i pezzi in un unico tar
            for a in assets:
                _download_stream(a["browser_download_url"], tar_out,
                                 base=done, total=total, progress_cb=progress_cb)
                done += int(a.get("size", 0))
        with tarfile.open(tar_path) as tf:
            # Salta i companion di macOS (._*, .DS_Store) che il tar di macOS può
            # includere: DataManager li scambierebbe per dati (._*.nc) e netCDF
            # fallirebbe con "Unknown file format".
            members = [m for m in tf.getmembers() if not _is_apple_junk(m.name)]
            tf.extractall(data_dir, members=members)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _is_apple_junk(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("._") or name == ".DS_Store"


def _remove_apple_junk(data_dir: Path) -> None:
    """Ripulisce i file companion di macOS già finiti in data/ (self-heal per
    installazioni fatte prima di questo fix)."""
    if not data_dir.exists():
        return
    for pat in ("._*", ".DS_Store"):
        for p in data_dir.rglob(pat):
            try:
                p.unlink()
            except OSError:
                pass


def available_sources(dm: DataManager) -> list:
    """Sorgenti disponibili per la demo (SRC107-SRC132)."""
    return sorted(s for s in dm.get_discovered_sources() if int(s[3:]) > 106)


def pick_source(dm: DataManager, version: str, rng: np.random.Generator) -> Optional[str]:
    """Sorgente a caso che possiede un file per la versione data."""
    sources = available_sources(dm)
    for s in rng.permutation(sources):
        if any(version in f.name and s in f.name for f in dm._nc_files):
            return str(s)
    return None


def load_field(dm: DataManager, source_id: str, version: str):
    files = [f for f in dm._nc_files if version in f.name and source_id in f.name]
    if not files:
        return None
    field = dm._nc_loader.load(str(files[0]),
                               concentration_var="Concentration - component 1")
    if field is None:
        return None
    coords = dm.get_source_coordinates(source_id)
    if coords:
        field.source_position = coords
    field.run_id = f"{source_id}_{version}"
    return field


# ─── Motore (indipendente dalla GUI, testabile headless) ─────────────────────

def make_agent_env(dm: DataManager, root_dir: Path, tech: str, version: str,
                   chunk: int, vmax: float, formation: str, source: str,
                   fcm_lr: float = FCM_LR) -> Tuple:
    """Costruisce (agente, vec_env, is_fcm, label) per la scelta dell'utente.

    - PPO: trova il run (singola/doppia corona, v_max), carica modello +
      VecNormalize e usa la config del modello (obs coerente col modello).
    - FCM: AdamFCMAgent(lr=40, sensor_range=50), env senza VecNormalize
      (obs grezze) dalla config base senza reward vento.
    """
    field = load_field(dm, source, version)
    if field is None:
        raise RuntimeError(f"Campo non caricabile per {source} {version}.")

    if tech == "FCM":
        config = load_config(str(root_dir / "utils" / "config" / "config_base_no_wind_reward.yaml"))
        env_cfg = make_env_config(config, chunk_id=chunk)
        vec_env = build_env_fcm(env_cfg, field, use_masking=MASKABLE_PPO_AVAILABLE,
                                data_manager=dm, wind_mapping=WIND_MAPPING,
                                current_mapping=CURRENT_MAPPING)
        agent = AdamFCMAgent(sensor_range=FCM_SENSOR_RANGE, lr=fcm_lr)
        agent.reset()
        label = f"FCM Adam (lr={int(fcm_lr)} m)"
        return agent, vec_env, True, label

    # PPO ---------------------------------------------------------------------
    trained = root_dir / "trained_models"
    if formation == "Double":
        run_dir = _find_dualcorona_run(trained, vmax=float(vmax), K=5)
        form_lbl = "double ring"
    else:
        run_dir = _find_velocity_run(trained, vmax=float(vmax), K=5)
        form_lbl = "single ring"
    if run_dir is None:
        raise RuntimeError(f"No PPO {form_lbl} model for max speed {vmax}.")

    model_path = run_dir / "models" / "final_model.zip"
    vec_norm_path = run_dir / "models" / "vec_normalize.pkl"
    if not model_path.exists():
        raise RuntimeError(f"final_model.zip missing in {run_dir/'models'}.")

    config = load_config(str(run_dir / "config.yaml"))
    env_cfg = make_env_config(config, chunk_id=chunk)
    vec_env = build_env(env_cfg, field, vec_norm_path,
                        use_masking=MASKABLE_PPO_AVAILABLE, data_manager=dm,
                        wind_data=None, current_data=None,
                        wind_mapping=WIND_MAPPING, current_mapping=CURRENT_MAPPING)
    model = load_model(str(model_path))
    label = f"PPO {form_lbl}, v_max={vmax:g}"
    return model, vec_env, False, label


def _find_noform_run(trained, vmax, K: int = 5):
    """Run del modello PPO 'no ring' (central-agent-only, mask_formation) per il v_max.
    Criterio: K livelli, max_velocity==vmax, corona singola (no sensor_range_2),
    mask_formation=True. Ritorna un Path o None (i no-ring esistono solo per v_max<=2)."""
    best = None
    for d in sorted(Path(trained).glob("ppo_*")):
        cfg_p = d / "config.yaml"
        if not cfg_p.exists() or not (d / "models" / "final_model.zip").exists():
            continue
        try:
            ag = load_config(str(cfg_p)).get("agent", {})
        except Exception:
            continue
        if (int(ag.get("n_velocity_levels", 1)) == K
                and abs(float(ag.get("max_velocity", 1.0)) - float(vmax)) < 1e-6
                and ag.get("sensor_range_2") is None
                and bool(ag.get("mask_formation", False))):
            best = d
    return best


def _build_one(kind: str, dm, root_dir: Path, version: str, chunk: int, vmax: float, field):
    """Costruisce (agent, vec_env, is_fcm, label) per UN modello su un campo dato.
    Ritorna None se il modello non è disponibile per il v_max scelto (es. no-ring a v_max>2)."""
    if kind == "fcm":
        config = load_config(str(root_dir / "utils" / "config" / "config_base_no_wind_reward.yaml"))
        env_cfg = make_env_config(config, chunk_id=chunk)
        env_cfg.sensor_range = FCM_SENSOR_RANGE
        vec_env = build_env_fcm(env_cfg, field, use_masking=MASKABLE_PPO_AVAILABLE,
                                data_manager=dm, wind_mapping=WIND_MAPPING,
                                current_mapping=CURRENT_MAPPING)
        lr = max(1.0, round(float(vmax) * 10.0))          # passo FCM (m) ≈ v_max*10
        agent = AdamFCMAgent(sensor_range=FCM_SENSOR_RANGE, lr=lr)
        agent.reset()
        return agent, vec_env, True, f"FCM (lr={int(lr)} m)"

    trained = root_dir / "trained_models"
    if kind == "single":
        run_dir = _find_velocity_run(trained, vmax=float(vmax), K=5)
    elif kind == "double":
        run_dir = _find_dualcorona_run(trained, vmax=float(vmax), K=5)
    elif kind == "noform":
        run_dir = _find_noform_run(trained, vmax=float(vmax), K=5)
    else:
        return None
    if run_dir is None:
        return None
    model_path = run_dir / "models" / "final_model.zip"
    vn = run_dir / "models" / "vec_normalize.pkl"
    if not model_path.exists():
        return None
    config = load_config(str(run_dir / "config.yaml"))
    env_cfg = make_env_config(config, chunk_id=chunk)
    vec_env = build_env(env_cfg, field, vn, use_masking=MASKABLE_PPO_AVAILABLE,
                        data_manager=dm, wind_data=None, current_data=None,
                        wind_mapping=WIND_MAPPING, current_mapping=CURRENT_MAPPING)
    model = load_model(str(model_path))
    return model, vec_env, False, MODEL_LABEL[kind]


def make_agents(dm, root_dir: Path, kinds, version: str, chunk: int, vmax: float, source: str):
    """Costruisce gli agenti selezionati sullo STESSO scenario, con START COMUNE.
    Il primo modello disponibile fa lo spawn naturale; gli altri sono forzati sullo
    stesso punto (monkeypatch di _spawn_on_plume). Ritorna (agents, unavailable)."""
    import copy as _copy
    base_field = load_field(dm, source, version)
    if base_field is None:
        raise RuntimeError(f"Campo non caricabile per {source} {version}.")
    agents, unavailable, common_start = [], [], None
    for kind in kinds:
        built = _build_one(kind, dm, root_dir, version, chunk, vmax, _copy.deepcopy(base_field))
        if built is None:
            unavailable.append(kind)
            continue
        agent, vec_env, is_fcm, label = built
        inner = get_inner_env(vec_env)
        if common_start is None:
            obs = vec_env.reset()                          # spawn naturale = start comune
            common_start = (float(inner.state.x), float(inner.state.y))
        else:
            inner._spawn_on_plume = lambda s=common_start: (s[0], s[1])
            obs = vec_env.reset()
        agents.append(dict(kind=kind, agent=agent, vec_env=vec_env, is_fcm=is_fcm,
                           color=MODEL_COLOR[kind], label=label, obs=obs,
                           done=False, steps=0, term=None,
                           traj=[(float(inner.state.x), float(inner.state.y))]))
    return agents, unavailable


def step_once(agent, vec_env, obs, is_fcm) -> Tuple[np.ndarray, bool, dict]:
    """Un passo: predice, (FCM: aggiorna la velocità adattiva), avanza.

    Ritorna (nuova_obs, done, info). Replica la meccanica di run_episode /
    run_episode_fcm di inference.py.
    """
    inner = get_inner_env(vec_env)
    masks = vec_env.env_method("action_masks")[0] if MASKABLE_PPO_AVAILABLE else None
    if masks is not None:
        action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
    else:
        action, _ = agent.predict(obs, deterministic=True)

    if is_fcm:
        # Passo adattivo Adam: la velocità del passo corrente esce dall'ottimizzatore.
        inner.config.max_velocity = agent._last_step / inner.config.dt

    obs, _, dones, infos = vec_env.step(action)
    return obs, bool(dones[0]), infos[0]


def resolve_termination(info: dict) -> str:
    t = info.get("termination_reason")
    if t in {"success", "boundary", "land", "timeout"}:
        return t
    if info.get("source_reached", False):
        return "success"
    if info.get("on_land", False):
        return "land"
    if info.get("out_of_bounds", False):
        return "boundary"
    return "timeout"


# ─── Tema scuro ──────────────────────────────────────────────────────────────
DARK_BG     = "#1b1f2a"   # sfondo finestra / pannelli / figura
DARK_PANEL  = "#232a3a"   # bottoni / bordi (leggermente più chiari)
DARK_FIELD  = "#2a3244"   # sfondo dei menu a tendina
DARK_FG     = "#e6e9ef"   # testo principale
DARK_MUTED  = "#95a0b3"   # testo/assi secondari
DARK_ACCENT = "#3b82f6"   # blu accento (bottoni attivi, barra di caricamento)
UI_FONT    = ("", 13)     # font base dei controlli (menu in alto)
UI_FONT_SM = ("", 11)     # etichette piccole sopra i menu


def _apply_dark_theme(root) -> None:
    """Tema scuro coerente per i widget ttk. Usa il tema 'clam' perché, a
    differenza di 'aqua'/'vista', è completamente ricolorabile."""
    from tkinter import ttk
    root.configure(bg=DARK_BG)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=DARK_BG, foreground=DARK_FG,
                    fieldbackground=DARK_FIELD, bordercolor=DARK_PANEL,
                    lightcolor=DARK_PANEL, darkcolor=DARK_BG, font=UI_FONT)
    style.configure("TFrame", background=DARK_BG)
    style.configure("TLabel", background=DARK_BG, foreground=DARK_FG, font=UI_FONT)
    style.configure("TCombobox", fieldbackground=DARK_FIELD, background=DARK_FIELD,
                    foreground=DARK_FG, arrowcolor=DARK_MUTED, bordercolor=DARK_PANEL,
                    padding=(11, 9), relief="flat", font=UI_FONT,
                    selectbackground=DARK_FIELD, selectforeground=DARK_FG)
    style.map("TCombobox",
              fieldbackground=[("readonly", DARK_FIELD), ("focus", DARK_FIELD),
                               ("active", DARK_FIELD), ("disabled", DARK_BG)],
              background=[("readonly", DARK_FIELD), ("active", DARK_FIELD),
                          ("pressed", DARK_FIELD), ("disabled", DARK_BG)],
              foreground=[("readonly", DARK_FG), ("disabled", DARK_MUTED)],
              arrowcolor=[("disabled", DARK_PANEL), ("pressed", DARK_MUTED),
                          ("active", DARK_MUTED), ("focus", DARK_MUTED),
                          ("readonly", DARK_MUTED)],
              selectbackground=[("readonly", DARK_FIELD)],
              selectforeground=[("readonly", DARK_FG)])
    style.configure("Horizontal.TProgressbar", background=DARK_ACCENT,
                    troughcolor=DARK_FIELD, bordercolor=DARK_PANEL)

    # ── Stili "minimal" per la barra configurazioni ──────────────────────────
    style.configure("Card.TFrame", background=DARK_PANEL)
    style.configure("Field.TLabel", background=DARK_PANEL, foreground=DARK_MUTED,
                    font=UI_FONT_SM)
    style.configure("Muted.TLabel", background=DARK_BG, foreground=DARK_MUTED,
                    font=UI_FONT)
    style.configure("TButton", background=DARK_PANEL, foreground=DARK_FG,
                    padding=(20, 11), relief="flat", borderwidth=0, font=UI_FONT)
    style.map("TButton",
              background=[("active", DARK_FIELD), ("pressed", DARK_FIELD),
                          ("disabled", DARK_BG)],
              foreground=[("disabled", DARK_MUTED)])
    style.configure("Accent.TButton", background=DARK_ACCENT, foreground="#ffffff",
                    padding=(24, 11), relief="flat", borderwidth=0,
                    focuscolor=DARK_ACCENT, font=UI_FONT)
    style.map("Accent.TButton",
              background=[("active", "#2f6fd0"), ("pressed", "#2a63bb"),
                          ("disabled", DARK_FIELD)],
              foreground=[("disabled", DARK_MUTED)])
    # La lista a discesa del Combobox è un widget Tk classico (non ttk).
    root.option_add("*TCombobox*Listbox.background", DARK_FIELD)
    root.option_add("*TCombobox*Listbox.foreground", DARK_FG)
    root.option_add("*TCombobox*Listbox.selectBackground", DARK_ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", UI_FONT)


def draw_scene(ax, inner, title: str) -> None:
    """Disegna la scena corrente nell'axes dato, con lo stesso stile dei plot
    standard del progetto (mare azzurro, terra bianca, plume YlOrRd) — cfr.
    plot_trajectory in inference.py. Nessuna dipendenza da pyplot."""
    ax.clear()
    field = inner.field
    conc = field.get_current_field()
    xc, yc = field.x_coords, field.y_coords

    # Extent con correzione di mezzo pixel (i coords sono centri cella).
    dx = float(xc[1] - xc[0]) if len(xc) > 1 else 10.0
    dy = float(yc[1] - yc[0]) if len(yc) > 1 else 10.0
    extent = [float(xc[0]) - dx / 2, float(xc[-1]) + dx / 2,
              float(yc[0]) - dy / 2, float(yc[-1]) + dy / 2]

    # 1) Mare (sfondo azzurro uniforme); 2) terra bianca via land_mask.
    ax.set_facecolor("#87CEEB")
    land_mask = getattr(field, "land_mask", None)
    if land_mask is not None:
        land = np.ma.masked_where(~land_mask, np.ones_like(conc))
        ax.imshow(land, origin="lower", extent=extent,
                  cmap=ListedColormap(["#FFFFFF"]), alpha=1.0, zorder=1)
        plume_mask = land_mask | (conc < 0.01)
    else:
        plume_mask = conc < 0.01

    # 3) Plume di concentrazione (mascherato su terra e dove conc ~ 0).
    conc_masked = np.ma.masked_where(plume_mask, conc)
    ax.imshow(conc_masked, origin="lower", extent=extent, cmap="YlOrRd",
              alpha=0.9, vmin=0, vmax=max(float(conc.max()), 0.1), zorder=2)

    # Traiettoria dell'agente.
    if len(inner.trajectory) > 1:
        traj = np.array(inner.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="#1f4fd6", linewidth=1.8,
                alpha=0.9, label="Trajectory", zorder=4)

    # Agente + freccia di direzione.
    ax.scatter(inner.state.x, inner.state.y, c="#0b3d91", s=90, marker="o",
               edgecolors="white", linewidths=1.2, zorder=6, label="Agent")
    if inner.state.vx != 0 or inner.state.vy != 0:
        ax.arrow(inner.state.x, inner.state.y, inner.state.vx * 50, inner.state.vy * 50,
                 head_width=30, head_length=20, fc="#0b3d91", ec="#0b3d91", zorder=5)
        # Velocità del passo corrente (modulo di (vx, vy), in m/s) accanto al pallino.
        speed = float(np.hypot(inner.state.vx, inner.state.vy))
        ax.annotate(f"{speed:.2f} m/s", (inner.state.x, inner.state.y),
                    textcoords="offset points", xytext=(12, 10),
                    fontsize=9, fontweight="bold", color="#0b3d91",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#0b3d91",
                              alpha=0.8), zorder=8)

    # Sorgente (stella gialla) e raggio di successo.
    sx, sy = inner.source_position
    ax.scatter(sx, sy, c="yellow", s=220, marker="*", edgecolors="black",
               zorder=7, label="Source")
    ax.add_patch(Circle((sx, sy), inner.config.source_distance_threshold,
                        fill=False, color="red", linestyle="--", zorder=7))

    ax.set_xlabel("X (m)", color=DARK_FG)
    ax.set_ylabel("Y (m)", color=DARK_FG)
    ax.set_title(title, color=DARK_FG)
    ax.tick_params(colors=DARK_MUTED)
    for _sp in ax.spines.values():
        _sp.set_color(DARK_MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(float(xc[0]), float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.legend(loc="upper right", fontsize=8, facecolor=DARK_PANEL,
              edgecolor=DARK_MUTED, labelcolor=DARK_FG)


def draw_scene_multi(ax, agents: list, title: str) -> None:
    """Come draw_scene ma con PIÙ agenti sovrapposti: campo dal frame più avanzato
    (fra gli agenti ancora in corsa), poi ogni agente con la sua scia+pallino colorati
    (una X quando ha finito). Start comune e sorgente. Stesso stile canonico."""
    ax.clear()
    running = [a for a in agents if not a["done"]]
    ref = max(running or agents, key=lambda a: a["steps"])
    rinner = get_inner_env(ref["vec_env"])
    field = rinner.field
    conc = field.get_current_field()
    xc, yc = field.x_coords, field.y_coords
    dx = float(xc[1] - xc[0]) if len(xc) > 1 else 10.0
    dy = float(yc[1] - yc[0]) if len(yc) > 1 else 10.0
    extent = [float(xc[0]) - dx / 2, float(xc[-1]) + dx / 2,
              float(yc[0]) - dy / 2, float(yc[-1]) + dy / 2]

    ax.set_facecolor("#87CEEB")
    land_mask = getattr(field, "land_mask", None)
    if land_mask is not None:
        land = np.ma.masked_where(~land_mask, np.ones_like(conc))
        ax.imshow(land, origin="lower", extent=extent,
                  cmap=ListedColormap(["#FFFFFF"]), alpha=1.0, zorder=1)
        plume_mask = land_mask | (conc < 0.01)
    else:
        plume_mask = conc < 0.01
    ax.imshow(np.ma.masked_where(plume_mask, conc), origin="lower", extent=extent,
              cmap="YlOrRd", alpha=0.9, vmin=0, vmax=max(float(conc.max()), 0.1), zorder=2)

    start = agents[0]["traj"][0]
    ax.scatter(start[0], start[1], c="white", s=80, marker="o", edgecolors="black",
               linewidths=1.2, zorder=6, label="Start")
    sx, sy = rinner.source_position
    ax.scatter(sx, sy, c="yellow", s=220, marker="*", edgecolors="black", zorder=8, label="Source")
    ax.add_patch(Circle((sx, sy), rinner.config.source_distance_threshold,
                        fill=False, color="red", linestyle="--", zorder=7))

    dt = float(getattr(rinner.config, "dt", 10.0))
    for idx, a in enumerate(agents):
        tr = np.array(a["traj"])
        suffix = f"— {a['term'].upper()} ({a['steps']})" if (a["done"] and a["term"]) \
                 else f"— {a['steps']} steps"
        ax.plot(tr[:, 0], tr[:, 1], "-", color=a["color"], linewidth=1.8, alpha=0.9,
                zorder=4, label=f"{a['label']} {suffix}")
        ax.scatter(tr[-1, 0], tr[-1, 1], color=a["color"], s=90,
                   marker=("X" if a["done"] else "o"), edgecolors="white",
                   linewidths=1.0, zorder=6)
        # Freccia di direzione + velocità del passo corrente (solo agenti in corsa).
        if not a["done"] and len(tr) >= 2:
            dxp, dyp = float(tr[-1, 0] - tr[-2, 0]), float(tr[-1, 1] - tr[-2, 1])
            seg = (dxp * dxp + dyp * dyp) ** 0.5
            if seg > 1e-9:
                vx, vy = dxp / dt, dyp / dt          # m/s (coerente con inner.state.v)
                ax.arrow(tr[-1, 0], tr[-1, 1], vx * 50, vy * 50, head_width=30,
                         head_length=20, fc=a["color"], ec=a["color"], zorder=5,
                         length_includes_head=True)
                ax.annotate(f"{seg / dt:.2f} m/s", (tr[-1, 0], tr[-1, 1]),
                            textcoords="offset points", xytext=(11, 8 + (idx - 1.5) * 13),
                            fontsize=8, fontweight="bold", color=a["color"],
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec=a["color"], alpha=0.85), zorder=8)

    ax.set_xlabel("X (m)", color=DARK_FG)
    ax.set_ylabel("Y (m)", color=DARK_FG)
    ax.set_title(title, color=DARK_FG)
    ax.tick_params(colors=DARK_MUTED)
    for _sp in ax.spines.values():
        _sp.set_color(DARK_MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(float(xc[0]), float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.legend(loc="upper right", fontsize=8, facecolor=DARK_PANEL,
              edgecolor=DARK_MUTED, labelcolor=DARK_FG)


# ─── GUI ─────────────────────────────────────────────────────────────────────

def build_gui(root, dm, fps: float = 15.0) -> None:
    """Popola `root` con la GUI della simulazione usando il DataManager `dm` già
    caricato. NON crea la finestra e NON avvia il mainloop: lo fa il chiamante
    (main(), dopo le fasi di caricamento)."""
    _import_heavy()
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root_dir = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng()
    delay_ms = int(1000.0 / max(fps, 1.0))

    root.title("HYDRAS — Live Simulation")
    _apply_dark_theme(root)
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    # Barra-configurazioni come "card" (pannello più chiaro, staccato dai bordi).
    ctrl = ttk.Frame(root, style="Card.TFrame", padding=(26, 22))
    ctrl.grid(row=0, column=0, sticky="ew", padx=14, pady=(16, 10))
    ctrl.columnconfigure(5, weight=1)     # spacer: spinge i bottoni a destra

    ver_var = tk.StringVar(value=DEFAULTS["version"])
    chunk_var = tk.StringVar(value=DEFAULTS["chunk"])
    vmax_var = tk.StringVar(value=DEFAULTS["vmax"])
    # Un checkbox per modello: l'utente sceglie un GRUPPO di modelli da confrontare.
    model_vars = {k: tk.BooleanVar(value=(k in DEFAULT_MODELS)) for k, _, _ in MODEL_DEFS}

    def field(col, label, var, values):
        """Un campo = etichetta minuscola in alto + menu a tendina sotto, come celle
        affiancate nella barra. Ritorna (cella, combobox)."""
        cell = ttk.Frame(ctrl, style="Card.TFrame")
        cell.grid(row=0, column=col, padx=(0, 32), sticky="w")
        ttk.Label(cell, text=label.upper(), style="Field.TLabel").pack(anchor="w")
        cb = ttk.Combobox(cell, textvariable=var, values=values, state="readonly",
                          width=13)
        cb.pack(anchor="w", pady=(7, 0))
        return cell, cb

    # v_max coperti: singola/doppia 0.1–5; FCM = passo v_max*10; no-ring solo <=2.
    SPEEDS = ["0.1", "0.4", "0.7", "1", "1.2", "1.5", "1.7", "1.9", "2", "3", "4", "5"]

    # Cella MODELS: 4 checkbox (2x2), etichette col colore della rispettiva traiettoria.
    models_cell = ttk.Frame(ctrl, style="Card.TFrame")
    models_cell.grid(row=0, column=0, padx=(0, 36), sticky="w")
    ttk.Label(models_cell, text="MODELS", style="Field.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w")
    model_checks = []
    for i, (k, lab, color) in enumerate(MODEL_DEFS):
        chk = tk.Checkbutton(models_cell, text=lab, variable=model_vars[k],
                             onvalue=True, offvalue=False, bg=DARK_PANEL, fg=color,
                             selectcolor=DARK_FIELD, activebackground=DARK_PANEL,
                             activeforeground=color, highlightthickness=0, bd=0,
                             font=UI_FONT, anchor="w")
        chk.grid(row=1 + i // 2, column=i % 2, sticky="w", padx=(0, 18), pady=(5, 0))
        model_checks.append(chk)

    ver_cell, ver_cb = field(1, "Wind", ver_var, VERSIONS)
    chunk_cell, chunk_cb = field(2, "Time Chunk", chunk_var, list(CHUNK_BY_LABEL.keys()))
    vmax_cell, vmax_cb = field(3, "Max Speed", vmax_var, SPEEDS)

    # Centro: la simulazione live occupa la maggior parte della finestra. Il canvas
    # viene messo in griglia solo a caricamento completato (finalize_loading);
    # durante il caricamento la stessa cella (row 1) ospita la schermata di attesa.
    fig = Figure(figsize=(9, 7), facecolor=DARK_BG)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().configure(bg=DARK_BG, highlightthickness=0)
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    def show_idle():
        """Schermata inerte (nessuna simulazione): identica all'avvio e dopo il
        reset. Imposta il facecolor in modo esplicito perché ax.clear() NON lo
        ripristina — senza questo, dopo un episodio lo sfondo resterebbe blu."""
        ax.clear()
        ax.set_facecolor(DARK_BG)
        ax.set_xticks([]); ax.set_yticks([])
        for _sp in ax.spines.values():
            _sp.set_color(DARK_PANEL)
        ax.set_title("No simulation running", color=DARK_MUTED)
        canvas.draw_idle()

    # Pulsanti a destra nella stessa barra (l'etichetta vuota li allinea coi menu).
    btns = ttk.Frame(ctrl, style="Card.TFrame")
    btns.grid(row=0, column=6, sticky="e")
    ttk.Label(btns, text=" ", style="Field.TLabel").grid(row=0, column=0, columnspan=2)
    start_btn = ttk.Button(btns, text="Start", style="Accent.TButton")
    start_btn.grid(row=1, column=0, padx=(0, 8), pady=(4, 0))
    cancel_btn = ttk.Button(btns, text="Cancel")
    cancel_btn.grid(row=1, column=1, pady=(4, 0))

    status = ttk.Label(root, text="Ready — choose the options and press Start.",
                       style="Muted.TLabel", padding=(18, 12))
    status.grid(row=2, column=0, sticky="w")

    state = {"agents": [], "running": False, "steps": 0, "dm": dm,
             "load_token": None, "scenario": ""}

    def set_controls(enabled: bool):
        st = "readonly" if enabled else "disabled"
        for cb in (ver_cb, chunk_cb, vmax_cb):
            cb.configure(state=st)
        for chk in model_checks:
            chk.configure(state=("normal" if enabled else "disabled"))
        start_btn.configure(state=("normal" if enabled else "disabled"))

    def cleanup_env():
        for a in state.get("agents", []):
            try:
                a["vec_env"].close()
            except Exception:
                pass
        state["agents"] = []

    def reset_program():
        """Ripristina le configurazioni ai default e riabilita i controlli."""
        cleanup_env()
        state.update(running=False, steps=0, load_token=None)
        for k, _, _ in MODEL_DEFS:
            model_vars[k].set(k in DEFAULT_MODELS)
        ver_var.set(DEFAULTS["version"]); chunk_var.set(DEFAULTS["chunk"])
        vmax_var.set(DEFAULTS["vmax"])
        set_controls(True)

    def finish():
        gstep = max((a["steps"] for a in state["agents"]), default=0)
        draw_scene_multi(ax, state["agents"],
                         f"{state['scenario']}  —  finished ({gstep} steps)")
        canvas.draw_idle()
        outcomes = "   ".join(
            f"{MODEL_LABEL[a['kind']]}: {(a['term'] or 'timeout').upper()} ({a['steps']})"
            for a in state["agents"])
        # Fine naturale: MANTIENE le configurazioni scelte; chiude gli ambienti e
        # riabilita i controlli, lasciando l'ultimo frame a schermo.
        cleanup_env()
        state.update(running=False, steps=0)
        set_controls(True)
        status.configure(text=f"Finished — {outcomes}. Settings kept — press Start to run again.")

    def step():
        if not state["running"]:
            return
        any_running = False
        for a in state["agents"]:
            if a["done"]:
                continue
            try:
                obs, done, info = step_once(a["agent"], a["vec_env"], a["obs"], a["is_fcm"])
            except Exception as e:
                status.configure(text=f"Error during the simulation: {e}")
                reset_program()
                return
            a["obs"] = obs
            a["steps"] += 1
            pos = info.get("position")            # posizione dell'info (valida anche a done)
            if pos is not None:
                a["traj"].append((float(pos[0]), float(pos[1])))
            if done:
                a["done"] = True
                a["term"] = resolve_termination(info)
            else:
                any_running = True
        gstep = max((a["steps"] for a in state["agents"]), default=0)
        draw_scene_multi(ax, state["agents"], f"{state['scenario']}  —  step {gstep}")
        canvas.draw_idle()
        if any_running:
            root.after(delay_ms, step)
        else:
            state["running"] = False
            finish()

    def start():
        kinds = [k for k, _, _ in MODEL_DEFS if model_vars[k].get()]
        if not kinds:
            status.configure(text="Select at least one model.")
            return
        version = ver_var.get()
        chunk = CHUNK_BY_LABEL[chunk_var.get()]
        vmax = float(vmax_var.get())

        source = pick_source(state["dm"], version, rng)
        if source is None:
            status.configure(text=f"No source available for wind {version}.")
            return

        set_controls(False)
        status.configure(text=f"Loading {len(kinds)} model(s) … scenario {source} "
                              f"{version} {chunk_var.get()}")

        # Caricamento in un THREAD di sfondo (make_agents non tocca Tk/matplotlib).
        token = object()
        state["load_token"] = token             # invalidato da Cancel/reset
        result = {}

        def _load():
            try:
                result["value"] = make_agents(state["dm"], root_dir, kinds,
                                              version, chunk, vmax, source)
            except Exception as e:              # marshallato nel poll (thread GUI)
                result["error"] = e

        def _await_load():
            if state.get("load_token") is not token:   # Cancel/nuovo Start nel frattempo
                v = result.get("value")
                if v is not None:
                    for a in v[0]:
                        try:
                            a["vec_env"].close()
                        except Exception:
                            pass
                return
            if "error" in result:
                status.configure(text=f"Loading error: {result['error']}")
                set_controls(True)
                return
            if "value" not in result:
                root.after(50, _await_load)     # ancora in caricamento
                return
            agents, unavailable = result["value"]
            if not agents:
                status.configure(text=f"None of the selected models is available "
                                      f"for max speed {vmax:g}.")
                set_controls(True)
                return
            state.update(agents=agents, running=True, steps=0,
                         scenario=f"{source} {version} {chunk_var.get()}")
            note = ("   ·   n/a @v_max: " + ", ".join(MODEL_LABEL[k] for k in unavailable)) \
                if unavailable else ""
            status.configure(text=f"Running {len(agents)} model(s): {state['scenario']}{note}")
            root.after(delay_ms, step)

        threading.Thread(target=_load, daemon=True).start()
        root.after(50, _await_load)

    def cancel():
        """Interrompe la simulazione (se in corso) e ripristina le configurazioni.
        Non chiude il programma: l'end state è sempre la simulazione interrotta."""
        was_running = state["running"]
        show_idle()
        reset_program()
        status.configure(text=("Simulation cancelled. " if was_running else "")
                              + "Settings reset — ready for a new run.")

    start_btn.configure(command=start)
    cancel_btn.configure(command=cancel)

    def on_close():
        state["running"] = False
        cleanup_env()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # GUI pronta: mostra il canvas, abilita i controlli, schermata inerte.
    canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 6))
    set_controls(True)
    show_idle()
    status.configure(text="Ready — choose the options and press Start.")


def main() -> None:
    """Entry point unico: bootstrapper + GUI in un solo file. Una schermata di
    caricamento con 5 fasi (requisiti → script → dati → modelli → ambiente), poi la
    simulazione nella stessa finestra. Ogni fase è saltata se già soddisfatta
    (idempotente e non distruttivo). Gli import pesanti sono differiti a dopo che
    requisiti e script sono presenti (_import_heavy)."""
    ap = argparse.ArgumentParser(description="HYDRAS — simulazione live")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="frame al secondo della visualizzazione (default 15)")
    args = ap.parse_args()

    import tkinter as tk
    from tkinter import ttk

    root_dir = INSTALL_DIR
    root = tk.Tk()
    root.title("HYDRAS — Startup")
    root.geometry("900x680")
    _apply_dark_theme(root)
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    loading = ttk.Frame(root, padding=40)
    loading.grid(row=1, column=0, sticky="nsew")
    cap_label = ttk.Label(loading, text="Starting…", font=("", 13))
    cap_label.pack(pady=(120, 14))
    pbar = ttk.Progressbar(loading, mode="determinate", maximum=100, length=420)
    pbar.pack()
    pct_label = ttk.Label(loading, text="0%")
    pct_label.pack(pady=(8, 0))

    loader = {"dm": None, "error": None, "done": False, "installing": False,
              "caption": "Starting…", "pct": 0.0, "env_start": None}

    def rep(caption=None, pct=None):
        if caption is not None:
            loader["caption"] = caption
        if pct is not None:
            loader["pct"] = pct

    def worker():
        cache = {"path": None}
        try:
            # 1) Requisiti ------------------------------------------------------
            rep("Installing requirements…", 2)
            if missing_packages():
                req = root_dir / "requirements.txt"
                if not req.exists():
                    download_file(REQUIREMENTS_URL, req)
                # pip è lungo e (su pipe) non espone la % del singolo download: il
                # poll() stima una percentuale che AVANZA dai pacchetti visti + tempo,
                # e la didascalia mostra il pacchetto corrente ("Downloading torch…").
                loader["pkgs"] = set()
                loader["install_start"] = time.perf_counter()
                loader["installing"] = True

                def _pip_cb(status):
                    loader["pkgs"].add(status)        # stati distinti ≈ progresso (per la %)
                    rep(status)

                install_requirements(root_dir, progress_cb=_pip_cb)
                loader["installing"] = False
                importlib.invalidate_caches()
                if missing_packages():
                    raise RuntimeError("Requirements still missing after pip: "
                                       + ", ".join(missing_packages()))
            rep(pct=15)

            # 2) Script: solo le dipendenze effettive di live_sim (inference.py +
            # utils/). Si escludono explainability.py, train_ppo.py e
            # run_adaptive_sweeps.py, che live_sim non importa mai.
            rep("Downloading scripts…", 15)
            if not scripts_present(root_dir):
                tar = _get_tarball(cache, cb=lambda f: rep(pct=15 + 10 * f))
                _extract_prefixes(tar, root_dir, ("src/inference.py", "utils/"))
                if not scripts_present(root_dir):
                    raise RuntimeError("Project scripts missing after extraction.")
            rep(pct=30)

            # Requisiti e script ci sono: ora si possono caricare numpy/inference/
            # DataManager (e i finder dei modelli usati nella fase 4).
            _import_heavy()

            # 3) Dati (da GitHub Release, con progresso in GB) ------------------
            rep("Downloading data…", 30)
            if not data_present(root_dir):
                _GB = 1073741824

                def data_prog(done, tot):
                    rep(f"Downloading data… {done/_GB:.1f} / {tot/_GB:.1f} GB",
                        30 + 35 * (done / tot))

                download_data_from_release(root_dir, DATA_RELEASE_TAG,
                                           progress_cb=data_prog)
            # Ripulisce eventuali companion di macOS (anche da installazioni
            # precedenti): DataManager li scambierebbe per file .nc.
            _remove_apple_junk(root_dir / "data")
            rep(pct=65)

            # 4) Modelli --------------------------------------------------------
            rep("Downloading models…", 65)
            if missing_models(root_dir):
                tar = _get_tarball(cache, cb=lambda f: rep(pct=65 + 8 * f))
                _extract_prefixes(tar, root_dir, ("trained_models/",))
                if missing_models(root_dir):
                    raise RuntimeError("PPO models missing after extraction.")
            rep(pct=75)
            if cache["path"]:
                cache["path"].unlink(missing_ok=True)

            # 5) Ambiente -------------------------------------------------------
            rep("Loading environment…", 75)
            loader["env_start"] = time.perf_counter()
            loader["dm"] = DataManager(data_dir=str(root_dir / "data"), preload_all=False,
                                       sources_csv="Coordinate_Sorgenti_FaseII.csv")
        except Exception as e:                 # marshallato nel poll (thread principale)
            loader["error"] = e
        finally:
            loader["done"] = True

    def poll():
        if loader["error"] is not None:
            cap_label.configure(text=f"Startup error: {loader['error']}")
            pct_label.configure(text="")
            return
        if loader["done"]:
            pbar["value"] = 100; pct_label.configure(text="100%")
            cap_label.configure(text=loader["caption"])

            def finalize():
                loading.destroy()
                root.title("HYDRAS — Live Simulation")
                build_gui(root, loader["dm"], args.fps)

            root.after(200, finalize)          # mostra brevemente il 100%
            return
        if loader.get("installing"):           # pip: % stimata da pacchetti + tempo
            n = len(loader.get("pkgs") or ())
            el = time.perf_counter() - (loader.get("install_start") or time.perf_counter())
            pct = 2.0 + 13.0 * (1.0 - 1.0 / (1.0 + (n + el / 15.0) / 12.0))
        elif loader["env_start"] is not None:  # fase ambiente opaca: anima a tempo
            el = time.perf_counter() - loader["env_start"]
            pct = 75.0 + min(24.0, 24.0 * el / 13.0)
        else:
            pct = loader["pct"]
        pbar["value"] = pct; pct_label.configure(text=f"{int(pct)}%")
        cap_label.configure(text=loader["caption"])
        root.after(100, poll)

    threading.Thread(target=worker, daemon=True).start()
    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
