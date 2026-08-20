"""
Esegue in SEQUENZA i due sweep "adattivi", con la nuova logica di spawn distribuito
e SENZA generazione video (più rapido; i video si fanno eventualmente dopo):

  1) Sweep FCM Adam     — learning rate (passo) crescente: lr ∈ {10,20,30,40,50} m,
                          sensor_range = 50 m (ottimale FCM).
                          Output: thesis/evaluations/evaluations_FCM/fcm_adaptive/lr_{lr}/

  2) Sweep PPO velocità adattiva — v_max ∈ {1,2,3,4,5} m/s, K=5 livelli (Discrete(8*5)=40):
                          l'agente sceglie la velocità in (0, v_max] ad ogni passo.
                          Catena: v1 WARM-START dal minimal, poi v2 da v1, v3 da v2, ...
                          (ogni stadio parte dall'ULTIMO modello addestrato).
                          SOLO TRAINING — i modelli vanno in trained_models/ppo_*.
                          Le inferenze si faranno dopo, separatamente.

Uso:
  python3 src/run_adaptive_sweeps.py                # DEFAULT (canonico): SEQUENZA COMPLETA prof, in automatico e in
                                                    #   sequenza -> 1) singola-corona 0.7/0.4/0.1 (catena da v1)
                                                    #   2) doppia-corona 0.7/0.4/0.1  3) doppia-corona 1.2/1.5/1.7/1.9
                                                    #   [TRAINING] log live in trained_models/train_prof.log
  STAGE=ppo_mid python3 src/run_adaptive_sweeps.py  # solo catena v_max INTERMEDI 1.2/1.5/1.7/1.9 (fine-tuning da v1)
  STAGE=ppo python3 src/run_adaptive_sweeps.py      # catena piena v_max 1→5
  STAGE=ppo_low python3 src/run_adaptive_sweeps.py  # solo singola BASSE velocità 0.7/0.4/0.1 (discendente da v1) [TRAINING]
  STAGE=fcm python3 src/run_adaptive_sweeps.py      # sweep FCM (lr 10→50)
  STAGE=fcm_low python3 src/run_adaptive_sweeps.py  # sweep FCM basse velocità (lr 7/4/1) [solo inferenza]
  STAGE=dc_mid python3 src/run_adaptive_sweeps.py   # solo DOPPIA corona intermedi 1.2/1.5/1.7/1.9 (warm-start da single_vX) [TRAINING]
  STAGE=dc_low python3 src/run_adaptive_sweeps.py   # solo DOPPIA corona bassi 0.7/0.4/0.1 (serve prima STAGE=ppo_low) [TRAINING]
  STAGE=noform python3 src/run_adaptive_sweeps.py   # agente SOLO senza formazione: 1 finetuning da SINGOLA corona v2 (mask_formation) [TRAINING]
"""
import os
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import run_inference_fcm                       # noqa: E402
from src.train_ppo import train, load_config                     # noqa: E402

DATA_DIR    = str(ROOT / "data")
# Config "migliore" dalle analisi precedenti: reward minimal (base_no_wind_reward),
# sensori a 20 m, singola corona (no sensor_range_2). È il config del modello minimal (96.9%).
MINIMAL_CONFIG = str(ROOT / "utils" / "config" / "config_base_no_wind_reward.yaml")
BASE_CONFIG = MINIMAL_CONFIG   # base sia per FCM sia per PPO
FCM_CONFIG  = MINIMAL_CONFIG

N_EPISODES  = 2     # episodi per (sorgente, vento, chunk) in inferenza


# ─── Utility di logging ───────────────────────────────────────────────────────

def _banner(title: str, char: str = "=") -> None:
    line = char * 84
    print(f"\n{line}\n{title}\n{line}", flush=True)


def _fmt_dt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _read_sr(output_dir) -> float:
    """Legge il Global Success Rate (%) dal log.txt prodotto dall'inferenza."""
    log = Path(output_dir) / "log.txt"
    if log.exists():
        for ln in log.read_text(errors="ignore").splitlines():
            if "Global Success Rate" in ln:
                m = re.search(r"([\d.,]+)\s*%", ln)
                if m:
                    return float(m.group(1).replace(",", "."))
    return float("nan")


# ─── 1) Sweep FCM Adam ────────────────────────────────────────────────────────

def fcm_sweep(lrs=None):
    out_root = ROOT / "thesis" / "evaluations" / "evaluations_FCM" / "fcm_adaptive"
    lrs = lrs if lrs is not None else [10, 20, 30, 40, 50]
    results = []
    for i, lr in enumerate(lrs, 1):
        _banner(f"  [FCM {i}/{len(lrs)}]  Adam — passo lr = {lr} m   "
                f"(sensor_range = 50 m, spawn distribuito, no video)")
        t0 = time.time()
        out = out_root / f"lr_{lr}"
        run_inference_fcm(
            config_path=FCM_CONFIG, data_dir=DATA_DIR, output_dir=str(out),
            n_episodes=N_EPISODES, sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2], sensor_range=50.0, lr=float(lr),
            save_videos=False,
        )
        sr, dt = _read_sr(out), time.time() - t0
        sr_s = f"{sr:.1f}%" if sr == sr else "n/d"
        results.append((f"lr={lr}m", sr_s, dt))
        print(f"\n  ✓ [FCM {i}/{len(lrs)}] lr={lr}m  →  SR = {sr_s}   ({_fmt_dt(dt)})\n"
              f"    output: {out}", flush=True)
    return results


# ─── 2) Sweep PPO velocità adattiva (catena di fine-tuning) ───────────────────

K_VELOCITY_LEVELS  = 5
VMAX_LIST          = [1, 2, 3, 4, 5]      # m/s
TIMESTEPS_FINETUNE = 2_000_000            # stadi successivi (a catena)
FINETUNE_LR        = 1.0e-5               # LR costante per il fine-tuning (come la catena del report v8)
FINETUNE_ALPHA     = 0.8                  # peso scenari difficili (V1/V2 × Q1/2,Q3/4): hard 8× easy
N_ENVS             = 2

# v_max=1: warm-start dal minimal Discrete(8) -> Discrete(8*K) invece che da zero.
# Parte dal comportamento del minimal (sempre velocità max) e impara solo a rallentare.
MINIMAL_MODEL       = ROOT / "trained_models" / "ppo_20260516_143937" / "models" / "final_model.zip"
TIMESTEPS_WARMSTART = 2_000_000           # basta un fine-tuning: la navigazione è già appresa
WARMSTART_LR        = 1.0e-4              # più alto del fine-tuning: deve plasmare la testa di velocità

# Esperimento "raggio 10 m": fine-tuning del v5 con raggio di successo ridotto a 10 m.
# Passo max 50 m ≫ 10 m → l'agente è OBBLIGATO a rallentare vicino alla sorgente per
# centrarla → dovrebbe emergere la modulazione di velocità.
V5_MODEL_R10   = ROOT / "trained_models" / "ppo_20260629_065041" / "models" / "final_model.zip"
R10_RADIUS     = 10        # m - nuovo raggio di successo
R10_STAGNATION = 5         # m - stagnation soglia scalata (non penalizza l'avvicinamento lento)
R10_LR         = 1.0e-4    # alto: deve creare un comportamento nuovo (rallentare), non rifinire
R10_TIMESTEPS  = 2_000_000

# Esperimento "doppia corona": warm-start da v1 (singola corona, 116-dim) verso la doppia
# corona (196-dim, +sensor_range_2). Test EQUO del valore della 2a corona: parte dal
# comportamento di v1 (corona-2 azzerata) e impara a usarla solo se aiuta.
V1_MODEL_DC  = ROOT / "trained_models" / "ppo_20260628_192155" / "models" / "final_model.zip"
DC_LR        = 1.0e-4      # alto: deve imparare a usare la corona-2 (comportamento nuovo)
DC_TIMESTEPS = 2_000_000

# Opzione 2 (doppia corona a ogni v_max): per X in {2,3,4,5} warm-start INDIPENDENTE della
# doppia corona dal single_vX omologo (STESSO protocollo di dual_v1: lr DC_LR, 2M step, α 0.8,
# corona-2 azzerata). Serve a misurare se il guadagno di STEP (-18% a v1) regge a ogni v_max
# (gli step, a differenza dell'SR, non hanno soffitto). NON è una catena: ogni dual_vX si
# stacca dal proprio single_vX, quindi i 4 addestramenti sono indipendenti (parallelizzabili).
DC_CHAIN_VMAX = [2, 3, 4, 5]
# Doppia corona anche ai v_max INTERMEDI e BASSI (warm-start dal single_vX omologo, che
# deve esistere: intermedi già addestrati; bassi creati prima da STAGE=ppo_low).
DC_MID_VMAX = [1.2, 1.5, 1.7, 1.9]        # richiesta prof: doppia corona intermedi
DC_LOW_VMAX = [0.7, 0.4, 0.1]             # richiesta prof: doppia corona bassi


def ppo_sweep():
    prev_model = None                      # v1: warm-start dal minimal; poi catena
    results = []

    for i, vmax in enumerate(VMAX_LIST, 1):
        # Config dedicato: K livelli di velocità, v_max corrente, singola corona (116-dim)
        cfg = load_config(BASE_CONFIG)
        cfg['agent']['n_velocity_levels'] = K_VELOCITY_LEVELS
        cfg['agent']['max_velocity'] = float(vmax)
        cfg['agent'].pop('sensor_range_2', None)
        cfg['agent']['sensor_range'] = cfg['agent'].get('sensor_range', 20)
        cfg.setdefault('training', {})['target_kl'] = 0.02   # evita early stopping sistematico

        # Tutti gli stadi girano a LR costante. Lo schedule del config è tarato su 6M:
        # su 2M resterebbe bloccato a 3e-4 → lo rimuoviamo.
        #  - v1: warm-start dal minimal, LR più alto (plasma la testa di velocità);
        #  - v2..v5: fine-tuning a catena dall'ultimo modello, LR basso.
        is_first = prev_model is None
        timesteps = TIMESTEPS_WARMSTART if is_first else TIMESTEPS_FINETUNE
        cfg['training'].pop('lr_schedule', None)
        cfg['training']['learning_rate'] = WARMSTART_LR if is_first else FINETUNE_LR

        # Eval multi-scenario: ATTIVA solo su v1 warm-start (curva SR in diretta ogni
        # eval_freq step, best model in models/best/) per validare il transfer dal
        # minimal. v2..v5 senza eval (più veloci). v1 usa gli eval_scenarios del config
        # base (mix V0/V1/V2).
        if not is_first:
            cfg['training']['eval_scenarios'] = []

        # Curriculum α a FASE UNICA con alpha alto: oversampling degli scenari difficili
        # (_HARD = vento V1/V2 × chunk Q1/2,Q3/4) per tutta la durata del fine-tuning.
        # Sostituisce lo schedule α 6M del config (che su 2M resterebbe a α=0, uniforme).
        cfg['training']['scenario_curriculum'] = [{'end': timesteps, 'alpha': FINETUNE_ALPHA}]

        cfg_path = str(ROOT / "utils" / "config" / f"config_vmax{vmax}.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        levels = [round((j + 1) / K_VELOCITY_LEVELS * vmax, 2) for j in range(K_VELOCITY_LEVELS)]
        kind = (f"warm-start da minimal (lr {WARMSTART_LR:g}, α={FINETUNE_ALPHA:g})" if is_first
                else f"fine-tuning (lr {FINETUNE_LR:g}, α={FINETUNE_ALPHA:g})")
        start_desc = str(MINIMAL_MODEL) if is_first else prev_model
        _banner(f"  [PPO {i}/{len(VMAX_LIST)}]  v_max = {vmax} m/s   "
                f"(livelli {levels} m/s, K={K_VELOCITY_LEVELS})")
        print(f"    tipo      : {kind}")
        print(f"    timesteps : {timesteps:,}")
        print(f"    parte da  : {start_desc}", flush=True)

        t0 = time.time()
        # --- Solo training (catena). Le inferenze si faranno dopo, separatamente. ---
        print(f"\n  ▸ training v_max={vmax} m/s ...", flush=True)
        _, run_dir = train(
            config_path=cfg_path, output_dir=str(ROOT / "trained_models"),
            n_envs=N_ENVS, total_timesteps=timesteps, seed=42,
            data_dir=DATA_DIR,
            resume_from=(None if is_first else prev_model),
            warm_start_from=(str(MINIMAL_MODEL) if is_first else None),
        )
        model_path = str(run_dir / "models" / "final_model.zip")
        prev_model = model_path             # catena: il prossimo stadio parte da qui

        dt = time.time() - t0
        results.append((f"v_max={vmax}m/s", "addestrato", dt))
        print(f"\n  ✓ [PPO {i}/{len(VMAX_LIST)}] v_max={vmax} m/s addestrato   ({_fmt_dt(dt)})\n"
              f"    modello: {model_path}", flush=True)

    return results


# ─── 2b) Sweep PPO v_max INTERMEDI fra 1 e 2 m/s (catena di fine-tuning) ───────
# Il salto di SR più marcato della catena è fra v_max=1 e v_max=2 (es. vento V2:
# 89,7 → 99,2; SR aggregata 97,1 → 99,7). Per risolvere la curva in quell'intervallo
# si addestrano modelli a v_max intermedi. Protocollo IDENTICO ai passaggi
# v2←v1←... della catena originale: fine-tuning puro (resume) dell'ULTIMO modello,
# LR FINETUNE_LR, 2M step, α 0.8, singola corona. Ancora = modello single v_max=1
# ESISTENTE (l'architettura Discrete(8*K) è già la sua → resume, non warm-start).
# Catena:  v1(esistente) → 1.2 → 1.5 → 1.7 → 1.9   (ogni stadio parte dall'ultimo).
# SOLO training; le inferenze si fanno dopo con la stessa pipeline della tabella.
VMAX_MID_LIST = [1.2, 1.5, 1.7, 1.9]      # m/s, fra v1 e v2
# Basse velocità (< v1): catena DISCENDENTE v1(esistente) → 0.7 → 0.4 → 0.1, stesso
# protocollo degli intermedi (fine-tuning puro dell'ultimo modello, ancora = single v1).
# Servono a estendere la curva SR–v_max sotto 1 m/s (richiesta 0.1–2.0 m/s).
VMAX_LOW_LIST = [0.7, 0.4, 0.1]           # m/s, sotto v1


def ppo_intermediate_sweep(default_list=None, label="MID"):
    default_list = default_list if default_list is not None else VMAX_MID_LIST
    # Override via env (usati dal micro-run di calibrazione, non dal run reale):
    #   MID_VMAX="1.2"  MID_TIMESTEPS=20000  MID_OUTDIR=/tmp/...
    vmax_list = [float(x) for x in os.environ.get(
        "MID_VMAX", ",".join(str(v) for v in default_list)).split(",") if x.strip()]
    timesteps = int(os.environ.get("MID_TIMESTEPS", TIMESTEPS_FINETUNE))
    out_dir = os.environ.get("MID_OUTDIR", str(ROOT / "trained_models"))

    anchor = _find_single_vmax_model(1)
    if anchor is None:
        print("  [ERRORE] modello single v_max=1 non trovato: catena non ancorabile.", flush=True)
        return [("intermedi", "no-base", 0.0)]

    prev_model = str(anchor)
    results = []
    print(f"  Ancora catena (single v_max=1): {anchor}")
    print(f"  v_max intermedi: {vmax_list}   |   timesteps/stadio: {timesteps:,}   |   out: {out_dir}", flush=True)

    for i, vmax in enumerate(vmax_list, 1):
        # Config identico al ramo single-corona di ppo_sweep, sempre in FINE-TUNING.
        cfg = load_config(BASE_CONFIG)
        cfg['agent']['n_velocity_levels'] = K_VELOCITY_LEVELS
        cfg['agent']['max_velocity'] = float(vmax)
        cfg['agent'].pop('sensor_range_2', None)
        cfg['agent']['sensor_range'] = cfg['agent'].get('sensor_range', 20)
        cfg.setdefault('training', {})['target_kl'] = 0.02
        cfg['training'].pop('lr_schedule', None)
        cfg['training']['learning_rate'] = FINETUNE_LR
        cfg['training']['eval_scenarios'] = []
        cfg['training']['scenario_curriculum'] = [{'end': timesteps, 'alpha': FINETUNE_ALPHA}]

        cfg_path = str(ROOT / "utils" / "config" / f"config_vmax{vmax}.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        levels = [round((j + 1) / K_VELOCITY_LEVELS * vmax, 2) for j in range(K_VELOCITY_LEVELS)]
        _banner(f"  [PPO {label} {i}/{len(vmax_list)}]  v_max = {vmax} m/s   "
                f"(livelli {levels} m/s, K={K_VELOCITY_LEVELS})")
        print(f"    tipo      : fine-tuning (lr {FINETUNE_LR:g}, α={FINETUNE_ALPHA:g})")
        print(f"    timesteps : {timesteps:,}")
        print(f"    parte da  : {prev_model}", flush=True)

        t0 = time.time()
        print(f"\n  ▸ training v_max={vmax} m/s ...", flush=True)
        _, run_dir = train(
            config_path=cfg_path, output_dir=out_dir,
            n_envs=N_ENVS, total_timesteps=timesteps, seed=42,
            data_dir=DATA_DIR,
            resume_from=prev_model,
            warm_start_from=None,
        )
        model_path = str(run_dir / "models" / "final_model.zip")
        prev_model = model_path             # catena: il prossimo stadio parte da qui
        dt = time.time() - t0
        results.append((f"v_max={vmax}m/s", "addestrato", dt))
        print(f"\n  ✓ [PPO {label} {i}/{len(vmax_list)}] v_max={vmax} m/s addestrato   ({_fmt_dt(dt)})\n"
              f"    modello: {model_path}", flush=True)
    return results


def ppo_low_sweep():
    """Catena DISCENDENTE v1 → 0.8 → 0.5 → 0.2 (protocollo identico agli intermedi)."""
    return ppo_intermediate_sweep(default_list=VMAX_LOW_LIST, label="LOW")


def _summary(title, rows, col="esito"):
    """Stampa una tabella riassuntiva (etichetta, valore, durata)."""
    print(f"\n  {title}")
    print(f"  {'-'*44}")
    print(f"  {'configurazione':<18}{col:>10}{'durata':>10}")
    print(f"  {'-'*44}")
    for label, val, dt in rows:
        print(f"  {label:<18}{str(val):>10}{_fmt_dt(dt):>10}")
    print(f"  {'-'*44}")


def radius10_experiment():
    """Fine-tuning del v5 con raggio di successo 10 m per forzare la modulazione.

    Resume dal v5 (stesso Discrete(40)), LR alto (deve creare il rallentamento, non
    rifinirlo), α=0.8 invariato, raggio 10 m + stagnation scalata. SOLO training:
    l'inferenza con il pannello velocità-vs-distanza si lancia con
    `python3 src/inference.py r10`.
    """
    cfg = load_config(BASE_CONFIG)
    cfg['agent']['n_velocity_levels'] = K_VELOCITY_LEVELS
    cfg['agent']['max_velocity'] = 5.0
    cfg['agent'].pop('sensor_range_2', None)
    cfg['agent']['sensor_range'] = cfg['agent'].get('sensor_range', 20)
    cfg.setdefault('training', {})['target_kl'] = 0.02
    cfg['training']['eval_scenarios'] = []
    cfg['training'].pop('lr_schedule', None)
    cfg['training']['learning_rate'] = R10_LR
    cfg['training']['scenario_curriculum'] = [{'end': R10_TIMESTEPS, 'alpha': FINETUNE_ALPHA}]
    # Cambio chiave: raggio di successo 10 m + stagnation coerente (5 m)
    rew = cfg.setdefault('environment', {}).setdefault('reward', {})
    rew['distance_threshold'] = R10_RADIUS
    rew['stagnation_distance_threshold'] = R10_STAGNATION

    cfg_path = str(ROOT / "utils" / "config" / "config_vmax5_r10.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    _banner(f"  [ESPERIMENTO RAGGIO {R10_RADIUS} m]  v_max=5, modulazione forzata")
    print(f"    resume da : {V5_MODEL_R10}")
    print(f"    raggio    : {R10_RADIUS} m   (era 50)   |   stagnation: {R10_STAGNATION} m")
    print(f"    lr        : {R10_LR:g}   |   α: {FINETUNE_ALPHA:g}   |   timesteps: {R10_TIMESTEPS:,}", flush=True)

    t0 = time.time()
    _, run_dir = train(
        config_path=cfg_path, output_dir=str(ROOT / "trained_models"),
        n_envs=N_ENVS, total_timesteps=R10_TIMESTEPS, seed=42,
        data_dir=DATA_DIR, resume_from=str(V5_MODEL_R10),
    )
    model_path = str(run_dir / "models" / "final_model.zip")
    print(f"\n  ✓ esperimento raggio {R10_RADIUS} m addestrato  ({_fmt_dt(time.time()-t0)})\n"
          f"    modello: {model_path}\n"
          f"    inferenza: python3 src/inference.py r10", flush=True)
    return [(f"r{R10_RADIUS}", "addestrato", time.time() - t0)]


def dualcorona_experiment():
    """Warm-start della DOPPIA corona da v1 (singola corona) — test equo del 2° anello.

    Parte da v1 (Discrete(40), obs 116), aggiunge la corona-2 (obs 196) coi pesi d'ingresso
    della corona-2 azzerati → all'avvio si comporta come v1 (97%); impara a usare la 2a
    corona solo se aiuta. Eval attiva (un solo run). Inferenza separata dopo.
    """
    cfg = load_config(BASE_CONFIG)
    cfg['agent']['n_velocity_levels'] = K_VELOCITY_LEVELS
    cfg['agent']['max_velocity'] = 1.0
    cfg['agent']['sensor_range'] = 20
    cfg['agent']['sensor_range_2'] = 50          # DOPPIA corona (obs 196)
    cfg.setdefault('training', {})['target_kl'] = 0.02
    cfg['training'].pop('lr_schedule', None)
    cfg['training']['learning_rate'] = DC_LR
    cfg['training']['scenario_curriculum'] = [{'end': DC_TIMESTEPS, 'alpha': FINETUNE_ALPHA}]
    # Intorno di successo standard a 50 m (esplicito) + stagnation standard.
    rew = cfg.setdefault('environment', {}).setdefault('reward', {})
    rew['distance_threshold'] = 50
    rew['stagnation_distance_threshold'] = 20.0
    # eval_scenarios lasciati attivi (dal config base): curva SR in diretta

    cfg_path = str(ROOT / "utils" / "config" / "config_dualcorona_v1.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    _banner("  [ESPERIMENTO DOPPIA CORONA]  warm-start da v1 (116 -> 196 dim)")
    print(f"    base       : {V1_MODEL_DC}")
    print(f"    sensori    : corona 1 @ 20 m  +  corona 2 @ 50 m")
    print(f"    lr         : {DC_LR:g}   |   α: {FINETUNE_ALPHA:g}   |   timesteps: {DC_TIMESTEPS:,}", flush=True)

    t0 = time.time()
    _, run_dir = train(
        config_path=cfg_path, output_dir=str(ROOT / "trained_models"),
        n_envs=N_ENVS, total_timesteps=DC_TIMESTEPS, seed=42,
        data_dir=DATA_DIR, warm_start_dualcorona_from=str(V1_MODEL_DC),
    )
    model_path = str(run_dir / "models" / "final_model.zip")
    print(f"\n  ✓ doppia corona (warm-start da v1) addestrata  ({_fmt_dt(time.time()-t0)})\n"
          f"    modello: {model_path}", flush=True)
    return [("dual-corona", "addestrato", time.time() - t0)]


def _find_single_vmax_model(vmax, K: int = K_VELOCITY_LEVELS):
    """final_model.zip del modello SINGOLA corona della catena PPO per il dato v_max.

    Criterio: K livelli di velocità, max_velocity==vmax, corona singola (sensor_range_2
    assente) e raggio di successo 50 m (esclude l'esperimento r10). A parità prende il più
    recente (le dir ppo_<timestamp> sono ordinabili). Ritorna un Path o None.
    """
    best = None
    for d in sorted((ROOT / "trained_models").glob("ppo_*")):
        cfg_p = d / "config.yaml"
        fm = d / "models" / "final_model.zip"
        if not cfg_p.exists() or not fm.exists():
            continue
        try:
            c = load_config(str(cfg_p))
        except Exception:
            continue
        ag = c.get('agent', {})
        thr = c.get('environment', {}).get('reward', {}).get('distance_threshold', 50)
        if (int(ag.get('n_velocity_levels', 1)) == K
                and abs(float(ag.get('max_velocity', 1.0)) - float(vmax)) < 1e-6
                and ag.get('sensor_range_2') is None
                and not ag.get('mask_formation', False)   # esclude i modelli no-formation
                and abs(float(thr) - 50.0) < 1e-6):
            best = fm
    return best


def dualcorona_chain_experiment(vmax_list=None):
    """Doppia corona a più v_max, ciascuna warm-startata dal single_vX omologo.

    Replica a ogni livello il protocollo di dual_v1 (warm-start della corona-2 azzerata,
    lr DC_LR, 2M step, α 0.8, sensor_range_2=50, raggio 50 m). NON è una catena: ogni
    dual_vX si stacca dal proprio single_vX. Serve a misurare il guadagno di STEP per
    livello. SOLO training; le inferenze (dualcorona_vX) si lanciano dopo, separatamente.
    `vmax_list` default = DC_CHAIN_VMAX (2..5); usata anche per gli intermedi e i bassi.
    """
    vmax_list = vmax_list if vmax_list is not None else DC_CHAIN_VMAX
    results = []
    for i, vmax in enumerate(vmax_list, 1):
        single = _find_single_vmax_model(vmax)
        if single is None:
            print(f"  [SKIP] single_v{vmax} non trovato "
                  f"(K={K_VELOCITY_LEVELS}, corona singola, raggio 50 m).", flush=True)
            results.append((f"dual-v{vmax}", "no-base", 0.0))
            continue

        cfg = load_config(BASE_CONFIG)
        cfg['agent']['n_velocity_levels'] = K_VELOCITY_LEVELS
        cfg['agent']['max_velocity'] = float(vmax)
        cfg['agent']['sensor_range'] = 20
        cfg['agent']['sensor_range_2'] = 50          # DOPPIA corona (obs 196)
        cfg.setdefault('training', {})['target_kl'] = 0.02
        cfg['training'].pop('lr_schedule', None)
        cfg['training']['learning_rate'] = DC_LR
        cfg['training']['scenario_curriculum'] = [{'end': DC_TIMESTEPS, 'alpha': FINETUNE_ALPHA}]
        cfg['training']['eval_scenarios'] = []       # eval OFF (come catena singola v2..v5); inference dopo
        rew = cfg.setdefault('environment', {}).setdefault('reward', {})
        rew['distance_threshold'] = 50
        rew['stagnation_distance_threshold'] = 20.0

        cfg_path = str(ROOT / "utils" / "config" / f"config_dualcorona_v{vmax}.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        _banner(f"  [DOPPIA CORONA v{vmax} — {i}/{len(vmax_list)}]  "
                f"warm-start da single_v{vmax} (116 -> 196 dim)")
        print(f"    base       : {single}")
        print(f"    v_max      : {vmax} m/s   |   sensori: corona 1 @ 20 m + corona 2 @ 50 m")
        print(f"    lr         : {DC_LR:g}   |   α: {FINETUNE_ALPHA:g}   |   timesteps: {DC_TIMESTEPS:,}", flush=True)

        t0 = time.time()
        _, run_dir = train(
            config_path=cfg_path, output_dir=str(ROOT / "trained_models"),
            n_envs=N_ENVS, total_timesteps=DC_TIMESTEPS, seed=42,
            data_dir=DATA_DIR, warm_start_dualcorona_from=str(single),
        )
        model_path = str(run_dir / "models" / "final_model.zip")
        dt = time.time() - t0
        results.append((f"dual-v{vmax}", "addestrato", dt))
        print(f"\n  ✓ doppia corona v{vmax} (warm-start da single_v{vmax}) addestrata  ({_fmt_dt(dt)})\n"
              f"    modello: {model_path}", flush=True)
    return results


class _Tee:
    """Duplica la scrittura su più stream (console + file di log), con flush immediato
    così il log si aggiorna LIVE (seguibile con `tail -f`). Delega isatty/fileno/ecc. al
    primo stream (la console) per restare 'trasparente' verso chi lo interroga."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._streams[0].isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        # Attributi non definiti (fileno, encoding, ...) delegati alla console reale.
        return getattr(self._streams[0], name)


NOFORM_VMAX = 2.0   # base per il no-formation: SINGOLA corona v2 (rimozione più contenuta e
                    # interpretabile della doppia; velocità di crociera realistica).


def noformation_experiment(vmax=NOFORM_VMAX):
    """Agente 'solo' (senza formazione): UNA finetuning del modello SINGOLA corona v{vmax}
    con i sensori direzionali AZZERATI (mask_formation=True). Si parte dalla singola corona
    (8 sensori), non dalla doppia: la rimozione è più contenuta, il confronto 'formazione
    vs nessuna formazione' è pulito, e il warm-start ha meno ingressi 'morti' (80 anziché
    160) → adattamento migliore e gap più equo. L'architettura NON cambia (obs 116,
    Discrete(40)) → resume_from dal modello con formazione, così il confronto è a parità di
    tutto tranne la formazione. Per il capitolo Explainability (importanza della formazione:
    ablation congelata vs riaddestramento). UNA sola finetuning; inferenza held-out dopo."""
    base = _find_single_vmax_model(vmax)
    if base is None:
        print(f"  [SKIP] singola corona v{vmax:g} non trovata (serve come base per il no-formation).",
              flush=True)
        return [("no-formation", "no-base", 0.0)]

    cfg = load_config(BASE_CONFIG)
    cfg['agent']['n_velocity_levels'] = K_VELOCITY_LEVELS
    cfg['agent']['max_velocity'] = float(vmax)
    cfg['agent']['sensor_range'] = 20
    cfg['agent'].pop('sensor_range_2', None)       # SINGOLA corona (obs 116, forma invariata)
    cfg['agent']['mask_formation'] = True           # <-- azzera i sensori della formazione
    cfg.setdefault('training', {})['target_kl'] = 0.02
    cfg['training'].pop('lr_schedule', None)
    cfg['training']['learning_rate'] = DC_LR
    cfg['training']['scenario_curriculum'] = [{'end': DC_TIMESTEPS, 'alpha': FINETUNE_ALPHA}]
    cfg['training']['eval_scenarios'] = []         # eval OFF; inferenza held-out dopo, separata
    rew = cfg.setdefault('environment', {}).setdefault('reward', {})
    rew['distance_threshold'] = 50
    rew['stagnation_distance_threshold'] = 20.0

    cfg_path = str(ROOT / "utils" / "config" / f"config_noformation_v{vmax:g}.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    _banner(f"  [AGENTE SOLO (no-formation) v{vmax:g}]  finetuning da SINGOLA corona, formazione azzerata")
    print(f"    base       : {base}")
    print(f"    sensori    : AZZERATI (mask_formation=True) — restano conc. propria, memorie, vento/corrente")
    print(f"    lr         : {DC_LR:g}   |   α: {FINETUNE_ALPHA:g}   |   timesteps: {DC_TIMESTEPS:,}", flush=True)

    t0 = time.time()
    _, run_dir = train(
        config_path=cfg_path, output_dir=str(ROOT / "trained_models"),
        n_envs=N_ENVS, total_timesteps=DC_TIMESTEPS, seed=42,
        data_dir=DATA_DIR, resume_from=str(base),
    )
    model_path = str(run_dir / "models" / "final_model.zip")
    print(f"\n  ✓ agente solo (no-formation) addestrato  ({_fmt_dt(time.time()-t0)})\n"
          f"    modello: {model_path}", flush=True)
    return [("no-formation", "addestrato", time.time() - t0)]


def run_prof_training_sequence():
    """Sequenza COMPLETA richiesta dal prof, AUTOMATICA e CONSECUTIVA (finito uno parte l'altro):
      1) SINGOLA corona basse velocità: catena discendente v1 -> 0.7 -> 0.4 -> 0.1
         (ogni stadio fine-tuning del precedente).
      2) DOPPIA corona basse: 0.7/0.4/0.1, ogni v_max warm-start dal single_vX omologo
         (creato al passo 1) — stesso protocollo dei modelli 1-5.
      3) DOPPIA corona intermedie: 1.2/1.5/1.7/1.9, ogni v_max warm-start dal single_vX
         intermedio (già esistente) — stesso protocollo.
    L'ordine garantisce anche la dipendenza (la fase 2 richiede i single-low della fase 1).
    SOLO training; le inferenze si lanciano dopo.

    Tutto l'output (banner di fase + tabelle di progresso SB3) viene scritto LIVE anche su
    trained_models/train_prof.log, così puoi seguire l'avanzamento con `tail -f`.
    """
    log_path = ROOT / "trained_models" / "train_prof.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", buffering=1, encoding="utf-8")   # line-buffered
    # Tee SOLO su stdout (banner + tabelle SB3). NON tocco stderr: così la progress bar
    # tqdm di SB3 (che scrive su stderr) resta visibile sul terminale.
    orig_out = sys.stdout
    sys.stdout = _Tee(orig_out, logf)

    t0 = time.time()
    all_res = []
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]  LOG LIVE → {log_path}", flush=True)
        print(f"    (segui con:  tail -f {log_path} )", flush=True)

        _banner("  ###  SEQUENZA PROF 1/3 — SINGOLA CORONA 0.7/0.4/0.1 (catena discendente da v1)  ###", "#")
        all_res.append(("--- 1) singola corona 0.7/0.4/0.1 ---", "", 0.0))
        all_res += ppo_low_sweep()

        _banner("  ###  SEQUENZA PROF 2/3 — DOPPIA CORONA 0.7/0.4/0.1 (warm-start da single_vX)  ###", "#")
        all_res.append(("--- 2) doppia corona 0.7/0.4/0.1 ---", "", 0.0))
        all_res += dualcorona_chain_experiment(vmax_list=DC_LOW_VMAX)

        _banner("  ###  SEQUENZA PROF 3/3 — DOPPIA CORONA 1.2/1.5/1.7/1.9 (warm-start da single_vX)  ###", "#")
        all_res.append(("--- 3) doppia corona 1.2/1.5/1.7/1.9 ---", "", 0.0))
        all_res += dualcorona_chain_experiment(vmax_list=DC_MID_VMAX)

        _banner("  SEQUENZA PROF COMPLETATA", "#")
        _summary("Riepilogo sequenza (1 singola-low -> 2 doppia-low -> 3 doppia-mid):", all_res, col="stato")
        print(f"\n  [{time.strftime('%H:%M:%S')}]  Tempo totale sequenza: {_fmt_dt(time.time() - t0)}\n", flush=True)
    finally:
        sys.stdout = orig_out
        logf.close()
    return all_res


def main():
    stage = os.environ.get("STAGE", "prof")   # prof(DEFAULT) | all | fcm | fcm_low | ppo | ppo_mid | ppo_low | r10 | dc | dc2 | dc_mid | dc_low
    t_start = time.time()
    fcm_res, ppo_res = [], []

    stage_labels = {
        "all":     "FCM + PPO (v_max 1→5)",
        "fcm":     "FCM Adam (passo 10→50 m)",
        "fcm_low": "FCM Adam BASSE velocità (passo 7/4/1 m ≈ v_max 0.7/0.4/0.1)",
        "fcm_mid": "FCM Adam INTERMEDI (passo 12/15/17/19 m ≈ v_max 1.2/1.5/1.7/1.9)",
        "ppo":     "PPO velocità adattiva (v_max 1→5, catena)",
        "ppo_mid": "PPO v_max INTERMEDI (1.2/1.5/1.7/1.9, catena da v1)",
        "ppo_low": "PPO v_max BASSI (0.7/0.4/0.1, catena discendente da v1)",
        "r10":     "PPO raggio 10 m (modulazione forzata)",
        "dc":      "Doppia corona (warm-start da v1)",
        "dc2":     "Doppia corona a ogni v_max (opzione 2)",
        "dc_mid":  "Doppia corona INTERMEDI (1.2/1.5/1.7/1.9, warm-start da single_vX)",
        "dc_low":  "Doppia corona BASSI (0.7/0.4/0.1, warm-start da single_vX)",
        "noform":  "Agente SOLO (no-formation): finetuning singola corona v2, formazione azzerata",
        "prof":    "SEQUENZA PROF: 1) singola 0.7/0.4/0.1  2) doppia 0.7/0.4/0.1  3) doppia 1.2/1.5/1.7/1.9",
    }
    _banner("  SWEEP ADATTIVI   (spawn distribuito, no video)", "#")
    label = stage_labels.get(stage, stage.upper())
    if stage in ("all", "fcm", "fcm_low", "fcm_mid"):
        print(f"  Stadio: {label}   |   episodi/scenario: {N_EPISODES}   |   sorgenti test: SRC107–SRC132")
    else:
        print(f"  Stadio: {label}   |   SOLO TRAINING (le inferenze si fanno dopo, separatamente)")

    if stage in ("all", "fcm"):
        _banner("  ###  STADIO 1/2 — SWEEP FCM ADAM (passo 10→50 m, sensor 50 m)  ###", "#")
        fcm_res = fcm_sweep()

    if stage == "fcm_low":
        _banner("  ###  SWEEP FCM ADAM BASSE VELOCITÀ (passo 7/4/1 m, sensor 50 m)  ###", "#")
        fcm_res = fcm_sweep(lrs=[7, 4, 1])

    if stage == "fcm_mid":
        _banner("  ###  SWEEP FCM ADAM INTERMEDI (passo 12/15/17/19 m, sensor 50 m)  ###", "#")
        fcm_res = fcm_sweep(lrs=[12, 15, 17, 19])

    if stage in ("all", "ppo"):
        _banner("  ###  STADIO 2/2 — SWEEP PPO VELOCITÀ ADATTIVA (v_max 1→5 m/s, catena)  ###", "#")
        ppo_res = ppo_sweep()

    if stage == "ppo_mid":
        _banner("  ###  SWEEP PPO v_max INTERMEDI (1.2/1.5/1.7/1.9, catena da v1)  ###", "#")
        ppo_res = ppo_intermediate_sweep()

    if stage == "ppo_low":
        _banner("  ###  SWEEP PPO v_max BASSI (0.7/0.4/0.1, catena discendente da v1)  ###", "#")
        ppo_res = ppo_low_sweep()

    if stage == "r10":
        _banner("  ###  ESPERIMENTO RAGGIO 10 m (modulazione forzata)  ###", "#")
        ppo_res = radius10_experiment()

    if stage == "dc":
        _banner("  ###  ESPERIMENTO DOPPIA CORONA (warm-start da v1)  ###", "#")
        ppo_res = dualcorona_experiment()

    if stage == "dc2":
        _banner("  ###  DOPPIA CORONA A OGNI v_max (opzione 2: warm-start da single_vX)  ###", "#")
        ppo_res = dualcorona_chain_experiment()

    if stage == "dc_mid":
        _banner("  ###  DOPPIA CORONA v_max INTERMEDI (1.2/1.5/1.7/1.9, warm-start da single_vX)  ###", "#")
        ppo_res = dualcorona_chain_experiment(vmax_list=DC_MID_VMAX)

    if stage == "dc_low":
        _banner("  ###  DOPPIA CORONA v_max BASSI (0.7/0.4/0.1, warm-start da single_vX)  ###", "#")
        ppo_res = dualcorona_chain_experiment(vmax_list=DC_LOW_VMAX)

    if stage == "noform":
        _banner("  ###  AGENTE SOLO (NO-FORMATION): finetuning singola corona v2, sensori azzerati  ###", "#")
        ppo_res = noformation_experiment()

    if stage == "prof":
        ppo_res = run_prof_training_sequence()

    # --- Riepilogo finale ---
    _banner("  RIEPILOGO FINALE", "#")
    if fcm_res:
        _summary("FCM Adam — passo crescente:", fcm_res, col="SR")
    if ppo_res:
        _summary("PPO — velocità adattiva (v_max crescente):", ppo_res, col="stato")
    print(f"\n  Tempo totale: {_fmt_dt(time.time() - t_start)}\n")


if __name__ == "__main__":
    main()
