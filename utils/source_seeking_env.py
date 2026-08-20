"""
HYDRAS Source Seeking - Gymnasium Environment
Ambiente Gymnasium completo per l'addestramento di agenti RL
nella localizzazione di sorgenti di inquinante in ambienti marini.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from .data_loader import (
    ConcentrationField, DataManager, DomainConfig, WindData, CurrentData
)


@dataclass
class AgentState:
    """Stato corrente dell'agente."""
    x: float  # Posizione x (UTM)
    y: float  # Posizione y (UTM)
    vx: float = 0.0  # Velocità x
    vy: float = 0.0  # Velocità y
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


@dataclass
class SourceSeekingConfig:
    """Configurazione dell'ambiente."""
    # Domain
    xmin: float = 619000
    xmax: float = 622000
    ymin: float = 4794500
    ymax: float = 4797000
    resolution: float = 10

    # Agent
    max_velocity: float = 1.0  # m/s (velocità AUV)

    # Memory - ultimi N step (concentrazioni passate + spostamenti passati)
    memory_length: int = 9  # 9 step passati

    # Episode
    dt: float = 10.0  # s  (con max_velocity=1 m/s -> spostamento max 10 m/step)
    max_steps: int = 1080  # 3 ore: 10800s / 10s = 1080 steps

    # Spawn
    spawn_start_frame: int = 352  # frame di partenza (25% della simulazione, Chunk 0) - sovrascitto da chunk_id
    spawn_conc_threshold: float = 0.5  # soglia minima concentrazione per spawn
    spawn_min_distance: float = 300.0   # m - floor assoluto di d_min (no spawn troppo facili)
    spawn_min_percentile: float = 50.0  # percentile (mediana) delle distanze del plume per d_min (scala con la diffusione)
    spawn_max_distance: float = 3000.0  # m - limite superiore di d_min..d_max (disco esterno)
    spawn_min_width: float = 500.0      # m - ampiezza minima dell'intervallo (d_min, d_max)
    chunk_id: int = 0  # supportati: 0 = spawn @1/4, 1 = spawn @1/2, 2 = spawn @3/4 della simulazione

    # Reward
    source_distance_threshold: float = 50  # m (intorno di successo)
    source_found_reward: float = 100.0
    step_penalty: float = -0.1
    boundary_penalty: float = -10.0
    distance_reward_multiplier: float = 100.0  # Moltiplicatore per reward distanza (scala per potenziale quadratico)
    concentration_gradient_reward_positive: float = 0.05  # reward per aumento concentrazione
    concentration_gradient_reward_negative: float = -0.05  # penalty per diminuzione concentrazione
    
    # Wind alignment reward (seguire il vento controcorrente verso sorgente)
    wind_alignment_reward: float = 0.05  # reward se movimento è controcorrente al vento
    wind_alignment_penalty: float = -0.05  # penalty se movimento è a favore del vento
    # Current alignment reward (controcorrente = upstream = verso sorgente)
    current_alignment_reward: float = 0.05
    current_alignment_penalty: float = -0.05
    # Stagnation penalty (scia sbagliata / esplorazione circolare)
    stagnation_window: int = 50          # step da osservare
    stagnation_distance_threshold: float = 20.0  # m - miglioramento minimo richiesto nella finestra
    stagnation_penalty: float = -0.5    # penalità se nessun progresso nella finestra
    # Stagnation penalty direzionale (oscillazione ripetuta sullo stesso asse)
    directional_stagnation_threshold: float = 0.15  # efficienza < soglia → oscillazione
    directional_stagnation_penalty: float = -1.0    # penalità per oscillazione direzionale

    # Retreat penalty multiplier: allontanarsi dalla sorgente costa N× di più che avvicinarsi
    # (nella stessa zona di distanza). Scoraggia il bypass della sorgente.
    retreat_penalty_multiplier: float = 3.0

    # Reward mode:
    #   "full"                 → reward completa
    #   "base"                 → reward semplificata: senza zone_mult, retreat_asymmetry, stagnazione
    #   "base_no_wind_reward"  → come "base" ma senza wind/current alignment (modello di riferimento)
    reward_mode: str = "full"

    # Distanza di misurazione dei sensori direzionali (8 direzioni).
    # Variabile per lo sweep sperimentale; il modello base è trainato con 20m.
    sensor_range: float = 20.0
    sensor_range_2: Optional[float] = None  # seconda corona di sensori; None = disabilitata (116-dim obs)

    # "Agente solo" (senza formazione): azzera i sensori direzionali della formazione
    # (correnti + storia), lasciando all'agente solo la concentrazione nella propria
    # posizione, le memorie, il vento/corrente e le feature posizionali. Serve
    # all'esperimento sull'importanza della formazione (Cap. Explainability). La forma
    # dell'osservazione NON cambia (i canali sono azzerati), così il modello resta
    # warm-startabile da uno addestrato con la formazione.
    mask_formation: bool = False

    @classmethod
    def from_config(cls, config: dict, chunk_id: int = 0) -> 'SourceSeekingConfig':
        """Costruisce SourceSeekingConfig da un dict YAML (output di yaml.safe_load)."""
        domain = config.get('domain', {})
        agent = config.get('agent', {})
        env = config.get('environment', {})
        reward = env.get('reward', {})
        spawn = env.get('spawn', {})
        return cls(
            xmin=domain.get('xmin', 619000),
            xmax=domain.get('xmax', 622000),
            ymin=domain.get('ymin', 4794500),
            ymax=domain.get('ymax', 4797000),
            resolution=domain.get('grid_resolution', 10),
            max_velocity=agent.get('max_velocity', 1.0),
            memory_length=agent.get('memory_length', 9),
            n_discrete_actions=agent.get('n_discrete_actions', 8),
            n_velocity_levels=agent.get('n_velocity_levels', 1),
            sensor_range=float(agent.get('sensor_range', 20.0)),
            sensor_range_2=float(agent['sensor_range_2']) if 'sensor_range_2' in agent else None,
            mask_formation=bool(agent.get('mask_formation', False)),
            dt=env.get('dt', 10),
            max_steps=env.get('max_episode_steps', 1080),
            chunk_id=chunk_id,
            source_found_reward=reward.get('source_reached_bonus', 100.0),
            step_penalty=reward.get('step_penalty', -0.1),
            boundary_penalty=reward.get('boundary_penalty', -10.0),
            source_distance_threshold=reward.get('distance_threshold', 50.0),
            distance_reward_multiplier=reward.get('distance_reward_multiplier', 100.0),
            retreat_penalty_multiplier=reward.get('retreat_penalty_multiplier', 3.0),
            land_proximity_threshold=reward.get('land_proximity_threshold', 30.0),
            land_proximity_penalty_max=reward.get('land_proximity_penalty_max', -15.0),
            concentration_gradient_reward_positive=reward.get('concentration_gradient_reward_positive', 0.05),
            concentration_gradient_reward_negative=reward.get('concentration_gradient_reward_negative', -0.05),
            wind_alignment_reward=reward.get('wind_alignment_reward', 0.05),
            wind_alignment_penalty=reward.get('wind_alignment_penalty', -0.05),
            current_alignment_reward=reward.get('current_alignment_reward', 0.05),
            current_alignment_penalty=reward.get('current_alignment_penalty', -0.05),
            stagnation_window=reward.get('stagnation_window', 15),
            stagnation_distance_threshold=reward.get('stagnation_distance_threshold', 20.0),
            stagnation_penalty=reward.get('stagnation_penalty', -1.5),
            directional_stagnation_threshold=reward.get('directional_stagnation_threshold', 0.15),
            directional_stagnation_penalty=reward.get('directional_stagnation_penalty', -0.5),
            reward_mode=reward.get('reward_mode', 'full'),
            spawn_min_land_distance=spawn.get('min_land_distance', 50.0),
            spawn_start_frame=spawn.get('start_frame', 352),
            spawn_conc_threshold=spawn.get('conc_threshold', 0.5),
            spawn_min_distance=spawn.get('min_distance', 300.0),
            spawn_min_percentile=spawn.get('min_percentile', 50.0),
            spawn_max_distance=spawn.get('max_distance', 3000.0),
            spawn_min_width=spawn.get('min_width', 500.0),
        )

    # Land avoidance
    land_proximity_threshold: float = 30.0  # m - distanza dalla terra per penalità progressiva
    land_proximity_penalty_max: float = -15.0  # penalità massima per vicinanza terra
    spawn_min_land_distance: float = 50.0  # m - distanza minima dalla terra per spawn

    # Action
    action_type: str = "discrete"  # "discrete" (N/S/E/W + diagonali)
    n_discrete_actions: int = 8  # 8 direzioni: 4 cardinali + 4 diagonali
    n_velocity_levels: int = 1   # K livelli di velocità in (0, max_velocity]; 1 = passo fisso.
                                 # Con K>1 lo spazio azioni diventa Discrete(8*K): l'agente
                                 # sceglie direzione E velocità. Livelli = max_velocity*{1/K..K/K}.


class SourceSeekingEnv(gym.Env):
    """
    Ambiente Gymnasium per il source seeking di inquinanti marini.

    L'agente (AUV) deve navigare in un campo di concentrazione
    per trovare la sorgente dell'inquinante.

    Observation Space (116 valori):
        - 1  concentrazione corrente
        - 9  concentrazioni passate
        - 18 spostamenti passati (Δx, Δy) in metri
        - 8  sensori concentrazione @ sensor_range (8 direzioni)
        - 72 sensori concentrazione passati (9 timestep x 8 direzioni)
        - 2  componenti vento (u, v)
        - 2  componenti corrente (u, v)
        - 2  vettore (Δx, Δy) verso posizione di max concentrazione vista
        - 1  valore di max concentrazione vista
        - 1  step dall'ultimo contatto col plume (normalizzato)
        (normalizzazione delegata a VecNormalize)

    Action Space:
        - Discrete(8): N, S, E, W, NE, SE, NW, SW

    Reward:
        - Bonus sorgente raggiunta + time bonus (terminale dominante)
        - Reward distanza zone-based con retreat asymmetry (segnale dominante)
        - Reward gradiente concentrazione (±0.05)
        - Reward allineamento vento/corrente (±0.05)
        - Penalità per step (-0.1)
        - Penalità bordi (-10), terra, stagnation
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[SourceSeekingConfig] = None,
        concentration_field: Optional[ConcentrationField] = None,
        wind_data: Optional[WindData] = None,
        current_data: Optional[CurrentData] = None,
        source_id: str = "SRC001",
        render_mode: Optional[str] = None,
        data_dir: Optional[str] = None,
        randomize_field: bool = False,
        data_manager: Optional['DataManager'] = None,
        wind_mapping: Optional[Dict[str, str]] = None,
        current_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Args:
            config: Configurazione dell'ambiente
            concentration_field: Campo di concentrazione pre-caricato
            wind_data: Dati di vento pre-caricati
            current_data: Dati di corrente pre-caricati
            source_id: ID della sorgente (es. 'SRC001', 'SRC042', 'SRC132')
            render_mode: Modalità di rendering
            data_dir: Directory con file NC (per randomize_field)
            randomize_field: Se True, sceglie un NC random ad ogni reset
            data_manager: DataManager per caricamenti dinamici (opzionale)
            wind_mapping: Dict con mappatura run_id -> wind_filename (opzionale)
            current_mapping: Dict con mappatura run_id -> current_filename (opzionale)
            **kwargs: Parametri aggiuntivi per la configurazione
        """
        super().__init__()

        self.config = config or SourceSeekingConfig(**kwargs)
        self.source_id = source_id
        self.render_mode = render_mode
        self.randomize_field = randomize_field
        self._current_run_id = None  # Salvato dopo reset() con concentrazione random
        
        # Wind e current mapping per caricamenti dinamici
        self.wind_mapping = wind_mapping or {}
        self.current_mapping = current_mapping or {}
        
        # Setup dominio
        self.domain = DomainConfig(
            xmin=self.config.xmin,
            xmax=self.config.xmax,
            ymin=self.config.ymin,
            ymax=self.config.ymax,
            resolution=self.config.resolution
        )

        # Data Manager per gestione NC files
        self._data_manager: Optional[DataManager] = data_manager
        if data_manager is None and data_dir:
            self._data_manager = DataManager(
                data_dir=data_dir,
                domain_config=self.domain,
                preload_all=False,  # NON precaricare - carica on-demand per risparmiare RAM
            )

        # Campo di concentrazione
        if concentration_field is not None:
            self.field = concentration_field
        else:
            self._init_field()

        # Posizione sorgente (sempre da coordinate hardcodate nel campo)
        if self.field.source_position is not None:
            self.source_position = np.array(self.field.source_position)
        else:
            raise ValueError(
                f"ConcentrationField non ha source_position impostata. "
                f"Controlla che il file NC contenga S1/S2/S3 nel nome "
                f"o passa un campo con source_position."
            )

        # Dati di vento e corrente
        self.wind_data = wind_data
        self.current_data = current_data

        # Stato agente
        self.state: Optional[AgentState] = None
        self.steps = 0
        self.prev_concentration = 0.0
        self.prev_distance = 0.0
        self._prev_position: Optional[np.ndarray] = None

        # Memory buffer per concentrazioni passate (usata nell'osservazione)
        self._concentration_memory: List[float] = [0.0] * self.config.memory_length

        # Memory buffer per spostamenti passati (Δx, Δy)
        self._displacement_memory: List[Tuple[float, float]] = [(0.0, 0.0)] * self.config.memory_length

        # Memory buffer per concentrazioni direzionali passate (9 timestep x 8 direzioni)
        # Ogni elemento è una lista di 8 float (uno per direzione)
        self._directional_conc_memory: List[List[float]] = [[0.0] * 8 for _ in range(self.config.memory_length)]
        if self.config.sensor_range_2 is not None:
            self._directional_conc_memory_2: List[List[float]] = [[0.0] * 8 for _ in range(self.config.memory_length)]

        # Cache wind/current components: popolata una volta per step, letta da _compute_reward e _get_observation
        self._cached_wind_components: Optional[Tuple[float, float]] = None
        self._cached_current_components: Optional[Tuple[float, float]] = None

        # Buffer distanze per stagnation penalty
        self._distance_history: List[float] = []

        # Feature 3: posizione e valore della concentrazione massima rilevata finora
        self._max_concentration_seen: float = 0.0
        self._max_conc_position: Tuple[float, float] = (0.0, 0.0)

        # Feature 4: step dall'ultimo contatto con il plume
        self._steps_since_plume_contact: int = 0

        # Allowed sources per curriculum learning (sarà impostato dal CurriculumCallback)
        # Default vuoto: richiede che sia impostato dal training script
        self.allowed_sources: List[str] = []

        # Weighted scenario sampling: dict "V2_1" → probability (impostato dal training script)
        self.scenario_weights: Optional[Dict[str, float]] = None

        # History per analisi
        self.trajectory: List[np.ndarray] = []
        self.concentration_history: List[float] = []

        # Setup spaces
        self._setup_observation_space()
        self._setup_action_space()

        # Rendering
        self._fig = None
        self._ax = None

    def set_scenario_weights(self, weights: Optional[Dict[str, float]]) -> None:
        """Aggiorna i pesi per il campionamento scenari (chiamato dal callback di curriculum)."""
        self.scenario_weights = weights

    def _init_field(self):
        """Inizializza il campo di concentrazione."""
        if not self._data_manager:
            raise ValueError(
                "Nessun DataManager configurato. "
                "Passa data_dir al costruttore o concentration_field direttamente."
            )
        if self.randomize_field:
            self.field, self.source_id = self._data_manager.get_random_field()
        else:
            self.field = self._data_manager.get_concentration_field(source_id=self.source_id)

    def _setup_observation_space(self):
        """Configura lo spazio delle osservazioni.

        Osservazione (196 valori):
        - 1   concentrazione corrente
        - 9   concentrazioni passate (memory_length)
        - 18  spostamenti passati (9 * 2)
        - 8   sensori corona 1 @ sensor_range    (correnti)
        - 8   sensori corona 2 @ sensor_range_2  (correnti)
        - 72  sensori corona 1 passati (9 timestep x 8 direzioni)
        - 72  sensori corona 2 passati (9 timestep x 8 direzioni)
        - 2   vento corrente (u, v)
        - 2   corrente marina (u, v)
        - 4   max conc position + value + steps since plume
        """
        dual = self.config.sensor_range_2 is not None
        obs_dim = (1 + self.config.memory_length + self.config.memory_length * 2
                   + 8 + (8 if dual else 0)
                   + self.config.memory_length * 8 + (self.config.memory_length * 8 if dual else 0)
                   + 2 + 2 + 4)
        # single ring: 1+9+18+8+72+2+2+4 = 116; dual ring: +8+72 = 196

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _setup_action_space(self):
        """Configura lo spazio delle azioni.
        
        8 azioni discrete:
          0 = Nord  (+y)
          1 = Sud   (-y)
          2 = Est   (+x)
          3 = Ovest (-x)
          4 = NordEst (+x,+y)
          5 = SudEst  (+x,-y)
          6 = NordOvest (-x,+y)
          7 = SudOvest  (-x,-y)
        """
        self.action_space = spaces.Discrete(
            self.config.n_discrete_actions * self.config.n_velocity_levels)

    def spawn_disk_bounds(self) -> Tuple[float, float]:
        """(d_min, d_max) del disco di spawn per il frame/scenario corrente.

        Rispecchia il calcolo dei bordi in _spawn_on_plume; usato per disegnare il disco
        virtuale di spawn nelle spawn map. Imposta il frame dal chunk_id (come reset()).
        """
        if self.field.n_timesteps > 1 and self.config.chunk_id in (0, 1, 2):
            nt = self.field.n_timesteps
            frame = {0: nt // 4, 1: nt // 2, 2: (nt * 3) // 4}[self.config.chunk_id]
            self._start_time_idx = min(frame, nt - 1)
            self.field.set_time(self._start_time_idx)
        elif self.field.n_timesteps > 1:
            self.field.set_time(getattr(self, '_start_time_idx', self.config.spawn_start_frame))

        field_data = np.nan_to_num(self.field.get_current_field(), nan=0.0)
        vi = np.where(field_data > self.config.spawn_conc_threshold)
        if len(vi[0]) == 0:
            raise ValueError("No plume at spawn frame.")
        y_coords = self.field.y_coords[vi[0]]
        x_coords = self.field.x_coords[vi[1]]
        distances = np.sqrt((x_coords - self.source_position[0])**2 +
                            (y_coords - self.source_position[1])**2)
        land_safe = np.array([
            self._min_distance_to_land(x_coords[i], y_coords[i]) >= self.config.spawn_min_land_distance
            for i in range(len(x_coords))
        ])
        if np.any(land_safe):
            distances = distances[land_safe]

        d_cap = float(self.config.spawn_max_distance)
        d_disk = distances[distances <= d_cap]
        if d_disk.size == 0:
            d_disk = distances
        d_max = min(d_cap, float(d_disk.max()))
        d_min = max(float(self.config.spawn_min_distance),
                    float(np.percentile(d_disk, self.config.spawn_min_percentile)))
        if d_max - d_min < self.config.spawn_min_width:
            d_min = max(float(self.config.spawn_min_distance), d_max - self.config.spawn_min_width)
        if d_min >= d_max:
            d_min = float(d_disk.min())
        return d_min, d_max

    def _spawn_on_plume(self) -> Tuple[float, float]:
        """Spawna dentro il plume nel disco [d_min, d_max] con distribuzione uniforme
        per distanza dalla sorgente.

        Logica:
        1. Si raccolgono tutte le celle del plume (conc >= soglia) sufficientemente
           lontane dalla terra.
        2. Si fissa una distanza massima ``spawn_max_distance`` (es. 2.5 km) e si calcola,
           per lo specifico scenario, una distanza minima ``d_min`` = max(floor di config,
           minima distanza realmente disponibile).
        3. Si campiona una distanza obiettivo in modo UNIFORME su [d_min, d_max] e si
           sceglie a caso una cella valida prossima a quella distanza. Questo evita che
           l'agente parta sempre dalla stessa zona (es. sempre l'anello più esterno) e
           distribuisce i punti di partenza su tutto l'intervallo.
        """
        # Imposta il timestep al frame configurato (quello calcolato dal chunk_id nel reset())
        if self.field.n_timesteps > 1:
            spawn_frame = getattr(self, '_start_time_idx', self.config.spawn_start_frame)
            self.field.set_time(spawn_frame)

        # Ottieni tutti i punti dentro il plume
        field_data = np.nan_to_num(self.field.get_current_field(), nan=0.0)
        plume_mask = field_data > self.config.spawn_conc_threshold
        valid_indices = np.where(plume_mask)

        if len(valid_indices[0]) == 0:
            raise ValueError(
                f"No plume found at spawn frame {spawn_frame} "
                f"with threshold {self.config.spawn_conc_threshold}."
            )

        # Coordinate di tutti i punti nel plume e distanze dalla sorgente
        y_coords = self.field.y_coords[valid_indices[0]]
        x_coords = self.field.x_coords[valid_indices[1]]
        distances = np.sqrt(
            (x_coords - self.source_position[0])**2 +
            (y_coords - self.source_position[1])**2
        )

        # Filtra punti troppo vicini alla terra
        land_safe_mask = np.array([
            self._min_distance_to_land(x_coords[i], y_coords[i]) >= self.config.spawn_min_land_distance
            for i in range(len(x_coords))
        ])
        if np.any(land_safe_mask):
            x_coords = x_coords[land_safe_mask]
            y_coords = y_coords[land_safe_mask]
            distances = distances[land_safe_mask]

        # Disco di spawn: [d_min, d_max].
        #  - d_max è un limite superiore fisso da config (es. 3 km), abbassato alla
        #    massima estensione del plume se questo non arriva così lontano.
        #  - d_min SCALA con la diffusione del plume dalla sorgente: è un percentile
        #    basso delle distanze delle celle di plume. Plume molto diffuso → le celle
        #    sono distribuite lontano → percentile alto → d_min alta; plume compatto →
        #    celle vicine → percentile basso → d_min bassa (una d_min alta sarebbe
        #    inutile/irraggiungibile). Un floor assoluto evita lo spawn nel raggio di
        #    successo.
        d_cap = float(self.config.spawn_max_distance)
        in_disk = distances <= d_cap
        if not np.any(in_disk):
            in_disk = np.ones_like(distances, dtype=bool)   # plume non raggiunge d_cap
        x_disk = x_coords[in_disk]
        y_disk = y_coords[in_disk]
        d_disk = distances[in_disk]

        d_max = min(d_cap, float(d_disk.max()))
        d_min = float(np.percentile(d_disk, self.config.spawn_min_percentile))
        d_min = max(float(self.config.spawn_min_distance), d_min)
        # L'intervallo non deve essere troppo stretto, altrimenti gli agenti partono
        # quasi tutti alla stessa distanza: garantisci un'ampiezza minima abbassando d_min
        # (fino al floor) quando d_max - d_min è insufficiente.
        if d_max - d_min < self.config.spawn_min_width:
            d_min = max(float(self.config.spawn_min_distance),
                        d_max - self.config.spawn_min_width)
        if d_min >= d_max:                                  # safety: garantisci d_min < d_max
            d_min = float(d_disk.min())

        # Campionamento uniforme per distanza: scegli una distanza obiettivo uniforme in
        # [d_min, d_max], poi una cella valida prossima a quella distanza.
        for _ in range(20):
            target_d = float(self.np_random.uniform(d_min, d_max))
            band = np.abs(d_disk - target_d) <= 50.0   # tolleranza ±50 m
            cand = np.where(band)[0]
            if len(cand) == 0:
                cand = np.array([int(np.argmin(np.abs(d_disk - target_d)))])
            idx = int(self.np_random.choice(cand))
            x = float(x_disk[idx]) + float(self.np_random.uniform(-5, 5))
            y = float(y_disk[idx]) + float(self.np_random.uniform(-5, 5))
            if (self.config.xmin <= x <= self.config.xmax and
                    self.config.ymin <= y <= self.config.ymax and
                    self.field.get_concentration(x, y) >= self.config.spawn_conc_threshold and
                    self._min_distance_to_land(x, y) >= self.config.spawn_min_land_distance):
                return (x, y)

        # Fallback: cella esatta della griglia più vicina a una distanza uniforme
        target_d = float(self.np_random.uniform(d_min, d_max))
        idx = int(np.argmin(np.abs(d_disk - target_d)))
        return (float(x_disk[idx]), float(y_disk[idx]))


    def _get_observation(self) -> np.ndarray:
        """Costruisce il vettore di osservazione (116 valori) RAW.

        Struttura:
        - [0]       : concentrazione corrente
        - [1:10]    : 9 concentrazioni passate
        - [10:28]   : 9 spostamenti passati (Δx, Δy) in metri
        - [28:36]   : 8 sensori corona 1 @ sensor_range    (correnti)
        - [36:44]   : 8 sensori corona 2 @ sensor_range_2  (correnti)
        - [44:116]  : 9*8=72 sensori corona 1 passati
        - [116:188] : 9*8=72 sensori corona 2 passati
        - [188:190] : vento corrente (u, v) in m/s
        - [190:192] : corrente corrente (u, v) in m/s
        - [192:194] : (Δx, Δy) verso posizione di concentrazione massima vista (m)
        - [194]     : valore di concentrazione massima vista
        - [195]     : step dall'ultimo contatto col plume (normalizzato su max_steps)
        """
        obs = []

        # Concentrazione al centro (1 valore)
        center_conc = self.field.get_concentration(self.state.x, self.state.y)
        obs.append(center_conc)

        # 9 concentrazioni passate
        for past_conc in self._concentration_memory:
            obs.append(past_conc)

        # 9 spostamenti passati (Δx, Δy) in metri
        for dx, dy in self._displacement_memory:
            obs.append(dx)
            obs.append(dy)

        x, y = self.state.x, self.state.y
        
        # Corona 1 corrente: 8 sensori @ sensor_range
        obs.extend(self._compute_directional_sensors(self.config.sensor_range))

        # Corona 2 corrente (opzionale): 8 sensori @ sensor_range_2
        if self.config.sensor_range_2 is not None:
            obs.extend(self._compute_directional_sensors(self.config.sensor_range_2))

        # Corona 1 storica: 9 * 8 = 72 valori
        for timestep_sensors in self._directional_conc_memory:
            obs.extend(timestep_sensors)

        # Corona 2 storica (opzionale): 9 * 8 = 72 valori
        if self.config.sensor_range_2 is not None:
            for timestep_sensors in self._directional_conc_memory_2:
                obs.extend(timestep_sensors)

        # Vento corrente (u, v)
        if self._cached_wind_components is not None:
            obs.append(self._cached_wind_components[0])
            obs.append(self._cached_wind_components[1])
        elif self.wind_data is not None:
            wind_u, wind_v = self.wind_data.get_wind_components()
            obs.append(wind_u)
            obs.append(wind_v)
        else:
            obs.append(0.0)
            obs.append(0.0)

        # Corrente corrente (u, v)
        if self._cached_current_components is not None:
            obs.append(self._cached_current_components[0])
            obs.append(self._cached_current_components[1])
        elif self.current_data is not None:
            current_u, current_v = self.current_data.get_current_components(x, y)
            obs.append(current_u)
            obs.append(current_v)
        else:
            obs.append(0.0)
            obs.append(0.0)

        # Feature 3: vettore verso posizione di concentrazione massima + valore massimo
        obs.append(self._max_conc_position[0] - x)
        obs.append(self._max_conc_position[1] - y)
        obs.append(self._max_concentration_seen)

        # Feature 4: step dall'ultimo contatto col plume (normalizzato)
        obs.append(float(self._steps_since_plume_contact) / self.config.max_steps)

        obs = np.array(obs, dtype=np.float32)
        if self.config.mask_formation:
            # "Agente solo": azzera i sensori della formazione (correnti + storia).
            # I primi 28 valori (conc. propria + 9 conc. passate + 9 spostamenti) e gli
            # ultimi 8 (vento, corrente, feature posizionali) restano intatti; tutto ci\`o
            # che sta in mezzo sono i sensori direzionali (singola o doppia corona).
            obs[28:-8] = 0.0
        return obs

    def _compute_directional_sensors(self, sensor_range: float = None) -> List[float]:
        """Calcola 8 sensori di concentrazione direzionali al raggio dato.

        Args:
            sensor_range: raggio in metri (default: config.sensor_range)
        Returns:
            Lista di 8 float (uno per direzione)
        """
        if sensor_range is None:
            sensor_range = self.config.sensor_range
        conc_sensors = []
        x, y = self.state.x, self.state.y

        for action_idx in range(8):
            dx_dir, dy_dir = self._ACTION_MAP[action_idx]
            sense_x = x + dx_dir * sensor_range
            sense_y = y + dy_dir * sensor_range
            # Se il punto è su terra o fuori dominio, concentrazione = 0
            out_of_bounds = (sense_x < self.config.xmin or sense_x > self.config.xmax or
                             sense_y < self.config.ymin or sense_y > self.config.ymax)
            if out_of_bounds or self.field.is_land(sense_x, sense_y):
                conc_sensors.append(0.0)
            else:
                conc = self.field.get_concentration(sense_x, sense_y)
                conc_sensors.append(float(np.nan_to_num(conc, nan=0.0)))

        return conc_sensors

    @staticmethod
    def _distance_zone_multiplier(distance: float) -> float:
        """Moltiplicatore inversamente proporzionale alla distanza dalla sorgente.

        mult = max(1.0, 2000 / distance)

        Cresce in modo continuo e senza cap: più l'agente è vicino,
        più il reward di approccio (e la penalità di retreat) aumentano.
        Flat a 1.0× oltre i 2000 m; a 100 m vale 20×, a 50 m vale 40×.
        """
        d = max(distance, 1.0)  # evita divisione per zero
        return float(max(1.0, 2000.0 / d))

    # Mappa azioni discrete: 8 direzioni (4 cardinali + 4 diagonali)
    # Usa cos(45°) = sin(45°) = 1/√2 ≈ 0.7071 per normalizzazione corretta
    _DIAG = 1.0 / np.sqrt(2.0)
    # Cascata di ring (min_dist, max_dist) provati in ordine per lo spawn.
    # Si scende al ring successivo solo se nessun punto del plume rientra in quello corrente.
    _SPAWN_DISTANCE_CASCADE: List[Tuple[float, float]] = [
        (2000, 2500),
        (1500, 2000),
        (1000, 1500),
        (500,  1000),
        (250,   500),
        (50,    250),
        (0,      50),
    ]

    _ACTION_MAP = {
        0: (0.0,   1.0),       # Nord  (+y)
        1: (0.0,  -1.0),       # Sud   (-y)
        2: (1.0,   0.0),       # Est   (+x)
        3: (-1.0,  0.0),       # Ovest (-x)
        4: (_DIAG,  _DIAG),     # NordEst
        5: (_DIAG, -_DIAG),     # SudEst
        6: (-_DIAG, _DIAG),     # NordOvest
        7: (-_DIAG,-_DIAG),     # SudOvest
    }

    def _decode_action(self, action) -> Tuple[int, float]:
        """Decodifica l'azione in (indice direzione, velocità).

        Con n_velocity_levels = K, l'azione appartiene a {0, ..., 8K-1} ed è
        codificata come  action = dir_idx * K + vel_idx:
          - dir_idx = action // K   -> una delle 8 direzioni
          - vel_idx = action %  K   -> livello di velocità
        La velocità è la (vel_idx+1)-esima frazione di max_velocity:
          velocity = (vel_idx + 1) / K * max_velocity   (K-partizione di (0, v_max]).
        Con K = 1 si riduce al passo fisso (velocity = max_velocity).
        """
        K = self.config.n_velocity_levels
        action_int = int(action)
        dir_idx = action_int // K
        vel_idx = action_int % K
        velocity = (vel_idx + 1) / K * self.config.max_velocity
        return dir_idx, velocity

    def _apply_action(self, action):
        """Applica l'azione all'agente: direzione (8) + velocità (K livelli).

        Spostamento = direzione * velocità * dt, con velocità scelta dall'agente
        in (0, max_velocity] tramite il livello codificato nell'azione.
        """
        dir_idx, velocity = self._decode_action(action)
        dx_dir, dy_dir = self._ACTION_MAP[dir_idx]

        vx = dx_dir * velocity
        vy = dy_dir * velocity

        # Aggiorna stato
        self.state.vx = vx
        self.state.vy = vy
        self.state.x += vx * self.config.dt
        self.state.y += vy * self.config.dt



    def _check_boundary(self) -> bool:
        """Verifica se l'agente è fuori dal dominio."""
        return (
            self.state.x < self.config.xmin or
            self.state.x > self.config.xmax or
            self.state.y < self.config.ymin or
            self.state.y > self.config.ymax
        )

    def _check_on_land(self) -> bool:
        """Verifica se l'agente è sulla terra (usa land_mask del campo)."""
        return getattr(self, '_on_land', False)

    def _is_time_varying(self) -> bool:
        """
        Verifica se il campo di concentrazione è time-varying.
        La concentrazione guida il tempo della simulazione; vento e corrente
        vengono poi sincronizzati in minuti reali.
        """
        return self.field.n_timesteps > 1

    def _check_source_reached(self) -> bool:
        """Verifica se l'agente ha raggiunto la sorgente.

        Considera sia la posizione corrente, sia il segmento tra posizione
        precedente e corrente: evita falsi "quasi-hit" dovuti al campionamento
        discreto degli step (10 m) vicino alla soglia di successo.
        """
        if self.state is None:
            return False

        threshold = float(self.config.source_distance_threshold)
        sx, sy = float(self.source_position[0]), float(self.source_position[1])
        cx, cy = float(self.state.x), float(self.state.y)

        # Check standard sulla posizione corrente.
        distance = np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
        if distance <= threshold:
            return True

        # Check continuo sul segmento precedente->corrente.
        prev_pos = self._prev_position
        if prev_pos is None:
            return False

        x0, y0 = float(prev_pos[0]), float(prev_pos[1])
        x1, y1 = cx, cy
        seg_dx = x1 - x0
        seg_dy = y1 - y0
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-12:
            return False

        t = ((sx - x0) * seg_dx + (sy - y0) * seg_dy) / seg_len_sq
        t = float(np.clip(t, 0.0, 1.0))
        closest_x = x0 + t * seg_dx
        closest_y = y0 + t * seg_dy
        closest_dist = np.sqrt((closest_x - sx) ** 2 + (closest_y - sy) ** 2)
        return closest_dist <= threshold

    def action_masks(self) -> np.ndarray:
        """Ritorna una maschera booleana delle azioni valide.
        
        Un'azione è invalida se:
        1. Porta l'agente su terra (land collision)
        2. Porta l'agente fuori dal dominio (boundary)
        
        Usata da MaskablePPO per evitare azioni che causano terminazione.
        
        Returns:
            np.ndarray di shape (n_actions,) con True per azioni valide
        """
        n_actions = self.config.n_discrete_actions * self.config.n_velocity_levels
        if self.state is None:
            # Prima del reset, tutte le azioni sono valide
            return np.ones(n_actions, dtype=bool)

        masks = np.ones(n_actions, dtype=bool)

        # Masking CONGIUNTO: ogni azione (direzione, velocità) è valida solo se il
        # movimento risultante resta nel dominio e non finisce su terra. Una direzione
        # può quindi essere valida a bassa velocità e invalida ad alta velocità.
        for action_idx in range(n_actions):
            dir_idx, velocity = self._decode_action(action_idx)
            step_size = velocity * self.config.dt
            dx_dir, dy_dir = self._ACTION_MAP[dir_idx]
            new_x = self.state.x + dx_dir * step_size
            new_y = self.state.y + dy_dir * step_size

            if (new_x < self.config.xmin or new_x > self.config.xmax or
                new_y < self.config.ymin or new_y > self.config.ymax):
                masks[action_idx] = False
                continue
            if self.field.is_land(new_x, new_y):
                masks[action_idx] = False

        # Se tutte le azioni sono mascherate, evita crash di MaskablePPO
        # e lascia al reward il compito di penalizzare scelte non valide.
        if not masks.any():
            return np.ones(n_actions, dtype=bool)

        return masks

    def _min_distance_to_land(self, x: float, y: float) -> float:
        """Ritorna la distanza minima dalla terra usando la distance map precomputata.
        
        Usa la EDT (Euclidean Distance Transform) calcolata una volta sola
        al caricamento del campo — O(1) invece di O(160) chiamate a is_land().
        
        Returns:
            Distanza in metri dalla terra più vicina (0 se sulla terra, max 100m)
        """
        dist = self.field.get_land_distance(x, y)
        return min(dist, 100.0)  # Cap a 100m per coerenza con il comportamento precedente

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calcola il reward basato su:
        1. Bonus sorgente raggiunta (terminale dominante)
        2. Penalità terra e bordi (terminale)
        3. Reward distanza zone-based + retreat asymmetry (segnale dominante)
        4. Reward gradiente concentrazione (hint ±0.05)
        5. Reward allineamento vento/corrente (hint ±0.05)
        6. Penalità tempo (-0.1 per step)
        7. Penalità progressiva avvicinamento terra
        8. Stagnation penalty (sospesa entro 250m dalla sorgente)
        """
        reward = 0.0
        info = {}

        # Stato corrente
        current_distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )

        # Verifica terra tramite maschera
        on_land = self.field.is_land(self.state.x, self.state.y)
        current_conc = self.field.get_concentration(self.state.x, self.state.y)

        # ============================================================
        # 1. BONUS SORGENTE RAGGIUNTA
        # ============================================================
        if self._check_source_reached():
            time_bonus = max(0, (self.config.max_steps - self.steps) / self.config.max_steps * 50)
            total_bonus = self.config.source_found_reward + time_bonus
            reward += total_bonus
            info['source_found'] = self.config.source_found_reward
            info['time_bonus'] = time_bonus

        # Traccia se l'agente è su terra (per terminazione)
        if on_land:
            self._on_land = True
        else:
            self._on_land = False

        # ============================================================
        # 2. PENALITÀ USCITA DAL DOMINIO
        # ============================================================
        if self._check_boundary():
            reward += self.config.boundary_penalty
            info['boundary'] = self.config.boundary_penalty

        # ============================================================
        # 4. REWARD DISTANZA
        # ============================================================
        reward_mode = self.config.reward_mode
        approaching = current_distance <= self.prev_distance

        if reward_mode == "full":
            # Potenziale quadratico + zone multiplier + retreat asymmetry
            _QUAD_MAX = 3000.0
            raw_delta = (
                (_QUAD_MAX - current_distance) ** 2 - (_QUAD_MAX - self.prev_distance) ** 2
            ) / (_QUAD_MAX ** 2)
            zone_mult = self._distance_zone_multiplier(current_distance)
            if approaching:
                distance_reward = raw_delta * zone_mult * self.config.distance_reward_multiplier
            else:
                distance_reward = (
                    raw_delta * zone_mult
                    * self.config.retreat_penalty_multiplier
                    * self.config.distance_reward_multiplier
                )
            info['zone_multiplier'] = zone_mult
        else:
            # Base: delta distanza lineare, simmetrico, niente zone multiplier
            distance_reward = (
                (self.prev_distance - current_distance)
                * self.config.distance_reward_multiplier / 3000.0
            )
            info['zone_multiplier'] = 1.0

        reward += distance_reward
        info['distance_reward'] = distance_reward
        info['approaching'] = approaching

        # ============================================================
        # 5. REWARD GRADIENTE CONCENTRAZIONE
        # ============================================================
        conc_gradient = current_conc - self.prev_concentration
        if conc_gradient > 0:
            reward += self.config.concentration_gradient_reward_positive
            info['conc_gradient_reward'] = self.config.concentration_gradient_reward_positive
        else:
            reward += self.config.concentration_gradient_reward_negative
            info['conc_gradient_reward'] = self.config.concentration_gradient_reward_negative

        # ============================================================
        # 5b. REWARD ALLINEAMENTO VENTO (controvento = upwind = verso sorgente)
        # ============================================================
        if reward_mode == "base_no_wind_reward":
            info['wind_alignment_reward'] = 0.0
            info['current_alignment_reward'] = 0.0
        elif self._cached_wind_components is not None:
            wind_u, wind_v = self._cached_wind_components
            wind_norm = np.sqrt(wind_u ** 2 + wind_v ** 2)
            if wind_norm > 1e-6:
                alignment = (self.state.vx * wind_u + self.state.vy * wind_v) / wind_norm
                if alignment < 0:
                    reward += self.config.wind_alignment_reward
                    info['wind_alignment_reward'] = self.config.wind_alignment_reward
                elif alignment > 0:
                    reward += self.config.wind_alignment_penalty
                    info['wind_alignment_reward'] = self.config.wind_alignment_penalty
            else:
                info['wind_alignment_reward'] = 0.0
        else:
            info['wind_alignment_reward'] = 0.0

        # ============================================================
        # 5c. REWARD ALLINEAMENTO CORRENTE (controcorrente = upstream = verso sorgente)
        # ============================================================
        if reward_mode == "base_no_wind_reward":
            pass  # già azzerato nel blocco 5b
        elif self._cached_current_components is not None:
            current_u, current_v = self._cached_current_components
            current_norm = np.sqrt(current_u ** 2 + current_v ** 2)
            if current_norm > 1e-6:
                # dot product tra direzione movimento e direzione corrente
                # negativo = controcorrente (upstream) = reward
                alignment = (self.state.vx * current_u + self.state.vy * current_v) / current_norm
                if alignment < 0:
                    reward += self.config.current_alignment_reward
                    info['current_alignment_reward'] = self.config.current_alignment_reward
                elif alignment > 0:
                    reward += self.config.current_alignment_penalty
                    info['current_alignment_reward'] = self.config.current_alignment_penalty
            else:
                info['current_alignment_reward'] = 0.0
        else:
            info['current_alignment_reward'] = 0.0

        # ============================================================
        # 6. PENALITÀ TEMPO (time efficiency)
        # ============================================================
        reward += self.config.step_penalty  # -0.1
        info['time_penalty'] = self.config.step_penalty

        # ============================================================
        # 7. PENALITÀ PROGRESSIVA AVVICINAMENTO TERRA
        # Land proximity penalty (penalità progressiva vicino alla terra)
        dist_to_land = self._min_distance_to_land(self.state.x, self.state.y)
        if (dist_to_land < self.config.land_proximity_threshold 
                and not on_land):
            # Penalità lineare: 0 a threshold, max a 0
            proximity_penalty = self.config.land_proximity_penalty_max * (
                (self.config.land_proximity_threshold - dist_to_land) / self.config.land_proximity_threshold
            )
            reward += proximity_penalty
            info['land_proximity_penalty'] = proximity_penalty

        # ============================================================
        # 8. STAGNATION PENALTY — solo in reward_mode "full"
        # ============================================================
        near_source_search = current_distance < 250.0
        if reward_mode == "full":
            self._distance_history.append(current_distance)
            if len(self._distance_history) > self.config.stagnation_window:
                self._distance_history.pop(0)

            if not near_source_search and len(self._distance_history) == self.config.stagnation_window:
                window_start = self._distance_history[0]
                window_best = min(self._distance_history)
                improvement = window_start - window_best
                if improvement < self.config.stagnation_distance_threshold:
                    reward += self.config.stagnation_penalty
                    info['stagnation_penalty'] = self.config.stagnation_penalty
                else:
                    info['stagnation_penalty'] = 0.0
            else:
                info['stagnation_penalty'] = 0.0
        else:
            info['stagnation_penalty'] = 0.0

        # ============================================================
        # 8b. STAGNATION PENALTY DIREZIONALE — solo in reward_mode "full"
        # ============================================================
        if reward_mode == "full" and not near_source_search and self.steps >= self.config.memory_length:
            dxs = [d[0] for d in self._displacement_memory]
            dys = [d[1] for d in self._displacement_memory]
            net_disp = np.sqrt(sum(dxs) ** 2 + sum(dys) ** 2)
            total_traveled = sum(np.sqrt(dx ** 2 + dy ** 2) for dx, dy in self._displacement_memory)
            if total_traveled > 1e-6:
                efficiency = net_disp / total_traveled
                if efficiency < self.config.directional_stagnation_threshold:
                    reward += self.config.directional_stagnation_penalty
                    info['directional_stagnation_penalty'] = self.config.directional_stagnation_penalty
                else:
                    info['directional_stagnation_penalty'] = 0.0
            else:
                info['directional_stagnation_penalty'] = 0.0
        else:
            info['directional_stagnation_penalty'] = 0.0

        # Aggiorna valori precedenti
        self.prev_concentration = current_conc
        self.prev_distance = current_distance

        # Info aggiuntive
        info['total_reward'] = reward
        info['distance_to_source'] = current_distance
        info['concentration'] = current_conc
        info['on_land'] = on_land
        info['steps'] = self.steps

        return reward, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset dell'ambiente.

        Args:
            seed: Seed per la riproducibilità
            options: Opzioni aggiuntive (es. 'spawn_position')

        Returns:
            observation: Osservazione iniziale
            info: Informazioni aggiuntive
        """
        super().reset(seed=seed)

        # Curriculum/Inference: scegli sorgente random (training) o usa field fornito (inference)
        if self.randomize_field and self._data_manager:
            # TRAINING MODE: randomizza sorgente e carica field random
            # Se allowed_sources non yet populated da curriculum, usa sorgenti reali dal DataManager
            available_sources = self.allowed_sources if self.allowed_sources else self._data_manager.get_discovered_sources()

            if not available_sources:
                raise ValueError("Nessuna sorgente disponibile da allowed_sources o DataManager")

            # Weighted scenario sampling: campiona (version, chunk_id) dai pesi se configurato
            if self.scenario_weights:
                keys = list(self.scenario_weights.keys())
                probs = np.array([self.scenario_weights[k] for k in keys], dtype=float)
                probs /= probs.sum()
                chosen_idx = int(self.np_random.choice(len(keys), p=probs))
                chosen_key = keys[chosen_idx]
                version_str, chunk_str = chosen_key.rsplit('_', 1)
                self.config.chunk_id = int(chunk_str)
                source = self.np_random.choice(available_sources)
                self.field, self._current_run_id = self._data_manager.get_random_field_for_source_version(source, version_str)
            else:
                source = self.np_random.choice(available_sources)
                self.field, self._current_run_id = self._data_manager.get_random_field_for_source(source)
            # Estrai source_id dal run_id (es. 'SRC042_V1' -> 'SRC042')
            self.source_id = self._current_run_id.split('_')[0]
            
            # Aggiorna posizione sorgente (sempre da coordinate hardcodate)
            if self.field.source_position is not None:
                self.source_position = np.array(self.field.source_position)
            else:
                raise ValueError(
                    f"Campo per {source} non ha source_position. "
                    f"Controlla DataManager.SOURCE_CONFIGS e nome file NC."
                )
        elif self.field and hasattr(self.field, 'run_id') and self.field.run_id:
            # INFERENCE MODE: il field è già fornito con run_id settato
            self._current_run_id = self.field.run_id
            self.source_id = self._current_run_id.split('_')[0] if '_' in self._current_run_id else self._current_run_id
        
        # Carica dati di vento e corrente corretti per questo run_id
        # (fatto qui FUORI dal blocco randomize_field per funzionare sia in training che inference)
        if self._current_run_id and self._data_manager and self.wind_mapping:
            self.wind_data = self._data_manager.get_wind_data_for_run(
                self._current_run_id, self.wind_mapping
            )
        # Carica corrente dinamicamente se current_mapping è disponibile
        if self._current_run_id and self._data_manager and self.current_mapping:
            self.current_data = self._data_manager.get_current_data_for_run(
                self._current_run_id, self.current_mapping
            )

        # Determina spawn_start_frame: usa chunk_id
        # chunk_id=0: spawn a 1/4 della simulazione (inizio plume)
        # chunk_id=1: spawn a 1/2 della simulazione (plume maturo)
        # chunk_id=2: spawn a 3/4 della simulazione (plume tardi/disperso)
        spawn_frame = self.config.spawn_start_frame
        if self.field.n_timesteps > 1 and self.config.chunk_id in [0, 1, 2]:
            if self.config.chunk_id == 0:
                spawn_frame = self.field.n_timesteps // 4      # Q1/4
            elif self.config.chunk_id == 1:
                spawn_frame = self.field.n_timesteps // 2      # Q1/2
            else:  # chunk_id == 2
                spawn_frame = (self.field.n_timesteps * 3) // 4  # Q3/4

        # Imposta timestep al frame calcolato
        if self.field.n_timesteps > 1:
            self._start_time_idx = min(spawn_frame, self.field.n_timesteps - 1)
            self.field.set_time(self._start_time_idx)
        else:
            self._start_time_idx = 0

        # Determina posizione iniziale
        if options and 'spawn_position' in options:
            spawn_pos = options['spawn_position']
        else:
            spawn_pos = self._spawn_on_plume()

        # Inizializza stato agente
        self.state = AgentState(x=spawn_pos[0], y=spawn_pos[1])
        self._prev_position = self.state.position.copy()
        self._on_land = False  # BUG #5 FIX: Inizializza flag terra
        self.steps = 0

        # Inizializza valori per il reward
        self.prev_concentration = self.field.get_concentration(self.state.x, self.state.y)
        if np.isnan(self.prev_concentration):
            self.prev_concentration = 0.0
        self.prev_distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )

        # Reset history
        self.trajectory = [self.state.position.copy()]
        self.concentration_history = [self.prev_concentration]

        # Reset memoria concentrazioni passate (9 valori)
        self._concentration_memory = [0.0] * self.config.memory_length

        # Reset memoria spostamenti passati (9 coppie Δx, Δy)
        self._displacement_memory = [(0.0, 0.0)] * self.config.memory_length

        # Reset memoria concentrazioni direzionali passate (9 timestep x 8 direzioni)
        self._directional_conc_memory = [[0.0] * 8 for _ in range(self.config.memory_length)]
        if self.config.sensor_range_2 is not None:
            self._directional_conc_memory_2 = [[0.0] * 8 for _ in range(self.config.memory_length)]

        # Reset buffer distanze per stagnation penalty
        self._distance_history = []

        # Reset feature 3: inizializza max concentration alla posizione di spawn
        spawn_conc = self.field.get_concentration(spawn_pos[0], spawn_pos[1])
        self._max_concentration_seen = float(spawn_conc) if not np.isnan(spawn_conc) else 0.0
        self._max_conc_position = (float(spawn_pos[0]), float(spawn_pos[1]))

        # Reset feature 4
        self._steps_since_plume_contact = 0

        # Sincronizza vento e corrente al timestep di start usando TEMPO REALE (minuti)
        # dt_conc = 2 min/frame, dt_wind = 60 min/frame, dt_current = 2 min/frame
        # set_time_from_minutes() converte automaticamente: time_idx = time_minutes / dt
        if self._is_time_varying():
            # Calcola tempo reale dalla concentrazione (dt_conc = 2 minuti per frame)
            time_minutes = self._start_time_idx * 2.0
            
            # Sincronizza vento (dt=60 min)
            if self.wind_data is not None:
                self.wind_data.set_time_from_minutes(time_minutes)
            
            # Sincronizza corrente (dt=2 min, come conc)
            if self.current_data is not None:
                self.current_data.set_time_from_minutes(time_minutes)

        observation = self._get_observation()
        info = {
            'spawn_position': spawn_pos,
            'source_position': self.source_position.tolist(),
            'source_id': self.source_id,
            'initial_distance': self.prev_distance,
            'initial_concentration': self.prev_concentration,
            'start_time_idx': self._start_time_idx  # Aggiunto per tracking frame nei plot
        }
        
        # Salva l'info dict per accesso esterno (in run_episode)
        self.info_reset = info

        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue un passo di simulazione.

        Args:
            action: Azione da eseguire

        Returns:
            observation: Nuova osservazione
            reward: Reward ottenuto
            terminated: True se episodio terminato (successo/fallimento)
            truncated: True se episodio troncato (max steps)
            info: Informazioni aggiuntive
        """
        self.steps += 1

        # Converti azione se necessario
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)

        # Registra posizione prima dell'azione
        old_x, old_y = self.state.x, self.state.y
        self._prev_position = np.array([old_x, old_y], dtype=np.float64)

        # Applica azione
        self._apply_action(action)

        # Calcola spostamento e aggiorna memoria (Δx, Δy)
        dx = self.state.x - old_x
        dy = self.state.y - old_y
        self._displacement_memory.pop(0)
        self._displacement_memory.append((dx, dy))

        # Avanza il tempo del campo se time-varying (partendo dal frame di start)
        if self._is_time_varying():
            # Calcola il frame di concentrazione a partire dal tempo simulato
            # time_offset è il numero di step × dt (in secondi) convertito a minuti
            # dt_conc = 2 minuti per frame
            time_offset_minutes = (self.steps * self.config.dt / 60.0)  # secondi → minuti
            time_minutes = self._start_time_idx * 2.0 + time_offset_minutes
            
            # Concentrazione: usa frame float (interpolazione temporale)
            time_offset_frames = time_offset_minutes / 2.0  # dt_conc = 2 min per frame
            time_idx = self._start_time_idx + time_offset_frames
            time_idx = max(0.0, min(time_idx, float(self.field.n_timesteps - 1)))
            self.field.set_time(time_idx)
            
            # Vento e corrente: sincronizza con tempo reale in minuti
            # set_time_from_minutes() clipa automaticamente agli estremi
            if self.wind_data is not None:
                self.wind_data.set_time_from_minutes(time_minutes)
            
            if self.current_data is not None:
                self.current_data.set_time_from_minutes(time_minutes)

        # Cache wind/current components una volta per step (riusati da _compute_reward e _get_observation)
        self._cached_wind_components = self.wind_data.get_wind_components() if self.wind_data is not None else None
        self._cached_current_components = (
            self.current_data.get_current_components(self.state.x, self.state.y)
            if self.current_data is not None else None
        )

        # Calcola reward
        reward, reward_info = self._compute_reward(action)

        # Registra traiettoria
        self.trajectory.append(self.state.position.copy())
        conc_now = self.field.get_concentration(self.state.x, self.state.y)
        self.concentration_history.append(conc_now)

        # Aggiorna memoria concentrazioni (FIFO: rimuovi più vecchio, aggiungi attuale)
        self._concentration_memory.pop(0)
        self._concentration_memory.append(conc_now)

        # Aggiorna memoria concentrazioni direzionali (FIFO: 9 timestep x 8 direzioni)
        self._directional_conc_memory.pop(0)
        self._directional_conc_memory.append(self._compute_directional_sensors(self.config.sensor_range))
        if self.config.sensor_range_2 is not None:
            self._directional_conc_memory_2.pop(0)
            self._directional_conc_memory_2.append(self._compute_directional_sensors(self.config.sensor_range_2))

        # Aggiorna feature 3 e 4
        if conc_now > self.config.spawn_conc_threshold:
            self._steps_since_plume_contact = 0
            if conc_now > self._max_concentration_seen:
                self._max_concentration_seen = float(conc_now)
                self._max_conc_position = (float(self.state.x), float(self.state.y))
        else:
            self._steps_since_plume_contact += 1

        # Controlla terminazione
        terminated = False
        truncated = False
        termination_reason = None

        source_reached = self._check_source_reached()
        out_of_bounds = self._check_boundary()
        on_land = self._check_on_land()

        if source_reached:
            terminated = True
            termination_reason = 'success'
        elif out_of_bounds:
            terminated = True
            termination_reason = 'boundary'
        elif on_land:
            terminated = True  # Termina se va sulla terra
            termination_reason = 'land'

        if not terminated and self.steps >= self.config.max_steps:
            truncated = True
            termination_reason = 'timeout'

        if termination_reason is None:
            termination_reason = 'running'

        # Osservazione
        observation = self._get_observation()

        # Info
        info = {
            **reward_info,
            'steps': self.steps,
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'source_reached': source_reached,
            'is_success': bool(source_reached),
            'out_of_bounds': out_of_bounds,
            'on_land': on_land,
            'terminated': terminated,
            'truncated': truncated,
            'termination_reason': termination_reason,
            'end_time_idx': int(self.field._current_time_idx) if hasattr(self.field, '_current_time_idx') else self._start_time_idx  # Salva il frame finale prima del reset
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renderizza l'ambiente."""
        if self.render_mode is None:
            return None

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            return None

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 8))

        self._ax.clear()

        # Plot campo di concentrazione
        field_data = self.field.get_current_field()
        extent = [self.config.xmin, self.config.xmax,
                  self.config.ymin, self.config.ymax]

        im = self._ax.imshow(
            field_data,
            extent=extent,
            origin='lower',
            cmap='YlOrRd',
            aspect='auto',
            alpha=0.7
        )

        # Plot sorgente
        self._ax.scatter(
            self.source_position[0],
            self.source_position[1],
            c='red', s=200, marker='*',
            label='Source', zorder=10
        )

        # Plot soglia sorgente
        circle = Circle(
            self.source_position,
            self.config.source_distance_threshold,
            fill=False, color='red', linestyle='--'
        )
        self._ax.add_patch(circle)

        # Plot traiettoria
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self._ax.plot(
                traj[:, 0], traj[:, 1],
                'b-', linewidth=2, alpha=0.7,
                label='Trajectory'
            )

        # Plot agente
        self._ax.scatter(
            self.state.x, self.state.y,
            c='blue', s=100, marker='o',
            label='Agent', zorder=11
        )

        # Plot direzione
        if self.state.vx != 0 or self.state.vy != 0:
            self._ax.arrow(
                self.state.x, self.state.y,
                self.state.vx * 50, self.state.vy * 50,
                head_width=30, head_length=20,
                fc='blue', ec='blue'
            )

        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_title(
            f'HYDRAS Source Seeking - Step {self.steps}\n'
            f'Concentration: {self.concentration_history[-1]:.1f} g/m³'
        )
        self._ax.legend(loc='upper right')

        plt.tight_layout()

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        else:  # rgb_array
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        """Chiude l'ambiente."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# Registra l'ambiente con Gymnasium
gym.register(
    id='SourceSeeking-v0',
    entry_point='utils.source_seeking_env:SourceSeekingEnv',
)


class ScenarioWeightWrapper(gym.Wrapper):
    """Wrapper esterno che espone set_scenario_weights come metodo accessibile via SubprocVecEnv.env_method.

    Gymnasium 1.2+ non ha __getattr__ delegation nei Wrapper, quindi env_method
    non riesce a raggiungere metodi degli env interni. Questo wrapper li espone esplicitamente.
    """

    def set_scenario_weights(self, weights: Optional[Dict[str, float]]) -> None:
        inner = self.env
        while hasattr(inner, 'env'):
            inner = inner.env
        inner.scenario_weights = weights

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


if __name__ == "__main__":
    # Test dell'ambiente
    print("Testing SourceSeekingEnv...")

    env = SourceSeekingEnv(
        source_id='SRC001',
        render_mode=None,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Test alcuni step con azioni random
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended at step {i+1}")
            print(f"  Reason: {'Source reached' if info['source_reached'] else 'Out of bounds' if info['out_of_bounds'] else 'Max steps'}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")

    env.close()
    print("\nEnvironment test completed!")