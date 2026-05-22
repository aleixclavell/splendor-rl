# TFM-splendor: Aprenentatge per Reforç al joc Splendor

**Treball Final de Màster** — Entrenament d'agents d'RL per jugar a Splendor, un joc de taula de gestió de recursos.

S'exploren i comparen dos algorismes:
- **PPO** (Proximal Policy Optimization) via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Discrete SAC** (Soft Actor-Critic per a accions discretes) via [Tianshou](https://github.com/thu-ml/tianshou)

Tots dos s'entrenen amb aprenentatge per currículum (oponents progressius: Random → Heurístic → Self-play).

---

## Estructura del repositori

```
TFM-splendor/
├── Splendor-AI/        # Motor del joc — MIT © roeey777 (vegeu atribució)
│   └── src/splendor/   # Entorn Gym de Splendor + agents genèrics
├── tfm_splendor/       # Paquet d'entrenament
│   ├── agents/         # Agents heurístics (H1–H3) + wrappers de models entrenats
│   └── entrenament/    # Wrappers Gym, callbacks, política DSAC, tuning Optuna
├── scripts/            # Scripts d'entrada per a l'entrenament
│   ├── ppo_splendor.py
│   ├── dsac_splendor.py
│   ├── pipeline_entrenament.py   # Pipeline de currículum
│   ├── run_optuna.py
│   └── avaluar_runs.py
├── jugar_vs_model.py   # Jugar interactivament contra un agent entrenat
├── requirements.txt
└── pyproject.toml
```

---

## Instal·lació

**Pas 1** — Instal·lar el motor del joc:
```bash
pip install -e Splendor-AI/
```

**Pas 2** — Instal·lar el paquet d'entrenament:
```bash
pip install -e .
```

**Pas 3** — Instal·lar les dependències restants:
```bash
pip install -r requirements.txt
```

> Requereix Python 3.10+.

---

## Ús

### Entrenar un agent PPO
```bash
python scripts/ppo_splendor.py
```

### Entrenar un agent Discrete SAC
```bash
python scripts/dsac_splendor.py
```

### Executar la pipeline de currículum
```bash
python scripts/pipeline_entrenament.py
```

### Cerca d'hiperparàmetres (Optuna)
```bash
python scripts/run_optuna.py
```

### Jugar contra un model entrenat
```bash
python jugar_vs_model.py
```

---

## Atribució

El motor del joc (`Splendor-AI/`) està basat en el treball de [roeey777](https://github.com/roeey777/Splendor-AI), llicenciat sota la **llicència MIT**. El motor original va ser desenvolupat per estudiants de la Universitat de Melbourne per al curs COMP90054, amb permís del Prof. Nir Lipovetzky.
