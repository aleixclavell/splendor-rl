"""
Avaluació post-entrenament sobre totes les runs existents.

Per a cada pipeline a runs/, agafa el model de l'última etapa i l'avalua
contra RandomAgent, H1Agent i H2Agent (400 episodis cadascun per defecte).
Els resultats es guarden a runs/<pipeline>/eval_extesa/resultats.json.

Ús:
    python avaluar_runs.py
    python avaluar_runs.py --runs_dir runs --n_episodes 400 --force
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional

from codecarbon import EmissionsTracker
from sb3_contrib import MaskablePPO
from splendor.agents.generic.random import RandomAgent

from tfm_splendor.agents.H1Agent import H1Agent
from tfm_splendor.agents.H2Agent import H2Agent
from tfm_splendor.agents.H3Agent import H3Agent
from tfm_splendor.entrenament.evaluation_utils import avaluar_model
from dsac_splendor import carregar_model_dsac


_STAGE_RE = re.compile(r"^e(\d+)[pd]_")

OPONENTS_AVALUACIO = [
    ("Random", lambda: [RandomAgent(1)]),
    ("H1",     lambda: [H1Agent(1)]),
    ("H2",     lambda: [H2Agent(1)]),
    ("H1 i H2",  lambda: [H1Agent(1), H2Agent(2)]),
]

def _trobar_model_final(pipeline_dir: str) -> Optional[tuple[str, str]]:
    """Retorna (model_path, mode) de l'última etapa, o None si no n'hi ha."""
    stage_dirs = [
        d for d in os.listdir(pipeline_dir)
        if os.path.isdir(os.path.join(pipeline_dir, d)) and _STAGE_RE.match(d)
    ]
    if not stage_dirs:
        return None

    stage_dirs.sort(key=lambda d: int(_STAGE_RE.match(d).group(1)))
    model_dir = os.path.join(pipeline_dir, stage_dirs[-1], "model")

    if not os.path.isdir(model_dir):
        return None

    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".zip"):
            return os.path.join(model_dir, fname), "ppo"
        if fname.endswith(".pth"):
            return os.path.join(model_dir, fname), "dsac"

    return None


_PPO_CUSTOM_OBJECTS = {
    "clip_range": 0.2,
    "lr_schedule": 3e-4,
}


def _carregar_model(model_path: str, mode: str):
    if mode == "dsac":
        return carregar_model_dsac(model_path, opponents=[H1Agent(1)])
    return MaskablePPO.load(model_path, custom_objects=_PPO_CUSTOM_OBJECTS)


def avaluar_pipeline(
    pipeline_dir: str,
    n_episodes: int = 200,
    skip_existing: bool = True,
) -> Optional[dict]:
    """
    Avalua el model final d'una pipeline contra RandomAgent, H1Agent i H2Agent.
    Guarda els resultats a <pipeline_dir>/eval_extesa/resultats.json.
    Retorna el diccionari de resultats, o None si s'ha saltat.
    """
    eval_dir = os.path.join(pipeline_dir, "eval_extesa")
    results_file = os.path.join(eval_dir, "resultats.json")

    if skip_existing and os.path.exists(results_file):
        print(f"  [skip] {os.path.basename(pipeline_dir)} — ja avaluat")
        return None

    trobat = _trobar_model_final(pipeline_dir)
    if trobat is None:
        print(f"  [skip] {os.path.basename(pipeline_dir)} — no s'ha trobat cap model")
        return None

    model_path, mode = trobat
    pipeline_nom = os.path.basename(pipeline_dir)

    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline_nom}  [{mode.upper()}]")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    model = _carregar_model(model_path, mode)
    os.makedirs(eval_dir, exist_ok=True)

    os.environ["CODECARBON_COUNTRY_ISO_CODE"] = "ESP"
    os.environ["CODECARBON_REGION"] = "catalonia"

    resultats = {}
    for nom, oponents_fn in OPONENTS_AVALUACIO:
        opponents = oponents_fn()
        monitor_file = os.path.join(eval_dir, f"monitor_vs_{nom.lower()}")
        emissions_file = os.path.join(eval_dir, f"emissions_vs_{nom.lower()}.csv")
        print(f"\n  vs {nom}Agent ({n_episodes} episodis):")

        tracker = EmissionsTracker(
            project_name=f"Avaluació {pipeline_nom} vs {nom}",
            measure_power_secs=10,
            output_file=emissions_file,
            log_level="error",
        )
        tracker.start()
        try:
            metrics = avaluar_model(
                model,
                opponents=opponents,
                n_episodes=n_episodes,
                monitor_file=monitor_file,
                mode=mode,
                print_summary=True,
            )
        finally:
            emissions_kg = tracker.stop() or 0.0

        metrics["emissions_kg_co2"] = emissions_kg
        resultats[f"vs_{nom}"] = metrics

    output = {
        "pipeline": pipeline_nom,
        "model_path": model_path,
        "model_type": mode,
        "n_episodes": n_episodes,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "resultats": resultats,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultats guardats a: {results_file}")

    return output


def avaluar_run(
    run_name: str,
    runs_dir: str = "runs",
    n_episodes: int = 200,
    skip_existing: bool = True,
) -> Optional[dict]:
    """
    Avalua el model final d'un run concret identificat pel seu nom de directori.

    Args:
        run_name:       Nom del directori de la pipeline (p. ex. "pl_20240101_123456").
        runs_dir:       Directori arrel on hi ha les pipelines (default: "runs").
        n_episodes:     Episodis d'avaluació per oponent (default: 200).
        skip_existing:  Si True, salta si ja existeixen resultats.

    Returns:
        Diccionari amb els resultats, o None si s'ha saltat.
    """
    pipeline_dir = os.path.join(runs_dir, run_name)
    if not os.path.isdir(pipeline_dir):
        raise FileNotFoundError(f"No s'ha trobat el run: {pipeline_dir}")
    return avaluar_pipeline(pipeline_dir, n_episodes=n_episodes, skip_existing=skip_existing)


def avaluar_tots_els_runs(
    runs_dir: str = "runs",
    n_episodes: int = 200,
    skip_existing: bool = True,
) -> list[dict]:
    """
    Escaneja tots els directoris de pipelines a runs_dir i avalua el model final
    de cadascun contra RandomAgent, H1Agent, H2Agent i H3Agent.

    Args:
        runs_dir:       Directori arrel on hi ha les pipelines (default: "runs").
        n_episodes:     Episodis d'avaluació per oponent (default: 200).
        skip_existing:  Si True, salta les pipelines que ja tinguin resultats.

    Returns:
        Llista de diccionaris amb els resultats de cada pipeline avaluada.
    """
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"No s'ha trobat el directori: {runs_dir}")

    pipeline_dirs = sorted([
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("pl_")
    ])

    print(f"\nAvaluant {len(pipeline_dirs)} pipelines a '{runs_dir}'...")
    tots_resultats = []

    for pipeline_dir in pipeline_dirs:
        result = avaluar_pipeline(pipeline_dir, n_episodes=n_episodes, skip_existing=skip_existing)
        if result is not None:
            tots_resultats.append(result)

    print(f"\n{'='*60}")
    print(f"Avaluació completada: {len(tots_resultats)} pipelines noves avaluades")

    if tots_resultats:
        print("\nResum:")
        header = f"  {'Pipeline':<30} {'Tipus':<6}  " + "  ".join(
            f"vs {nom:<8}" for nom, _ in OPONENTS_AVALUACIO
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in tots_resultats:
            winrates = "  ".join(
                f"{r['resultats'][f'vs_{nom}']['winrate']:>8.0%}"
                for nom, _ in OPONENTS_AVALUACIO
            )
            print(f"  {r['pipeline']:<30} {r['model_type'].upper():<6}  {winrates}")

    return tots_resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalua models entrenats contra Random, H1 i H2")
    parser.add_argument("--runs_dir", default="runs", help="Directori de pipelines (default: runs)")
    parser.add_argument("--n_episodes", type=int, default=400, help="Episodis per oponent (default: 400)")
    parser.add_argument("--force", action="store_true", help="Reavalua tot, ignorant resultats existents")
    parser.add_argument("--pipeline", default=None, help="Avalua només una pipeline específica (nom del directori)")
    args = parser.parse_args()

    if args.pipeline:
        avaluar_run(args.pipeline, runs_dir=args.runs_dir, n_episodes=args.n_episodes, skip_existing=not args.force)
    else:
        avaluar_tots_els_runs(
            runs_dir=args.runs_dir,
            n_episodes=args.n_episodes,
            skip_existing=not args.force,
        )
