import csv
import os

from tfm_splendor.entrenament.evaluation_utils import avaluar_model
from tfm_splendor.entrenament.artifact_utils import registrar_hiperparametres, preparar_execucio


def _safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_codecarbon_csv(emissions_csv_path):
    """Llegeix l'última fila del CSV de CodeCarbon i retorna les mètriques com a dict."""
    if not emissions_csv_path or not os.path.exists(emissions_csv_path):
        return {}
    with open(emissions_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
    if last is None:
        return {}
    wanted_keys = (
        "emissions", "energy_consumed", "cpu_energy", "gpu_energy", "ram_energy",
        "cpu_power", "gpu_power", "ram_power", "emissions_rate",
    )
    metrics = {}
    for key in wanted_keys:
        value = _safe_float(last.get(key))
        if value is not None:
            metrics[key] = value
    return metrics
