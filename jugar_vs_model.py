"""
Llança una partida gràfica de Splendor: tu (jugador 1) contra un model SB3 entrenat.

Ús:
    python jugar_vs_model.py
    python jugar_vs_model.py C:\ruta\al\model.zip
"""

import sys
from pathlib import Path

import splendor.splendor.gym  # noqa: F401 — registra l'entorn splendor-v1
from sb3_contrib import MaskablePPO
from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.game import Game
from splendor.splendor.splendor_displayer import GUIDisplayer
from splendor.splendor.splendor_model import SplendorGameRule

from tfm_splendor.agents.TrainedPPOAgent import TrainedPPOAgent

DEFAULT_MODEL = Path(
    r"C:\TFM\rivals\thebest.zip"
)

if __name__ == "__main__":
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL

    print(f"Carregant model: {model_path}")
    model = MaskablePPO.load(str(model_path))

    ai_agent = TrainedPPOAgent(0, model)
    human_placeholder = RandomAgent(1)  # substituït per interactivitat

    displayer = GUIDisplayer(half_scale=False, delay=3, no_highlighting=True)
    #llavor random per a la generació de les cartes i tokens, per a que sigui no reproducible
    llavor = None
    game = Game(
        SplendorGameRule,
        [ai_agent, human_placeholder],
        num_of_agent=2,
        seed=llavor,
        displayer=displayer,
        #agents_namelist=[model_path.stem, "Human"],
        agents_namelist=['Agent', "Humà"],
        interactive=True,
    )
    game.Run()
