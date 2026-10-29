"""Generación del catálogo, las bibliotecas y el historial simulado de PlayNow.

Los videojuegos existen, pero sus atributos son estimaciones para representar
distintas experiencias de juego. Los perfiles, las sesiones y los comentarios
también se construyen aquí, sin consultar plataformas ni reseñas reales.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

# La raíz del proyecto permite importar el motor al ejecutar este archivo desde data.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playnow.engine import SEED, aggregate_lexicon, russell_quadrant  # noqa: E402

DATA = Path(__file__).resolve().parent

# Cada fila se convierte en una ficha con el orden indicado en COLUMNS:
# identificación, título, género e indicador indie; minutos hasta la diversión,
# intervalo de guardado, sesión mínima y facilidad de pausa; costo de retomar,
# carga cognitiva, exigencia motora y narrativa; valencia, activación y frustración;
# adecuación a relajación, partida ligera, historia y reto; modalidad y micrófono.
# Los grados de adecuación y las cargas van de 0 a 1; valencia y activación, de -1 a 1.
GAMES = [
    ("hades", "Hades", "roguelike", 1, 3, 12, 20, 0.55, 0.15, 0.45, 0.80, 0.35, 0.35, 0.75, 0.45, 0.10, 0.35, 0.25, 0.90, "Solo", 0),
    ("stardew", "Stardew Valley", "simulacion", 1, 5, 15, 20, 0.95, 0.10, 0.25, 0.10, 0.30, 0.70, -0.55, 0.05, 0.90, 0.55, 0.20, 0.05, "Solo", 0),
    ("elden", "Elden Ring", "accion-rpg", 0, 15, 25, 45, 0.35, 0.80, 0.70, 0.85, 0.40, -0.20, 0.55, 0.85, 0.05, 0.10, 0.35, 0.90, "Solo", 0),
    ("bg3", "Baldur's Gate 3", "rpg", 0, 25, 20, 75, 0.70, 0.75, 0.80, 0.20, 0.95, 0.40, 0.15, 0.25, 0.15, 0.10, 0.95, 0.30, "Coop", 0),
    ("rocket", "Rocket League", "deportes", 0, 1, 7, 15, 0.40, 0.10, 0.30, 0.90, 0.05, 0.25, 0.85, 0.55, 0.05, 0.40, 0.00, 0.90, "PvP", 1),
    ("cyberpunk", "Cyberpunk 2077", "rpg", 0, 20, 15, 50, 0.65, 0.60, 0.55, 0.45, 0.80, 0.10, 0.35, 0.30, 0.10, 0.15, 0.85, 0.35, "Solo", 0),
    ("celeste", "Celeste", "plataformas", 1, 2, 8, 20, 0.80, 0.25, 0.40, 0.85, 0.45, 0.15, 0.40, 0.70, 0.15, 0.25, 0.40, 0.75, "Solo", 0),
    ("slay", "Slay the Spire", "roguelike", 1, 4, 15, 25, 0.95, 0.20, 0.65, 0.15, 0.15, 0.20, -0.10, 0.25, 0.35, 0.40, 0.10, 0.55, "Solo", 0),
    ("disco", "Disco Elysium", "rpg", 1, 10, 10, 45, 0.90, 0.70, 0.85, 0.05, 0.98, 0.05, -0.25, 0.15, 0.20, 0.05, 0.95, 0.10, "Solo", 0),
    ("forza", "Forza Horizon 5", "carreras", 0, 3, 10, 20, 0.85, 0.15, 0.20, 0.45, 0.10, 0.75, 0.20, 0.10, 0.55, 0.60, 0.05, 0.25, "Solo", 0),
    ("rdr2", "Red Dead Redemption 2", "accion", 0, 40, 20, 60, 0.40, 0.72, 0.50, 0.40, 0.92, 0.30, -0.05, 0.15, 0.20, 0.15, 0.90, 0.20, "Solo", 0),
    ("hollow", "Hollow Knight", "metroidvania", 1, 8, 15, 30, 0.55, 0.55, 0.45, 0.70, 0.40, 0.10, 0.25, 0.60, 0.10, 0.20, 0.40, 0.70, "Solo", 0),
    ("unpacking", "Unpacking", "narrativa", 1, 2, 10, 18, 0.90, 0.16, 0.16, 0.12, 0.38, 0.42, -0.20, 0.06, 0.60, 0.66, 0.22, 0.04, "Solo", 0),
    ("shorthike", "A Short Hike", "aventura", 1, 1, 8, 20, 0.90, 0.08, 0.15, 0.20, 0.25, 0.85, -0.40, 0.05, 0.85, 0.55, 0.20, 0.05, "Solo", 0),
    ("vampire", "Vampire Survivors", "roguelike", 1, 1, 15, 15, 0.70, 0.08, 0.15, 0.35, 0.05, 0.40, 0.45, 0.15, 0.40, 0.85, 0.00, 0.30, "Solo", 0),
    ("minecraft", "Minecraft", "sandbox", 0, 5, 10, 25, 0.95, 0.15, 0.35, 0.20, 0.15, 0.55, -0.20, 0.10, 0.70, 0.60, 0.10, 0.15, "Coop", 0),
    ("witcher", "The Witcher 3", "rpg", 0, 20, 15, 50, 0.60, 0.65, 0.55, 0.40, 0.90, 0.25, 0.20, 0.25, 0.15, 0.15, 0.92, 0.30, "Solo", 0),
    ("gow", "God of War", "accion", 0, 12, 15, 40, 0.50, 0.50, 0.40, 0.65, 0.75, 0.20, 0.45, 0.35, 0.10, 0.20, 0.75, 0.55, "Solo", 0),
    ("animal", "Animal Crossing: New Horizons", "simulacion", 0, 3, 5, 15, 0.98, 0.05, 0.15, 0.05, 0.20, 0.80, -0.65, 0.02, 0.95, 0.50, 0.15, 0.00, "Solo", 0),
    ("fc24", "EA Sports FC 24", "deportes", 0, 2, 12, 15, 0.30, 0.10, 0.25, 0.70, 0.00, 0.10, 0.70, 0.60, 0.05, 0.35, 0.00, 0.85, "PvP", 1),
    ("civ6", "Civilization VI", "estrategia", 0, 15, 20, 60, 0.80, 0.40, 0.90, 0.10, 0.20, 0.15, 0.05, 0.20, 0.15, 0.20, 0.15, 0.45, "Solo", 0),
    ("portal2", "Portal 2", "puzles", 0, 3, 15, 25, 0.85, 0.20, 0.55, 0.25, 0.40, 0.55, 0.10, 0.15, 0.25, 0.45, 0.35, 0.30, "Coop", 0),
    ("journey", "Journey", "aventura", 1, 2, 20, 25, 0.40, 0.15, 0.15, 0.20, 0.70, 0.75, -0.20, 0.05, 0.70, 0.30, 0.75, 0.05, "Solo", 0),
    ("gris", "Gris", "plataformas", 1, 2, 15, 20, 0.80, 0.10, 0.15, 0.20, 0.65, 0.40, -0.35, 0.05, 0.75, 0.30, 0.70, 0.05, "Solo", 0),
    ("deadcells", "Dead Cells", "roguelike", 1, 2, 12, 20, 0.50, 0.20, 0.40, 0.85, 0.15, 0.15, 0.70, 0.65, 0.10, 0.30, 0.10, 0.90, "Solo", 0),
    ("amongus", "Among Us", "party", 1, 2, 10, 15, 0.20, 0.05, 0.25, 0.20, 0.10, 0.30, 0.55, 0.20, 0.15, 0.70, 0.05, 0.25, "PvP", 1),
    ("ittakes", "It Takes Two", "cooperativo", 0, 8, 15, 40, 0.25, 0.35, 0.35, 0.45, 0.70, 0.65, 0.35, 0.15, 0.20, 0.25, 0.80, 0.30, "Coop", 1),
    ("overcooked", "Overcooked! 2", "cooperativo", 0, 2, 10, 20, 0.30, 0.10, 0.40, 0.75, 0.05, 0.20, 0.90, 0.55, 0.05, 0.35, 0.05, 0.70, "Coop", 1),
    ("skyrim", "The Elder Scrolls V: Skyrim", "rpg", 0, 15, 15, 45, 0.75, 0.45, 0.50, 0.35, 0.70, 0.35, 0.10, 0.20, 0.25, 0.30, 0.70, 0.30, "Solo", 0),
    ("tetris", "Tetris Effect", "puzles", 0, 1, 5, 15, 0.90, 0.05, 0.30, 0.35, 0.00, 0.60, -0.15, 0.10, 0.70, 0.55, 0.00, 0.25, "Solo", 0),
    ("dorf", "Dorfromantik", "puzles", 1, 1, 10, 15, 0.98, 0.05, 0.30, 0.05, 0.05, 0.65, -0.70, 0.02, 0.95, 0.40, 0.05, 0.00, "Solo", 0),
    ("dave", "Dave the Diver", "aventura", 1, 5, 15, 25, 0.85, 0.15, 0.30, 0.25, 0.45, 0.60, -0.10, 0.10, 0.55, 0.70, 0.40, 0.10, "Solo", 0),
    ("cult", "Cult of the Lamb", "roguelike", 1, 5, 15, 25, 0.70, 0.20, 0.40, 0.45, 0.40, 0.25, 0.30, 0.25, 0.30, 0.55, 0.35, 0.45, "Solo", 0),
    ("hades2", "Hades II", "roguelike", 1, 4, 12, 25, 0.55, 0.20, 0.50, 0.80, 0.45, 0.30, 0.70, 0.50, 0.10, 0.30, 0.35, 0.90, "Solo", 0),
    ("lol", "League of Legends", "moba", 0, 8, 30, 35, 0.05, 0.35, 0.55, 0.80, 0.05, -0.10, 0.90, 0.80, 0.00, 0.15, 0.05, 0.95, "PvP", 1),
    ("valorant", "Valorant", "fps", 0, 3, 25, 30, 0.05, 0.25, 0.45, 0.90, 0.05, -0.05, 0.85, 0.75, 0.00, 0.20, 0.00, 0.95, "PvP", 1),
    ("fortnite", "Fortnite", "battle-royale", 0, 3, 20, 25, 0.10, 0.15, 0.30, 0.75, 0.05, 0.20, 0.80, 0.50, 0.05, 0.40, 0.00, 0.70, "PvP", 1),
    ("gta", "Grand Theft Auto V", "accion", 0, 10, 15, 30, 0.60, 0.30, 0.30, 0.45, 0.40, 0.15, 0.40, 0.25, 0.25, 0.50, 0.35, 0.40, "Solo", 0),
    ("tlou", "The Last of Us Part I", "accion", 0, 15, 20, 45, 0.45, 0.60, 0.40, 0.45, 0.90, 0.05, 0.25, 0.30, 0.10, 0.10, 0.95, 0.25, "Solo", 0),
    ("spiderman", "Marvel's Spider-Man", "accion", 0, 8, 15, 30, 0.60, 0.30, 0.30, 0.60, 0.55, 0.45, 0.40, 0.20, 0.20, 0.35, 0.60, 0.40, "Solo", 0),
    ("horizon", "Horizon Zero Dawn", "accion-rpg", 0, 15, 15, 40, 0.55, 0.50, 0.45, 0.55, 0.70, 0.30, 0.30, 0.25, 0.15, 0.20, 0.75, 0.40, "Solo", 0),
    ("mhw", "Monster Hunter: World", "accion", 0, 20, 25, 40, 0.20, 0.55, 0.60, 0.70, 0.20, 0.10, 0.40, 0.45, 0.05, 0.15, 0.15, 0.80, "Coop", 1),
    ("diablo", "Diablo IV", "accion-rpg", 0, 8, 15, 30, 0.40, 0.25, 0.35, 0.60, 0.20, 0.05, 0.45, 0.35, 0.15, 0.40, 0.20, 0.60, "Solo", 0),
    ("wow", "World of Warcraft", "mmorpg", 0, 20, 20, 40, 0.30, 0.40, 0.55, 0.40, 0.45, 0.10, 0.35, 0.40, 0.10, 0.20, 0.40, 0.45, "Coop", 1),
    ("spirit", "Spiritfarer", "narrativa", 1, 8, 15, 30, 0.90, 0.20, 0.25, 0.15, 0.75, 0.55, -0.45, 0.05, 0.80, 0.30, 0.80, 0.05, "Solo", 0),
    ("coffee", "Coffee Talk", "narrativa", 1, 1, 10, 15, 0.98, 0.05, 0.15, 0.05, 0.60, 0.70, -0.60, 0.02, 0.90, 0.35, 0.55, 0.00, "Solo", 0),
    ("goose", "Untitled Goose Game", "puzles", 1, 1, 10, 15, 0.90, 0.05, 0.20, 0.15, 0.15, 0.75, 0.10, 0.05, 0.40, 0.85, 0.10, 0.10, "Solo", 0),
    ("outer", "Outer Wilds", "aventura", 1, 15, 5, 40, 0.50, 0.85, 0.75, 0.20, 0.60, 0.45, 0.15, 0.35, 0.10, 0.15, 0.70, 0.40, "Solo", 0),
    ("obra", "Return of the Obra Dinn", "puzles", 1, 10, 10, 40, 0.90, 0.80, 0.90, 0.05, 0.40, 0.10, -0.05, 0.20, 0.05, 0.10, 0.45, 0.35, "Solo", 0),
    ("baba", "Baba Is You", "puzles", 1, 5, 10, 25, 0.95, 0.35, 0.95, 0.05, 0.05, 0.20, 0.00, 0.15, 0.10, 0.40, 0.05, 0.40, "Solo", 0),
    ("breach", "Into the Breach", "estrategia", 1, 2, 10, 20, 0.98, 0.15, 0.70, 0.10, 0.10, 0.15, 0.05, 0.20, 0.25, 0.50, 0.10, 0.45, "Solo", 0),
    ("ftl", "FTL: Faster Than Light", "roguelike", 1, 3, 15, 25, 0.85, 0.25, 0.60, 0.15, 0.20, 0.10, 0.20, 0.40, 0.20, 0.45, 0.15, 0.55, "Solo", 0),
    ("rimworld", "RimWorld", "simulacion", 1, 15, 10, 45, 0.90, 0.35, 0.85, 0.15, 0.25, 0.05, 0.10, 0.30, 0.10, 0.25, 0.20, 0.40, "Solo", 0),
    ("cities", "Cities: Skylines", "simulacion", 0, 10, 10, 40, 0.95, 0.25, 0.70, 0.10, 0.10, 0.40, -0.15, 0.10, 0.45, 0.40, 0.05, 0.10, "Solo", 0),
    ("mariokart", "Mario Kart 8 Deluxe", "carreras", 0, 1, 5, 12, 0.70, 0.05, 0.15, 0.55, 0.00, 0.70, 0.60, 0.15, 0.20, 0.85, 0.00, 0.45, "Coop", 0),
    ("botw", "The Legend of Zelda: Breath of the Wild", "aventura", 0, 10, 15, 35, 0.75, 0.40, 0.40, 0.45, 0.45, 0.55, 0.15, 0.15, 0.35, 0.40, 0.50, 0.35, "Solo", 0),
    ("totk", "The Legend of Zelda: Tears of the Kingdom", "aventura", 0, 12, 15, 40, 0.70, 0.45, 0.50, 0.45, 0.50, 0.50, 0.20, 0.20, 0.30, 0.35, 0.55, 0.40, "Solo", 0),
    ("dread", "Metroid Dread", "accion", 0, 5, 12, 25, 0.45, 0.40, 0.40, 0.75, 0.30, 0.10, 0.50, 0.45, 0.05, 0.25, 0.25, 0.75, "Solo", 0),
    ("cuphead", "Cuphead", "plataformas", 1, 2, 8, 15, 0.40, 0.20, 0.25, 0.95, 0.10, -0.10, 0.70, 0.90, 0.00, 0.15, 0.05, 0.95, "Solo", 0),
    ("ori", "Ori and the Blind Forest", "plataformas", 1, 5, 15, 25, 0.70, 0.30, 0.30, 0.55, 0.60, 0.45, 0.15, 0.35, 0.25, 0.30, 0.65, 0.40, "Solo", 0),
    ("inside", "Inside", "plataformas", 1, 2, 20, 25, 0.55, 0.20, 0.25, 0.30, 0.70, -0.15, -0.10, 0.25, 0.20, 0.25, 0.80, 0.15, "Solo", 0),
    ("limbo", "Limbo", "plataformas", 1, 2, 15, 20, 0.60, 0.15, 0.25, 0.30, 0.55, -0.20, -0.15, 0.20, 0.25, 0.30, 0.60, 0.15, "Solo", 0),
    ("stray", "Stray", "aventura", 1, 3, 12, 25, 0.80, 0.15, 0.20, 0.20, 0.55, 0.65, -0.15, 0.08, 0.55, 0.50, 0.60, 0.10, "Solo", 0),
    ("cocoon", "Cocoon", "puzles", 1, 3, 12, 25, 0.85, 0.25, 0.55, 0.15, 0.30, 0.40, -0.05, 0.10, 0.30, 0.45, 0.30, 0.25, "Solo", 0),
    ("balatro", "Balatro", "roguelike", 1, 1, 8, 15, 0.95, 0.08, 0.45, 0.10, 0.05, 0.35, 0.15, 0.15, 0.40, 0.80, 0.00, 0.40, "Solo", 0),
    ("satisfactory", "Satisfactory", "simulacion", 1, 15, 10, 45, 0.90, 0.40, 0.75, 0.20, 0.10, 0.30, 0.00, 0.15, 0.20, 0.35, 0.05, 0.20, "Coop", 0),
    ("factorio", "Factorio", "simulacion", 1, 15, 10, 50, 0.90, 0.45, 0.90, 0.15, 0.05, 0.10, 0.10, 0.25, 0.10, 0.20, 0.05, 0.30, "Solo", 0),
    ("deeprock", "Deep Rock Galactic", "cooperativo", 1, 5, 20, 25, 0.35, 0.20, 0.35, 0.65, 0.15, 0.40, 0.55, 0.30, 0.15, 0.45, 0.10, 0.60, "Coop", 1),
]


def _comment(game: dict) -> str:
    """Redacta un comentario simulado a partir de las características del juego."""
    chunks = []
    # Se elige una frase de uso según la carga, la activación y la sesión mínima.
    if game["carga_cognitiva"] <= 0.35 and game["arousal"] <= 0.15 and game["duracion_minima_sesion_min"] <= 25:
        chunks.append(
            "Ideal después del trabajo, partida corta para desconectar cuando llego cansado, sin pensar mucho."
        )
    elif game["duracion_minima_sesion_min"] >= 50:
        chunks.append(
            "No rinde en media hora. Demasiado largo. Necesitas un bloque largo, mejor el fin de semana."
        )
    else:
        chunks.append("Se puede abrir un rato, aunque no es la opción más suave.")
    # Se añaden observaciones sobre volver al juego y sobre la frustración prevista.
    if game["costo_reenganche"] >= 0.55:
        chunks.append("Si lo dejas olvidas los controles, cuesta retomar y pierdes el hilo de la trama.")
    else:
        chunks.append("Retomar fácil, controles simples, se vuelve a entrar y no olvidas los controles.")
    if game["riesgo_frustracion"] >= 0.55:
        chunks.append("Es frustrante, punitivo y da rabia cuando te castiga.")
    else:
        chunks.append("Se siente justo y se puede dejar sin castigo.")
    return " ".join(chunks)


def _variant(game: dict) -> str:
    """Crea una segunda versión del comentario con una observación sobre la historia."""
    if game["densidad_narrativa"] >= 0.75:
        story = "La historia pide atención y no es una partida corta."
    else:
        story = "No depende de una trama larga."
    return _comment(game) + " " + story


# Nombres de las columnas, en el mismo orden que los valores de cada fila de GAMES.
COLUMNS = [
    "game_id",
    "titulo",
    "genero",
    "es_indie",
    "minutos_hasta_diversion",
    "intervalo_guardado_min",
    "duracion_minima_sesion_min",
    "facilidad_pausa",
    "costo_reenganche",
    "carga_cognitiva",
    "exigencia_motora",
    "densidad_narrativa",
    "valencia",
    "arousal",
    "riesgo_frustracion",
    "mu_relajacion",
    "mu_ligero",
    "mu_historia",
    "mu_reto",
    "modalidad",
    "requiere_microfono",
]


def catalog_frame() -> pd.DataFrame:
    """Convierte las filas de videojuegos en un catálogo con su origen identificado."""
    rows = []
    for raw in GAMES:
        record = dict(zip(COLUMNS, raw))
        record["cuadrante"] = russell_quadrant(record["valencia"], record["arousal"])
        record["fuente_atributos"] = "estimacion_de_diseno_no_medicion"
        rows.append(record)
    return pd.DataFrame(rows)


def comments_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """Genera dos comentarios por juego y los identifica como texto sintético."""
    rows = []
    for game in catalog.to_dict(orient="records"):
        for index, text in enumerate((_comment(game), _variant(game)), start=1):
            rows.append(
                {
                    "comment_id": f"{game['game_id']}_{index}",
                    "game_id": game["game_id"],
                    "fuente": "corpus_sintetico",
                    "texto": text,
                }
            )
    return pd.DataFrame(rows)


# Biblioteca de Andrés, el perfil usado para comparar los escenarios.
# Cada par indica las horas jugadas y los días desde la última sesión.
LIBRARY_U01 = {
    "hades": (140, 2),
    "stardew": (36, 1),
    "elden": (28, 24),
    "bg3": (10, 20),
    "rdr2": (8, 35),
    "rocket": (32, 5),
    "celeste": (14, 9),
    "slay": (20, 4),
    "disco": (4, 27),
    "unpacking": (0, 0),
    "coffee": (6, 3),
    "vampire": (11, 6),
    "civ6": (48, 15),
    "ittakes": (5, 40),
    "deeprock": (16, 8),
    "balatro": (9, 2),
    "witcher": (7, 22),
    "dorf": (3, 12),
}


def _users() -> pd.DataFrame:
    """Define personas simuladas, su perfil, juego refugio y pulso habitual."""
    rows = [
        ("U01", "Andres", "alta_carga", "stardew", 64),
        ("U02", "Laura", "alta_carga", "coffee", 60),
        ("U03", "Camilo", "alta_carga", "vampire", 66),
        ("U04", "Sofia", "fin_de_semana", "bg3", 61),
        ("U05", "Mateo", "fin_de_semana", "witcher", 63),
        ("U06", "Valentina", "fin_de_semana", "rdr2", 62),
        ("U07", "Diego", "competitivo", "rocket", 70),
        ("U08", "Sara", "competitivo", "valorant", 68),
        ("U09", "Julian", "competitivo", "hades", 67),
        ("U10", "Nina", "alta_carga", "dorf", 65),
        ("U11", "Pedro", "alta_carga", "stardew", 63),
        ("U12", "Lucia", "fin_de_semana", "disco", 60),
        ("U13", "Hugo", "competitivo", "deadcells", 72),
        ("U14", "Irene", "alta_carga", "unpacking", 61),
        ("U15", "Tomas", "fin_de_semana", "tlou", 64),
    ]
    return pd.DataFrame(
        rows,
        columns=["user_id", "nombre", "perfil_ocupacional", "juego_refugio", "hr_baseline"],
    )


def _libraries(catalog: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada persona sus juegos, horas acumuladas y días sin jugar."""
    all_ids = catalog["game_id"].tolist()
    # Las primeras bibliotecas se definen a mano para representar usos distintos.
    assignments = {
        "U01": LIBRARY_U01,
        "U02": {
            "coffee": (40, 1),
            "unpacking": (18, 2),
            "stardew": (22, 3),
            "dorf": (8, 4),
            "animal": (15, 6),
            "gris": (5, 8),
            "hades": (12, 10),
            "bg3": (3, 30),
        },
        "U03": {
            "vampire": (60, 1),
            "unpacking": (14, 2),
            "stardew": (10, 5),
            "balatro": (20, 3),
            "slay": (16, 7),
            "coffee": (4, 2),
            "elden": (6, 18),
        },
        "U04": {
            "bg3": (80, 2),
            "disco": (20, 6),
            "witcher": (30, 10),
            "rdr2": (15, 12),
            "tlou": (12, 8),
            "stardew": (5, 20),
        },
        "U05": {
            "witcher": (90, 3),
            "skyrim": (40, 8),
            "bg3": (12, 14),
            "cyberpunk": (18, 9),
            "rdr2": (10, 16),
        },
        "U06": {
            "rdr2": (70, 4),
            "tlou": (25, 7),
            "bg3": (9, 11),
            "spirit": (6, 5),
            "gow": (14, 13),
        },
        "U07": {
            "rocket": (120, 1),
            "fc24": (40, 2),
            "valorant": (30, 3),
            "hades": (15, 6),
            "mariokart": (10, 4),
        },
        "U08": {
            "valorant": (150, 1),
            "lol": (80, 2),
            "rocket": (20, 5),
            "fortnite": (25, 4),
            "cuphead": (8, 9),
        },
        "U09": {
            "hades": (100, 1),
            "hades2": (20, 3),
            "deadcells": (30, 4),
            "celeste": (12, 8),
            "hollow": (18, 6),
        },
        "U10": {
            "dorf": (8, 2),
            "coffee": (2, 4),
            "unpacking": (0, 0),
            "stardew": (1, 15),
            "tetris": (3, 6),
        },
    }
    # Las cinco bibliotecas restantes se sortean con una semilla fija.
    rng_py = random.Random(SEED)
    extra_users = ["U11", "U12", "U13", "U14", "U15"]
    for user_id in extra_users:
        picks = rng_py.sample(all_ids, 8)
        assignments[user_id] = {game_id: (rng_py.randint(0, 40), rng_py.randint(0, 30)) for game_id in picks}
    # Se guarda una fila por relación persona-juego para facilitar los cruces.
    rows = []
    for user_id, games in assignments.items():
        for game_id, (hours, days) in games.items():
            rows.append(
                {
                    "user_id": user_id,
                    "game_id": game_id,
                    "horas_jugadas": hours,
                    "dias_sin_jugar": days,
                }
            )
    return pd.DataFrame(rows)


def _sessions(library: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Construye sesiones de ejemplo con duración, señales simuladas y cierre."""
    rng = random.Random(SEED)
    rows = []
    session_id = 1
    # Estos ejemplos dan al componente de vecinos sesiones repetidas por juego
    # con las que calcular satisfacción y formas de cierre.
    example_sessions = [
        ("U02", "unpacking", "GUARDADO_FIN_NATURAL", 5, 30, 78, 32, 5.5),
        ("U02", "unpacking", "GUARDADO_FIN_NATURAL", 5, 28, 80, 30, 6.0),
        ("U03", "unpacking", "GUARDADO_FIN_NATURAL", 4, 25, 76, 34, 6.5),
        ("U11", "unpacking", "GUARDADO_FIN_NATURAL", 5, 32, 74, 36, 7.0),
        ("U14", "unpacking", "GUARDADO_FIN_NATURAL", 5, 20, 70, 40, 7.5),
        ("U02", "stardew", "GUARDADO_FIN_NATURAL", 5, 40, 72, 38, 7.0),
        ("U03", "stardew", "GUARDADO_FIN_NATURAL", 4, 35, 75, 33, 6.5),
        ("U02", "elden", "SALIDA_PREMATURA_FRUSTRACION", 2, 18, 88, 28, 5.0),
        ("U03", "elden", "SALIDA_PREMATURA_FRUSTRACION", 1, 12, 90, 26, 4.5),
        ("U04", "bg3", "GUARDADO_FIN_NATURAL", 5, 150, 68, 48, 8.0),
        ("U05", "bg3", "GUARDADO_FIN_NATURAL", 5, 120, 70, 46, 7.5),
        ("U06", "rdr2", "GUARDADO_FIN_NATURAL", 5, 130, 66, 50, 8.0),
        ("U07", "rocket", "GUARDADO_FIN_NATURAL", 4, 25, 82, 35, 7.0),
        ("U08", "rocket", "GUARDADO_FIN_NATURAL", 4, 20, 84, 33, 6.5),
        ("U09", "hades", "GUARDADO_FIN_NATURAL", 5, 30, 80, 36, 7.0),
        ("U01", "hades", "GUARDADO_FIN_NATURAL", 4, 28, 79, 37, 6.5),
        ("U01", "stardew", "GUARDADO_FIN_NATURAL", 5, 35, 77, 34, 6.0),
        ("U01", "civ6", "CAMBIO_JUEGO", 2, 20, 86, 30, 5.0),
    ]
    for user_id, game_id, event, satisfaction, minutes_played, hr, hrv, sleep in example_sessions:
        rows.append(
            _session_row(
                session_id, user_id, game_id, minutes_played, hr, hrv, sleep, event, satisfaction, "ejemplo"
            )
        )
        session_id += 1

    # Se agregan sesiones para una muestra de relaciones con horas jugadas.
    # El perfil define las probabilidades de cierre y la satisfacción simulada.
    for record in library.itertuples(index=False):
        if record.horas_jugadas <= 0 or rng.random() > 0.35:
            continue
        profile = users.loc[users["user_id"] == record.user_id, "perfil_ocupacional"].iloc[0]
        if profile == "alta_carga":
            event = "GUARDADO_FIN_NATURAL" if rng.random() > 0.25 else "SALIDA_PREMATURA_FRUSTRACION"
            satisfaction = 4 if event.startswith("GUARDADO") else 2
        elif profile == "fin_de_semana":
            event = "GUARDADO_FIN_NATURAL"
            satisfaction = rng.choice([4, 5])
        else:
            event = rng.choice(["GUARDADO_FIN_NATURAL", "CAMBIO_JUEGO"])
            satisfaction = rng.choice([3, 4, 5])
        rows.append(
            _session_row(
                session_id,
                record.user_id,
                record.game_id,
                rng.randint(10, 90),
                rng.randint(64, 92),
                rng.randint(28, 55),
                round(rng.uniform(4.5, 8.0), 1),
                event,
                satisfaction,
                "muestreo",
            )
        )
        session_id += 1
    return pd.DataFrame(rows)


def _session_row(session_id, user_id, game_id, minutes_played, hr, hrv, sleep, event, satisfaction, origin) -> dict:
    """Arma una fila de sesión con los campos usados por el motor y el cuaderno."""
    return {
        "session_id": f"S{session_id:03d}",
        "user_id": user_id,
        "game_id": game_id,
        "evento_apertura": "INICIO",
        "minutos_libres": int(minutes_played) + 5,
        "minutos_jugados": minutes_played,
        "hr": hr,
        "hrv": hrv,
        # El campo llamado gsr se calcula desde HRV; no representa una lectura
        # independiente de conductancia de la piel, aunque conserve ese nombre.
        "gsr": round(max(0.1, (90 - hrv) / 80), 3),
        "horas_sueno": sleep,
        "evento_cierre": event,
        "satisfaccion": satisfaction,
        "origen": origin,
    }


def main() -> None:
    """Genera las cinco tablas y reemplaza los CSV de la carpeta de datos."""
    catalog = catalog_frame()
    comments = comments_frame(catalog)
    # El léxico convierte los comentarios en puntajes y los agrega al catálogo.
    lexicon = aggregate_lexicon(comments)
    catalog = catalog.drop(columns=[c for c in ("score_post_trabajo", "score_juego_confort", "score_facil_retomar") if c in catalog.columns])
    catalog = catalog.merge(lexicon, on="game_id", how="left")
    # Primero se definen personas y bibliotecas; después se construye el historial.
    users = _users()
    library = _libraries(catalog)
    sessions = _sessions(library, users)
    # Cada tabla se guarda sin el índice de pandas: sus identificadores ya son columnas.
    catalog.to_csv(DATA / "catalogo_videojuegos_profundo.csv", index=False)
    comments.to_csv(DATA / "comentarios_foros_videojuegos.csv", index=False)
    users.to_csv(DATA / "usuarios_perfiles_ocupacionales.csv", index=False)
    library.to_csv(DATA / "biblioteca_usuario.csv", index=False)
    sessions.to_csv(DATA / "sesiones_contextuales_wearable.csv", index=False)
    print(f"juegos={len(catalog)} comentarios={len(comments)} usuarios={len(users)} relaciones_biblioteca={len(library)} sesiones={len(sessions)}")


if __name__ == "__main__":
    main()
