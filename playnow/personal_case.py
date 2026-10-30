"""Caso personal de Jairo para demostrar PlayNow sin alterar el experimento académico.

Los atributos profundos son estimaciones de diseño construidas a partir de su
biblioteca y preferencias declaradas. Este módulo no modifica los CSV del
prototipo ni participa en las pruebas de sus 15 usuarios simulados.
"""

from __future__ import annotations

import pandas as pd


# Perfiles de juego usados para caracterizar la biblioteca personal de forma
# consistente. Son aproximaciones explícitas para la demostración, no mediciones.
ARCHETYPES = {
    "aventura_narrativa": {
        "genero": "aventura_narrativa", "es_indie": 0, "minutos_hasta_diversion": 10,
        "intervalo_guardado_min": 15, "duracion_minima_sesion_min": 35, "facilidad_pausa": 0.70,
        "costo_reenganche": 0.26, "carga_cognitiva": 0.46, "exigencia_motora": 0.42,
        "densidad_narrativa": 0.86, "valencia": 0.30, "arousal": 0.16,
        "riesgo_frustracion": 0.20, "mu_relajacion": 0.26, "mu_ligero": 0.38,
        "mu_historia": 0.94, "mu_reto": 0.25, "modalidad": "Solo",
        "score_post_trabajo": 0.46, "score_facil_retomar": 0.56, "score_juego_confort": 0.84,
    },
    "mundo_abierto": {
        "genero": "mundo_abierto", "es_indie": 0, "minutos_hasta_diversion": 9,
        "intervalo_guardado_min": 14, "duracion_minima_sesion_min": 40, "facilidad_pausa": 0.68,
        "costo_reenganche": 0.30, "carga_cognitiva": 0.46, "exigencia_motora": 0.53,
        "densidad_narrativa": 0.68, "valencia": 0.42, "arousal": 0.28,
        "riesgo_frustracion": 0.24, "mu_relajacion": 0.34, "mu_ligero": 0.50,
        "mu_historia": 0.80, "mu_reto": 0.40, "modalidad": "Solo",
        "score_post_trabajo": 0.58, "score_facil_retomar": 0.64, "score_juego_confort": 0.82,
    },
    "partida_rapida": {
        "genero": "partida_rapida", "es_indie": 0, "minutos_hasta_diversion": 3,
        "intervalo_guardado_min": 8, "duracion_minima_sesion_min": 12, "facilidad_pausa": 0.90,
        "costo_reenganche": 0.08, "carga_cognitiva": 0.28, "exigencia_motora": 0.45,
        "densidad_narrativa": 0.08, "valencia": 0.56, "arousal": 0.28,
        "riesgo_frustracion": 0.16, "mu_relajacion": 0.52, "mu_ligero": 0.96,
        "mu_historia": 0.08, "mu_reto": 0.36, "modalidad": "Solo",
        "score_post_trabajo": 0.92, "score_facil_retomar": 0.93, "score_juego_confort": 0.76,
    },
    "social": {
        "genero": "social", "es_indie": 0, "minutos_hasta_diversion": 4,
        "intervalo_guardado_min": 10, "duracion_minima_sesion_min": 15, "facilidad_pausa": 0.82,
        "costo_reenganche": 0.10, "carga_cognitiva": 0.32, "exigencia_motora": 0.55,
        "densidad_narrativa": 0.05, "valencia": 0.60, "arousal": 0.48,
        "riesgo_frustracion": 0.28, "mu_relajacion": 0.38, "mu_ligero": 0.90,
        "mu_historia": 0.05, "mu_reto": 0.54, "modalidad": "PvP",
        "score_post_trabajo": 0.74, "score_facil_retomar": 0.90, "score_juego_confort": 0.70,
    },
    "exploracion_suave": {
        "genero": "exploracion", "es_indie": 0, "minutos_hasta_diversion": 6,
        "intervalo_guardado_min": 10, "duracion_minima_sesion_min": 25, "facilidad_pausa": 0.86,
        "costo_reenganche": 0.14, "carga_cognitiva": 0.31, "exigencia_motora": 0.27,
        "densidad_narrativa": 0.34, "valencia": 0.62, "arousal": -0.18,
        "riesgo_frustracion": 0.10, "mu_relajacion": 0.78, "mu_ligero": 0.76,
        "mu_historia": 0.38, "mu_reto": 0.16, "modalidad": "Solo",
        "score_post_trabajo": 0.88, "score_facil_retomar": 0.82, "score_juego_confort": 0.90,
    },
    "reto": {
        "genero": "accion_desafiante", "es_indie": 0, "minutos_hasta_diversion": 8,
        "intervalo_guardado_min": 15, "duracion_minima_sesion_min": 30, "facilidad_pausa": 0.54,
        "costo_reenganche": 0.32, "carga_cognitiva": 0.58, "exigencia_motora": 0.84,
        "densidad_narrativa": 0.30, "valencia": 0.05, "arousal": 0.76,
        "riesgo_frustracion": 0.68, "mu_relajacion": 0.08, "mu_ligero": 0.24,
        "mu_historia": 0.25, "mu_reto": 0.94, "modalidad": "Solo",
        "score_post_trabajo": 0.20, "score_facil_retomar": 0.46, "score_juego_confort": 0.34,
    },
    "terror": {
        "genero": "survival_horror", "es_indie": 0, "minutos_hasta_diversion": 10,
        "intervalo_guardado_min": 18, "duracion_minima_sesion_min": 35, "facilidad_pausa": 0.58,
        "costo_reenganche": 0.30, "carga_cognitiva": 0.56, "exigencia_motora": 0.60,
        "densidad_narrativa": 0.58, "valencia": -0.28, "arousal": 0.72,
        "riesgo_frustracion": 0.52, "mu_relajacion": 0.05, "mu_ligero": 0.20,
        "mu_historia": 0.60, "mu_reto": 0.60, "modalidad": "Solo",
        "score_post_trabajo": 0.20, "score_facil_retomar": 0.45, "score_juego_confort": 0.30,
    },
    "retro": {
        "genero": "retro", "es_indie": 0, "minutos_hasta_diversion": 3,
        "intervalo_guardado_min": 10, "duracion_minima_sesion_min": 20, "facilidad_pausa": 0.82,
        "costo_reenganche": 0.12, "carga_cognitiva": 0.32, "exigencia_motora": 0.43,
        "densidad_narrativa": 0.25, "valencia": 0.56, "arousal": 0.18,
        "riesgo_frustracion": 0.24, "mu_relajacion": 0.60, "mu_ligero": 0.66,
        "mu_historia": 0.32, "mu_reto": 0.42, "modalidad": "Solo",
        "score_post_trabajo": 0.78, "score_facil_retomar": 0.86, "score_juego_confort": 0.82,
    },
}


# Ajustes concretos a partir del historial y preferencias declaradas por Jairo.
# Permiten que títulos de un mismo arquetipo no se comporten como copias exactas.
PERSONAL_TUNING = {
    "gta_san_andreas": {"carga_cognitiva": 0.30, "mu_ligero": 0.94, "facilidad_pausa": 0.90, "score_post_trabajo": 0.94, "score_juego_confort": 0.96},
    "gta_v": {"carga_cognitiva": 0.34, "mu_ligero": 0.88, "facilidad_pausa": 0.86, "score_post_trabajo": 0.88, "score_juego_confort": 0.98},
    "ea_fc_26": {"carga_cognitiva": 0.31, "mu_ligero": 0.93, "exigencia_motora": 0.42, "score_juego_confort": 0.76},
    "minecraft": {"carga_cognitiva": 0.24, "mu_relajacion": 0.88, "mu_ligero": 0.86, "score_juego_confort": 0.86},
    "mortal_kombat": {"carga_cognitiva": 0.35, "mu_ligero": 0.87, "mu_reto": 0.58, "score_juego_confort": 0.70},
    "ratchet_rift_apart": {"carga_cognitiva": 0.32, "mu_ligero": 0.93, "facilidad_pausa": 0.88, "score_juego_confort": 0.86},
    "dave_the_diver": {"carga_cognitiva": 0.27, "mu_relajacion": 0.84, "mu_ligero": 0.88, "score_juego_confort": 0.90},
    "spider_man_miles": {"carga_cognitiva": 0.36, "mu_ligero": 0.78, "score_juego_confort": 0.88},
    "marvel_wolverine": {"carga_cognitiva": 0.42, "mu_historia": 0.88, "score_juego_confort": 0.84},
    "zelda_minish_cap": {"mu_ligero": 0.76, "score_juego_confort": 0.82},
    "zelda_ocarina": {"mu_ligero": 0.62, "score_juego_confort": 0.80},
}


# (identificador, título, arquetipo, horas estimadas, días desde la última sesión).
# Las horas son rangos de demostración derivados del estado declarado, no telemetría.
PERSONAL_LIBRARY = [
    ("ace_combat_zero", "Ace Combat Zero The Belkan War", "retro", 30, 600),
    ("batman_arkham", "Batman Arkham saga", "mundo_abierto", 45, 180),
    ("beyond_two_souls", "Beyond Two Souls", "aventura_narrativa", 14, 500),
    ("cyberpunk_2077", "Cyberpunk 2077", "mundo_abierto", 70, 240),
    ("dave_the_diver", "Dave the Diver", "exploracion_suave", 0, 999),
    ("days_gone", "Days Gone", "mundo_abierto", 58, 280),
    ("death_stranding", "Death Stranding Director's Cut", "exploracion_suave", 0, 999),
    ("demons_souls", "Demon's Souls", "reto", 0, 999),
    ("detroit_become_human", "Detroit: Become Human", "aventura_narrativa", 22, 380),
    ("ea_fc_26", "EA Sports FC 26", "social", 12, 8),
    ("final_fantasy_xvi", "Final Fantasy XVI", "aventura_narrativa", 0, 999),
    ("ghost_tsushima", "Ghost of Tsushima Director's Cut", "mundo_abierto", 0, 999),
    ("gta_chinatown_wars", "Grand Theft Auto: Chinatown Wars", "retro", 18, 700),
    ("gta_online", "Grand Theft Auto Online", "social", 35, 30),
    ("gta_san_andreas", "Grand Theft Auto: San Andreas", "mundo_abierto", 12, 4),
    ("gta_v", "Grand Theft Auto V", "mundo_abierto", 120, 45),
    ("gta_vice_city", "Grand Theft Auto: Vice City", "retro", 16, 70),
    ("heavy_rain", "Heavy Rain", "aventura_narrativa", 16, 620),
    ("horizon_forbidden_west", "Horizon Forbidden West", "mundo_abierto", 28, 210),
    ("horizon_zero_dawn", "Horizon Zero Dawn", "mundo_abierto", 45, 320),
    ("kingdom_come_ii", "Kingdom Come: Deliverance II", "mundo_abierto", 0, 999),
    ("mafia_definitive", "Mafia Definitive Edition", "aventura_narrativa", 14, 300),
    ("mafia_ii", "Mafia II", "aventura_narrativa", 16, 420),
    ("mafia_iii", "Mafia III", "mundo_abierto", 0, 999),
    ("spider_man_2", "Marvel's Spider-Man 2", "mundo_abierto", 22, 120),
    ("spider_man_miles", "Marvel's Spider-Man: Miles Morales", "mundo_abierto", 14, 180),
    ("spider_man", "Marvel's Spider-Man", "mundo_abierto", 38, 250),
    ("marvel_wolverine", "Marvel's Wolverine", "aventura_narrativa", 2, 3),
    ("minecraft", "Minecraft", "exploracion_suave", 20, 40),
    ("mortal_kombat", "Mortal Kombat Legacy Kollection", "partida_rapida", 3, 90),
    ("pokemon_lets_go", "Pokémon: Let's Go, Pikachu!", "exploracion_suave", 0, 999),
    ("ratchet_rift_apart", "Ratchet & Clank: Rift Apart", "partida_rapida", 0, 999),
    ("red_dead_online", "Red Dead Online", "social", 15, 300),
    ("red_dead_undead", "Red Dead Redemption: Undead Nightmare", "mundo_abierto", 0, 999),
    ("resident_evil_2", "Resident Evil 2 Remake", "terror", 0, 999),
    ("resident_evil_4_2005", "Resident Evil 4 (2005)", "terror", 12, 500),
    ("resident_evil_4_remake", "Resident Evil 4 Remake", "terror", 0, 999),
    ("resident_evil_village", "Resident Evil Village", "terror", 0, 999),
    ("returnal", "Returnal", "reto", 0, 999),
    ("last_of_us_part_i", "The Last of Us Part I", "aventura_narrativa", 25, 220),
    ("last_of_us_part_ii", "The Last of Us Part II", "aventura_narrativa", 28, 260),
    ("zelda_link_past", "The Legend of Zelda: A Link to the Past", "retro", 0, 999),
    ("zelda_breath_wild", "The Legend of Zelda: Breath of the Wild", "mundo_abierto", 8, 80),
    ("zelda_four_swords", "The Legend of Zelda: Four Swords", "retro", 0, 999),
    ("zelda_ocarina", "The Legend of Zelda: Ocarina of Time", "retro", 5, 120),
    ("zelda_minish_cap", "The Legend of Zelda: The Minish Cap", "retro", 0, 999),
    ("uncharted_2", "Uncharted 2: Among Thieves", "aventura_narrativa", 15, 700),
    ("uncharted_3", "Uncharted 3: Drake's Deception", "aventura_narrativa", 14, 650),
    ("uncharted_4", "Uncharted 4: A Thief's End", "aventura_narrativa", 22, 400),
    ("uncharted_drakes_fortune", "Uncharted: Drake's Fortune", "aventura_narrativa", 12, 750),
    ("uncharted_golden_abyss", "Uncharted: Golden Abyss", "aventura_narrativa", 0, 999),
    ("uncharted_lost_legacy", "Uncharted: The Lost Legacy", "aventura_narrativa", 0, 999),
    ("until_dawn", "Until Dawn", "terror", 10, 450),
]


def personal_case_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve tablas independientes para el caso personal de Jairo."""
    catalog_rows = []
    library_rows = []
    for index, (game_id, title, archetype_name, hours, days_since_playing) in enumerate(PERSONAL_LIBRARY):
        attributes = ARCHETYPES[archetype_name].copy()
        attributes.update(PERSONAL_TUNING.get(game_id, {}))
        # Variaciones pequeñas evitan tratar dos juegos distintos como el mismo
        # punto del catálogo cuando se calcula el agrupamiento auxiliar.
        variation = ((index % 7) - 3) * 0.008
        attributes["carga_cognitiva"] = round(min(1.0, max(0.0, attributes["carga_cognitiva"] + variation)), 3)
        attributes["arousal"] = round(min(1.0, max(-1.0, attributes["arousal"] - variation)), 3)
        attributes.update(
            {
                "game_id": game_id,
                "titulo": title,
                "requiere_microfono": int(attributes["modalidad"] == "PvP"),
                "fuente_atributos": "biblioteca_personal_y_estimacion_de_diseno",
                "riesgo_frustracion_lexico": attributes["riesgo_frustracion"],
            }
        )
        catalog_rows.append(attributes)
        library_rows.append(
            {
                "user_id": "JA01",
                "game_id": game_id,
                "horas_jugadas": hours,
                "dias_sin_jugar": days_since_playing,
            }
        )

    users = pd.DataFrame(
        [{"user_id": "JA01", "nombre": "Jairo", "perfil_ocupacional": "alta_carga", "juego_refugio": "gta_v", "hr_baseline": 62}]
    )
    sessions = pd.DataFrame(columns=["user_id", "game_id", "evento_cierre", "satisfaccion"])
    return pd.DataFrame(catalog_rows), pd.DataFrame(library_rows), users, sessions
