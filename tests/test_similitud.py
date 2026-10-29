"""Pruebas de cercanía afectiva, reenganche y recomendaciones de referencia."""

from itertools import product
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from playnow.engine import (
    PlayNowEngine,
    _apply_reengage,
    _russell_similarity,
    fatigue_from_wearable,
)


class CercaniaAfectivaTests(unittest.TestCase):
    def test_puntos_iguales_tienen_cercania_uno(self):
        for valencia, activacion in product((-1.0, 0.0, 1.0), repeat=2):
            with self.subTest(valencia=valencia, activacion=activacion):
                self.assertEqual(
                    _russell_similarity(valencia, activacion, valencia, activacion), 1.0
                )

    def test_esquinas_opuestas_tienen_cercania_cero(self):
        self.assertAlmostEqual(_russell_similarity(-1.0, -1.0, 1.0, 1.0), 0.0)
        self.assertAlmostEqual(_russell_similarity(-1.0, 1.0, 1.0, -1.0), 0.0)

    def test_se_usa_la_diagonal_completa_del_plano(self):
        # Recorrer un lado del cuadrado no equivale a recorrer toda su diagonal.
        self.assertAlmostEqual(
            _russell_similarity(-1.0, -1.0, 1.0, -1.0), 1.0 - 1.0 / np.sqrt(2.0)
        )

    def test_cercania_acotada_y_simetrica(self):
        puntos = list(product((-1.0, 0.0, 1.0), repeat=2))
        for origen, destino in product(puntos, repeat=2):
            with self.subTest(origen=origen, destino=destino):
                valor = _russell_similarity(*origen, *destino)
                self.assertGreaterEqual(valor, 0.0)
                self.assertLessEqual(valor, 1.0)
                self.assertAlmostEqual(valor, _russell_similarity(*destino, *origen))

    def test_bono_de_reenganche_se_satura_en_uno(self):
        self.assertEqual(_apply_reengage(0.95, -0.20), 1.0)
        self.assertAlmostEqual(_apply_reengage(0.40, -0.10), 0.50)

    def test_penalizacion_de_reenganche_se_acota_en_cero(self):
        self.assertEqual(_apply_reengage(0.20, 0.70), 0.0)
        self.assertAlmostEqual(_apply_reengage(0.80, 0.20), 0.60)


class RecomendacionesTests(unittest.TestCase):
    # Minutos, pulso, HRV, sueño, intención, juegos esperados y puntajes documentados.
    ESCENARIOS = (
        (35, 92, 28, 4.5, "relajarme", ("unpacking", "stardew", "coffee"), (0.9458, 0.8390, 0.8307)),
        (180, 66, 52, 8.0, "historia", ("bg3", "disco", "witcher"), (0.9657, 0.8443, 0.7617)),
        (50, 78, 40, 6.5, "reto", ("rocket", "hades", "celeste"), (0.7817, 0.7619, 0.7441)),
    )

    @classmethod
    def setUpClass(cls):
        datos = Path(__file__).resolve().parents[1] / "data"
        catalogo = pd.read_csv(datos / "catalogo_videojuegos_profundo.csv")
        cls.bibliotecas = pd.read_csv(datos / "biblioteca_usuario.csv")
        cls.usuarios = pd.read_csv(datos / "usuarios_perfiles_ocupacionales.csv")
        sesiones = pd.read_csv(datos / "sesiones_contextuales_wearable.csv")
        cls.motor = PlayNowEngine(catalogo, cls.bibliotecas, cls.usuarios, sesiones)

    def test_top_tres_y_puntajes_de_referencia(self):
        pulso_base = float(self.usuarios.set_index("user_id").loc["U01", "hr_baseline"])
        for minutos, pulso, hrv, sueno, intencion, juegos, puntajes in self.ESCENARIOS:
            with self.subTest(intencion=intencion):
                fatiga = fatigue_from_wearable(pulso, hrv, sueno, pulso_base)
                lista = self.motor.rank("U01", minutos, fatiga, intencion).head(3)
                self.assertEqual(lista["game_id"].tolist(), list(juegos))
                for actual, esperado in zip(lista["score"], puntajes):
                    self.assertAlmostEqual(actual, esperado, places=4)

    def test_limites_y_biblioteca_en_los_quince_perfiles(self):
        for persona in self.usuarios.itertuples(index=False):
            propios = set(
                self.bibliotecas.loc[
                    self.bibliotecas["user_id"] == persona.user_id, "game_id"
                ]
            )
            for minutos, pulso, hrv, sueno, intencion, _, _ in self.ESCENARIOS:
                with self.subTest(persona=persona.user_id, intencion=intencion):
                    fatiga = fatigue_from_wearable(pulso, hrv, sueno, persona.hr_baseline)
                    lista = self.motor.rank(persona.user_id, minutos, fatiga, intencion)
                    self.assertFalse(lista.empty)
                    self.assertTrue(set(lista["game_id"]).issubset(propios))
                    for columna in ("similitud", "similitud_russell", "score"):
                        self.assertTrue(lista[columna].between(0.0, 1.0).all(), columna)


if __name__ == "__main__":
    unittest.main()
