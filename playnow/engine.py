"""Ranking engine for PlayNow.

The candidate set is always the user's owned library. Cluster labels explain
families of games. They do not decide the order of the list.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skfuzzy import control as ctrl

SEED = 42

CONTENT_COLUMNS = [
    "carga_cognitiva",
    "exigencia_motora",
    "densidad_narrativa",
    "valencia_01",
    "arousal_01",
    "mu_relajacion",
    "mu_ligero",
    "mu_historia",
    "mu_reto",
    "facilidad_pausa",
    "score_post_trabajo",
    "score_juego_confort",
    "score_facil_retomar",
]

INTENTION_MU = {
    "relajarme": "mu_relajacion",
    "ligero": "mu_ligero",
    "historia": "mu_historia",
    "reto": "mu_reto",
}

INTENTION_TARGETS = {
    "relajarme": {
        "label": "Relajarme / Desconectar",
        "carga_cognitiva": 0.22,
        "exigencia_motora": 0.18,
        "densidad_narrativa": 0.25,
        "valencia": 0.65,
        "arousal": -0.60,
        "mu_relajacion": 0.92,
        "mu_ligero": 0.45,
        "mu_historia": 0.15,
        "mu_reto": 0.05,
        "facilidad_pausa": 0.90,
        "score_post_trabajo": 0.90,
        "score_juego_confort": 0.88,
        "score_facil_retomar": 0.85,
    },
    "ligero": {
        "label": "Partida rápida / Ligera",
        "carga_cognitiva": 0.35,
        "exigencia_motora": 0.40,
        "densidad_narrativa": 0.20,
        "valencia": 0.45,
        "arousal": 0.05,
        "mu_relajacion": 0.35,
        "mu_ligero": 0.90,
        "mu_historia": 0.15,
        "mu_reto": 0.25,
        "facilidad_pausa": 0.80,
        "score_post_trabajo": 0.70,
        "score_juego_confort": 0.60,
        "score_facil_retomar": 0.80,
    },
    "historia": {
        "label": "Sumergirme en una historia",
        "carga_cognitiva": 0.62,
        "exigencia_motora": 0.25,
        "densidad_narrativa": 0.90,
        "valencia": 0.25,
        "arousal": 0.10,
        "mu_relajacion": 0.20,
        "mu_ligero": 0.15,
        "mu_historia": 0.95,
        "mu_reto": 0.20,
        "facilidad_pausa": 0.45,
        "score_post_trabajo": 0.25,
        "score_juego_confort": 0.40,
        "score_facil_retomar": 0.30,
    },
    "reto": {
        "label": "Reto exigente / Competitivo",
        "carga_cognitiva": 0.55,
        "exigencia_motora": 0.85,
        "densidad_narrativa": 0.20,
        "valencia": 0.15,
        "arousal": 0.80,
        "mu_relajacion": 0.05,
        "mu_ligero": 0.25,
        "mu_historia": 0.10,
        "mu_reto": 0.95,
        "facilidad_pausa": 0.35,
        "score_post_trabajo": 0.20,
        "score_juego_confort": 0.25,
        "score_facil_retomar": 0.55,
    },
}

POST_WORK_POS = (
    "despues del trabajo",
    "para desconectar",
    "cuando llego cansado",
    "partida corta",
    "sin pensar mucho",
    "ideal de noche",
)
POST_WORK_NEG = (
    "no rinde en media hora",
    "mejor el fin de semana",
    "demasiado largo",
    "necesitas un bloque largo",
)
RETAKE_POS = (
    "retomar facil",
    "no olvidas los controles",
    "se vuelve a entrar",
    "controles simples",
)
RETAKE_NEG = (
    "si lo dejas olvidas",
    "cuesta retomar",
    "pierdes el hilo",
    "olvidas la trama",
)
TILT_TERMS = (
    "frustrante",
    "punitivo",
    "rabia",
    "rage",
    "injusto",
    "muy castigador",
)


def russell_quadrant(valence: float, arousal: float) -> str:
    if valence >= 0 and arousal >= 0:
        return "HVHA"
    if valence >= 0 and arousal < 0:
        return "HVLA"
    if valence < 0 and arousal >= 0:
        return "LVHA"
    return "LVLA"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(character for character in normalized if unicodedata.category(character) != "Mn")


def _hits(text: str, phrases: tuple[str, ...]) -> int:
    folded = _fold(text)
    return sum(1 for phrase in phrases if _fold(phrase) in folded)


def lexicon_scores(text: str) -> dict[str, float]:
    """Map a simulated forum comment to three implicit scores in [0, 1]."""
    post_pos = _hits(text, POST_WORK_POS)
    post_neg = _hits(text, POST_WORK_NEG)
    retake_pos = _hits(text, RETAKE_POS)
    retake_neg = _hits(text, RETAKE_NEG)
    tilt = _hits(text, TILT_TERMS)
    post = (post_pos + 0.5) / (post_pos + post_neg + 1.0)
    retake = (retake_pos + 0.5) / (retake_pos + retake_neg + 1.0)
    frustration = min(1.0, tilt / 2.0)
    confort = max(0.0, min(1.0, 1.0 - frustration))
    return {
        "score_post_trabajo": round(post, 3),
        "score_facil_retomar": round(retake, 3),
        "score_juego_confort": round(confort, 3),
        "riesgo_frustracion_lexico": round(frustration, 3),
    }


def aggregate_lexicon(comments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id, group in comments.groupby("game_id"):
        parts = [lexicon_scores(text) for text in group["texto"]]
        frame = pd.DataFrame(parts)
        summary = frame.mean().round(3).to_dict()
        summary["game_id"] = game_id
        rows.append(summary)
    return pd.DataFrame(rows)


class FuzzySessionModel:
    """Mamdani model: free minutes and fatigue -> admissible cognitive load."""

    def __init__(self) -> None:
        minutos = ctrl.Antecedent(np.arange(0, 241, 1), "minutos")
        fatiga = ctrl.Antecedent(np.arange(0, 101, 1), "fatiga")
        carga = ctrl.Consequent(np.arange(0, 101, 1), "carga_max")

        minutos["corto"] = fuzz.trapmf(minutos.universe, [0, 0, 25, 55])
        minutos["medio"] = fuzz.trapmf(minutos.universe, [35, 55, 80, 110])
        minutos["largo"] = fuzz.trapmf(minutos.universe, [90, 130, 240, 240])

        fatiga["baja"] = fuzz.trapmf(fatiga.universe, [0, 0, 20, 40])
        fatiga["media"] = fuzz.trapmf(fatiga.universe, [30, 45, 60, 75])
        fatiga["alta"] = fuzz.trapmf(fatiga.universe, [65, 80, 100, 100])

        carga["baja"] = fuzz.trapmf(carga.universe, [0, 0, 20, 40])
        carga["media"] = fuzz.trapmf(carga.universe, [30, 45, 55, 70])
        carga["alta"] = fuzz.trapmf(carga.universe, [60, 80, 100, 100])

        rules = [
            ctrl.Rule(fatiga["alta"] | minutos["corto"], carga["baja"]),
            ctrl.Rule(fatiga["baja"] & minutos["largo"], carga["alta"]),
            ctrl.Rule(fatiga["baja"] & minutos["medio"], carga["media"]),
            ctrl.Rule(fatiga["media"] & minutos["medio"], carga["media"]),
            ctrl.Rule(fatiga["media"] & minutos["largo"], carga["alta"]),
            ctrl.Rule(fatiga["baja"] & minutos["corto"], carga["baja"]),
        ]
        self._simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def admissible_load(self, minutes: float, fatigue: float) -> float:
        self._simulation.input["minutos"] = float(np.clip(minutes, 0, 240))
        self._simulation.input["fatiga"] = float(np.clip(fatigue, 0, 100))
        self._simulation.compute()
        return float(self._simulation.output["carga_max"]) / 100.0


def fatigue_from_wearable(hr: float, hrv: float, sleep_hours: float, hr_baseline: float = 62.0) -> float:
    """User-dependent fatigue proxy. Higher HR and lower HRV or sleep raise fatigue."""
    hr_term = np.clip((hr - hr_baseline) / 30.0, 0, 1)
    hrv_term = np.clip((55.0 - hrv) / 40.0, 0, 1)
    sleep_term = np.clip((7.0 - sleep_hours) / 4.0, 0, 1)
    return float(np.clip(100.0 * (0.45 * hr_term + 0.35 * hrv_term + 0.20 * sleep_term), 0, 100))


class PlayNowEngine:
    def __init__(
        self,
        catalog: pd.DataFrame,
        library: pd.DataFrame,
        users: pd.DataFrame,
        sessions: pd.DataFrame,
    ) -> None:
        self.catalog = catalog.copy()
        self.library = library.copy()
        self.users = users.copy()
        self.sessions = sessions.copy()
        self.fuzzy = FuzzySessionModel()
        self.scaler = MinMaxScaler()
        self._prepare_catalog()
        self.cluster_model, self.elbow = self._fit_clusters()

    def _prepare_catalog(self) -> None:
        frame = self.catalog
        frame["valencia_01"] = (frame["valencia"] + 1.0) / 2.0
        frame["arousal_01"] = (frame["arousal"] + 1.0) / 2.0
        if "cuadrante" not in frame.columns:
            frame["cuadrante"] = [
                russell_quadrant(v, a) for v, a in zip(frame["valencia"], frame["arousal"])
            ]
        matrix = frame[CONTENT_COLUMNS].to_numpy(dtype=float)
        self._matrix = self.scaler.fit_transform(matrix)
        self._by_id = frame.set_index("game_id")

    def _fit_clusters(self) -> tuple[KMeans, pd.DataFrame]:
        standard = StandardScaler().fit_transform(self.catalog[CONTENT_COLUMNS].to_numpy(dtype=float))
        pca = PCA(n_components=0.95, random_state=SEED)
        reduced = pca.fit_transform(standard)
        self.pca = pca
        self.embedding = reduced
        inertias = []
        for k in range(2, 11):
            model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            model.fit(reduced)
            inertias.append({"k": k, "inertia": float(model.inertia_)})
        elbow = pd.DataFrame(inertias)
        chosen = _elbow_k(elbow)
        final = KMeans(n_clusters=chosen, random_state=SEED, n_init=10)
        labels = final.fit_predict(reduced)
        self.catalog["cluster_id"] = labels
        self._by_id = self.catalog.set_index("game_id")
        return final, elbow

    def library_of(self, user_id: str) -> pd.DataFrame:
        owned = self.library[self.library["user_id"] == user_id]
        merged = owned.merge(self.catalog, on="game_id", how="inner")
        return merged

    def collaborative_scores(self, user_id: str) -> dict[str, float]:
        profile = self.users.loc[self.users["user_id"] == user_id, "perfil_ocupacional"].iloc[0]
        neighbors = self.users[
            (self.users["perfil_ocupacional"] == profile) & (self.users["user_id"] != user_id)
        ]["user_id"]
        relevant = self.sessions[self.sessions["user_id"].isin(neighbors)]
        scores: dict[str, float] = {}
        if relevant.empty:
            return scores
        for game_id, group in relevant.groupby("game_id"):
            if len(group) < 2:
                continue
            success = group["evento_cierre"].eq("GUARDADO_FIN_NATURAL").astype(float)
            early = group["evento_cierre"].eq("SALIDA_PREMATURA_FRUSTRACION").astype(float)
            satisfaction = group["satisfaccion"].astype(float) / 5.0
            scores[str(game_id)] = float(
                np.clip(satisfaction.mean() * 0.7 + success.mean() * 0.3 - early.mean() * 0.25, 0, 1)
            )
        return scores

    def rank(
        self,
        user_id: str,
        minutes: float,
        fatigue: float,
        intention: str,
        apply_backlog_slot: bool = True,
        use_collaborative: bool = True,
    ) -> pd.DataFrame:
        if intention not in INTENTION_TARGETS:
            raise KeyError(f"Unknown intention: {intention}")
        owned = self.library_of(user_id)
        if owned.empty:
            return owned
        target = INTENTION_TARGETS[intention]
        carga_max = self.fuzzy.admissible_load(minutes, fatigue)
        target_vector = _target_vector(target, carga_max)
        target_scaled = self.scaler.transform(target_vector.reshape(1, -1))
        collab = self.collaborative_scores(user_id) if use_collaborative else {}
        gamma_active = len(collab) >= 2
        alpha, beta, gamma = (0.55, 0.30, 0.15) if gamma_active else (0.65, 0.35, 0.0)

        rows = []
        for record in owned.itertuples(index=False):
            game_scaled = self.scaler.transform(_game_vector(record).reshape(1, -1))
            cosine = float(cosine_similarity(target_scaled, game_scaled)[0, 0])
            cosine = float(np.clip(cosine, 0.0, 1.0))
            russell = 1.0 - _va_distance(record.valencia, record.arousal, target["valencia"], target["arousal"]) / 2.0
            raw_similarity = 0.55 * cosine + 0.45 * russell
            penalty = _reengage_adjustment(record, minutes, fatigue)
            similarity = _apply_reengage(raw_similarity, penalty)
            phi = _viability(record, minutes, carga_max)
            fuzzy_fit = float(np.clip(1.0 - abs(record.carga_cognitiva - carga_max), 0.0, 1.0))
            membership = float(getattr(record, INTENTION_MU[intention]))
            raw_collab = collab.get(record.game_id, 0.0) if gamma_active else 0.0
            # Neighbors only break ties among games that already fit the intention.
            collab_score = raw_collab if membership >= 0.45 else 0.0
            score = phi * (alpha * similarity + beta * fuzzy_fit + gamma * collab_score)
            rows.append(
                {
                    "game_id": record.game_id,
                    "titulo": record.titulo,
                    "horas_jugadas": record.horas_jugadas,
                    "dias_sin_jugar": record.dias_sin_jugar,
                    "es_indie": int(record.es_indie),
                    "cluster_id": int(record.cluster_id),
                    "cuadrante": record.cuadrante,
                    "score": round(score, 4),
                    "phi_viabilidad": round(phi, 4),
                    "similitud": round(similarity, 4),
                    "similitud_coseno": round(cosine, 4),
                    "similitud_russell": round(russell, 4),
                    "ajuste_difuso": round(fuzzy_fit, 4),
                    "carga_max": round(carga_max, 4),
                    "penalizacion_reenganche": round(penalty, 4),
                    "score_colaborativo": round(collab_score, 4),
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "explicacion": _explain(record, minutes, fatigue, target, phi, similarity, penalty, collab_score),
                }
            )
        ranked = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        ranked["puesto_base"] = ranked.index + 1
        if apply_backlog_slot:
            ranked = _promote_backlog_gem(ranked)
        ranked["puesto"] = ranked.index + 1
        return ranked

    def baseline(self, user_id: str) -> pd.DataFrame:
        owned = self.library_of(user_id)
        ordered = owned.sort_values("horas_jugadas", ascending=False).reset_index(drop=True)
        ordered["puesto"] = ordered.index + 1
        return ordered[["puesto", "game_id", "titulo", "horas_jugadas", "dias_sin_jugar", "es_indie"]]

    def list_metrics(self, ranking: pd.DataFrame, k: int = 3) -> dict[str, float]:
        top = ranking.head(k)
        if len(top) < 2:
            diversity = 0.0
        else:
            vectors = []
            for game_id in top["game_id"]:
                row = self._by_id.loc[game_id]
                vectors.append([row.valencia, row.arousal, row.carga_cognitiva, row.densidad_narrativa])
            matrix = np.vstack(vectors)
            distances = []
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
            diversity = float(np.mean(distances))
        backlog = float((top["horas_jugadas"] < 2).mean())
        indie = float(top["es_indie"].max()) if len(top) else 0.0
        return {
            "diversidad_intra_lista": round(diversity, 4),
            "cobertura_backlog": round(backlog, 4),
            "exposicion_indie": round(indie, 4),
        }


def _elbow_k(elbow: pd.DataFrame) -> int:
    ks = elbow["k"].to_numpy(dtype=float)
    values = elbow["inertia"].to_numpy(dtype=float)
    start = np.array([ks[0], values[0]])
    end = np.array([ks[-1], values[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    distances = []
    for k, value in zip(ks, values):
        point = np.array([k, value])
        offset = start - point
        # Scalar determinant for the 2D cross product; compatible with recent NumPy.
        cross = line[0] * offset[1] - line[1] * offset[0]
        distance = abs(cross) / line_norm
        distances.append(distance)
    return int(ks[int(np.argmax(distances))])


def _target_vector(target: dict[str, float], carga_max: float) -> np.ndarray:
    carga = min(target["carga_cognitiva"], carga_max + 0.05)
    values = [
        carga,
        target["exigencia_motora"],
        target["densidad_narrativa"],
        (target["valencia"] + 1.0) / 2.0,
        (target["arousal"] + 1.0) / 2.0,
        target["mu_relajacion"],
        target["mu_ligero"],
        target["mu_historia"],
        target["mu_reto"],
        target["facilidad_pausa"],
        target["score_post_trabajo"],
        target["score_juego_confort"],
        target["score_facil_retomar"],
    ]
    return np.array(values, dtype=float)


def _game_vector(record: object) -> np.ndarray:
    return np.array([getattr(record, column) for column in CONTENT_COLUMNS], dtype=float)


def _va_distance(v1: float, a1: float, v2: float, a2: float) -> float:
    return float(np.hypot(v1 - v2, a1 - a2))


def _viability(record: object, minutes: float, carga_max: float) -> float:
    if record.duracion_minima_sesion_min > minutes + 8:
        return 0.08
    overflow = max(0.0, record.carga_cognitiva - carga_max)
    phi = max(0.12, 1.0 - 0.85 * overflow)
    if record.minutos_hasta_diversion > minutes * 0.55:
        phi *= 0.72
    if record.modalidad == "PvP" and minutes < 25:
        phi *= 0.8
    return float(np.clip(phi, 0.05, 1.0))


def _apply_reengage(raw_similarity: float, penalty: float) -> float:
    """Fold reengage into similarity without swallowing a weekend bonus.

    A positive penalty stays inside [0, 1]. A negative penalty is a bonus for a
    long, rested block. Leaving it uncapped lets a paused game rise when the
    content match was already at the ceiling.
    """
    adjusted = raw_similarity - penalty
    if penalty >= 0:
        return float(np.clip(adjusted, 0.0, 1.0))
    return float(max(0.0, adjusted))


def _reengage_adjustment(record: object, minutes: float, fatigue: float) -> float:
    staleness = min(1.0, record.dias_sin_jugar / 21.0)
    fatigue_n = fatigue / 100.0
    penalty = record.costo_reenganche * staleness * (0.35 + 0.65 * fatigue_n)
    if minutes >= 120 and fatigue < 35:
        return -0.22 * record.costo_reenganche
    return float(penalty)


def _explain(record: object, minutes: float, fatigue: float, target: dict, phi: float, similarity: float, penalty: float, collab: float) -> str:
    reasons = [f"Se parece a la intención '{target['label']}' (similitud {similarity:.2f})."]
    if phi < 0.2:
        reasons.append(
            f"Casi no cabe: pide unos {int(record.duracion_minima_sesion_min)} min y hay {int(minutes)} libres."
        )
    elif phi > 0.75:
        reasons.append(f"Cabe en los {int(minutes)} minutos libres.")
    else:
        reasons.append("Cabe con algo de fricción de tiempo o de carga cognitiva.")
    if penalty > 0.15:
        reasons.append(
            f"Llevas {int(record.dias_sin_jugar)} días sin abrirlo y, con la fatiga actual ({fatigue:.0f}/100), retomarlo pesa."
        )
    elif penalty < 0:
        reasons.append("Hay un bloque largo y energía alta, así que conviene rescatar un juego pausado.")
    if collab > 0.65:
        reasons.append("Personas con una carga ocupacional parecida cerraron bien una sesión con este juego.")
    if record.horas_jugadas < 1:
        reasons.append("Está en tu biblioteca y casi no lo has jugado.")
    return " ".join(reasons)


def _promote_backlog_gem(ranking: pd.DataFrame) -> pd.DataFrame:
    """Keep one viable unplayed title in the top 3 when the score is close."""
    if len(ranking) < 3:
        return ranking
    top = ranking.head(3)
    if (top["horas_jugadas"] < 2).any():
        return ranking
    gems = ranking[(ranking["horas_jugadas"] < 1) & (ranking["phi_viabilidad"] >= 0.45)]
    if gems.empty:
        return ranking
    gem = gems.iloc[0]
    threshold = float(ranking.iloc[2]["score"]) * 0.86
    if float(gem["score"]) < threshold or float(gem["similitud"]) < 0.62:
        return ranking
    gem_row = ranking.loc[[gem.name]]
    without = ranking.drop(index=gem.name)
    head = pd.concat([without.iloc[:2], gem_row], ignore_index=True)
    tail = without.iloc[2:]
    return pd.concat([head, tail], ignore_index=True)
