"""Motor de recomendación de videojuegos para una sesión de juego.

Combina la intención de la persona con el tiempo disponible, la fatiga simulada
y las características de cada juego. Los candidatos salen de su biblioteca.
Los grupos de juegos ayudan a describir el catálogo. La lista se ordena por el
puntaje de la sesión y puede dar espacio a un juego que la persona casi no usa.
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

# Fijar la semilla permite repetir los cálculos que tienen un componente aleatorio.
SEED = 42

# Características que se comparan entre el juego y la experiencia buscada.
# "arousal" significa activación: desde una experiencia tranquila hasta una intensa.
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

# Cada intención se vincula con el grado de adecuación del juego a esa experiencia.
INTENTION_MU = {
    "relajarme": "mu_relajacion",
    "ligero": "mu_ligero",
    "historia": "mu_historia",
    "reto": "mu_reto",
}

# Perfiles de referencia definidos para las cuatro respuestas de la sesión.
# Son valores de diseño: por ejemplo, relajarse pide poca carga y una pausa fácil.
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

# Frases que se buscan en los comentarios para describir el uso de cada juego.
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
    """Ubica el juego según su valencia y activación en el plano de Russell.

    H indica un valor alto y L uno bajo; V corresponde a valencia y A a activación.
    Por ejemplo, HVLA describe valencia positiva y activación baja.
    """
    if valence >= 0 and arousal >= 0:
        return "HVHA"
    if valence >= 0 and arousal < 0:
        return "HVLA"
    if valence < 0 and arousal >= 0:
        return "LVHA"
    return "LVLA"


def _fold(text: str) -> str:
    """Pasa el texto a minúsculas y quita las tildes para comparar frases."""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(character for character in normalized if unicodedata.category(character) != "Mn")


def _hits(text: str, phrases: tuple[str, ...]) -> int:
    """Cuenta cuántas frases del listado aparecen al menos una vez en el texto."""
    folded = _fold(text)
    return sum(1 for phrase in phrases if _fold(phrase) in folded)


def lexicon_scores(text: str) -> dict[str, float]:
    """Obtiene puntajes de uso y frustración a partir de un comentario simulado."""
    post_pos = _hits(text, POST_WORK_POS)
    post_neg = _hits(text, POST_WORK_NEG)
    retake_pos = _hits(text, RETAKE_POS)
    retake_neg = _hits(text, RETAKE_NEG)
    tilt = _hits(text, TILT_TERMS)
    # Los términos 0.5 y 1 evitan divisiones por cero y dan 0.5 si no hay pistas.
    post = (post_pos + 0.5) / (post_pos + post_neg + 1.0)
    retake = (retake_pos + 0.5) / (retake_pos + retake_neg + 1.0)
    # Dos expresiones de frustración bastan para llegar al máximo de esta escala.
    frustration = min(1.0, tilt / 2.0)
    confort = max(0.0, min(1.0, 1.0 - frustration))
    return {
        "score_post_trabajo": round(post, 3),
        "score_facil_retomar": round(retake, 3),
        "score_juego_confort": round(confort, 3),
        "riesgo_frustracion_lexico": round(frustration, 3),
    }


def aggregate_lexicon(comments: pd.DataFrame) -> pd.DataFrame:
    """Resume cada juego con el promedio de los puntajes de sus comentarios."""
    rows = []
    for game_id, group in comments.groupby("game_id"):
        parts = [lexicon_scores(text) for text in group["texto"]]
        frame = pd.DataFrame(parts)
        summary = frame.mean().round(3).to_dict()
        summary["game_id"] = game_id
        rows.append(summary)
    return pd.DataFrame(rows)


class FuzzySessionModel:
    """Estima cuánta exigencia mental admite la sesión con reglas difusas.

    El tiempo y la fatiga pueden pertenecer parcialmente a varias categorías.
    Las reglas de Mamdani combinan esos grados para obtener una carga admisible.
    """

    def __init__(self) -> None:
        # Dos entradas: minutos libres y fatiga de 0 a 100. La salida es la carga.
        minutos = ctrl.Antecedent(np.arange(0, 241, 1), "minutos")
        fatiga = ctrl.Antecedent(np.arange(0, 101, 1), "fatiga")
        carga = ctrl.Consequent(np.arange(0, 101, 1), "carga_max")

        # Los trapecios se solapan: una sesión no cambia de categoría de golpe.
        minutos["corto"] = fuzz.trapmf(minutos.universe, [0, 0, 25, 55])
        minutos["medio"] = fuzz.trapmf(minutos.universe, [35, 55, 80, 110])
        minutos["largo"] = fuzz.trapmf(minutos.universe, [90, 130, 240, 240])

        fatiga["baja"] = fuzz.trapmf(fatiga.universe, [0, 0, 20, 40])
        fatiga["media"] = fuzz.trapmf(fatiga.universe, [30, 45, 60, 75])
        fatiga["alta"] = fuzz.trapmf(fatiga.universe, [65, 80, 100, 100])

        carga["baja"] = fuzz.trapmf(carga.universe, [0, 0, 20, 40])
        carga["media"] = fuzz.trapmf(carga.universe, [30, 45, 55, 70])
        carga["alta"] = fuzz.trapmf(carga.universe, [60, 80, 100, 100])

        # Poco tiempo o mucha fatiga favorecen una carga baja; un bloque largo
        # permite mayor exigencia cuando la fatiga es baja o media.
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
        """Aplica las reglas y devuelve la carga admisible en una escala de 0 a 1."""
        # Se acotan las entradas al rango definido por los conjuntos difusos.
        self._simulation.input["minutos"] = float(np.clip(minutes, 0, 240))
        self._simulation.input["fatiga"] = float(np.clip(fatigue, 0, 100))
        self._simulation.compute()
        # El centroide resume la salida difusa en un número, que se divide por 100.
        return float(self._simulation.output["carga_max"]) / 100.0


def fatigue_from_wearable(hr: float, hrv: float, sleep_hours: float, hr_baseline: float = 62.0) -> float:
    """Calcula un indicador simulado de fatiga entre 0 y 100.

    El pulso se compara con la frecuencia habitual de la persona. Para la
    variabilidad cardíaca (HRV) y el sueño se usan referencias fijas. Un pulso
    elevado, menor HRV o menos sueño aumentan el resultado de esta heurística.
    """
    hr_term = np.clip((hr - hr_baseline) / 30.0, 0, 1)
    hrv_term = np.clip((55.0 - hrv) / 40.0, 0, 1)
    sleep_term = np.clip((7.0 - sleep_hours) / 4.0, 0, 1)
    # Se combinan las tres señales con pesos de diseño y se limita el resultado.
    return float(np.clip(100.0 * (0.45 * hr_term + 0.35 * hrv_term + 0.20 * sleep_term), 0, 100))


class PlayNowEngine:
    """Prepara el catálogo y ordena la biblioteca según el contexto de la sesión."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        library: pd.DataFrame,
        users: pd.DataFrame,
        sessions: pd.DataFrame,
    ) -> None:
        # Las copias permiten preparar los datos sin alterar las tablas recibidas.
        self.catalog = catalog.copy()
        self.library = library.copy()
        self.users = users.copy()
        self.sessions = sessions.copy()
        self.fuzzy = FuzzySessionModel()
        self.scaler = MinMaxScaler()
        self._prepare_catalog()
        self.cluster_model, self.elbow = self._fit_clusters()

    def _prepare_catalog(self) -> None:
        """Prepara los vectores de características y las búsquedas por juego."""
        frame = self.catalog
        # Valencia y activación pasan de [-1, 1] a [0, 1] antes de comparar vectores.
        frame["valencia_01"] = (frame["valencia"] + 1.0) / 2.0
        frame["arousal_01"] = (frame["arousal"] + 1.0) / 2.0
        if "cuadrante" not in frame.columns:
            frame["cuadrante"] = [
                russell_quadrant(v, a) for v, a in zip(frame["valencia"], frame["arousal"])
            ]
        matrix = frame[CONTENT_COLUMNS].to_numpy(dtype=float)
        # Se aprende una escala por característica a partir del catálogo completo.
        self._matrix = self.scaler.fit_transform(matrix)
        self._by_id = frame.set_index("game_id")

    def _fit_clusters(self) -> tuple[KMeans, pd.DataFrame]:
        """Agrupa juegos parecidos y guarda los datos de la curva del codo."""
        # La estandarización pone las características en una escala comparable.
        standard = StandardScaler().fit_transform(self.catalog[CONTENT_COLUMNS].to_numpy(dtype=float))
        # PCA reduce las dimensiones conservando al menos el 95 % de la varianza.
        pca = PCA(n_components=0.95, random_state=SEED)
        reduced = pca.fit_transform(standard)
        self.pca = pca
        self.embedding = reduced
        # La inercia mide la dispersión de los juegos respecto al centro de su grupo.
        # Se prueban de 2 a 10 grupos para observar cuánto mejora la agrupación.
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
        """Cruza la biblioteca de la persona con las características del catálogo."""
        # Este cruce delimita los candidatos: solo entran juegos de su biblioteca.
        owned = self.library[self.library["user_id"] == user_id]
        merged = owned.merge(self.catalog, on="game_id", how="inner")
        return merged

    def collaborative_scores(self, user_id: str) -> dict[str, float]:
        """Resume la experiencia de otras personas del mismo perfil ocupacional."""
        profile = self.users.loc[self.users["user_id"] == user_id, "perfil_ocupacional"].iloc[0]
        neighbors = self.users[
            (self.users["perfil_ocupacional"] == profile) & (self.users["user_id"] != user_id)
        ]["user_id"]
        relevant = self.sessions[self.sessions["user_id"].isin(neighbors)]
        scores: dict[str, float] = {}
        if relevant.empty:
            return scores
        for game_id, group in relevant.groupby("game_id"):
            # Se requieren al menos dos sesiones de vecinos para puntuar un juego.
            if len(group) < 2:
                continue
            success = group["evento_cierre"].eq("GUARDADO_FIN_NATURAL").astype(float)
            early = group["evento_cierre"].eq("SALIDA_PREMATURA_FRUSTRACION").astype(float)
            satisfaction = group["satisfaccion"].astype(float) / 5.0
            # Se promedian satisfacción y cierres naturales, y se descuentan
            # las salidas por frustración. El resultado queda entre 0 y 1.
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
        """Calcula las recomendaciones y conserva el detalle de cada puntaje.

        La intención define la experiencia buscada. El tiempo y la fatiga
        ajustan la exigencia admisible y el esfuerzo de retomar cada juego.
        Los indicadores permiten consultar el efecto de los vecinos y de la
        promoción de juegos poco jugados por separado.
        """
        if intention not in INTENTION_TARGETS:
            raise KeyError(f"Intención no reconocida: {intention}")
        owned = self.library_of(user_id)
        if owned.empty:
            return owned
        # La intención se convierte en un vector comparable con el de los juegos.
        target = INTENTION_TARGETS[intention]
        carga_max = self.fuzzy.admissible_load(minutes, fatigue)
        target_vector = _target_vector(target, carga_max)
        target_scaled = self.scaler.transform(target_vector.reshape(1, -1))
        collab = self.collaborative_scores(user_id) if use_collaborative else {}
        gamma_active = len(collab) >= 2
        # Si hay evidencia para dos juegos de vecinos, se activa su aporte.
        # Estos pesos son decisiones de diseño, no parámetros aprendidos.
        alpha, beta, gamma = (0.55, 0.30, 0.15) if gamma_active else (0.65, 0.35, 0.0)

        rows = []
        for record in owned.itertuples(index=False):
            game_scaled = self.scaler.transform(_game_vector(record).reshape(1, -1))
            # El coseno compara la orientación de los vectores: cuánto se parece
            # el perfil del juego a la experiencia que la persona busca hoy.
            cosine = float(cosine_similarity(target_scaled, game_scaled)[0, 0])
            cosine = float(np.clip(cosine, 0.0, 1.0))
            # La cercanía emocional usa la diagonal completa del plano [-1, 1]²
            # como distancia máxima y queda entre 0 y 1.
            russell = _russell_similarity(record.valencia, record.arousal, target["valencia"], target["arousal"])
            raw_similarity = 0.55 * cosine + 0.45 * russell
            # Retomar un juego puede restar afinidad o dar un bono si hay tiempo
            # y poca fatiga. La implementación actual lo aplica a la similitud.
            penalty = _reengage_adjustment(record, minutes, fatigue)
            similarity = _apply_reengage(raw_similarity, penalty)
            phi = _viability(record, minutes, carga_max)
            # Una carga cognitiva cercana a la admisible recibe mayor puntaje.
            fuzzy_fit = float(np.clip(1.0 - abs(record.carga_cognitiva - carga_max), 0.0, 1.0))
            membership = float(getattr(record, INTENTION_MU[intention]))
            raw_collab = collab.get(record.game_id, 0.0) if gamma_active else 0.0
            # La señal de vecinos se usa si la adecuación a la intención es al menos 0.45.
            collab_score = raw_collab if membership >= 0.45 else 0.0
            # La viabilidad reduce la suma de contenido, ajuste de carga y vecinos
            # cuando el juego encaja peor con el tiempo o la carga de la sesión.
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
        # Se conserva el puesto por puntaje antes de promover un juego poco jugado.
        if apply_backlog_slot:
            ranked = _promote_backlog_gem(ranked)
        ranked["puesto"] = ranked.index + 1
        return ranked

    def baseline(self, user_id: str) -> pd.DataFrame:
        """Ordena por horas jugadas para comparar con la recomendación contextual."""
        owned = self.library_of(user_id)
        ordered = owned.sort_values("horas_jugadas", ascending=False).reset_index(drop=True)
        ordered["puesto"] = ordered.index + 1
        return ordered[["puesto", "game_id", "titulo", "horas_jugadas", "dias_sin_jugar", "es_indie"]]

    def list_metrics(self, ranking: pd.DataFrame, k: int = 3) -> dict[str, float]:
        """Describe diversidad, juegos poco jugados y presencia indie en los primeros k."""
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
            # La diversidad es la distancia media entre cada par de juegos en
            # valencia, activación, carga cognitiva y densidad narrativa.
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
            diversity = float(np.mean(distances))
        # Se mide la fracción con menos de dos horas y si aparece al menos un indie.
        backlog = float((top["horas_jugadas"] < 2).mean())
        indie = float(top["es_indie"].max()) if len(top) else 0.0
        return {
            "diversidad_intra_lista": round(diversity, 4),
            "cobertura_backlog": round(backlog, 4),
            "exposicion_indie": round(indie, 4),
        }


def _elbow_k(elbow: pd.DataFrame) -> int:
    """Elige el número de grupos en el punto más alejado de la recta de referencia."""
    ks = elbow["k"].to_numpy(dtype=float)
    values = elbow["inertia"].to_numpy(dtype=float)
    start = np.array([ks[0], values[0]])
    end = np.array([ks[-1], values[-1]])
    # La recta une el primer y el último punto de la curva de inercia.
    line = end - start
    line_norm = np.linalg.norm(line)
    distances = []
    for k, value in zip(ks, values):
        point = np.array([k, value])
        offset = start - point
        # El determinante en dos dimensiones, dividido por la longitud del
        # vector de referencia, da la distancia perpendicular de cada punto.
        cross = line[0] * offset[1] - line[1] * offset[0]
        distance = abs(cross) / line_norm
        distances.append(distance)
    # La mayor distancia señala el cambio de pendiente elegido como codo.
    return int(ks[int(np.argmax(distances))])


def _target_vector(target: dict[str, float], carga_max: float) -> np.ndarray:
    """Construye el perfil buscado con el mismo orden de atributos que el catálogo."""
    # La exigencia deseada se limita a la carga admisible más un margen de 0.05.
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
    """Extrae las características del juego en el orden usado para la comparación."""
    return np.array([getattr(record, column) for column in CONTENT_COLUMNS], dtype=float)


def _va_distance(v1: float, a1: float, v2: float, a2: float) -> float:
    """Calcula la distancia euclidiana entre dos puntos de valencia y activación."""
    return float(np.hypot(v1 - v2, a1 - a2))


def _russell_similarity(v1: float, a1: float, v2: float, a2: float) -> float:
    """Convierte la distancia afectiva en cercanía: 1 si coinciden y 0 en los extremos."""
    # De (-1, -1) a (1, 1), la distancia máxima es la diagonal: 2 * raíz de 2.
    max_distance = 2.0 * np.sqrt(2.0)
    distance = _va_distance(v1, a1, v2, a2)
    return float(np.clip(1.0 - distance / max_distance, 0.0, 1.0))


def _viability(record: object, minutes: float, carga_max: float) -> float:
    """Reduce el puntaje si el juego exige demasiado tiempo o esfuerzo mental."""
    # Si la sesión mínima supera el tiempo disponible más ocho minutos,
    # se mantiene como candidato, pero con un factor de viabilidad muy bajo.
    if record.duracion_minima_sesion_min > minutes + 8:
        return 0.08
    # Solo se penaliza la parte de carga que supera lo admisible para la sesión.
    overflow = max(0.0, record.carga_cognitiva - carga_max)
    phi = max(0.12, 1.0 - 0.85 * overflow)
    # También pesa tardar en llegar a la diversión o jugar una partida
    # competitiva contra otras personas (PvP) con menos de 25 minutos libres.
    if record.minutos_hasta_diversion > minutes * 0.55:
        phi *= 0.72
    if record.modalidad == "PvP" and minutes < 25:
        phi *= 0.8
    return float(np.clip(phi, 0.05, 1.0))


def _apply_reengage(raw_similarity: float, penalty: float) -> float:
    """Aplica a la similitud el esfuerzo de retomar el juego.

    Un ajuste positivo resta y uno negativo suma un bono por disponer de tiempo
    y poca fatiga. En ambos casos el resultado se limita al intervalo [0, 1].
    """
    adjusted = raw_similarity - penalty
    return float(np.clip(adjusted, 0.0, 1.0))


def _reengage_adjustment(record: object, minutes: float, fatigue: float) -> float:
    """Estima cuánto pesa volver a un juego después de dejarlo un tiempo."""
    # Los días sin jugar aumentan el esfuerzo hasta un máximo de tres semanas.
    staleness = min(1.0, record.dias_sin_jugar / 21.0)
    fatigue_n = fatigue / 100.0
    penalty = record.costo_reenganche * staleness * (0.35 + 0.65 * fatigue_n)
    if minutes >= 120 and fatigue < 35:
        # Con al menos dos horas y poca fatiga, el costo se convierte en un bono.
        return -0.22 * record.costo_reenganche
    return float(penalty)


def _explain(record: object, minutes: float, fatigue: float, target: dict, phi: float, similarity: float, penalty: float, collab: float) -> str:
    """Redacta los motivos de la recomendación con los valores ya calculados."""
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
    """Da espacio a un juego casi sin usar si encaja y su puntaje es cercano."""
    if len(ranking) < 3:
        return ranking
    # Si ya hay un juego con menos de dos horas en los tres primeros, no se cambia.
    top = ranking.head(3)
    if (top["horas_jugadas"] < 2).any():
        return ranking
    # Se consideran juegos con menos de una hora y viabilidad de al menos 0.45.
    gems = ranking[(ranking["horas_jugadas"] < 1) & (ranking["phi_viabilidad"] >= 0.45)]
    if gems.empty:
        return ranking
    gem = gems.iloc[0]
    # Debe alcanzar el 86 % del puntaje del tercero y una similitud mínima de 0.62.
    threshold = float(ranking.iloc[2]["score"]) * 0.86
    if float(gem["score"]) < threshold or float(gem["similitud"]) < 0.62:
        return ranking
    # Se ubica ese candidato en el tercer puesto y se conserva el resto del orden.
    gem_row = ranking.loc[[gem.name]]
    without = ranking.drop(index=gem.name)
    head = pd.concat([without.iloc[:2], gem_row], ignore_index=True)
    tail = without.iloc[2:]
    return pd.concat([head, tail], ignore_index=True)
