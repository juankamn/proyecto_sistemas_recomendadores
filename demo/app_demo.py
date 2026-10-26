"""Interfaz para explorar recomendaciones con una biblioteca y un contexto simulados.

Los controles representan el tiempo de agenda y las señales de un reloj.
La persona elige qué le provoca jugar; el motor compartido con el cuaderno
calcula las recomendaciones y sus explicaciones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Se parte de la ubicación del archivo para encontrar el motor y los datos.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playnow.engine import INTENTION_TARGETS, PlayNowEngine, fatigue_from_wearable

DATA = ROOT / "data"

# Escenarios de ejemplo: tiempo libre, pulso, variabilidad cardíaca, sueño e intención.
PRESETS = {
    "Martes 21:30, relajarme": {"minutes": 35, "hr": 92, "hrv": 28, "sleep": 4.5, "intention": "relajarme"},
    "Sábado 16:00, historia": {"minutes": 180, "hr": 66, "hrv": 52, "sleep": 8.0, "intention": "historia"},
    "Viernes 20:00, reto": {"minutes": 50, "hr": 78, "hrv": 40, "sleep": 6.5, "intention": "reto"},
}


@st.cache_resource
def load_engine() -> tuple[PlayNowEngine, pd.DataFrame]:
    """Carga los CSV y prepara el motor una vez para reutilizarlo en la interfaz."""
    catalog = pd.read_csv(DATA / "catalogo_videojuegos_profundo.csv")
    library = pd.read_csv(DATA / "biblioteca_usuario.csv")
    users = pd.read_csv(DATA / "usuarios_perfiles_ocupacionales.csv")
    sessions = pd.read_csv(DATA / "sesiones_contextuales_wearable.csv")
    return PlayNowEngine(catalog, library, users, sessions), users


def apply_preset(name: str) -> None:
    """Copia un escenario a los controles que conserva la sesión de Streamlit."""
    preset = PRESETS[name]
    st.session_state["minutes"] = preset["minutes"]
    st.session_state["hr"] = preset["hr"]
    st.session_state["hrv"] = preset["hrv"]
    st.session_state["sleep"] = preset["sleep"]
    st.session_state["intention"] = preset["intention"]


def main() -> None:
    """Organiza los controles y muestra las recomendaciones para la sesión elegida."""
    st.set_page_config(page_title="PlayNow", layout="wide")
    engine, users = load_engine()

    st.title("PlayNow")
    st.caption(
        "Recomienda qué juego de tu biblioteca abrir ahora. "
        "La agenda, el reloj y las plataformas están simulados."
    )

    if "minutes" not in st.session_state:
        apply_preset("Martes 21:30, relajarme")

    # El selector muestra nombres legibles; el motor trabaja con los identificadores.
    names = {row.user_id: f"{row.nombre} ({row.perfil_ocupacional})" for row in users.itertuples(index=False)}
    user_id = st.sidebar.selectbox("Persona", list(names), format_func=lambda key: names[key])
    profile = users.loc[users["user_id"] == user_id].iloc[0]
    baseline_hr = float(profile["hr_baseline"])

    st.sidebar.subheader("Escenarios")
    for name in PRESETS:
        st.sidebar.button(name, on_click=apply_preset, args=(name,), width="stretch")

    # Estos controles permiten explorar las señales que se obtendrían de las
    # integraciones; la pregunta de intención representa la respuesta de la persona.
    st.sidebar.subheader("Agenda y reloj simulados")
    minutes = st.sidebar.slider("Minutos libres", min_value=10, max_value=240, key="minutes")
    hr = st.sidebar.slider("Frecuencia cardíaca", min_value=50, max_value=120, key="hr")
    hrv = st.sidebar.slider("HRV", min_value=15, max_value=80, key="hrv")
    sleep = st.sidebar.slider("Horas de sueño", min_value=3.0, max_value=9.0, step=0.5, key="sleep")
    intention = st.sidebar.radio(
        "¿Qué te provoca jugar hoy?",
        list(INTENTION_TARGETS),
        format_func=lambda key: INTENTION_TARGETS[key]["label"],
        key="intention",
    )

    # Se estima la fatiga, se ordena la biblioteca y se calcula una lista por horas
    # como referencia para entender el efecto del contexto.
    fatigue = fatigue_from_wearable(hr, hrv, sleep, baseline_hr)
    ranking = engine.rank(user_id, minutes, fatigue, intention)
    baseline = engine.baseline(user_id)
    metrics = engine.list_metrics(ranking)
    load = float(ranking.iloc[0]["carga_max"]) if len(ranking) else 0.0

    st.subheader(profile["nombre"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Perfil", str(profile["perfil_ocupacional"]).replace("_", " "))
    c2.metric("Fatiga estimada", f"{fatigue:.0f}/100")
    c3.metric("Carga admisible", f"{load:.2f}")
    c4.metric("Juego refugio", str(profile["juego_refugio"]))
    st.caption(
        f"El pulso se compara con la frecuencia habitual de {profile['nombre']} "
        f"(FC habitual {baseline_hr:.0f}). HRV y sueño usan referencias fijas."
    )

    st.subheader("Recomendaciones de tu biblioteca")
    if ranking.empty:
        st.warning("Esta persona no tiene juegos en la biblioteca simulada.")
        return

    # Cada tarjeta combina el puntaje con las razones que generó el motor.
    cards = st.columns(3)
    for column, row in zip(cards, ranking.head(3).itertuples(index=False)):
        with column:
            st.markdown(f"### {row.puesto}. {row.titulo}")
            st.write(row.explicacion)
            st.caption(
                f"Puntaje {row.score:.2f} · viabilidad {row.phi_viabilidad:.2f} · "
                f"similitud {row.similitud:.2f} · difuso {row.ajuste_difuso:.2f} · "
                f"vecinos {row.score_colaborativo:.2f} · {int(row.horas_jugadas)} h"
            )

    # Ambas listas usan la misma biblioteca; solo PlayNow incorpora el contexto.
    left, right = st.columns(2)
    with left:
        st.subheader("PlayNow")
        st.dataframe(
            ranking.head(5)[["puesto", "titulo", "score", "horas_jugadas", "cluster_id"]],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Línea base: más horas")
        st.dataframe(baseline.head(5), hide_index=True, width="stretch")

    # Los indicadores describen la lista recomendada, sin medir satisfacción real.
    st.subheader("Lectura del prototipo")
    m1, m2, m3 = st.columns(3)
    m1.metric("Diversidad intra-lista", f"{metrics['diversidad_intra_lista']:.2f}")
    m2.metric("Juegos poco jugados", f"{metrics['cobertura_backlog']:.0%}")
    m3.metric("Entra un indie", "Sí" if metrics["exposicion_indie"] else "No")
    st.caption(
        "Estas cifras describen la lista de este prototipo. No son un estudio con jugadores reales. "
        "La comparación útil es que la línea base por horas no cambia al pasar de martes a sábado, y PlayNow sí."
    )


if __name__ == "__main__":
    main()
