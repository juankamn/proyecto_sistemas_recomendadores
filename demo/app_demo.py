"""Demo visual de PlayNow con una experiencia simple y una vista avanzada.

La primera pestaña representa la experiencia final: PlayNow toma las señales
simuladas de agenda, reloj y biblioteca, por lo que la persona solo responde
qué le provoca jugar. La segunda pestaña permite inspeccionar esas señales y
el resultado del motor durante la demostración académica.
"""

from __future__ import annotations

import html
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

# El modo final usa el escenario nocturno: el contexto llega de las integraciones simuladas.
FINAL_CONTEXT = {"user_id": "U01", **PRESETS["Martes 21:30, relajarme"]}


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


def inject_style() -> None:
    """Aplica una identidad visual propia sin alterar los componentes del motor."""
    st.markdown(
        """
        <style>
        :root { --ink: #f7f8ff; --muted: #d2daf2; --navy: #081225; --panel: #14264a;
                --violet: #9c72ff; --cyan: #4de1ff; --lime: #c4ff73; }
        .stApp { background: radial-gradient(circle at 15% 0%, #223b76 0, transparent 28%),
                           radial-gradient(circle at 94% 12%, #3d216d 0, transparent 24%), var(--navy); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(8, 18, 37, .96); border-bottom: 1px solid rgba(184, 192, 221, .12); }
        [data-testid="stHeader"] * { color: var(--ink) !important; }
        [data-testid="stSidebar"] { background: #0c1931; border-right: 1px solid rgba(184, 192, 221, .15); }
        [data-testid="stSidebar"] * { color: var(--ink); }
        [data-testid="stMetric"] { background: rgba(23, 41, 80, .88); border: 1px solid rgba(184, 192, 221, .16);
                                   border-radius: 16px; padding: 14px; }
        [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] { color: var(--ink) !important; }
        .stButton > button { border: 1px solid rgba(77, 225, 255, .58); border-radius: 13px; color: #fff !important;
                             background: linear-gradient(120deg, #294d8e, #513183); font-weight: 720; min-height: 52px; }
        .stButton > button:hover { border-color: var(--lime); background: linear-gradient(120deg, #3968b6, #6743a5); color: #fff !important; }
        .stButton > button[kind="primary"] { border-color: var(--lime); background: linear-gradient(120deg, #527c2c, #627432); box-shadow: 0 0 0 3px rgba(196, 255, 115, .16); }
        .stButton > button:disabled { color: #dce4fa !important; opacity: 1; cursor: default; }
        .hero { padding: 34px; margin: 10px 0 22px; border: 1px solid rgba(123, 147, 229, .33); border-radius: 24px;
                background: linear-gradient(125deg, rgba(31, 62, 123, .9), rgba(77, 43, 132, .82)); }
        .eyebrow { color: var(--lime); font-size: .78rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
        .hero h1 { margin: .25rem 0 .45rem; color: #fff; font-size: clamp(2.25rem, 5vw, 4.3rem); letter-spacing: -.06em; }
        .hero p { color: #dce6ff; font-size: 1.1rem; max-width: 700px; margin-bottom: 0; }
        .platforms { display: flex; flex-wrap: wrap; gap: 10px; margin: 28px 0 20px; }
        .platform { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px;
                    background: rgba(7, 15, 34, .5); border: 1px solid rgba(238, 242, 255, .18); color: #fff; font-weight: 650; }
        .platform img { height: 18px; width: 18px; }
        .context-card { min-height: 84px; box-sizing: border-box; padding: 18px 20px; border-radius: 16px; background: rgba(17, 31, 60, .94);
                        border: 1px solid rgba(184, 192, 221, .24); color: var(--muted); line-height: 1.55;
                        display: flex; align-items: center; justify-content: center; text-align: center; }
        .context-card span { display: block; }
        .context-card strong { color: var(--ink); }
        .choice-label { margin: 28px 0 10px; color: var(--lime); font-weight: 750; letter-spacing: .05em; text-transform: uppercase; }
        .game-card { min-height: 272px; box-sizing: border-box; padding: 24px; border-radius: 20px; border: 1px solid rgba(184, 192, 221, .25);
                     background: linear-gradient(150deg, rgba(29, 54, 102, .98), rgba(13, 28, 57, .98)); display: flex; flex-direction: column; }
        .game-card.featured { border-color: rgba(64, 217, 255, .72); box-shadow: 0 12px 35px rgba(0, 0, 0, .22); }
        .game-rank { color: var(--cyan); font-weight: 800; font-size: .8rem; letter-spacing: .09em; }
        .game-card h3 { color: white; font-size: 1.45rem; margin: 10px 0 12px; }
        .game-card p { color: #edf1ff; line-height: 1.65; margin: 2px 0 18px; }
        .score { color: var(--lime); font-size: 1.2rem; font-weight: 800; margin-top: auto; }
        .section-title { color: #fff; font-size: 1.55rem; font-weight: 750; margin: 30px 0 4px; }
        .section-copy { color: var(--muted); margin: 0 0 16px; }
        .awaiting-choice { margin-top: 32px; min-height: 120px; display: flex; align-items: center; justify-content: center;
                            color: #cbd7f4; border: 1px dashed rgba(77, 225, 255, .42); border-radius: 20px;
                            background: rgba(12, 27, 55, .45); font-size: 1.04rem; text-align: center; padding: 20px; }
        .selected-answer { margin: 24px 0 18px; padding: 15px 18px; border: 1px solid rgba(196, 255, 115, .65);
                           border-radius: 14px; background: rgba(95, 126, 48, .18); color: #efffd7; text-align: center; }
        .table-wrap { margin-top: 12px; overflow-x: auto; border: 1px solid rgba(184, 192, 221, .24); border-radius: 16px; }
        .data-table { width: 100%; border-collapse: collapse; background: rgba(14, 29, 59, .94); color: #f4f7ff; }
        .data-table th { padding: 12px 14px; color: #baf7ff; font-size: .78rem; text-align: left; letter-spacing: .04em; text-transform: uppercase; background: rgba(40, 74, 136, .62); }
        .data-table td { padding: 12px 14px; border-top: 1px solid rgba(184, 192, 221, .14); color: #e8edfb; }
        .data-table tr:nth-child(even) td { background: rgba(36, 59, 105, .34); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def platform_badges() -> None:
    """Muestra las plataformas simuladas conectadas a la biblioteca."""
    st.markdown(
        """
        <div class="platforms" aria-label="Plataformas conectadas">
          <span class="platform"><img src="https://cdn.simpleicons.org/playstation/ffffff" alt="">PlayStation</span>
          <span class="platform"><img src="https://imagedelivery.net/5MYSbk45M80qAwecrlKzdQ/88a8cf15-5116-48ff-2889-6ad183a3a400/preview" alt="">Xbox</span>
          <span class="platform"><img src="https://cdn.simpleicons.org/steam/ffffff" alt="">Steam</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_cards(ranking: pd.DataFrame) -> None:
    """Presenta los tres primeros resultados en tarjetas fáciles de recorrer."""
    cards = st.columns(3)
    for position, (column, row) in enumerate(zip(cards, ranking.head(3).itertuples(index=False)), start=1):
        title = html.escape(str(row.titulo))
        explanation = html.escape(str(row.explicacion))
        featured = " featured" if position == 1 else ""
        with column:
            st.markdown(
                f"""
                <article class="game-card{featured}">
                    <div class="game-rank">{position:02d} · PARA ESTA SESIÓN</div>
                    <h3>{title}</h3>
                    <p>{explanation}</p>
                    <div class="score">{row.score:.0%} de afinidad</div>
                </article>
                """,
                unsafe_allow_html=True,
            )


def compact_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> None:
    """Convierte una tabla breve en HTML para conservar la identidad visual del demo."""
    header = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in columns)
    rows = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"<div class='table-wrap'><table class='data-table'><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def final_experience(engine: PlayNowEngine, users: pd.DataFrame) -> None:
    """Muestra la experiencia final con una única pregunta explícita."""
    profile = users.loc[users["user_id"] == FINAL_CONTEXT["user_id"]].iloc[0]
    baseline_hr = float(profile["hr_baseline"])
    fatigue = fatigue_from_wearable(
        FINAL_CONTEXT["hr"], FINAL_CONTEXT["hrv"], FINAL_CONTEXT["sleep"], baseline_hr
    )

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Tu biblioteca, en el momento justo</div>
          <h1>¿A qué jugamos hoy?</h1>
          <p>PlayNow revisó tu tiempo disponible y el contexto de tu sesión para encontrar algo que encaje contigo.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    platform_badges()

    context_left, context_right = st.columns(2)
    with context_left:
        st.markdown(
            f"<div class='context-card'><span><strong>Contexto listo.</strong> Esta noche tienes {FINAL_CONTEXT['minutes']} minutos libres y tu biblioteca ya está sincronizada.</span></div>",
            unsafe_allow_html=True,
        )
    with context_right:
        st.markdown(
            f"<div class='context-card'><span><strong>Ritmo de hoy.</strong> Fatiga estimada: {fatigue:.0f}/100.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='choice-label'>Una sola pregunta para empezar</div>", unsafe_allow_html=True)
    st.markdown("## ¿Qué te provoca jugar hoy?")
    choices = list(INTENTION_TARGETS)
    selected = st.session_state.get("final_intention")
    choice_columns = st.columns(len(choices))
    for column, key in zip(choice_columns, choices):
        with column:
            if st.button(
                INTENTION_TARGETS[key]["label"],
                key=f"final_{key}",
                width="stretch",
                type="primary" if selected == key else "secondary",
                disabled=selected is not None,
            ):
                st.session_state["final_intention"] = key
                st.rerun()

    if selected is None:
        st.markdown(
            "<div class='awaiting-choice'>Elige una opción para descubrir qué juego encaja mejor con esta sesión.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='selected-answer'><strong>Intención elegida:</strong> {html.escape(INTENTION_TARGETS[selected]['label'])}. Para probar otra, reinicia tu elección.</div>",
        unsafe_allow_html=True,
    )
    _, reset_column, _ = st.columns([2, 1, 2])
    with reset_column:
        if st.button("↻ Elegir otra vez", key="reset_final", width="stretch"):
            del st.session_state["final_intention"]
            st.rerun()

    st.markdown("<div class='section-title'>Tu mejor sesión empieza aquí</div>", unsafe_allow_html=True)
    st.markdown("<p class='section-copy'>Estas opciones salen únicamente de tu biblioteca.</p>", unsafe_allow_html=True)
    ranking = engine.rank(FINAL_CONTEXT["user_id"], FINAL_CONTEXT["minutes"], fatigue, selected)
    recommendation_cards(ranking)


def advanced_experience(engine: PlayNowEngine, users: pd.DataFrame) -> None:
    """Conserva los controles del prototipo en una presentación más ordenada."""
    st.markdown("<div class='eyebrow'>Vista de exploración</div>", unsafe_allow_html=True)
    st.markdown("# PlayNow avanzado")
    st.caption("Explora cómo las señales simuladas cambian la recomendación sin alterar el motor.")

    if "minutes" not in st.session_state:
        apply_preset("Martes 21:30, relajarme")

    names = {row.user_id: f"{row.nombre} ({row.perfil_ocupacional})" for row in users.itertuples(index=False)}
    with st.sidebar:
        st.markdown("## Configurar sesión")
        user_id = st.selectbox("Persona", list(names), format_func=lambda key: names[key])
        st.markdown("### Escenarios rápidos")
        for name in PRESETS:
            st.button(name, on_click=apply_preset, args=(name,), width="stretch")
        st.markdown("### Señales simuladas")
        minutes = st.slider("Minutos libres", min_value=10, max_value=240, key="minutes")
        hr = st.slider("Frecuencia cardíaca", min_value=50, max_value=120, key="hr")
        hrv = st.slider("HRV", min_value=15, max_value=80, key="hrv")
        sleep = st.slider("Horas de sueño", min_value=3.0, max_value=9.0, step=0.5, key="sleep")
        intention = st.radio(
            "¿Qué te provoca jugar hoy?",
            list(INTENTION_TARGETS),
            format_func=lambda key: INTENTION_TARGETS[key]["label"],
            key="intention",
        )

    profile = users.loc[users["user_id"] == user_id].iloc[0]
    baseline_hr = float(profile["hr_baseline"])
    fatigue = fatigue_from_wearable(hr, hrv, sleep, baseline_hr)
    ranking = engine.rank(user_id, minutes, fatigue, intention)
    baseline = engine.baseline(user_id)
    metrics = engine.list_metrics(ranking)
    load = float(ranking.iloc[0]["carga_max"]) if len(ranking) else 0.0

    platform_badges()
    st.markdown(f"<div class='section-title'>Sesión de {html.escape(str(profile['nombre']))}</div>", unsafe_allow_html=True)
    st.caption("Las señales son simuladas para poder comparar escenarios del prototipo.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Perfil", str(profile["perfil_ocupacional"]).replace("_", " "))
    c2.metric("Fatiga estimada", f"{fatigue:.0f}/100")
    c3.metric("Carga admisible", f"{load:.2f}")
    c4.metric("Juego refugio", str(profile["juego_refugio"]))

    if ranking.empty:
        st.warning("Esta persona no tiene juegos en la biblioteca simulada.")
        return

    st.markdown("<div class='section-title'>Recomendaciones de esta sesión</div>", unsafe_allow_html=True)
    recommendation_cards(ranking)

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='section-title'>Ranking PlayNow</div>", unsafe_allow_html=True)
        ranking_view = ranking.head(5).loc[:, ["puesto", "titulo", "score", "horas_jugadas"]].copy()
        ranking_view["score"] = ranking_view["score"].map(lambda value: f"{value:.0%}")
        compact_table(
            ranking_view,
            ["puesto", "titulo", "score", "horas_jugadas"],
            {"puesto": "#", "titulo": "Juego", "score": "Afinidad", "horas_jugadas": "Horas jugadas"},
        )
    with right:
        st.markdown("<div class='section-title'>Referencia por horas jugadas</div>", unsafe_allow_html=True)
        compact_table(
            baseline.head(5),
            ["puesto", "titulo", "horas_jugadas"],
            {"puesto": "#", "titulo": "Juego", "horas_jugadas": "Horas jugadas"},
        )

    st.markdown("<div class='section-title'>Lectura de la lista</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Diversidad", f"{metrics['diversidad_intra_lista']:.2f}")
    m2.metric("Juegos poco jugados", f"{metrics['cobertura_backlog']:.0%}")
    m3.metric("Incluye un indie", "Sí" if metrics["exposicion_indie"] else "No")
    st.caption("Los indicadores describen esta lista simulada, no la satisfacción de personas reales.")


def main() -> None:
    """Organiza las dos maneras de recorrer el mismo motor de recomendaciones."""
    st.set_page_config(page_title="PlayNow", page_icon="🎮", layout="wide")
    inject_style()
    engine, users = load_engine()

    # Dos botones de navegación evitan los colores y los indicadores por defecto
    # de un selector, y permiten ocultar los controles técnicos en la vista final.
    if "demo_view" not in st.session_state:
        st.session_state["demo_view"] = "final"
    nav_final, nav_advanced = st.columns(2)
    with nav_final:
        if st.button("🎮 PlayNow", key="view_final", width="stretch", type="primary" if st.session_state["demo_view"] == "final" else "secondary"):
            st.session_state["demo_view"] = "final"
            st.rerun()
    with nav_advanced:
        if st.button("⚙️ Versión avanzada", key="view_advanced", width="stretch", type="primary" if st.session_state["demo_view"] == "advanced" else "secondary"):
            st.session_state["demo_view"] = "advanced"
            st.rerun()

    view = st.session_state["demo_view"]
    if view == "final":
        final_experience(engine, users)
    else:
        advanced_experience(engine, users)


if __name__ == "__main__":
    main()
