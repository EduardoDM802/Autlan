from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.montecarlo_engine import REQUIRED_FILES, run_model, to_csv_bytes


st.set_page_config(
    page_title="Cobertura AUTLAN",
    page_icon="AM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PLOT_TEMPLATE = "plotly_white"
COLORS = {
    "tec_blue": "#0039A6",
    "autlan_teal": "#007C92",
    "text": "#1F2933",
    "muted": "#667085",
    "light": "#F4F6F8",
    "border": "#D9E2EC",
    "white": "#FFFFFF",
    "gold": "#C79A24",
    "green": "#007C92",
    "esg_green": "#2E7D32",
    "red": "#C2410C",
    "blue": "#0039A6",
    "grid": "rgba(31,41,51,.10)",
}
LOGO_DIR = PROJECT_ROOT / "LOGOS"
TEC_LOGO_PATH = LOGO_DIR / "tec_logo.png"
AUTLAN_LOGO_PATH = LOGO_DIR / "autlan_logo.png"
AUTLAN_LOGO_FALLBACKS = [LOGO_DIR / "auntlan_logo.png"]


st.markdown(
    """
    <style>
      .stApp {
        background: #FFFFFF;
        color: #1F2933;
      }
      .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
      }
      h1, h2, h3, h4 {
        color: #0039A6;
        letter-spacing: -0.02em;
      }
      p, li, label, span {
        color: #1F2933;
      }
      [data-testid="stMetric"] {
        background: #F4F6F8;
        border: 1px solid #E5EAF0;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(31,41,51,.08);
      }
      [data-testid="stMetricLabel"] {
        color: #667085;
        font-weight: 700;
      }
      [data-testid="stMetricValue"] {
        color: #0039A6;
        font-weight: 800;
      }
      [data-testid="stMetricDelta"] svg { fill: #007C92; }
      [data-testid="stMetricDelta"] div { color: #007C92; }
      div[data-testid="stExpander"] {
        border: 1px solid #E5EAF0;
        border-radius: 14px;
        background: #FFFFFF;
        box-shadow: 0 6px 18px rgba(31,41,51,.05);
      }
      .section-note {
        color: #667085;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
      }
      .section-divider {
        height: 1px;
        width: 100%;
        margin: 28px 0 18px;
        background: linear-gradient(90deg, rgba(0,57,166,.28), rgba(0,124,146,.28), rgba(217,226,236,.12));
      }
      .section-kicker {
        color: #007C92;
        font-weight: 800;
        font-size: .78rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .executive-section h2 {
        margin-bottom: 4px;
      }
      .narrative-card,
      .macro-card,
      .scenario-card {
        background: #F4F6F8;
        border: 1px solid #E5EAF0;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 22px rgba(31,41,51,.06);
        height: 100%;
      }
      .narrative-card strong,
      .macro-card strong,
      .scenario-card strong {
        color: #0039A6;
      }
      .macro-card h3 {
        color: #0039A6;
        margin: 0 0 10px;
        font-size: 1.18rem;
      }
      .macro-card .label {
        color: #667085;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .06em;
        text-transform: uppercase;
      }
      .macro-card .value {
        color: #1F2933;
        font-size: 1rem;
        font-weight: 650;
        margin: 2px 0 10px;
      }
      .risk-text {
        color: #C2410C;
        font-weight: 750;
      }
      .strategy-text {
        color: #007C92;
        font-weight: 750;
      }
      .esg-card {
        background:
          linear-gradient(135deg, rgba(46,125,50,.08), rgba(0,57,166,.035)),
          #FFFFFF;
        border: 1px solid rgba(46,125,50,.22);
        border-left: 6px solid #2E7D32;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 22px rgba(31,41,51,.06);
        height: 100%;
      }
      .esg-card h3 {
        color: #2E7D32;
        margin: 0 0 8px;
        font-size: 1.16rem;
      }
      .esg-kicker {
        color: #2E7D32;
        font-weight: 850;
        font-size: .76rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .esg-stat {
        color: #1F2933;
        font-size: clamp(1.45rem, 2.2vw, 2rem);
        font-weight: 850;
        letter-spacing: -0.03em;
        margin: 4px 0;
      }
      .esg-substat {
        color: #0039A6;
        font-weight: 800;
        margin: 0 0 10px;
      }
      .esg-card p {
        color: #1F2933;
        margin: 0;
      }
      .esg-mini {
        margin-top: 14px;
      }
      .macro-hero {
        padding: 22px 24px;
        border: 1px solid #E5EAF0;
        border-left: 6px solid #007C92;
        border-radius: 18px;
        background:
          linear-gradient(135deg, rgba(0,57,166,.04), rgba(0,124,146,.08)),
          #FFFFFF;
        box-shadow: 0 8px 24px rgba(31,41,51,.06);
        margin-bottom: 20px;
      }
      .macro-hero h2 {
        margin: 0 0 8px;
        color: #0039A6;
      }
      .macro-hero p {
        margin: 0;
        color: #667085;
        font-size: 1.02rem;
      }
      .institutional-header {
        display: grid;
        grid-template-columns: minmax(110px, 180px) 1fr minmax(110px, 180px);
        gap: 24px;
        align-items: center;
        padding: 24px 28px 18px;
        margin-bottom: 20px;
        border: 1px solid #E5EAF0;
        border-radius: 22px;
        background:
          linear-gradient(135deg, rgba(0,57,166,.045), rgba(0,124,146,.060)),
          #FFFFFF;
        box-shadow: 0 12px 34px rgba(31,41,51,.08);
      }
      .header-logo-wrap {
        min-height: 78px;
        display: flex;
        align-items: center;
      }
      .header-logo-wrap.left { justify-content: flex-start; }
      .header-logo-wrap.right { justify-content: flex-end; }
      .header-logo {
        max-height: 74px;
        max-width: 170px;
        object-fit: contain;
      }
      .header-title {
        text-align: center;
      }
      .header-title h1 {
        margin: 0;
        color: #0039A6;
        font-size: clamp(1.8rem, 3vw, 2.75rem);
        font-weight: 850;
      }
      .header-title p {
        margin: 8px 0 0;
        color: #1F2933;
        font-size: 1.02rem;
        font-weight: 500;
      }
      .accent-bar {
        height: 5px;
        margin: 18px auto 0;
        width: min(520px, 85%);
        border-radius: 999px;
        background: linear-gradient(90deg, #0039A6, #007C92);
      }
      .logo-placeholder {
        border: 1px dashed #B9C4D3;
        border-radius: 14px;
        padding: 12px;
        min-height: 64px;
        min-width: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #667085;
        background: #F4F6F8;
        font-size: .85rem;
        text-align: center;
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #D9E2EC;
      }
      .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        color: #667085;
        font-weight: 700;
        padding: 10px 16px;
      }
      .stTabs [aria-selected="true"] {
        background: #F4F6F8;
        color: #0039A6;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid #E5EAF0;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(31,41,51,.04);
      }
      .stDownloadButton button {
        border-radius: 12px;
        border: 1px solid #007C92;
        color: #007C92;
        background: #FFFFFF;
        font-weight: 700;
      }
      .stDownloadButton button:hover {
        border-color: #0039A6;
        color: #0039A6;
        background: #F4F6F8;
      }
      @media (max-width: 760px) {
        .institutional-header {
          grid-template-columns: 1fr;
          text-align: center;
        }
        .header-logo-wrap.left,
        .header-logo-wrap.right {
          justify-content: center;
        }
        .header-logo {
          max-height: 58px;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_money(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value / 1e6:,.{decimals}f}M"


def fmt_price(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.{decimals}f}"


def fmt_fx(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:,.{decimals}f}%"


def prepare_display_money(df: pd.DataFrame, money_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].map(fmt_money)
    return out


def apply_fig_layout(fig: go.Figure, *, height: int = 430, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        title=title,
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["white"],
        font=dict(color=COLORS["text"]),
        title_font=dict(color=COLORS["tec_blue"], size=18),
        margin=dict(l=34, r=24, t=58 if title else 24, b=34),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS["text"]),
        ),
        hoverlabel=dict(bgcolor=COLORS["white"], font_color=COLORS["text"]),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


@st.cache_data(show_spinner=False)
def load_results(root: str):
    return run_model(root)


def chart_fcf_distribution(results) -> go.Figure:
    fcf_sin = results.simulations["fcf_sin_cob"] / 1e6
    fcf_con = results.simulations["fcf_con_cob"] / 1e6
    fig = go.Figure()
    fig.add_histogram(
        x=fcf_sin,
        nbinsx=70,
        histnorm="probability density",
        name="Sin cobertura",
        marker_color=COLORS["red"],
        opacity=0.55,
    )
    fig.add_histogram(
        x=fcf_con,
        nbinsx=70,
        histnorm="probability density",
        name="Con cobertura",
        marker_color=COLORS["green"],
        opacity=0.62,
    )
    for key, color, label in [
        ("fcf_sin_stats", COLORS["red"], "P50 sin"),
        ("fcf_con_stats", COLORS["green"], "P50 con"),
    ]:
        x = results.metrics[key]["p50"] / 1e6
        fig.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="FCF total (millones USD)", tickprefix="$", ticksuffix="M")
    fig.update_yaxes(title="Densidad")
    return apply_fig_layout(fig, title="Distribucion de FCF total")


def chart_diff_distribution(results) -> go.Figure:
    diff = results.simulations["diferencia_fcf"] / 1e6
    stats = results.metrics["diferencia_stats"]
    fig = go.Figure()
    fig.add_histogram(
        x=diff,
        nbinsx=70,
        histnorm="probability density",
        name="Mejora FCF",
        marker_color=COLORS["blue"],
        opacity=0.72,
    )
    for p, dash in [("p5", "dot"), ("p50", "dash"), ("p95", "dot")]:
        fig.add_vline(
            x=stats[p] / 1e6,
            line_dash=dash,
            line_color=COLORS["text"] if p == "p50" else COLORS["muted"],
            annotation_text=p.upper(),
        )
    fig.update_xaxes(title="Mejora de FCF (millones USD)", tickprefix="$", ticksuffix="M")
    fig.update_yaxes(title="Densidad")
    return apply_fig_layout(fig, title="Distribucion de mejora con cobertura")


def chart_paths(results, key: str, title: str, y_title: str, color: str) -> go.Figure:
    sim = results.simulations[key]
    months = results.metrics["meses_nombres"]
    rng = np.random.RandomState(7)
    sample_size = min(80, sim.shape[0])
    idx = rng.choice(sim.shape[0], size=sample_size, replace=False)
    fig = go.Figure()
    for i in idx:
        fig.add_trace(
            go.Scatter(
                x=months,
                y=sim[i, :],
                mode="lines",
                line=dict(color="rgba(155,165,180,.16)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=sim.mean(axis=0),
            mode="lines+markers",
            name="Media simulada",
            line=dict(color=color, width=3),
        )
    )
    fig.update_xaxes(title="Mes")
    fig.update_yaxes(title=y_title)
    return apply_fig_layout(fig, title=title)


def chart_history_forecast(history: pd.DataFrame, forecast: pd.DataFrame, value_col: str, title: str, y_title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["price"],
            mode="lines",
            name="Historico",
            line=dict(color="#8fa3bd", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["fecha"],
            y=forecast[value_col],
            mode="lines+markers",
            name="Forecast / forward",
            line=dict(color=color, width=3),
        )
    )
    fig.update_xaxes(title="Fecha")
    fig.update_yaxes(title=y_title)
    return apply_fig_layout(fig, title=title)


def chart_income_statement(results) -> go.Figure:
    df = results.tables["income_statement"]
    fig = go.Figure()
    fig.add_bar(
        x=df["concepto"],
        y=df["sin_cobertura"] / 1e6,
        name="Sin cobertura",
        marker_color=COLORS["red"],
    )
    fig.add_bar(
        x=df["concepto"],
        y=df["con_cobertura"] / 1e6,
        name="Con cobertura",
        marker_color=COLORS["green"],
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Millones USD", tickprefix="$", ticksuffix="M")
    return apply_fig_layout(fig, height=470, title="Estado de resultados promedio")


def chart_cascade(results) -> go.Figure:
    df = results.tables["cascade_summary"]
    fig = go.Figure()
    fig.add_bar(
        x=df["paso"],
        y=df["valor_minimo"] / 1e6,
        name="Minimo",
        marker_color=COLORS["blue"],
    )
    fig.add_bar(
        x=df["paso"],
        y=df["valor_promedio"] / 1e6,
        name="Promedio",
        marker_color=COLORS["gold"],
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Millones USD", tickprefix="$", ticksuffix="M")
    return apply_fig_layout(fig, title="Cascada de cobertura")


def chart_put_payoff(results) -> go.Figure:
    df = results.tables["payoff_put_oro"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["precio_oro"],
            y=df["payoff_put_neto_prima"] / 1e6,
            mode="lines",
            name="Payoff neto del put",
            line=dict(color=COLORS["gold"], width=3),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    fig.add_vline(
        x=results.metrics["strike_put"],
        line_dash="dash",
        line_color=COLORS["green"],
        annotation_text="Strike",
    )
    fig.update_xaxes(title="Precio oro final (USD/oz)", tickprefix="$")
    fig.update_yaxes(title="Millones USD", tickprefix="$", ticksuffix="M")
    return apply_fig_layout(fig, title="Payoff neto del put de oro")


def render_section(kicker: str, title: str, note: str | None = None) -> None:
    note_html = f"<p class='section-note'>{note}</p>" if note else ""
    st.markdown(
        f"""
        <div class="section-divider"></div>
        <div class="executive-section">
          <div class="section-kicker">{kicker}</div>
          <h2>{title}</h2>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def narrative_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="narrative-card">
          <strong>{title}</strong>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def macro_factor_card(
    title: str,
    indicator: str,
    impact: str,
    risk: str,
    hedge: str,
) -> None:
    st.markdown(
        f"""
        <div class="macro-card">
          <h3>{title}</h3>
          <div class="label">Indicador clave</div>
          <div class="value">{indicator}</div>
          <div class="label">Impacto financiero</div>
          <div class="value">{impact}</div>
          <div class="label">Riesgo</div>
          <div class="value risk-text">{risk}</div>
          <div class="label">Conexión con cobertura</div>
          <div class="value strategy-text">{hedge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def esg_indicator_card(*, compact: bool = False) -> None:
    card_class = "esg-card esg-mini" if compact else "esg-card"
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="esg-kicker">Indicador ESG vinculado a la estrategia</div>
          <h3>Energía propia / limpia</h3>
          <div class="esg-stat">25%–26% del consumo energético</div>
          <div class="esg-substat">Ahorros reportados 2025: US$10.1M</div>
          <p>Este factor fortalece la estrategia de cobertura porque reduce la exposición operativa a choques de precios energéticos y mejora la resiliencia del flujo de caja.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def macro_financials_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anio": [2022, 2023, 2024, 2025],
            "ventas_usd_m": [624.2, 365.4, 312.9, 322.7],
            "uafirda_usd_m": [214.2, 34.4, 36.2, 31.5],
        }
    )


def macro_steel_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anio": [2022, 2024],
            "produccion_acero_mt": [18.1, 13.8],
            "nota": ["Referencia 2022", "2024: caida de 16% anual"],
        }
    )


def macro_gold_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "escenario": ["Bajo: precio prom. 2025", "Alto: LBMA 5-jun-2026"],
            "onzas_2026": [20_000, 20_000],
            "precio_usd_oz": [3_431, 4_463],
            "potencial_usd_m": [68.6, 89.3],
        }
    )


def macro_fx_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impacto": ["Perdida cambiaria 2025", "Reduccion deuda 2024"],
            "valor_usd_m": [-11.1, 14.1],
            "tipo": ["RIF negativo", "Beneficio por depreciacion MXN"],
        }
    )


def esg_energy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "periodo": ["2025", "1T26"],
            "energia_propia_limpia_pct": [0.25, 0.26],
            "ahorro_usd_m": [10.1, np.nan],
        }
    )


def macro_matrix_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Factor": "Oro",
                "Variable afectada": "Ingresos Metallorum / EBITDA / FCF",
                "Direccion del impacto": "Mayor oro mejora caja; menor oro reduce proteccion natural.",
                "Riesgo": "Caida de precio o menor rampa de onzas.",
                "Cobertura relacionada": "Put/collar de oro.",
            },
            {
                "Factor": "MXN/USD",
                "Variable afectada": "Costos, deuda, RIF, utilidad neta y flujo de caja",
                "Direccion del impacto": "Peso fuerte encarece costos en USD; depreciacion puede reducir deuda MXN en USD.",
                "Riesgo": "Apreciacion del MXN hacia niveles adversos.",
                "Cobertura relacionada": "Forwards/collar FX.",
            },
            {
                "Factor": "Acero y ferroaleaciones",
                "Variable afectada": "Ventas ferroaleaciones / volumen / EBITDA / utilizacion",
                "Direccion del impacto": "Menor acero reduce demanda, precios y volumen.",
                "Riesgo": "Desaceleracion industrial o contraccion siderurgica.",
                "Cobertura relacionada": "No derivado directo; monitoreo y escenarios.",
            },
        ]
    )


def macro_scenarios_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Escenario": "Optimista",
                "Acero": "Mejora demanda siderurgica y recuperacion en ferroaleaciones.",
                "MXN/USD": "MXN 18.2-19.0 favorece costos medidos en USD.",
                "Oro": "Precios altos y rampa de Metallorum cerca de 20 mil oz.",
            },
            {
                "Escenario": "Base",
                "Acero": "Estabilidad gradual tras la contraccion de 2024.",
                "MXN/USD": "Rango 17.4-17.9 neutral para margenes.",
                "Oro": "Promedio de referencia 2025 y ejecucion operacional consistente.",
            },
            {
                "Escenario": "Pesimista",
                "Acero": "Nueva contraccion industrial presiona volumen y precios.",
                "MXN/USD": "Apreciacion hacia 17.09 o menor presiona costos y margen.",
                "Oro": "Correccion hacia 4,000 USD/oz o menor reduce caja potencial.",
            },
        ]
    )


def chart_macro_financials() -> go.Figure:
    df = macro_financials_data()
    fig = go.Figure()
    fig.add_bar(
        x=df["anio"],
        y=df["ventas_usd_m"],
        name="Ventas",
        marker_color=COLORS["tec_blue"],
        text=[f"${v:.1f}M" for v in df["ventas_usd_m"]],
        textposition="outside",
    )
    fig.add_bar(
        x=df["anio"],
        y=df["uafirda_usd_m"],
        name="UAFIRDA",
        marker_color=COLORS["autlan_teal"],
        text=[f"${v:.1f}M" for v in df["uafirda_usd_m"]],
        textposition="outside",
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Año", dtick=1)
    fig.update_yaxes(title="Millones USD", tickprefix="$", ticksuffix="M")
    return apply_fig_layout(fig, height=430, title="Ventas y UAFIRDA de Autlán")


def chart_macro_steel() -> go.Figure:
    df = macro_steel_data()
    fig = go.Figure()
    fig.add_bar(
        x=df["anio"],
        y=df["produccion_acero_mt"],
        marker_color=[COLORS["tec_blue"], COLORS["red"]],
        text=[f"{v:.1f} Mt" for v in df["produccion_acero_mt"]],
        textposition="outside",
        name="Produccion acero",
    )
    fig.add_annotation(
        x=2024,
        y=13.8,
        text="-16% en 2024",
        showarrow=True,
        arrowhead=2,
        ay=-50,
        font=dict(color=COLORS["red"], size=13),
    )
    fig.update_xaxes(title="Año", dtick=1)
    fig.update_yaxes(title="Millones de toneladas")
    return apply_fig_layout(fig, height=430, title="Producción mexicana de acero líquido")


def chart_macro_gold() -> go.Figure:
    df = macro_gold_data()
    fig = go.Figure()
    fig.add_bar(
        x=df["escenario"],
        y=df["potencial_usd_m"],
        marker_color=[COLORS["gold"], COLORS["autlan_teal"]],
        text=[f"${v:.1f}M" for v in df["potencial_usd_m"]],
        textposition="outside",
        name="Potencial bruto",
    )
    fig.update_yaxes(title="Millones USD", tickprefix="$", ticksuffix="M")
    return apply_fig_layout(fig, height=420, title="Potencial bruto teórico de oro 2026")


def chart_macro_fx() -> go.Figure:
    df = macro_fx_data()
    fig = go.Figure()
    fig.add_bar(
        x=df["impacto"],
        y=df["valor_usd_m"],
        marker_color=[COLORS["red"], COLORS["autlan_teal"]],
        text=[f"{v:+.1f}M" for v in df["valor_usd_m"]],
        textposition="outside",
        name="Impacto USD",
    )
    fig.add_hline(y=0, line_color=COLORS["muted"], line_dash="dot")
    fig.update_yaxes(title="Millones USD", tickprefix="$", ticksuffix="M")
    return apply_fig_layout(fig, height=420, title="Impactos reportados de tipo de cambio")


def chart_esg_energy() -> go.Figure:
    df = esg_energy_data()
    fig = go.Figure()
    fig.add_bar(
        x=df["periodo"],
        y=df["energia_propia_limpia_pct"],
        marker_color=[COLORS["esg_green"], COLORS["autlan_teal"]],
        text=[f"{v:.0%}" for v in df["energia_propia_limpia_pct"]],
        textposition="outside",
        name="Cobertura energética operativa",
    )
    fig.add_annotation(
        x="2025",
        y=0.25,
        text="Ahorro US$10.1M",
        showarrow=True,
        arrowhead=2,
        ay=-42,
        font=dict(color=COLORS["esg_green"], size=13),
    )
    fig.update_yaxes(title="Consumo energético cubierto", tickformat=".0%", range=[0, 0.32])
    return apply_fig_layout(fig, height=320, title="Energía propia / limpia")


def download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label=label,
        data=to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def _detect_image_mime(path: Path) -> str:
    data = path.read_bytes()[:256].lstrip()
    if data.startswith(b"<svg") or data.startswith(b"<?xml") or b"<svg" in data[:160]:
        return "image/svg+xml"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "application/octet-stream"


def _logo_data_uri(path: Path) -> str | None:
    try:
        mime = _detect_image_mime(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return None


def _resolve_logo(label: str, expected_path: Path, fallbacks: list[Path] | None = None) -> tuple[str | None, str | None]:
    if expected_path.exists():
        return _logo_data_uri(expected_path), None

    for fallback in fallbacks or []:
        if fallback.exists():
            return _logo_data_uri(fallback), None

    message = (
        f"No se encontro el logo de {label} en `{expected_path.relative_to(PROJECT_ROOT)}`. "
        "El dashboard continuara sin ese logo."
    )
    return None, message


def _logo_html(uri: str | None, alt: str) -> str:
    if uri:
        return f'<img class="header-logo" src="{uri}" alt="{alt}" />'
    return f'<div class="logo-placeholder">{alt}<br/>no disponible</div>'


def render_header(results) -> None:
    tec_logo_uri, tec_warning = _resolve_logo("Tec", TEC_LOGO_PATH)
    autlan_logo_uri, autlan_warning = _resolve_logo("Autlan", AUTLAN_LOGO_PATH, AUTLAN_LOGO_FALLBACKS)

    st.markdown(
        f"""
        <section class="institutional-header">
          <div class="header-logo-wrap left">
            {_logo_html(tec_logo_uri, "Logo Tec")}
          </div>
          <div class="header-title">
            <h1>Dashboard de Cobertura AUTLÁN</h1>
            <p>Simulación Monte Carlo y análisis de estrategia de cobertura</p>
            <div class="accent-bar"></div>
          </div>
          <div class="header-logo-wrap right">
            {_logo_html(autlan_logo_uri, "Logo Autlan")}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    for warning in [tec_warning, autlan_warning]:
        if warning:
            st.warning(warning)


def executive_tab(results) -> None:
    metrics = results.metrics
    st.markdown("<p class='section-note'>Lectura rapida de precios, distribucion de FCF y efecto esperado de la cobertura.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Oro spot modelo", fmt_price(metrics["oro_spot_modelo"]), help="Spot usado por montecarlo.py para iniciar la simulacion.")
    c2.metric("Ultimo oro historico", fmt_price(metrics["oro_spot_historico"]), help="Ultimo precio Mid del historico mensual.")
    c3.metric("USD/MXN spot", fmt_fx(metrics["fx_spot"]), help="Ultimo precio Mid del historico mensual.")
    c4.metric("Put oro ejercido", fmt_pct(metrics["prob_put_ejercido"]), help="Probabilidad de oro final por debajo del strike.")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("FCF medio sin cobertura", fmt_money(metrics["fcf_sin_stats"]["media"]), help="Media de FCF total sin cobertura.")
    c6.metric("FCF medio con cobertura", fmt_money(metrics["fcf_con_stats"]["media"]), delta=fmt_money(metrics["diferencia_stats"]["media"]), help="Media de FCF total con cobertura.")
    c7.metric("Mejora P50", fmt_money(metrics["diferencia_stats"]["p50"]), help="Mediana de la mejora con cobertura.")
    c8.metric("Pasan la cascada", fmt_pct(metrics["prob_pasa_todo"]), help="Escenarios con remanente no negativo al final de la cascada.")

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(chart_fcf_distribution(results), use_container_width=True)
        st.caption("Compara la distribucion completa del FCF total con y sin cobertura. Las lineas punteadas muestran la mediana.")
    with right:
        summary = results.tables["distribution_summary"]
        st.dataframe(
            prepare_display_money(summary, ["sin_cobertura", "con_cobertura", "diferencia"]),
            use_container_width=True,
            hide_index=True,
        )
        download_button("Descargar resumen de distribucion", summary, "resumen_distribucion_fcf.csv")

    st.plotly_chart(chart_diff_distribution(results), use_container_width=True)
    st.caption("La mejora se calcula como FCF con cobertura menos FCF sin cobertura en cada escenario simulado.")


def simulation_tab(results) -> None:
    st.markdown("<p class='section-note'>Distribuciones y trayectorias generadas por el mismo proceso Monte Carlo del modelo original.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_fcf_distribution(results), use_container_width=True)
    with col2:
        st.plotly_chart(chart_diff_distribution(results), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(chart_paths(results, "oro_sim", "Trayectorias simuladas de oro", "USD/oz", COLORS["gold"]), use_container_width=True)
        st.caption("Se muestra una muestra de trayectorias y la media de los 10,000 escenarios.")
    with col4:
        st.plotly_chart(chart_paths(results, "fx_sim", "Trayectorias simuladas USD/MXN", "MXN por USD", COLORS["blue"]), use_container_width=True)
        st.caption("El FX se simula con volatilidad historica mensual y drift de forwards del modelo.")

    with st.expander("Tabla resumen de percentiles", expanded=True):
        st.dataframe(
            prepare_display_money(results.tables["distribution_summary"], ["sin_cobertura", "con_cobertura", "diferencia"]),
            use_container_width=True,
            hide_index=True,
        )


def gold_tab(results) -> None:
    st.markdown("<p class='section-note'>Historico de oro, curva forward usada por el modelo y tabla descargable.</p>", unsafe_allow_html=True)
    hist = results.processed_data["gold_historical"]
    forward = results.processed_data["gold_forward_model"]
    contracts = results.processed_data["gold_forward_contracts"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Ultimo historico", fmt_price(results.metrics["oro_spot_historico"]), help="Ultimo precio historico detectado.")
    c2.metric("Spot modelo", fmt_price(results.metrics["oro_spot_modelo"]), help="Valor fijo que usa montecarlo.py para iniciar la simulacion.")
    c3.metric("Volatilidad anual", fmt_pct(results.metrics["oro_vol_anual"]), help="Volatilidad historica anualizada desde retornos mensuales.")

    st.plotly_chart(
        chart_history_forecast(hist, forward, "precio_forward", "Oro historico vs forward", "USD/oz", COLORS["gold"]),
        use_container_width=True,
    )
    st.caption("La curva forward se interpola desde contratos /GC detectados en `gold_forecasts.csv`.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Forward mensual del modelo")
        st.dataframe(forward, use_container_width=True, hide_index=True)
        download_button("Descargar forward oro", forward, "forward_oro_modelo.csv")
    with col2:
        st.subheader("Contratos detectados")
        st.dataframe(contracts, use_container_width=True, hide_index=True)
        download_button("Descargar contratos oro", contracts, "contratos_oro_detectados.csv")


def fx_tab(results) -> None:
    st.markdown("<p class='section-note'>Historico USD/MXN, curva forward del modelo y forwards disponibles en el CSV fuente.</p>", unsafe_allow_html=True)
    hist = results.processed_data["usdmxn_historical"]
    forward_model = results.processed_data["usdmxn_forward_model"]
    forward_csv = results.processed_data["usdmxn_forwards_csv_processed"]

    c1, c2, c3 = st.columns(3)
    c1.metric("USD/MXN spot", fmt_fx(results.metrics["fx_spot"]), help="Ultimo precio historico detectado.")
    c2.metric("Forward May-27 modelo", fmt_fx(forward_model["forward_usdmxn_modelo"].iloc[-1]), help="Curva derivada de swap points del modelo original.")
    c3.metric("Volatilidad anual", fmt_pct(results.metrics["fx_vol_anual"]), help="Volatilidad historica anualizada desde retornos mensuales.")

    st.plotly_chart(
        chart_history_forecast(hist, forward_model, "forward_usdmxn_modelo", "USD/MXN historico vs forward modelo", "MXN por USD", COLORS["blue"]),
        use_container_width=True,
    )
    st.caption("`montecarlo.py` no usa el CSV de forwards FX para el calculo; conserva swap points hardcodeados. El dashboard carga el CSV para inspeccion.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Forward USD/MXN del modelo")
        st.dataframe(forward_model, use_container_width=True, hide_index=True)
        download_button("Descargar forward USD/MXN modelo", forward_model, "forward_usdmxn_modelo.csv")
    with col2:
        st.subheader("CSV de forwards USD/MXN procesado")
        st.dataframe(forward_csv, use_container_width=True, hide_index=True)
        download_button("Descargar forwards USD/MXN procesados", forward_csv, "forward_usdmxn_csv_procesado.csv")


def hedge_tab(results) -> None:
    st.markdown("<p class='section-note'>Comparativo con/sin cobertura, cascada de proteccion y escenarios downside/base/upside.</p>", unsafe_allow_html=True)
    metrics = results.metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mejora promedio", fmt_money(metrics["diferencia_stats"]["media"]))
    c2.metric("Mejora P5", fmt_money(metrics["diferencia_stats"]["p5"]))
    c3.metric("Remanente cascada medio", fmt_money(metrics["remanente_cascada_stats"]["media"]))
    c4.metric("Remanente minimo", fmt_money(metrics["remanente_cascada_stats"]["min"]))

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.plotly_chart(chart_income_statement(results), use_container_width=True)
        st.caption("Promedios del estado de resultados simulado, manteniendo las formulas del archivo original.")
    with col2:
        st.plotly_chart(chart_cascade(results), use_container_width=True)
        st.caption("La cascada muestra el valor minimo y promedio al cierre de cada paso de cobertura.")

    col3, col4 = st.columns([1, 1])
    with col3:
        st.subheader("Escenarios")
        scenario = results.tables["scenario_summary"]
        st.dataframe(
            prepare_display_money(scenario, ["fcf_sin_cobertura", "fcf_con_cobertura", "mejora_fcf"]),
            use_container_width=True,
            hide_index=True,
        )
        download_button("Descargar escenarios", scenario, "escenarios_cobertura.csv")
    with col4:
        st.plotly_chart(chart_put_payoff(results), use_container_width=True)
        st.caption("Payoff neto del put de oro despues de prima, para precios finales hipoteticos.")

    with st.expander("Estado de resultados promedio", expanded=False):
        income = results.tables["income_statement"]
        st.dataframe(
            prepare_display_money(income, ["sin_cobertura", "con_cobertura", "diferencia"]),
            use_container_width=True,
            hide_index=True,
        )
        download_button("Descargar estado de resultados", income, "estado_resultados_promedio.csv")


def data_tab(results) -> None:
    st.markdown("<p class='section-note'>Validacion rapida de fuentes, columnas, fechas, missing values y tablas procesadas.</p>", unsafe_allow_html=True)

    st.subheader("Validacion de archivos")
    st.dataframe(results.tables["data_quality"], use_container_width=True, hide_index=True)
    download_button("Descargar validacion de archivos", results.tables["data_quality"], "validacion_archivos.csv")

    st.subheader("Missing values por columna")
    st.dataframe(results.tables["missing_values"], use_container_width=True, hide_index=True)

    st.subheader("Preview de CSV originales")
    for key, filename in REQUIRED_FILES.items():
        df = results.raw_data[key]
        with st.expander(f"{filename} - {len(df):,} filas, {len(df.columns):,} columnas", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
            download_button(f"Descargar {filename}", df, filename)

    st.subheader("Tablas procesadas")
    processed_names = {
        "gold_historical": "oro_historico_procesado.csv",
        "gold_monthly": "oro_mensual_procesado.csv",
        "gold_forward_model": "oro_forward_modelo.csv",
        "usdmxn_historical": "usdmxn_historico_procesado.csv",
        "usdmxn_monthly": "usdmxn_mensual_procesado.csv",
        "usdmxn_forward_model": "usdmxn_forward_modelo.csv",
        "usdmxn_forwards_csv_processed": "usdmxn_forwards_csv_procesado.csv",
    }
    for key, filename in processed_names.items():
        df = results.processed_data[key]
        with st.expander(filename, expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
            download_button(f"Descargar {filename}", df, filename)


def render_downloads_section(results) -> None:
    with st.expander("Validación de archivos y calidad de datos", expanded=False):
        st.dataframe(results.tables["data_quality"], use_container_width=True, hide_index=True)
        download_button("Descargar validación de archivos", results.tables["data_quality"], "validacion_archivos.csv")

    with st.expander("Missing values por columna", expanded=False):
        st.dataframe(results.tables["missing_values"], use_container_width=True, hide_index=True)
        download_button("Descargar missing values", results.tables["missing_values"], "missing_values.csv")

    with st.expander("CSV originales", expanded=False):
        for key, filename in REQUIRED_FILES.items():
            df = results.raw_data[key]
            st.markdown(f"**{filename}** — {len(df):,} filas, {len(df.columns):,} columnas")
            st.dataframe(df.head(30), use_container_width=True)
            download_button(f"Descargar {filename}", df, filename)

    with st.expander("Tablas procesadas del modelo", expanded=False):
        processed_names = {
            "gold_historical": "oro_historico_procesado.csv",
            "gold_monthly": "oro_mensual_procesado.csv",
            "gold_forward_model": "oro_forward_modelo.csv",
            "usdmxn_historical": "usdmxn_historico_procesado.csv",
            "usdmxn_monthly": "usdmxn_mensual_procesado.csv",
            "usdmxn_forward_model": "usdmxn_forward_modelo.csv",
            "usdmxn_forwards_csv_processed": "usdmxn_forwards_csv_procesado.csv",
        }
        for key, filename in processed_names.items():
            df = results.processed_data[key]
            st.markdown(f"**{filename}**")
            st.dataframe(df.head(40), use_container_width=True)
            download_button(f"Descargar {filename}", df, filename)


def dashboard_cobertura_tab(results) -> None:
    metrics = results.metrics

    with st.container():
        st.markdown("<p class='section-note'>Vista integrada del modelo Monte Carlo, resultados de cobertura, curvas de mercado y datos descargables.</p>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Oro spot modelo", fmt_price(metrics["oro_spot_modelo"]), help="Spot usado por montecarlo.py para iniciar la simulación.")
        c2.metric("Último oro histórico", fmt_price(metrics["oro_spot_historico"]), help="Último precio Mid del histórico mensual.")
        c3.metric("USD/MXN spot", fmt_fx(metrics["fx_spot"]), help="Último precio Mid del histórico mensual.")
        c4.metric("Put oro ejercido", fmt_pct(metrics["prob_put_ejercido"]), help="Probabilidad de oro final por debajo del strike.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("FCF medio sin cobertura", fmt_money(metrics["fcf_sin_stats"]["media"]))
        c6.metric("FCF medio con cobertura", fmt_money(metrics["fcf_con_stats"]["media"]), delta=fmt_money(metrics["diferencia_stats"]["media"]))
        c7.metric("Mejora P50", fmt_money(metrics["diferencia_stats"]["p50"]))
        c8.metric("Pasan la cascada", fmt_pct(metrics["prob_pasa_todo"]))

        esg_indicator_card(compact=True)

    render_section(
        "Distribuciones",
        "Resultado financiero simulado",
        "La distribución muestra cómo se desplaza el FCF cuando se incorpora la estrategia de cobertura.",
    )
    with st.container():
        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(chart_fcf_distribution(results), use_container_width=True)
            st.caption("Compara FCF total con y sin cobertura. Las líneas punteadas muestran la mediana.")
        with right:
            summary = results.tables["distribution_summary"]
            st.dataframe(
                prepare_display_money(summary, ["sin_cobertura", "con_cobertura", "diferencia"]),
                use_container_width=True,
                hide_index=True,
            )
            download_button("Descargar resumen de distribución", summary, "resumen_distribucion_fcf.csv")

    with st.container():
        st.plotly_chart(chart_diff_distribution(results), use_container_width=True)
        st.caption("La mejora se calcula como FCF con cobertura menos FCF sin cobertura en cada escenario simulado.")

    render_section(
        "Simulación",
        "Trayectorias de mercado",
        "Muestra una selección de trayectorias para oro y USD/MXN, más la media de los 10,000 escenarios.",
    )
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(chart_paths(results, "oro_sim", "Trayectorias simuladas de oro", "USD/oz", COLORS["gold"]), use_container_width=True)
        with col2:
            st.plotly_chart(chart_paths(results, "fx_sim", "Trayectorias simuladas USD/MXN", "MXN por USD", COLORS["blue"]), use_container_width=True)

    with st.expander("Tabla de percentiles de FCF", expanded=False):
        st.dataframe(
            prepare_display_money(results.tables["distribution_summary"], ["sin_cobertura", "con_cobertura", "diferencia"]),
            use_container_width=True,
            hide_index=True,
        )

    render_section(
        "Mercados",
        "Oro: histórico y curva forward",
        "El módulo de oro integra el histórico observado y la curva forward interpolada desde contratos /GC.",
    )
    with st.container():
        hist = results.processed_data["gold_historical"]
        forward = results.processed_data["gold_forward_model"]
        contracts = results.processed_data["gold_forward_contracts"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Último histórico oro", fmt_price(results.metrics["oro_spot_historico"]))
        c2.metric("Spot modelo oro", fmt_price(results.metrics["oro_spot_modelo"]))
        c3.metric("Volatilidad anual oro", fmt_pct(results.metrics["oro_vol_anual"]))
        st.plotly_chart(
            chart_history_forecast(hist, forward, "precio_forward", "Oro histórico vs forward", "USD/oz", COLORS["gold"]),
            use_container_width=True,
        )
        with st.expander("Tablas de oro", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown("**Forward mensual del modelo**")
                st.dataframe(forward, use_container_width=True, hide_index=True)
                download_button("Descargar forward oro", forward, "forward_oro_modelo.csv")
            with right:
                st.markdown("**Contratos detectados**")
                st.dataframe(contracts, use_container_width=True, hide_index=True)
                download_button("Descargar contratos oro", contracts, "contratos_oro_detectados.csv")

    render_section(
        "Mercados",
        "USD/MXN: histórico y forward",
        "El modelo conserva la lógica original: la curva forward de USD/MXN se deriva de swap points hardcodeados.",
    )
    with st.container():
        hist_fx = results.processed_data["usdmxn_historical"]
        forward_fx = results.processed_data["usdmxn_forward_model"]
        forward_csv = results.processed_data["usdmxn_forwards_csv_processed"]
        c1, c2, c3 = st.columns(3)
        c1.metric("USD/MXN spot", fmt_fx(results.metrics["fx_spot"]))
        c2.metric("Forward May-27 modelo", fmt_fx(forward_fx["forward_usdmxn_modelo"].iloc[-1]))
        c3.metric("Volatilidad anual FX", fmt_pct(results.metrics["fx_vol_anual"]))
        st.plotly_chart(
            chart_history_forecast(hist_fx, forward_fx, "forward_usdmxn_modelo", "USD/MXN histórico vs forward modelo", "MXN por USD", COLORS["blue"]),
            use_container_width=True,
        )
        with st.expander("Tablas USD/MXN", expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown("**Forward USD/MXN del modelo**")
                st.dataframe(forward_fx, use_container_width=True, hide_index=True)
                download_button("Descargar forward USD/MXN modelo", forward_fx, "forward_usdmxn_modelo.csv")
            with right:
                st.markdown("**CSV de forwards USD/MXN procesado**")
                st.dataframe(forward_csv, use_container_width=True, hide_index=True)
                download_button("Descargar forwards USD/MXN procesados", forward_csv, "forward_usdmxn_csv_procesado.csv")

    render_section(
        "Cobertura",
        "Estado de resultados, cascada y escenarios",
        "Resume el impacto esperado de la cobertura sobre FCF, remanente y downside.",
    )
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejora promedio", fmt_money(metrics["diferencia_stats"]["media"]))
        c2.metric("Mejora P5", fmt_money(metrics["diferencia_stats"]["p5"]))
        c3.metric("Remanente cascada medio", fmt_money(metrics["remanente_cascada_stats"]["media"]))
        c4.metric("Remanente mínimo", fmt_money(metrics["remanente_cascada_stats"]["min"]))

        col1, col2 = st.columns([1.15, 0.85])
        with col1:
            st.plotly_chart(chart_income_statement(results), use_container_width=True)
        with col2:
            st.plotly_chart(chart_cascade(results), use_container_width=True)

        col3, col4 = st.columns([1, 1])
        with col3:
            st.subheader("Escenarios downside/base/upside")
            scenario = results.tables["scenario_summary"]
            st.dataframe(
                prepare_display_money(scenario, ["fcf_sin_cobertura", "fcf_con_cobertura", "mejora_fcf"]),
                use_container_width=True,
                hide_index=True,
            )
            download_button("Descargar escenarios", scenario, "escenarios_cobertura.csv")
        with col4:
            st.plotly_chart(chart_put_payoff(results), use_container_width=True)

    render_section(
        "Datos",
        "Datos descargables y validaciones",
        "Los CSV originales y tablas procesadas quedan disponibles al final para revisión y descarga.",
    )
    render_downloads_section(results)


def macro_tab() -> None:
    st.markdown(
        """
        <div class="macro-hero">
          <h2>Situación Macroeconómica de Autlán</h2>
          <p>Factores externos que pueden afectar ingresos, costos, márgenes y flujo de caja.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        macro_factor_card(
            "Ciclo siderúrgico",
            "90% de ventas 2024 provinieron de ferroaleaciones; acero mexicano cayó 16% a 13.8 Mt.",
            "Ingresos, volumen, precios de ferroaleaciones, EBITDA y utilización de capacidad.",
            "Nueva desaceleración industrial reduce demanda y margen.",
            "No hay derivado directo; se gestiona con monitoreo, escenarios y liquidez.",
        )
    with cols[1]:
        macro_factor_card(
            "MXN/USD",
            "Autlán reporta en USD, pero una parte relevante de costos, gastos y deuda está en MXN.",
            "Costos, márgenes, deuda, RIF, utilidad neta y flujo de caja.",
            "Peso fuerte encarece costos medidos en USD; pérdida cambiaria 2025 de US$11.1M.",
            "Forwards/collars FX para estabilizar caja y exposición de costos/deuda.",
        )
    with cols[2]:
        macro_factor_card(
            "Precio del oro",
            "Metallorum puede producir cerca de 20 mil oz en 2026; oro 2025 promedió US$3,431/oz.",
            "Ingresos Metallorum, EBITDA, flujo de caja y desapalancamiento.",
            "Corrección del oro o retraso de rampa reduce caja esperada.",
            "Put/collar de oro preserva downside sin perder todo el upside.",
        )

    render_section(
        "ESG operativo",
        "Indicador ESG vinculado a la estrategia",
        "La integración energética opera como una cobertura natural de costos y resiliencia de flujo.",
    )
    esg_left, esg_right = st.columns([0.95, 1.05])
    with esg_left:
        esg_indicator_card()
        st.caption(
            "Aunque la cobertura financiera se enfoca en oro, FX y tasas, la integración energética funciona como una cobertura operativa natural. Reduce dependencia externa de energía, ayuda a estabilizar costos y complementa la estrategia de protección del FCF."
        )
    with esg_right:
        st.plotly_chart(chart_esg_energy(), use_container_width=True)
        st.caption("Comparativo de energía propia / limpia: 2025: 25%; 1T26: 26%.")

    render_section(
        "Contexto financiero",
        "Ventas, UAFIRDA y ciclo siderúrgico",
        "El negocio tradicional sigue expuesto al ciclo del acero y a la demanda de ferroaleaciones.",
    )
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.plotly_chart(chart_macro_financials(), use_container_width=True)
    with col2:
        st.plotly_chart(chart_macro_steel(), use_container_width=True)
    st.caption("Datos sintetizados del PDF macroeconómico: ventas/UAFIRDA 2022-2025 y producción mexicana de acero líquido.")

    render_section(
        "Variables de cobertura",
        "Oro y tipo de cambio",
        "El oro puede aportar caja relevante, mientras que el tipo de cambio mueve costos, deuda y RIF.",
    )
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(chart_macro_gold(), use_container_width=True)
        st.caption("Potencial bruto teórico: 20,000 oz por precios de referencia 2025 y LBMA 5-jun-2026; equivale aproximadamente a 21%-28% de las ventas consolidadas 2025.")
    with col4:
        st.plotly_chart(chart_macro_fx(), use_container_width=True)
        st.caption("El FX puede generar impactos opuestos: presión por pérdida cambiaria o alivio por deuda MXN al depreciarse el peso.")

    render_section(
        "Matriz ejecutiva",
        "Factores, riesgos y cobertura relacionada",
        "Relaciona cada factor macro con la variable financiera afectada y la herramienta de gestión de riesgo.",
    )
    matrix = macro_matrix_data()
    st.table(matrix)
    download_button("Descargar matriz macro", matrix, "matriz_macro_autlan.csv")

    render_section(
        "Escenarios macro",
        "Optimista, base y pesimista",
        "Estos escenarios sirven como narrativa de soporte para interpretar los resultados del Monte Carlo.",
    )
    scenarios = macro_scenarios_data()
    s1, s2, s3 = st.columns(3)
    for col, (_, row) in zip([s1, s2, s3], scenarios.iterrows()):
        with col:
            st.markdown(
                f"""
                <div class="scenario-card">
                  <h3>{row['Escenario']}</h3>
                  <p><strong>Acero:</strong> {row['Acero']}</p>
                  <p><strong>MXN/USD:</strong> {row['MXN/USD']}</p>
                  <p><strong>Oro:</strong> {row['Oro']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Ver tabla completa de escenarios", expanded=False):
        st.dataframe(scenarios, use_container_width=True, hide_index=True)
        download_button("Descargar escenarios macro", scenarios, "escenarios_macro_autlan.csv")

    render_section(
        "Conexión con cobertura",
        "Por qué se justifica el modelo Monte Carlo",
        None,
    )
    narrative_card(
        "Tesis ejecutiva",
        "La estrategia de cobertura se justifica porque Autlán enfrenta una doble exposición: por un lado, su negocio tradicional depende del ciclo siderúrgico; por otro, el oro y el tipo de cambio pueden modificar de forma significativa el flujo de caja. El modelo Monte Carlo permite cuantificar estos escenarios y evaluar si la cobertura estabiliza el FCF.",
    )


def main() -> None:
    try:
        results = load_results(str(PROJECT_ROOT))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Ejecuta el dashboard desde la raiz del proyecto o verifica que los cuatro CSV requeridos existan junto a `montecarlo.py`.")
        st.stop()
    except Exception as exc:
        st.error("No se pudo ejecutar el modelo Monte Carlo.")
        st.exception(exc)
        st.stop()

    render_header(results)

    tabs = st.tabs([
        "Dashboard de Cobertura",
        "Situación Macroeconómica",
    ])

    with tabs[0]:
        dashboard_cobertura_tab(results)
    with tabs[1]:
        macro_tab()


if __name__ == "__main__":
    main()
