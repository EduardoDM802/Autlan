"""Motor Monte Carlo para el dashboard ejecutivo de Autlan.

Este modulo replica la logica financiera existente en montecarlo.py, pero la
expone como funciones importables para Streamlit. No ejecuta graficas ni prints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "gold_forecasts": "gold_forecasts.csv",
    "gold_historical": "gold_historical_prices.csv",
    "usdmxn_forwards": "usdmxn_forwards_forecasts.csv",
    "usdmxn_historical": "usdmxn_historical_prices.csv",
}

MONTH_NAMES = [
    "Jun26",
    "Jul26",
    "Ago26",
    "Sep26",
    "Oct26",
    "Nov26",
    "Dic26",
    "Ene27",
    "Feb27",
    "Mar27",
    "Abr27",
    "May27",
]

MONTH_CODE_MAP = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


@dataclass(frozen=True)
class MonteCarloResults:
    """Contenedor de resultados del modelo y tablas para dashboard."""

    project_root: Path
    raw_data: dict[str, pd.DataFrame]
    processed_data: dict[str, pd.DataFrame]
    simulations: dict[str, np.ndarray]
    metrics: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def _project_root_from_cwd(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).resolve()


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Busca una columna por nombre exacto o normalizado."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    normalized = {_normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        found = normalized.get(_normalize_name(candidate))
        if found:
            return found
    return None


def _require_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    column = _find_column(df, candidates)
    if column is None:
        raise ValueError(
            f"No se pudo detectar la columna de {label}. "
            f"Candidatas: {candidates}. Columnas disponibles: {list(df.columns)}"
        )
    return column


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _detect_price_series(df: pd.DataFrame, label: str) -> pd.Series:
    """Detecta precio usando Mid/MID_PRICE o promedio Bid/Ask."""
    mid_col = _find_column(df, ["Mid", "MID_PRICE", "mid", "Price", "Close", "Last"])
    if mid_col is not None:
        mid = _numeric_series(df, mid_col)
        if mid.notna().any():
            return mid

    bid_col = _find_column(df, ["Bid", "BID", "bid"])
    ask_col = _find_column(df, ["Ask", "ASK", "ask"])
    if bid_col is not None and ask_col is not None:
        bid = _numeric_series(df, bid_col)
        ask = _numeric_series(df, ask_col)
        avg = (bid + ask) / 2
        if avg.notna().any():
            return avg

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        series = _numeric_series(df, col)
        if series.notna().any():
            return series

    raise ValueError(
        f"No se pudo detectar una columna numerica de precio para {label}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def _load_required_csvs(project_root: Path) -> dict[str, pd.DataFrame]:
    missing = [filename for filename in REQUIRED_FILES.values() if not (project_root / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "No se encontraron archivos requeridos: " + ", ".join(missing)
        )

    return {
        key: pd.read_csv(project_root / filename)
        for key, filename in REQUIRED_FILES.items()
    }


def _prepare_historical_prices(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    date_col = _require_column(df, ["Date", "date", "Fecha", "fecha"], f"fecha en {label}")
    out = df.copy()
    out["Date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["price"] = _detect_price_series(out, label)
    out = out[["Date", "price"]].dropna().sort_values("Date").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No hay observaciones validas de precio para {label}.")

    monthly = out.set_index("Date").resample("ME").last()
    monthly["ret"] = np.log(monthly["price"] / monthly["price"].shift(1))
    monthly = monthly.dropna().reset_index()

    if monthly.empty:
        raise ValueError(f"No hay suficientes datos mensuales para calcular volatilidad de {label}.")

    vol_mensual = float(monthly["ret"].std())
    vol_anual = vol_mensual * float(np.sqrt(12))
    spot = float(monthly["price"].iloc[-1])
    return out, monthly, vol_mensual, vol_anual, spot


def _mid_from_bid_ask(df: pd.DataFrame, label: str) -> pd.Series:
    mid_col = _find_column(df, ["mid", "Mid", "MID_PRICE", "TRDPRC_1", "Last"])
    bid_col = _find_column(df, ["BID", "Bid", "bid"])
    ask_col = _find_column(df, ["ASK", "Ask", "ask"])

    mid = None
    if bid_col is not None and ask_col is not None:
        bid = _numeric_series(df, bid_col)
        ask = _numeric_series(df, ask_col)
        avg = (bid + ask) / 2
        if avg.notna().any():
            mid = avg

    if mid is None and mid_col is not None:
        mid_candidate = _numeric_series(df, mid_col)
        if mid_candidate.notna().any():
            mid = mid_candidate

    if mid is None:
        raise ValueError(
            f"No se pudo construir precio medio para {label}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return mid


def _prepare_gold_forwards(gold_fut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    fut = gold_fut.copy()
    instrument_col = _require_column(fut, ["Instrument", "instrument", "RIC"], "instrumento de futuros de oro")
    fut["mid"] = _mid_from_bid_ask(fut, "futuros de oro")

    records: list[dict[str, Any]] = []
    for _, row in fut.iterrows():
        inst = str(row[instrument_col])
        if inst.startswith("/GC") and pd.notna(row["mid"]) and len(inst) == 6:
            month_code = inst[3]
            month = MONTH_CODE_MAP.get(month_code)
            if month is None:
                continue
            try:
                year = int("20" + inst[4:6])
            except ValueError:
                continue
            records.append(
                {
                    "fecha": pd.Timestamp(year=year, month=month, day=15),
                    "instrumento": inst,
                    "precio": float(row["mid"]),
                }
            )

    df_fut_oro = pd.DataFrame(records).sort_values("fecha").reset_index(drop=True)
    if df_fut_oro.empty or len(df_fut_oro) < 2:
        raise ValueError("No hay suficientes futuros /GC validos para construir la curva forward de oro.")

    horizon_dates = pd.date_range("2026-06-01", periods=12, freq="MS")
    forward_prices = np.interp(
        [date.timestamp() for date in horizon_dates],
        [date.timestamp() for date in df_fut_oro["fecha"]],
        df_fut_oro["precio"].values,
    )
    forward_table = pd.DataFrame(
        {
            "fecha": horizon_dates,
            "mes": MONTH_NAMES,
            "precio_forward": forward_prices,
        }
    )
    return fut, df_fut_oro, horizon_dates, forward_prices


def _prepare_fx_forwards_csv(fx_fwd: pd.DataFrame) -> pd.DataFrame:
    out = fx_fwd.copy()
    try:
        out["mid_detectado"] = _mid_from_bid_ask(out, "forwards USD/MXN")
    except ValueError:
        out["mid_detectado"] = np.nan

    date_col = _find_column(out, ["MATUR_DATE", "Matur Date", "Maturity", "EXPIR_DATE", "Date"])
    if date_col is not None:
        out["fecha_detectada"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out["fecha_detectada"] = pd.NaT
    return out


def _pago_fijo(saldo: float, meses_total: int, sofr: float, spread: float, n_meses: int) -> float:
    cuota = saldo / meses_total
    tasa_m = (sofr + spread) / 12
    total = 0.0
    saldo_m = float(saldo)
    for _ in range(n_meses):
        total += saldo_m * tasa_m + cuota
        saldo_m -= cuota
    return float(total)


def _pago_variable(
    saldo: float,
    meses_total: int,
    sofr_vec: np.ndarray,
    spread: float,
    n_meses: int,
    n: int,
) -> np.ndarray:
    cuota = saldo / meses_total
    tasa_m = (sofr_vec + spread) / 12
    total = np.zeros(n)
    saldo_m = np.full(n, float(saldo))
    for _ in range(n_meses):
        total += saldo_m * tasa_m + cuota
        saldo_m -= cuota
    return total


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "media": float(np.mean(values)),
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _make_distribution_table(
    fcf_sin_cob: np.ndarray,
    fcf_con_cob: np.ndarray,
    diferencia_fcf: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for label, fn in [
        ("Media", np.mean),
        ("Minimo", np.min),
        ("P5", lambda x: np.percentile(x, 5)),
        ("P25", lambda x: np.percentile(x, 25)),
        ("P50", lambda x: np.percentile(x, 50)),
        ("P75", lambda x: np.percentile(x, 75)),
        ("P95", lambda x: np.percentile(x, 95)),
        ("Maximo", np.max),
    ]:
        sin = float(fn(fcf_sin_cob))
        con = float(fn(fcf_con_cob))
        diff = con - sin
        rows.append(
            {
                "metrica": label,
                "sin_cobertura": sin,
                "con_cobertura": con,
                "diferencia": diff,
            }
        )
    return pd.DataFrame(rows)


def _make_data_quality(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, filename in REQUIRED_FILES.items():
        df = raw_data[key]
        date_col = _find_column(df, ["Date", "date", "Fecha", "fecha", "MATUR_DATE", "EXPIR_DATE"])
        date_min = pd.NaT
        date_max = pd.NaT
        invalid_dates = None
        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            date_min = dates.min()
            date_max = dates.max()
            invalid_dates = int(dates.isna().sum())
        rows.append(
            {
                "archivo": filename,
                "filas": len(df),
                "columnas": len(df.columns),
                "columnas_detectadas": ", ".join(map(str, df.columns)),
                "columna_fecha_detectada": date_col or "",
                "fecha_min": date_min,
                "fecha_max": date_max,
                "fechas_invalidas": invalid_dates,
                "missing_values": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def _make_missing_values_table(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, filename in REQUIRED_FILES.items():
        df = raw_data[key]
        for column, missing in df.isna().sum().items():
            rows.append(
                {
                    "archivo": filename,
                    "columna": column,
                    "missing": int(missing),
                    "missing_pct": float(missing / len(df)) if len(df) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_model(
    project_root: str | Path | None = None,
    *,
    n: int = 10_000,
    meses: int = 12,
    seed: int = 42,
) -> MonteCarloResults:
    """Ejecuta el modelo Monte Carlo replicando montecarlo.py."""
    root = _project_root_from_cwd(project_root)
    raw = _load_required_csvs(root)

    gold_hist, gold_monthly, oro_vol_mensual, oro_vol_anual, oro_spot_historico = _prepare_historical_prices(
        raw["gold_historical"], "oro historico"
    )
    gold_fut_processed, df_fut_oro, fechas_horizonte, precios_forward_oro = _prepare_gold_forwards(
        raw["gold_forecasts"]
    )
    drift_oro_mensual = np.concatenate(
        [[np.log(precios_forward_oro[0] / oro_spot_historico)], np.diff(np.log(precios_forward_oro))]
    )

    fx_hist, fx_monthly, fx_vol_mensual, fx_vol_anual, fx_spot = _prepare_historical_prices(
        raw["usdmxn_historical"], "USD/MXN historico"
    )
    fx_forward_csv_processed = _prepare_fx_forwards_csv(raw["usdmxn_forwards"])

    swap_pts = {1: 0.0475, 3: 0.138, 6: 0.27, 12: 0.555}
    meses_conocidos = np.array([0] + list(swap_pts.keys()))
    forwards_conocidos = np.array([fx_spot] + [fx_spot + p for p in swap_pts.values()])
    forwards_fx_mensual = np.interp(np.arange(1, 13), meses_conocidos, forwards_conocidos)
    drift_fx_mensual = np.concatenate(
        [[np.log(forwards_fx_mensual[0] / fx_spot)], np.diff(np.log(forwards_fx_mensual))]
    )

    # Parametros de montecarlo.py.
    ingresos_totales_1t26 = 98_386_000
    ingresos_metallorum_1t26 = 11_964_000
    ingresos_op_anual = (ingresos_totales_1t26 - ingresos_metallorum_1t26) * 4
    costo_ventas_anual = 83_463_000 * 4
    gastos_venta_anual = 5_905_000 * 4
    gastos_admin_anual = 10_656_000 * 4
    otros_gastos_anual = 1_808_000 * 4
    depreciacion_anual = 10_597_000 * 4
    capex_anual = 3_313_000 * 4
    tasa_isr = 0.30

    ebitda_trimestral = np.array([7_466, 7_873, 9_356, 6_777, 10_767])
    ebitda_media_anual = float(ebitda_trimestral.mean() * 4 * 1_000)
    ebitda_std_anual = float(ebitda_trimestral.std() * 4 * 1_000)

    saldo_c1 = 119_000_000
    saldo_c2 = 16_000_000
    meses_c1 = 66
    meses_c2 = 19
    spread_c1 = 0.0600
    spread_c2 = 0.0550
    sofr_fijo = 0.03593
    gafin_sin_ref = 34_129_000
    gafin_con_ref = 9_000_000

    onzas_cubiertas = 10_000
    onzas_libres = 10_000
    strike_put = 4_200.0
    prima_por_oz = 210.0
    prima_put_total = onzas_cubiertas * prima_por_oz
    ingresos_base = 98_386_000 * 4
    prima_put_peso = ingresos_base * 0.60 * 0.05
    n_meses_cascada = 7

    # El notebook sobrescribe el spot historico y usa 4,420 para simulacion.
    oro_spot_modelo = 4_420.0

    rng = np.random.RandomState(seed)

    oro_sim = np.zeros((n, meses))
    precio_anterior = np.full(n, oro_spot_modelo)
    for m in range(meses):
        z = rng.normal(0, 1, n)
        oro_sim[:, m] = precio_anterior * np.exp(
            drift_oro_mensual[m] - 0.5 * oro_vol_mensual**2 + oro_vol_mensual * z
        )
        precio_anterior = oro_sim[:, m]
    oro_anio = oro_sim[:, -1]

    fx_sim = np.zeros((n, meses))
    fx_anterior = np.full(n, fx_spot)
    for m in range(meses):
        z = rng.normal(0, 1, n)
        fx_sim[:, m] = fx_anterior * np.exp(
            drift_fx_mensual[m] - 0.5 * fx_vol_mensual**2 + fx_vol_mensual * z
        )
        fx_anterior = fx_sim[:, m]
    fx_anio = fx_sim[:, -1]

    sofr_kappa = 0.30
    sofr_theta = 0.035
    sofr_sigma_r = 0.010
    sofr_dt = 1 / 12
    sofr_actual = 0.03593

    sofr_sim = np.zeros((n, meses))
    sofr_anterior = np.full(n, sofr_actual)
    for m in range(meses):
        z = rng.normal(0, 1, n)
        sofr_sim[:, m] = (
            sofr_anterior
            + sofr_kappa * (sofr_theta - sofr_anterior) * sofr_dt
            + sofr_sigma_r * np.sqrt(sofr_dt) * z
        )
        sofr_sim[:, m] = np.clip(sofr_sim[:, m], 0.001, 0.15)
        sofr_anterior = sofr_sim[:, m]
    sofr_prom = sofr_sim.mean(axis=1)

    ingreso_oz_cub = np.where(
        oro_anio < strike_put,
        onzas_cubiertas * strike_put,
        onzas_cubiertas * oro_anio,
    )
    ingreso_neto_p1 = ingreso_oz_cub - prima_put_total

    pago_c1_fijo = _pago_fijo(saldo_c1, meses_c1, sofr_fijo, spread_c1, n_meses_cascada)
    pago_c2_fijo = _pago_fijo(saldo_c2, meses_c2, sofr_fijo, spread_c2, n_meses_cascada)
    pago_deuda_fijo = pago_c1_fijo + pago_c2_fijo

    pago_c1_var = _pago_variable(saldo_c1, meses_c1, sofr_prom, spread_c1, n_meses_cascada, n)
    pago_c2_var = _pago_variable(saldo_c2, meses_c2, sofr_prom, spread_c2, n_meses_cascada, n)
    pago_deuda_var = pago_c1_var + pago_c2_var

    prima_swap_sim = pago_deuda_var - pago_deuda_fijo
    remanente_p2 = ingreso_neto_p1 - prima_swap_sim - pago_deuda_fijo
    remanente_cascada = remanente_p2 - prima_put_peso

    pasa_fase1 = np.ones(n, dtype=bool)
    pasa_fase2 = remanente_p2 >= 0
    pasa_fase3 = remanente_cascada >= 0
    pasa_todo = pasa_fase1 & pasa_fase2 & pasa_fase3

    precio_oro_base = 4_420.0
    ingreso_oz_base = (onzas_cubiertas + onzas_libres) * precio_oro_base
    ingreso_oz_sim = (onzas_cubiertas + onzas_libres) * oro_anio
    ajuste_oro_sin = ingreso_oz_sim - ingreso_oz_base

    ingresos_sin_cob = ingresos_base + ajuste_oro_sin
    gastos_op_sin = ingresos_sin_cob * 0.60
    ebit_sin = ingresos_sin_cob - gastos_op_sin
    ajuste_gafin_sin = (sofr_prom - 0.0386) * (saldo_c1 + saldo_c2)
    gafin_sin_sim = gafin_sin_ref + ajuste_gafin_sin
    ebt_sin = ebit_sin - gafin_sin_sim
    isr_sin = np.where(ebt_sin > 0, ebt_sin * tasa_isr, 0)
    fcf_sin_cob = ebt_sin - isr_sin

    ingreso_oz_lib_sim = onzas_libres * oro_anio
    ingreso_oz_lib_base = onzas_libres * precio_oro_base
    ajuste_oro_con = ingreso_oz_lib_sim - ingreso_oz_lib_base
    ingresos_con_cob = (ingresos_base - onzas_cubiertas * precio_oro_base) + ajuste_oro_con
    gastos_op_con = ingresos_con_cob * 0.60
    ebit_con = ingresos_con_cob - gastos_op_con
    gafin_con_sim = np.full(n, float(gafin_con_ref))
    ebt_con = ebit_con - gafin_con_sim
    isr_con = np.where(ebt_con > 0, ebt_con * tasa_isr, 0)
    fcf_operativo_con = ebt_con - isr_con
    fcf_con_cob = fcf_operativo_con + remanente_cascada
    diferencia_fcf = fcf_con_cob - fcf_sin_cob

    forward_oro_table = pd.DataFrame(
        {
            "fecha": fechas_horizonte,
            "mes": MONTH_NAMES,
            "precio_forward": precios_forward_oro,
        }
    )
    forward_fx_model_table = pd.DataFrame(
        {
            "fecha": fechas_horizonte,
            "mes": MONTH_NAMES,
            "forward_usdmxn_modelo": forwards_fx_mensual,
        }
    )

    income_statement = pd.DataFrame(
        [
            ("Ingresos totales", ingresos_sin_cob.mean(), ingresos_con_cob.mean()),
            ("Gastos op. (60%)", gastos_op_sin.mean(), gastos_op_con.mean()),
            ("EBIT", ebit_sin.mean(), ebit_con.mean()),
            ("GAFIN", gafin_sin_sim.mean(), gafin_con_sim.mean()),
            ("EBT", ebt_sin.mean(), ebt_con.mean()),
            ("ISR (30%)", isr_sin.mean(), isr_con.mean()),
            ("FCF operativo", fcf_sin_cob.mean(), fcf_operativo_con.mean()),
            ("Remanente cascada", 0.0, remanente_cascada.mean()),
            ("FCF TOTAL", fcf_sin_cob.mean(), fcf_con_cob.mean()),
        ],
        columns=["concepto", "sin_cobertura", "con_cobertura"],
    )
    income_statement["diferencia"] = income_statement["con_cobertura"] - income_statement["sin_cobertura"]

    distribution_summary = _make_distribution_table(fcf_sin_cob, fcf_con_cob, diferencia_fcf)

    scenario_summary = pd.DataFrame(
        [
            {
                "escenario": "Downside P5",
                "oro_usd_oz": np.percentile(oro_anio, 5),
                "usdmxn": np.percentile(fx_anio, 5),
                "fcf_sin_cobertura": np.percentile(fcf_sin_cob, 5),
                "fcf_con_cobertura": np.percentile(fcf_con_cob, 5),
                "mejora_fcf": np.percentile(diferencia_fcf, 5),
            },
            {
                "escenario": "Base P50",
                "oro_usd_oz": np.percentile(oro_anio, 50),
                "usdmxn": np.percentile(fx_anio, 50),
                "fcf_sin_cobertura": np.percentile(fcf_sin_cob, 50),
                "fcf_con_cobertura": np.percentile(fcf_con_cob, 50),
                "mejora_fcf": np.percentile(diferencia_fcf, 50),
            },
            {
                "escenario": "Upside P95",
                "oro_usd_oz": np.percentile(oro_anio, 95),
                "usdmxn": np.percentile(fx_anio, 95),
                "fcf_sin_cobertura": np.percentile(fcf_sin_cob, 95),
                "fcf_con_cobertura": np.percentile(fcf_con_cob, 95),
                "mejora_fcf": np.percentile(diferencia_fcf, 95),
            },
        ]
    )

    cascade_summary = pd.DataFrame(
        [
            {
                "paso": "Paso 1 - Put oro",
                "escenarios_que_pasan": int(pasa_fase1.sum()),
                "probabilidad": float(pasa_fase1.mean()),
                "valor_promedio": float(ingreso_neto_p1.mean()),
                "valor_minimo": float(ingreso_neto_p1.min()),
            },
            {
                "paso": "Paso 2 - IRS + deuda",
                "escenarios_que_pasan": int(pasa_fase2.sum()),
                "probabilidad": float(pasa_fase2.mean()),
                "valor_promedio": float(remanente_p2.mean()),
                "valor_minimo": float(remanente_p2.min()),
            },
            {
                "paso": "Paso 3 - Put peso",
                "escenarios_que_pasan": int(pasa_fase3.sum()),
                "probabilidad": float(pasa_fase3.mean()),
                "valor_promedio": float(remanente_cascada.mean()),
                "valor_minimo": float(remanente_cascada.min()),
            },
        ]
    )

    payoff_prices = np.linspace(
        max(0.0, np.percentile(oro_anio, 1) * 0.85),
        np.percentile(oro_anio, 99) * 1.05,
        200,
    )
    payoff_put_oro = pd.DataFrame(
        {
            "precio_oro": payoff_prices,
            "payoff_put_bruto": np.maximum(strike_put - payoff_prices, 0) * onzas_cubiertas,
            "payoff_put_neto_prima": np.maximum(strike_put - payoff_prices, 0) * onzas_cubiertas
            - prima_put_total,
            "ingreso_cubierto_neto": np.maximum(strike_put, payoff_prices) * onzas_cubiertas
            - prima_put_total,
        }
    )

    metrics = {
        "n": n,
        "meses": meses,
        "seed": seed,
        "meses_nombres": MONTH_NAMES,
        "fechas_horizonte": fechas_horizonte,
        "oro_spot_historico": oro_spot_historico,
        "oro_spot_modelo": oro_spot_modelo,
        "oro_vol_mensual": oro_vol_mensual,
        "oro_vol_anual": oro_vol_anual,
        "fx_spot": fx_spot,
        "fx_vol_mensual": fx_vol_mensual,
        "fx_vol_anual": fx_vol_anual,
        "sofr_fijo": sofr_fijo,
        "sofr_actual": sofr_actual,
        "sofr_stats": _summary_stats(sofr_prom),
        "oro_stats": _summary_stats(oro_anio),
        "fx_stats": _summary_stats(fx_anio),
        "fcf_sin_stats": _summary_stats(fcf_sin_cob),
        "fcf_con_stats": _summary_stats(fcf_con_cob),
        "diferencia_stats": _summary_stats(diferencia_fcf),
        "prob_put_ejercido": float((oro_anio < strike_put).mean()),
        "prob_fcf_positivo_sin": float((fcf_sin_cob > 0).mean()),
        "prob_fcf_positivo_con": float((fcf_con_cob > 0).mean()),
        "prob_pasa_todo": float(pasa_todo.mean()),
        "strike_put": strike_put,
        "prima_por_oz": prima_por_oz,
        "prima_put_total": prima_put_total,
        "onzas_cubiertas": onzas_cubiertas,
        "onzas_libres": onzas_libres,
        "prima_put_peso": prima_put_peso,
        "pago_deuda_fijo": pago_deuda_fijo,
        "prima_swap_promedio": float(prima_swap_sim.mean()),
        "remanente_cascada_stats": _summary_stats(remanente_cascada),
        "ingresos_base": ingresos_base,
        "ingresos_op_anual": ingresos_op_anual,
        "costo_ventas_anual": costo_ventas_anual,
        "gastos_venta_anual": gastos_venta_anual,
        "gastos_admin_anual": gastos_admin_anual,
        "otros_gastos_anual": otros_gastos_anual,
        "depreciacion_anual": depreciacion_anual,
        "capex_anual": capex_anual,
        "tasa_isr": tasa_isr,
        "ebitda_media_anual": ebitda_media_anual,
        "ebitda_std_anual": ebitda_std_anual,
        "gafin_sin_ref": gafin_sin_ref,
        "gafin_con_ref": gafin_con_ref,
        "swap_pts_fx_modelo": swap_pts,
    }

    processed_data = {
        "gold_historical": gold_hist,
        "gold_monthly": gold_monthly,
        "gold_forecasts_raw_processed": gold_fut_processed,
        "gold_forward_contracts": df_fut_oro,
        "gold_forward_model": forward_oro_table,
        "usdmxn_historical": fx_hist,
        "usdmxn_monthly": fx_monthly,
        "usdmxn_forwards_csv_processed": fx_forward_csv_processed,
        "usdmxn_forward_model": forward_fx_model_table,
    }

    simulations = {
        "oro_sim": oro_sim,
        "oro_anio": oro_anio,
        "fx_sim": fx_sim,
        "fx_anio": fx_anio,
        "sofr_sim": sofr_sim,
        "sofr_prom": sofr_prom,
        "ingreso_neto_p1": ingreso_neto_p1,
        "prima_swap_sim": prima_swap_sim,
        "remanente_p2": remanente_p2,
        "remanente_cascada": remanente_cascada,
        "pasa_fase1": pasa_fase1,
        "pasa_fase2": pasa_fase2,
        "pasa_fase3": pasa_fase3,
        "pasa_todo": pasa_todo,
        "ingresos_sin_cob": ingresos_sin_cob,
        "ingresos_con_cob": ingresos_con_cob,
        "gastos_op_sin": gastos_op_sin,
        "gastos_op_con": gastos_op_con,
        "ebit_sin": ebit_sin,
        "ebit_con": ebit_con,
        "gafin_sin_sim": gafin_sin_sim,
        "gafin_con_sim": gafin_con_sim,
        "isr_sin": isr_sin,
        "isr_con": isr_con,
        "fcf_operativo_con": fcf_operativo_con,
        "fcf_sin_cob": fcf_sin_cob,
        "fcf_con_cob": fcf_con_cob,
        "diferencia_fcf": diferencia_fcf,
    }

    tables = {
        "income_statement": income_statement,
        "distribution_summary": distribution_summary,
        "scenario_summary": scenario_summary,
        "cascade_summary": cascade_summary,
        "payoff_put_oro": payoff_put_oro,
        "data_quality": _make_data_quality(raw),
        "missing_values": _make_missing_values_table(raw),
        "gold_forward_model": forward_oro_table,
        "usdmxn_forward_model": forward_fx_model_table,
    }

    return MonteCarloResults(
        project_root=root,
        raw_data=raw,
        processed_data=processed_data,
        simulations=simulations,
        metrics=metrics,
        tables=tables,
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convierte una tabla a CSV descargable."""
    return df.to_csv(index=False).encode("utf-8")
