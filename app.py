# 실행: streamlit run app.py

from __future__ import annotations

import datetime
import functools
import json
import math
import operator
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── 상수 ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
RS_MIDPOINT = 1.0
SCREENER_CONFIG_PATH = Path("screener_config.json")

_SCREENER_CFG_KEYS: list[str] = [
    "screener_pct_slider",
    "screener_3m_rs_pct_min",
    "screener_1m_rs_pct_min",
    "screener_rs_pct_filter_mode",
    "screener_slope_chk",
    "screener_phase_filter",
    "screener_etf_only",
    "screener_price_min",
    "screener_adv_min",
    "screener_mktcap_min",
    "screener_atr_pct_min",
    "screener_atr_multiple_max",
    "screener_ind_rank_max",
    "screener_sort_by",
]


def _load_screener_config() -> dict:
    try:
        with open(SCREENER_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_screener_config() -> None:
    cfg = {k: st.session_state.get(k) for k in _SCREENER_CFG_KEYS}
    with open(SCREENER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

SECTOR_NAME_MAP: dict[str, str] = {
    "SPY":  "S&P 500",
    "QQQ":  "NASDAQ 100",
    "XLK":  "Technology",
    "XLV":  "Health Care",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLE":  "Energy",
    "XLY":  "Consumer Disc.",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLC":  "Comm. Services",
}

SECTOR_TICKERS: list[str] = list(SECTOR_NAME_MAP.keys())  # SPY, QQQ + 11 SPDR ETF

BREADTH_SCORE_LABELS: dict[int, tuple[str, str]] = {
    5: ("매우 강세", "success"),
    4: ("강세",      "success"),
    3: ("중립",      "warning"),
    2: ("주의",      "warning"),
    1: ("약세",      "danger"),
    0: ("매우 약세", "danger"),
}

# Daily Summary 탭용 상수
MARKET_TICKERS: list[str] = ["SPY", "QQQ", "IWM"]

MARKET_TICKER_NAMES: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
}

# Watch List 탭 상수
WATCHLIST_PATH = OUTPUT_DIR / "watchlist.parquet"

# 종합 점수 가중치 (합계 = 100)
WATCHLIST_SCORE_WEIGHTS: dict[str, int] = {
    "rs_percentile":     20,   # RS 백분위
    "rs_line_phase":     20,   # RS Line Phase Soft Score
    "rs_line_newhigh":   20,   # RS Line 52W 신고가 여부
    "industry_strength": 20,   # 소속 산업(상위 절반 주도주) RS 강도
    "extension":         20,   # 50SMA ATR Multiple Extension 페널티
}

# 포트폴리오 탭 상수
RS_SIGNAL_THRESHOLDS: dict[str, dict] = {
    "strong":  {"min_percentile": 80, "slope_min":  0.01},
    "neutral": {"min_percentile": 50, "slope_min": -0.01},
}

PORTFOLIO_COLS_DISPLAY: list[str] = [
    "ticker", "qty", "avg_price_usd", "current_price_usd",
    "pnl_pct", "eval_amt_usd", "rs_rank", "rs_score", "rs_slope", "rs_signal",
]

BENCHMARK_TICKERS: set[str] = {"SPY", "QQQ", "IWM"}


# ─── 데이터 로딩 ───────────────────────────────────────────────────────────────
@st.cache_data
def load_sector_summary() -> pd.DataFrame:
    try:
        return pd.read_parquet(OUTPUT_DIR / "sector_summary.parquet")
    except Exception as e:
        st.error(f"sector_summary.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_theme_summary() -> pd.DataFrame:
    try:
        return pd.read_parquet(OUTPUT_DIR / "theme_summary.parquet")
    except Exception as e:
        st.error(f"theme_summary.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_screener() -> pd.DataFrame:
    try:
        return pd.read_parquet(OUTPUT_DIR / "stock_screener.parquet")
    except Exception as e:
        st.error(f"stock_screener.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_rs_full() -> pd.DataFrame:
    try:
        df = pd.read_parquet(OUTPUT_DIR / "rs_full.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"rs_full.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_breadth_metrics() -> pd.DataFrame:
    try:
        df = pd.read_parquet(OUTPUT_DIR / "breadth_metrics.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"breadth_metrics.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_portfolio_snapshot() -> pd.DataFrame:
    path = OUTPUT_DIR / "portfolio_snapshot.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"portfolio_snapshot.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_trade_history() -> pd.DataFrame:
    path = OUTPUT_DIR / "trade_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"trade_history.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_technical_indicators() -> pd.DataFrame:
    try:
        df = pd.read_parquet(OUTPUT_DIR / "technical_indicators.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"technical_indicators.parquet 로드 실패: {e}")
        st.stop()



# ─── 유틸 함수 ────────────────────────────────────────────────────────────────
def _slope_color(val: object) -> str:
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "background-color: #2ecc71; color: #000000"
    return "background-color: #e74c3c; color: #ffffff"


def _make_treemap(
    df: pd.DataFrame,
    label_col: str,
    ticker_col: str,
    title: str,
) -> go.Figure:
    """섹터/테마 공용 Treemap 생성. rs_score로 색상 인코딩."""
    plot_df = df.copy()
    plot_df["_size"] = 1
    plot_df["rs_score"] = pd.to_numeric(plot_df.get("rs_score"), errors="coerce")
    # NaN 방지: 셀 텍스트용 문자열 컬럼을 별도 생성
    plot_df["_score_text"] = plot_df["rs_score"].apply(
        lambda v: f"{v:.3f}" if pd.notna(v) else ""
    )

    custom_cols = ["_score_text", ticker_col] + [
        c for c in ["rs_score", "rs_slope"] if c in plot_df.columns and c != ticker_col
    ]

    # rs_score는 SPY 상대값(1.0 = 시장 수익률)이라 1.0을 중심으로 발산형 컬러 적용.
    # 단일 아웃라이어가 스케일을 압축하지 않도록 95% 분위 편차로 반폭(half-width) 산정.
    score_vals = plot_df["rs_score"].dropna()
    if not score_vals.empty:
        half = max(float((score_vals - 1.0).abs().quantile(0.95)), 0.05)
    else:
        half = 0.3
    range_color = [1.0 - half, 1.0 + half]

    fig = px.treemap(
        plot_df,
        path=[px.Constant(title), label_col],
        values="_size",
        color="rs_score",
        color_continuous_scale="RdYlGn",
        range_color=range_color,
        color_continuous_midpoint=1.0,
        custom_data=custom_cols,
        title=title,
    )

    idx = {col: i for i, col in enumerate(custom_cols)}
    hover_lines = [
        "<b>%{label}</b>",
        f"Ticker: %{{customdata[{idx[ticker_col]}]}}",
    ]
    if "rs_score" in idx:
        hover_lines.append(f"RS Score: %{{customdata[{idx['rs_score']}]:.3f}}")
    if "rs_slope" in idx:
        hover_lines.append(f"Slope: %{{customdata[{idx['rs_slope']}]:.4f}}")
    hover_lines.append("<extra></extra>")

    fig.update_traces(
        hovertemplate="<br>".join(hover_lines),
        texttemplate=f"%{{label}}<br>%{{customdata[{idx['_score_text']}]}}",
        textinfo="label",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="RS Score", thickness=12),
        margin=dict(t=40, l=0, r=0, b=0),
        height=300,
    )
    return fig


def _make_trend_chart(
    rs_full: pd.DataFrame,
    tickers: list[str],
    title: str,
    months: int | None = None,
    normalize: bool = False,
) -> go.Figure | None:
    """공용 RS Line 추이 차트.

    months: 최근 N개월만 표시 (None이면 전체)
    normalize: True면 각 티커를 첫날 값 기준 1.0으로 정규화
    """
    filtered = rs_full[rs_full["ticker"].isin(tickers)].copy()
    if filtered.empty or "rs_line" not in filtered.columns:
        return None

    if months is not None:
        cutoff = filtered["date"].max() - pd.DateOffset(months=months)
        filtered = filtered[filtered["date"] >= cutoff]

    if normalize:
        filtered = filtered.sort_values(["ticker", "date"])
        first_vals = filtered.groupby("ticker")["rs_line"].transform("first")
        filtered["rs_line"] = filtered["rs_line"] / first_vals

    y_label = "RS Line (정규화, 시작=1.0)" if normalize else "RS Line (vs SPY)"

    fig = px.line(
        filtered,
        x="date",
        y="rs_line",
        color="ticker",
        title=title,
        labels={"date": "날짜", "rs_line": y_label},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.add_hline(
        y=RS_MIDPOINT,
        line_dash="dot",
        line_color="gray",
        annotation_text="기준선 1.0",
        annotation_position="right",
    )
    fig.update_layout(
        title=dict(y=0.97, yanchor="top"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=40, l=0, r=0, b=120),
        height=550,
    )
    return fig


def _render_ranking_table(df: pd.DataFrame, height: int = 500) -> None:
    """rs_slope 색상 강조 + column_config가 적용된 순위표."""
    candidate_cols = [
        "rs_rank", "ticker", "sector_name", "theme", "rs_score", "rs_slope",
        "price_chg_pct", "price_chg_5d_pct",
        "price_chg_1m_pct", "price_chg_3m_pct", "price_chg_6m_pct", "price_chg_1y_pct",
        "rs_1m", "rs_3m", "rs_line_phase",
    ]
    display_cols = [c for c in candidate_cols if c in df.columns]
    display_df = df[display_cols].sort_values("rs_rank").reset_index(drop=True)

    col_config: dict = {
        "rs_rank":       st.column_config.NumberColumn("순위",       format="%d"),
        "rs_score":      st.column_config.NumberColumn("RS Score",   format="%.3f"),
        "rs_slope":      st.column_config.NumberColumn("RS Slope",   format="%.4f"),
        "price_chg_pct":    st.column_config.NumberColumn("1D 등락률", format="%.2f%%"),
        "price_chg_5d_pct": st.column_config.NumberColumn("1W 등락률", format="%.2f%%"),
        "price_chg_1m_pct": st.column_config.NumberColumn("1M 등락률", format="%.2f%%"),
        "price_chg_3m_pct": st.column_config.NumberColumn("3M 등락률", format="%.2f%%"),
        "price_chg_6m_pct": st.column_config.NumberColumn("6M 등락률", format="%.2f%%"),
        "price_chg_1y_pct": st.column_config.NumberColumn("1Y 등락률", format="%.2f%%"),
        "rs_1m":            st.column_config.NumberColumn("1개월 RS",  format="%.3f"),
        "rs_3m":         st.column_config.NumberColumn("3개월 RS",   format="%.3f"),
        "rs_line_phase": st.column_config.TextColumn("RS Phase"),
    }

    if "rs_slope" in display_df.columns:
        styled = display_df.style.map(_slope_color, subset=["rs_slope"])
        st.dataframe(styled, use_container_width=True, hide_index=True, column_config=col_config, height=height)
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=col_config, height=height)


# ─── Daily Summary 헬퍼 함수 ──────────────────────────────────────────────────
def _calc_daily_return(rs_full: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """실제 주가 등락률(price_chg_pct) 반환. 컬럼이 없으면 rs_score 변화율로 대체."""
    df = rs_full[rs_full["ticker"].isin(tickers)].copy()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "daily_pct"])

    latest_date = df["date"].max()
    today       = df[df["date"] == latest_date]

    # price_chg_pct가 있으면 직접 사용 (실제 주가 등락률)
    if "price_chg_pct" in df.columns:
        result = today[["ticker", "price_chg_pct"]].copy()
        result = result.rename(columns={"price_chg_pct": "daily_pct"})
        return result.reset_index(drop=True)

    # fallback: rs_score 전일 대비 변화율 (pipeline 재실행 전 임시)
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        result = today[["ticker"]].copy()
        result["daily_pct"] = float("nan")
        return result[["ticker", "daily_pct"]]

    today_s = df[df["date"] == dates[-1]].set_index("ticker")["rs_score"]
    prev_s  = df[df["date"] == dates[-2]].set_index("ticker")["rs_score"]
    merged = pd.DataFrame({"today": today_s, "prev": prev_s})
    merged["daily_pct"] = (merged["today"] / merged["prev"] - 1) * 100
    return merged.reset_index()[["ticker", "daily_pct"]]


def _calc_rank_change(rs_full: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """전일 대비 rs_rank 변동. rank_change = rank_prev - rank_today (양수 = 순위 상승)."""
    df = rs_full.copy()
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    if df.empty:
        return pd.DataFrame(columns=["ticker", "rank_today", "rank_prev", "rank_change"])

    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["ticker", "rank_today", "rank_prev", "rank_change"])

    today_s = df[df["date"] == dates[-1]].set_index("ticker")["rs_rank"]
    prev_s  = df[df["date"] == dates[-2]].set_index("ticker")["rs_rank"]

    result = pd.DataFrame({"rank_today": today_s, "rank_prev": prev_s}).dropna().astype(int)
    result["rank_change"] = result["rank_prev"] - result["rank_today"]
    result["_abs"] = result["rank_change"].abs()
    result = result.sort_values("_abs", ascending=False).drop(columns=["_abs"])
    return result.reset_index()[["ticker", "rank_today", "rank_prev", "rank_change"]]


def _get_market_character(
    daily_ret: pd.DataFrame,
    breadth_latest: pd.Series,
) -> tuple[str, str]:
    """규칙 기반 시장 성격 자동 판단 → (텍스트, streamlit color)."""
    def _pct(ticker: str) -> float:
        row = daily_ret[daily_ret["ticker"] == ticker]["daily_pct"]
        if row.empty or not pd.notna(row.values[0]):
            return 0.0
        return float(row.values[0])

    spy = _pct("SPY")
    qqq = _pct("QQQ")
    iwm = _pct("IWM")

    bs     = int(breadth_latest.get("breadth_score", 0)) if not breadth_latest.empty else 0
    prefix = "Breadth 건강 | " if bs >= 4 else ""

    if spy < -1.5:
        text, color = "⚠️ 시장 약세 — 현금 비중 점검 필요", "danger"
    elif spy > 0 and qqq > spy * 1.5:
        text, color = "🚀 대형 성장주 주도 강세 — 나스닥 아웃퍼폼", "success"
    elif spy > 0 and iwm > spy:
        text, color = "📈 광범위 강세 — 소형주 동반 상승, Breadth 건강", "success"
    elif spy > 0 and iwm < 0:
        text, color = "⚡ 대형주만 강세 — 소형주 소외, 폭 좁은 상승", "warning"
    elif abs(spy) < 0.3:
        text, color = "😐 방향성 없는 혼조 — 관망 유효", "secondary"
    else:
        text, color = "📊 혼조 — 섹터별 차별화 확인 필요", "warning"

    return f"{prefix}{text}", color


def _get_rs_line_new_highs(rs_full: pd.DataFrame, lookback: int = 252) -> list[str]:
    """오늘 RS Line이 최근 lookback일 신고가인 종목 리스트 (SPY 제외, rs_score 내림차순)."""
    df = rs_full[rs_full["ticker"] != "SPY"].copy()
    if df.empty or "rs_line" not in df.columns:
        return []

    df["date"]  = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    all_dates   = sorted(df["date"].unique())
    cutoff_date = all_dates[-lookback] if len(all_dates) >= lookback else all_dates[0]

    df_window = df[df["date"] >= cutoff_date]
    max_rs    = df_window.groupby("ticker")["rs_line"].max()
    today     = df_window[df_window["date"] == latest_date].set_index("ticker")

    new_highs = [
        t for t in today.index
        if t in max_rs.index
        and pd.notna(today.loc[t, "rs_line"])
        and pd.notna(max_rs[t])
        and today.loc[t, "rs_line"] >= max_rs[t] * 0.999
    ]

    if "rs_score" in today.columns:
        new_highs.sort(
            key=lambda t: float(today.loc[t, "rs_score"]) if t in today.index else 0.0,
            reverse=True,
        )
    return new_highs


def _get_sector_rotation_signals(
    rs_full: pd.DataFrame,
    sector_tickers: list[str],
    days: int = 3,
) -> list[dict]:
    """최근 days일 연속 RS 순위 상승/하락 섹터 감지."""
    df = rs_full[rs_full["ticker"].isin(sector_tickers)].copy()
    if df.empty:
        return []

    df["date"]    = pd.to_datetime(df["date"])
    all_dates     = sorted(df["date"].unique())
    if len(all_dates) < days + 1:
        return []

    recent_dates = all_dates[-(days + 1):]
    pivot = (
        df[df["date"].isin(recent_dates)]
        .pivot(index="date", columns="ticker", values="rs_rank")
        .sort_index()
    )

    signals: list[dict] = []
    for ticker in sector_tickers:
        if ticker not in pivot.columns:
            continue
        ranks = pivot[ticker].dropna().tolist()
        if len(ranks) < days + 1:
            continue
        diffs = [ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1)]
        if all(d < 0 for d in diffs):
            signals.append({"ticker": ticker, "direction": "up",   "days": days, "rank_today": int(ranks[-1])})
        elif all(d > 0 for d in diffs):
            signals.append({"ticker": ticker, "direction": "down", "days": days, "rank_today": int(ranks[-1])})
    return signals


# ─── 포트폴리오 탭 헬퍼 함수 ──────────────────────────────────────────────────

def _join_rs_to_portfolio(portfolio_df: pd.DataFrame, rs_full: pd.DataFrame) -> pd.DataFrame:
    """
    보유 종목에 rs_full 최신일 RS 데이터 join 후 rs_signal 컬럼 생성.

    Args:
        portfolio_df: 보유 종목 DataFrame (ticker 컬럼 포함)
        rs_full:      전체 RS 데이터

    Returns:
        rs_score, rs_rank, rs_slope, rs_percentile, rs_signal이 추가된 DataFrame
    """
    df = portfolio_df.copy()
    if rs_full.empty or "ticker" not in rs_full.columns:
        for col in ["rs_score", "rs_rank", "rs_slope", "rs_percentile"]:
            df[col] = float("nan")
        df["rs_signal"] = "⚪ 데이터없음"
        return df

    latest = pd.to_datetime(rs_full["date"]).max()
    rs_today = (
        rs_full[
            (pd.to_datetime(rs_full["date"]) == latest)
            & (~rs_full["ticker"].isin(BENCHMARK_TICKERS))
        ][["ticker", "rs_score", "rs_rank", "rs_slope", "rs_percentile"]]
        .drop_duplicates("ticker")
    )

    df = df.merge(rs_today, on="ticker", how="left")

    thresholds = RS_SIGNAL_THRESHOLDS

    def _signal(row: pd.Series) -> str:
        pct   = row.get("rs_percentile", float("nan"))
        slope = row.get("rs_slope",      float("nan"))
        if pd.isna(pct) or pd.isna(slope):
            return "⚪ 데이터없음"
        if pct >= thresholds["strong"]["min_percentile"] and slope >= thresholds["strong"]["slope_min"]:
            return "🟢 RS강세"
        if pct >= thresholds["neutral"]["min_percentile"] and slope >= thresholds["neutral"]["slope_min"]:
            return "🟡 RS보통"
        return "🔴 RS약세"

    df["rs_signal"] = df.apply(_signal, axis=1)
    return df


def _get_spy_cumulative_return(
    rs_full: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series | None:
    """
    SPY의 price_chg_pct를 이용해 기간 내 누적수익률(%) 계산.

    Args:
        rs_full:    전체 RS 데이터 (price_chg_pct 컬럼 필요)
        start_date: 조회 시작일
        end_date:   조회 종료일

    Returns:
        date indexed pd.Series (값: 시작일 대비 누적수익률 %, 첫 날 기준 0% 정규화)
        데이터 없으면 None
    """
    if "price_chg_pct" not in rs_full.columns:
        return None

    spy_df = rs_full[rs_full["ticker"] == "SPY"].copy()
    if spy_df.empty:
        return None

    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_df = spy_df[
        (spy_df["date"] >= pd.Timestamp(start_date))
        & (spy_df["date"] <= pd.Timestamp(end_date))
    ].sort_values("date").dropna(subset=["price_chg_pct"])

    if spy_df.empty:
        return None

    spy_df = spy_df.set_index("date")
    cum_val = (1 + spy_df["price_chg_pct"] / 100).cumprod()
    # 시작점을 0%로 정규화
    cum_ret = (cum_val / cum_val.iloc[0] - 1) * 100
    return cum_ret


def _calc_realized_pnl_by_ticker(trade_history: pd.DataFrame) -> pd.DataFrame:
    """
    종목별 누적 실현손익 집계.

    Args:
        trade_history: 체결 내역 DataFrame (side, ticker, realized_pnl_usd 컬럼 필요)

    Returns:
        DataFrame with columns: ticker, total_realized_pnl_usd
    """
    if trade_history.empty or "side" not in trade_history.columns:
        return pd.DataFrame(columns=["ticker", "total_realized_pnl_usd"])

    sells = trade_history[trade_history["side"] == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame(columns=["ticker", "total_realized_pnl_usd"])

    result = (
        sells.groupby("ticker")["realized_pnl_usd"]
        .sum()
        .reset_index()
        .rename(columns={"realized_pnl_usd": "total_realized_pnl_usd"})
    )
    return result


def _get_rs_warnings(
    portfolio_with_rs: pd.DataFrame,
    rs_full: pd.DataFrame,
) -> list[dict]:
    """
    보유 종목 중 RS 경고/기회 신호 감지.

    Args:
        portfolio_with_rs: _join_rs_to_portfolio() 결과 DataFrame
        rs_full:           전체 RS 이력 (5일 비교용)

    Returns:
        list of {"type": "danger"|"warning"|"success", "ticker": str, "msg": str}
    """
    warnings: list[dict] = []
    if portfolio_with_rs.empty:
        return warnings

    tickers = portfolio_with_rs["ticker"].tolist()
    df_hist = rs_full[rs_full["ticker"].isin(tickers)].copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    dates = sorted(df_hist["date"].unique())

    # 5 거래일 전 날짜
    prev_date = dates[-6] if len(dates) >= 6 else (dates[0] if dates else None)
    latest_date = dates[-1] if dates else None

    rank_today_map: dict[str, int] = {}
    rank_prev_map:  dict[str, int] = {}
    if latest_date is not None:
        today_rows = df_hist[df_hist["date"] == latest_date].set_index("ticker")
        rank_today_map = today_rows["rs_rank"].dropna().astype(int).to_dict()
    if prev_date is not None:
        prev_rows = df_hist[df_hist["date"] == prev_date].set_index("ticker")
        rank_prev_map = prev_rows["rs_rank"].dropna().astype(int).to_dict()

    for _, row in portfolio_with_rs.iterrows():
        ticker = row["ticker"]
        rs_rank    = row.get("rs_rank",    float("nan"))
        rs_slope   = row.get("rs_slope",   float("nan"))
        rs_pct     = row.get("rs_percentile", float("nan"))
        pnl_pct    = row.get("pnl_pct",    0.0)

        # 조건 1: RS 순위 급락 (전주 대비 20위 이상 하락)
        if ticker in rank_today_map and ticker in rank_prev_map:
            rank_drop = rank_today_map[ticker] - rank_prev_map[ticker]
            if rank_drop >= 20:
                warnings.append({
                    "type":   "danger",
                    "ticker": ticker,
                    "msg":    f"RS 순위 {rank_today_map[ticker]}위, 전주 대비 -{rank_drop} 급락",
                })

        # 조건 2: RS Slope 음전환 + 수익 중
        if not pd.isna(rs_slope) and rs_slope < 0 and not pd.isna(pnl_pct) and pnl_pct > 0:
            warnings.append({
                "type":   "warning",
                "ticker": ticker,
                "msg":    "수익 중이나 RS Slope 음전환, 모멘텀 약화",
            })

        # 조건 3: RS 상위권 + Slope 가속
        if not pd.isna(rs_pct) and rs_pct >= 95 and not pd.isna(rs_slope) and rs_slope > 0.02:
            warnings.append({
                "type":   "success",
                "ticker": ticker,
                "msg":    "RS 상위권 + Slope 가속, 홀딩 or 추가 매수 검토",
            })

    return warnings


# ─── Watch List 탭 헬퍼 함수 ──────────────────────────────────────────────────

def _load_watchlist() -> list[str]:
    """
    watchlist.parquet에서 오늘 날짜 Watch List 로드.
    파일이 없거나 오늘 데이터가 없으면 빈 리스트 반환 (에러 없음).

    Returns:
        list[str]: 대문자 정규화된 티커 리스트
    """
    try:
        df = pd.read_parquet(WATCHLIST_PATH)
    except (FileNotFoundError, OSError):
        return []
    except Exception:
        return []

    if df.empty or "date" not in df.columns or "ticker" not in df.columns:
        return []

    df["date"] = pd.to_datetime(df["date"])
    today = pd.Timestamp(datetime.date.today())
    today_df = df[df["date"].dt.normalize() == today.normalize()]
    return [str(t).upper() for t in today_df["ticker"].tolist()]


def _save_watchlist(tickers: list[str]) -> None:
    """
    Watch List를 watchlist.parquet에 날짜별로 저장 (오늘 데이터는 덮어쓰기).
    빈 리스트도 저장 가능 (오늘 Watch List 초기화 용도).

    Args:
        tickers: 저장할 티커 리스트
    """
    today = pd.Timestamp(datetime.date.today()).normalize()
    new_df = pd.DataFrame({
        "date":   [today] * len(tickers),
        "ticker": [str(t).upper() for t in tickers],
    })

    if WATCHLIST_PATH.exists():
        try:
            existing = pd.read_parquet(WATCHLIST_PATH)
            existing["date"] = pd.to_datetime(existing["date"])
            existing = existing[existing["date"].dt.normalize() != today]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(WATCHLIST_PATH, index=False)


_PHASE_SCORE_MAP: dict[str, float] = {
    "강세":      20.0,
    "약세":       0.0,
    "데이터 없음": 5.0,
}


def _calc_priority_score_phase(rs_line_phase) -> float:
    """RS Line Phase 기반 점수 (이진, 0~20점).

    강세 → 20점 / 약세 → 0점 / NaN 또는 미분류 → 5점 (중립)
    """
    if pd.isna(rs_line_phase) or rs_line_phase == "":
        return 5.0
    return _PHASE_SCORE_MAP.get(str(rs_line_phase), 5.0)



def _calc_priority_score_sector(sector_rs_rank) -> float:
    """소속 섹터 ETF RS 순위 기반 점수 (15점 만점)."""
    if pd.isna(sector_rs_rank):
        return 0.0
    r = int(sector_rs_rank)
    if r <= 3:
        return 15.0
    if r <= 6:
        return 12.0
    if r <= 9:
        return 8.0
    return 3.0


def _calc_priority_score_theme(theme_rs_rank, total_themes: int) -> float:
    """소속 테마 ETF RS 순위 백분위 기반 점수 (15점 만점)."""
    if pd.isna(theme_rs_rank) or total_themes == 0:
        return 0.0
    pct = theme_rs_rank / total_themes
    if pct <= 0.20:
        return 15.0
    if pct <= 0.40:
        return 12.0
    if pct <= 0.60:
        return 8.0
    if pct <= 0.80:
        return 4.0
    return 0.0


EXTENSION_MAX_MULTIPLE = 10.0  # 이 값 이상 extended면 0점 (매수 자제)


def _calc_priority_score_extension(atr_multiple: float) -> float:
    """
    50SMA로부터의 ATR Multiple 기반 Extension 페널티/점수 (0~20점).

    atr_multiple = (close - 50SMA) / 50SMA * 100 / ATR_pct
    값이 클수록 50SMA로부터 많이 떨어진(extended) 상태.

    점수 기준 (soft score):
      atr_multiple < 0                       → 0점  (50SMA 아래, 약세)
      0 <= atr_multiple <= 10.0              → 20 * (1 - atr_multiple/10)
                                                (0배=20점 → 10배=0점 선형 감소)
      atr_multiple > 10.0                    → 0점  (과도 extended, 매수 자제)
      NaN                                    → 5점  (데이터 없음, 중립)

    Args:
        atr_multiple (float): 50SMA로부터의 ATR Multiple

    Returns:
        float: 0.0 ~ 20.0
    """
    if pd.isna(atr_multiple):
        return 5.0
    if atr_multiple < 0 or atr_multiple > EXTENSION_MAX_MULTIPLE:
        return 0.0
    # 0 ~ 10배 구간: 선형 soft score (0배=20점, 10배=0점)
    return 20.0 * (1.0 - atr_multiple / EXTENSION_MAX_MULTIPLE)


def _calc_priority_score_industry(industry_strength: float) -> float:
    """
    소속 산업 RS 강도 점수 (0~20점).

    industry_strength = 해당 산업의 상위 절반 주도주 평균 RS 백분위 (0~100).
    백분위가 높을수록(주도 산업) 높은 점수.

    Args:
        industry_strength (float): 산업 상위 절반 주도주 평균 RS 백분위

    Returns:
        float: 0.0 ~ 20.0 (NaN → 0)
    """
    if pd.isna(industry_strength):
        return 0.0
    return min(float(industry_strength), 100.0) / 100.0 * 20.0


def _calc_priority_score_rs_line(
    rs_line_today: float,
    rs_line_52w_high: float,
) -> float:
    """
    RS Line 신고가 근접도 기반 점수 계산 (0~20점).

    이진(신고가 여부) 대신 오늘값/52주최고값 비율로 연속 점수화.
    신고가 초과 시 초과율에 비례한 보너스 점수 추가.

    점수 구조:
    - 기본 근접도 점수 (0~15점):
        ratio >= 1.0  → 15점  (신고가 달성)
        ratio >= 0.98 → 13점  (신고가 2% 이내)
        ratio >= 0.95 → 10점  (신고가 5% 이내)
        ratio >= 0.90 →  7점  (신고가 10% 이내)
        ratio >= 0.80 →  4점  (신고가 20% 이내)
        ratio <  0.80 →  0점
    - 신고가 보너스 (0~5점, ratio > 1.0일 때만):
        bonus = min((ratio - 1.0) * 100, 5.0)  (초과율 1%당 1점, 최대 5점)
    - 최종: min(base + bonus, 20.0)

    Args:
        rs_line_today    (float): 오늘 rs_line 값
        rs_line_52w_high (float): 최근 252일 rs_line 최고값

    Returns:
        float: 0.0 ~ 20.0
    """
    if pd.isna(rs_line_today) or pd.isna(rs_line_52w_high):
        return 0.0
    if rs_line_52w_high == 0:
        return 0.0

    ratio = rs_line_today / rs_line_52w_high

    # 기본 근접도 점수
    if ratio >= 1.0:
        base = 15.0
    elif ratio >= 0.98:
        base = 13.0
    elif ratio >= 0.95:
        base = 10.0
    elif ratio >= 0.90:
        base = 7.0
    elif ratio >= 0.80:
        base = 4.0
    else:
        base = 0.0

    # 신고가 초과 보너스
    bonus = min((ratio - 1.0) * 100, 5.0) if ratio > 1.0 else 0.0

    return min(base + bonus, 20.0)


def _calc_priority_score(row: pd.Series) -> float:
    """
    종목 1개의 종합 매수 우선순위 점수 계산 (0~100점).

    점수 구성 (WATCHLIST_SCORE_WEIGHTS 참조):
      ① rs_percentile    (20점): min(pct, 100) / 100 * 20
      ② rs_line_phase    (20점): _calc_priority_score_phase()
      ③ rs_line 근접도   (20점): _calc_priority_score_rs_line()
      ④ industry_strength(20점): _calc_priority_score_industry()
      ⑤ extension        (20점): _calc_priority_score_extension()
    NaN은 모두 0점 처리 (extension·phase는 NaN 시 5점 중립).

    Args:
        row: rs_percentile, rs_line_phase, rs_line_today, rs_line_52w_high,
             industry_rs_percentile, atr_multiple 필드를 포함하는 Series

    Returns:
        float (0.0 ~ 100.0)
    """
    # ① RS Percentile (20점)
    pct = row.get("rs_percentile", float("nan"))
    score_rs_pct = min(float(pct), 100) / 100 * 20 if not pd.isna(pct) else 0.0

    # ② RS Line Phase (20점) — 소프트 점수
    score_rs_phase = _calc_priority_score_phase(row.get("rs_line_phase", float("nan")))

    # ③ RS Line 근접도 (20점) — 연속 점수
    score_rs_line = _calc_priority_score_rs_line(
        row.get("rs_line_today", float("nan")),
        row.get("rs_line_52w_high", float("nan")),
    )

    # ④ 산업 강도 (20점) — 소속 산업 상위 절반 주도주 평균 RS 백분위
    score_industry = _calc_priority_score_industry(row.get("industry_rs_percentile", float("nan")))

    # ⑤ Extension 페널티 (20점)
    score_extension = _calc_priority_score_extension(row.get("atr_multiple", float("nan")))

    total = (
        score_rs_pct
        + score_rs_phase
        + score_rs_line
        + score_industry
        + score_extension
    )
    return float(round(min(total, 100.0), 1))


def _build_watchlist_data(
    tickers: list[str],
    rs_full: pd.DataFrame,
    sector_summary: pd.DataFrame,
    theme_summary: pd.DataFrame,
    screener: pd.DataFrame,
    tech_indicators: pd.DataFrame,
) -> pd.DataFrame:
    """
    Watch List 종목들의 RS 데이터 + 섹터/테마 강도 + 기술적 지표를 종합한 분석 DataFrame 생성.

    rs_full에 없는 ticker는 NaN으로 처리하고 st.warning으로 안내.

    Returns:
        ticker, rs_rank, rs_score, rs_percentile, rs_slope, rs_line_newhigh,
        rs_line_today, rs_line_52w_high, parent_etf, sector_etf, sector_rs_rank,
        sector_rs_score, theme_rs_rank, theme_rs_score, sma_50, dist_pct_50sma,
        atr_multiple, priority_score, priority_rank 컬럼 포함 DataFrame
    """
    tickers = [str(t).upper() for t in tickers]
    if not tickers:
        return pd.DataFrame()

    # 최신 날짜 RS 데이터
    latest_date = pd.to_datetime(rs_full["date"]).max()
    rs_today = (
        rs_full[pd.to_datetime(rs_full["date"]) == latest_date]
        .drop_duplicates("ticker")
        .set_index("ticker")
    )

    # rs_full에 없는 종목 경고
    missing = [t for t in tickers if t not in rs_today.index]
    if missing:
        st.warning(f"rs_full에 없는 종목: {', '.join(missing)}")

    # RS Line 52W 신고가 종목 집합 (한 번만 계산)
    newhigh_set = set(_get_rs_line_new_highs(rs_full))

    # RS Line 근접도 점수용: 최근 252일 rs_line 최고값 + 오늘 rs_line 값
    rs_52w_high_map: dict = {}
    today_rs_line_map: dict = {}
    if "rs_line" in rs_full.columns:
        all_dates = sorted(rs_full["date"].unique())
        lookback_date = all_dates[-252] if len(all_dates) >= 252 else all_dates[0]
        rs_window = rs_full[rs_full["date"] >= lookback_date]
        rs_52w_high_map = rs_window.groupby("ticker")["rs_line"].max().to_dict()
        today_rs_line_map = (
            rs_full[rs_full["date"] == latest_date]
            .drop_duplicates("ticker")
            .set_index("ticker")["rs_line"]
            .to_dict()
        )

    # 섹터/테마 lookup 인덱스.
    # 주의: sector/theme_summary의 rs_rank는 전체 유니버스 글로벌 순위라
    # 점수 임계값(섹터 1~13위, 테마 백분위)과 맞지 않는다. 따라서 ETF 집합 내
    # rs_score 기준 상대 순위(_rel_rank)를 새로 계산해 사용한다.
    sector_idx = sector_summary.drop_duplicates("ticker").copy()
    sector_idx["_rel_rank"] = sector_idx["rs_score"].rank(ascending=False, method="min")
    sector_idx = sector_idx.set_index("ticker")

    theme_idx = theme_summary.drop_duplicates("ticker").copy()
    theme_idx["_rel_rank"] = theme_idx["rs_score"].rank(ascending=False, method="min")
    theme_idx = theme_idx.set_index("ticker")
    total_themes = int(theme_summary["ticker"].nunique())

    def _rank_key(idx: pd.DataFrame, t: str) -> float:
        """_rel_rank NaN-safe 정렬 키 (NaN은 후순위로)."""
        v = idx.loc[t, "_rel_rank"]
        return float(v) if pd.notna(v) else float("inf")

    # 종목별 parent_etf / industry 매핑
    screener_idx = (
        screener.drop_duplicates("ticker").set_index("ticker")
        if "ticker" in screener.columns else pd.DataFrame()
    )

    # 산업 강도 lookup: 각 산업 상위 절반 주도주 평균 RS 백분위 (전체 산업 포함)
    industry_summary = _build_industry_summary(screener, min_count=1, exclude_junk=False)
    industry_strength_map: dict = {}
    industry_rank_map: dict = {}
    if not industry_summary.empty:
        industry_strength_map = industry_summary.set_index("industry")["rs_percentile"].to_dict()
        industry_rank_map = industry_summary.set_index("industry")["rs_rank"].to_dict()

    def _split_etfs(raw: object) -> list[str]:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        return [tok.strip().upper() for tok in str(raw).replace(",", " ").split() if tok.strip()]

    rows: list[dict] = []
    for ticker in tickers:
        rec: dict = {"ticker": ticker}

        # RS 기본 데이터
        if ticker in rs_today.index:
            r = rs_today.loc[ticker]
            rec["rs_rank"]       = r.get("rs_rank", float("nan"))
            rec["rs_score"]      = r.get("rs_score", float("nan"))
            rec["rs_percentile"] = r.get("rs_percentile", float("nan"))
            rec["rs_slope"]      = r.get("rs_slope", float("nan"))
            rec["rs_line_phase"] = r.get("rs_line_phase", None)
            rec["rs_1m"]         = r.get("rs_1m", float("nan"))
            rec["rs_3m"]         = r.get("rs_3m", float("nan"))
        else:
            rec["rs_rank"] = rec["rs_score"] = rec["rs_percentile"] = rec["rs_slope"] = float("nan")
            rec["rs_line_phase"] = None
            rec["rs_1m"] = rec["rs_3m"] = float("nan")

        rec["rs_line_newhigh"]   = ticker in newhigh_set
        rec["rs_line_today"]     = today_rs_line_map.get(ticker, float("nan"))
        rec["rs_line_52w_high"]  = rs_52w_high_map.get(ticker, float("nan"))

        # ── 소속 산업 강도 (score용) ──────────────────────────────────────
        stock_industry = ""
        if not screener_idx.empty and ticker in screener_idx.index and "industry" in screener_idx.columns:
            _ind = screener_idx.loc[ticker, "industry"]
            stock_industry = str(_ind).strip() if pd.notna(_ind) else ""
        rec["industry"]               = stock_industry or "N/A"
        rec["industry_rs_percentile"] = industry_strength_map.get(stock_industry, float("nan"))
        rec["industry_rank"]          = industry_rank_map.get(stock_industry, float("nan"))

        # parent_etf 토큰 분리
        raw_parent = screener_idx.loc[ticker, "parent_etf"] if (not screener_idx.empty and ticker in screener_idx.index) else ""
        etf_tokens = _split_etfs(raw_parent)

        # ── 섹터 ETF: SECTOR_TICKERS 중 상대 RS 순위가 가장 좋은(작은) 토큰 ──
        sector_candidates = [t for t in etf_tokens if t in SECTOR_TICKERS and t in sector_idx.index]
        if sector_candidates:
            best_sector = min(sector_candidates, key=lambda t: _rank_key(sector_idx, t))
            rec["sector_etf"]      = best_sector
            rec["sector_rs_rank"]  = sector_idx.loc[best_sector, "_rel_rank"]
            rec["sector_rs_score"] = sector_idx.loc[best_sector, "rs_score"]
        else:
            rec["sector_etf"] = "N/A"
            rec["sector_rs_rank"] = rec["sector_rs_score"] = float("nan")

        # ── 테마 ETF: 섹터 ETF가 아닌 토큰 중 상대 RS 순위 최고, 없으면 임의 테마 토큰 ──
        theme_pref = [t for t in etf_tokens if t in theme_idx.index and t not in SECTOR_TICKERS]
        theme_pool = theme_pref if theme_pref else [t for t in etf_tokens if t in theme_idx.index]
        if theme_pool:
            best_theme = min(theme_pool, key=lambda t: _rank_key(theme_idx, t))
            rec["parent_etf"]     = best_theme
            rec["theme_rs_rank"]  = theme_idx.loc[best_theme, "_rel_rank"]
            rec["theme_rs_score"] = theme_idx.loc[best_theme, "rs_score"]
        else:
            rec["parent_etf"] = "N/A"
            rec["theme_rs_rank"] = rec["theme_rs_score"] = float("nan")

        rec["_total_themes"] = total_themes
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 기술적 지표 join (최신 날짜 기준) — sma_50, dist_pct_50sma, atr_multiple
    if tech_indicators is not None and not tech_indicators.empty:
        tech_latest = tech_indicators["date"].max()
        tech_today = tech_indicators[tech_indicators["date"] == tech_latest][
            ["ticker", "sma_50", "dist_pct_50sma", "atr_multiple"]
        ]
        df = df.merge(tech_today, on="ticker", how="left")
    else:
        df["sma_50"] = df["dist_pct_50sma"] = df["atr_multiple"] = float("nan")

    # priority_score는 atr_multiple 등 기술적 지표가 join된 뒤 계산
    df["priority_score"] = df.apply(_calc_priority_score, axis=1)
    df["priority_rank"]  = df["priority_score"].rank(ascending=False, method="min").astype(int)
    return df


# ─── 사이드바 ──────────────────────────────────────────────────────────────────
def render_sidebar(rs_full: pd.DataFrame) -> None:
    with st.sidebar:
        st.title("⚙️ 설정")
        st.caption("IBD RS Score 기준 분석")

        st.divider()
        latest_date = pd.to_datetime(rs_full["date"]).max()
        st.caption(f"기준일: {latest_date.strftime('%Y-%m-%d')}")

        st.divider()
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


# ─── Tab 1: 섹터 분석 ─────────────────────────────────────────────────────────
def _show_sector_tab(sector_summary: pd.DataFrame, rs_full: pd.DataFrame) -> None:
    st.subheader("Sector RS Overview")

    df = sector_summary[sector_summary["ticker"].isin(SECTOR_TICKERS)].copy()
    if "sector_name" not in df.columns:
        df["sector_name"] = df["ticker"].map(SECTOR_NAME_MAP).fillna(df["ticker"])

    df = df.sort_values("rs_score", ascending=False).reset_index(drop=True)
    df["rs_rank"] = range(1, len(df) + 1)

    # 섹션 A: 섹터 히트맵
    with st.container():
        st.markdown("#### 섹터 RS 히트맵")
        fig_treemap = _make_treemap(df, "sector_name", "ticker", "섹터 RS")
        st.plotly_chart(fig_treemap, use_container_width=True, key="sector_treemap")

    # 섹션 B: 순위표 + RS Line 추이
    with st.container():
        st.markdown("#### 섹터 RS 순위표 & RS Line 추이")
        col1, col2 = st.columns([4, 6])

        with col1:
            _render_ranking_table(df, height=550)

        with col2:
            ctrl_left, ctrl_right = st.columns([6, 4])
            with ctrl_left:
                default_tickers = df.sort_values("rs_rank").head(3)["ticker"].tolist()
                chosen = st.multiselect(
                    "표시할 섹터 선택",
                    options=df["ticker"].tolist(),
                    default=default_tickers,
                    key="sector_multiselect",
                )
            with ctrl_right:
                period_options = {"1개월": 1, "3개월": 3, "6개월": 6, "12개월": 12}
                period_label = st.radio(
                    "기간",
                    options=list(period_options.keys()),
                    index=2,
                    horizontal=True,
                    key="sector_chart_period",
                )
            selected_months = period_options[period_label]

            if chosen:
                fig_line = _make_trend_chart(
                    rs_full, chosen,
                    f"섹터 RS Line 추이 (최근 {period_label}, 정규화)",
                    months=selected_months,
                    normalize=True,
                )
                if fig_line:
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("RS Line 데이터가 없습니다.")


# ─── Tab 1.5: 산업 분석 ───────────────────────────────────────────────────────
INDUSTRY_JUNK_LABELS: set[str] = {"", "Unknown", "Other", "Shell Companies", "N/A"}
# 산업 강도/추이 집계 시 RS 백분위 상위 몇 %를 주도주로 볼지 (순위표·히트맵·추이 공통)
INDUSTRY_LEADER_FRAC: float = 0.5


def _build_industry_summary(
    screener: pd.DataFrame,
    min_count: int = 5,
    exclude_junk: bool = True,
    top_frac: float = INDUSTRY_LEADER_FRAC,
) -> pd.DataFrame:
    """
    개별 종목의 industry 라벨을 기준으로 산업별 RS 지표 집계.

    RS 지표 평균은 각 산업의 RS 백분위 상위 top_frac 비율 종목(주도주 절반)만으로
    계산해 백분위가 낮은 아웃라이어 종목을 배제한다. 종목 수가 적은 산업도
    최소 1개는 집계에 포함된다. stock_count는 산업 전체 종목 수.

    Returns:
        DataFrame[industry, stock_count, rs_score, rs_percentile, rs_slope,
                  pct_positive_slope, rs_rank]  (rs_percentile 내림차순)
    """
    if screener.empty or "industry" not in screener.columns:
        return pd.DataFrame()

    df = screener.copy()
    df["industry"] = df["industry"].astype(str).str.strip()
    df = df[df["industry"] != ""]
    df = df[df["industry"].str.lower() != "nan"]
    if exclude_junk:
        df = df[~df["industry"].isin(INDUSTRY_JUNK_LABELS)]
    if df.empty:
        return pd.DataFrame()

    # 산업 전체 종목 수 (treemap 크기 / min_count 필터용)
    counts = df.groupby("industry")["ticker"].size().rename("stock_count")

    # rs_1m_pct / rs_3m_pct 사전 계산 (top 선별 전에 df에 추가)
    for col in ("rs_1m", "rs_3m"):
        if col in df.columns:
            df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    # 각 산업 RS 백분위 상위 top_frac 비율 종목만 선별 (NaN 백분위는 자동 후순위).
    # 산업별 상위 ceil(종목수 * top_frac)개 — 최소 1개 보장.
    df_sorted = df.sort_values("rs_percentile", ascending=False)
    grp = df_sorted.groupby("industry")
    within_rank = grp.cumcount()  # 0-based: 백분위 내림차순 순위
    grp_size = grp["ticker"].transform("size")
    keep_n = (grp_size * top_frac).apply(lambda x: max(10, math.ceil(x)))  # ceil, 최소 10개
    top = df_sorted[within_rank < keep_n]

    agg_dict: dict = {
        "rs_score":      ("rs_score",      "mean"),
        "rs_percentile": ("rs_percentile", "mean"),
        "rs_slope":      ("rs_slope",      "mean"),
    }
    for col in ("rs_1m", "rs_3m", "price_chg_pct", "price_chg_5d_pct", "rs_1m_pct", "rs_3m_pct"):
        if col in top.columns:
            agg_dict[col] = (col, "mean")
    summary = top.groupby("industry").agg(**agg_dict)
    # 상위 절반 종목 중 RS Slope 양수 비율
    summary["pct_positive_slope"] = (
        top.groupby("industry")["rs_slope"].apply(lambda s: (s > 0).mean() * 100)
    )
    if "rs_line_phase" in top.columns:
        summary["pct_above_ema"] = (
            top.groupby("industry")["rs_line_phase"]
            .apply(lambda s: s.isin(["강세"]).mean() * 100)
        )
    summary = summary.join(counts).reset_index()

    summary = summary[summary["stock_count"] >= min_count]
    if summary.empty:
        return pd.DataFrame()

    summary = summary.sort_values("rs_percentile", ascending=False).reset_index(drop=True)
    summary["rs_rank"] = range(1, len(summary) + 1)
    return summary


def _make_industry_treemap(df: pd.DataFrame) -> go.Figure:
    """산업 히트맵 — 크기=RS Score, 색=RS 백분위 평균."""
    plot_df = df.copy()
    fig = px.treemap(
        plot_df,
        path=[px.Constant("산업"), "industry"],
        values="rs_score",
        color="rs_percentile",
        color_continuous_scale="RdYlGn",
        range_color=[1, 99],
        custom_data=["stock_count", "rs_score", "rs_slope", "rs_percentile"],
        title="산업 RS 히트맵 (크기=RS Score, 색=RS 백분위 평균)",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "종목수: %{customdata[0]}<br>"
            "RS 백분위(평균): %{customdata[3]:.1f}<br>"
            "RS Score(평균): %{customdata[1]:.3f}<br>"
            "Slope(평균): %{customdata[2]:.4f}"
            "<extra></extra>"
        ),
        texttemplate="%{label}<br>%{customdata[3]:.0f}",
        textinfo="label",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="RS %ile", thickness=12),
        margin=dict(t=40, l=0, r=0, b=0),
        height=420,
    )
    return fig


def _make_industry_trend_chart(
    rs_full: pd.DataFrame,
    screener: pd.DataFrame,
    industries: list[str],
    title: str,
    months: int | None = None,
) -> go.Figure | None:
    """산업별 RS Line 추이 — 상위 절반 주도주 rs_line을 정규화 후 date별 median 집계.

    순위표·히트맵과 동일하게 각 산업 RS 백분위 상위 INDUSTRY_LEADER_FRAC 종목만 사용.
    """
    if not industries or "rs_line" not in rs_full.columns:
        return None

    cutoff = rs_full["date"].max() - pd.DateOffset(months=months) if months else None
    frames = []
    for ind in industries:
        # 순위표와 동일한 상위 절반 주도주 선별 (RS 백분위 내림차순)
        ind_members = screener[screener["industry"] == ind]
        if ind_members.empty:
            continue
        if "rs_percentile" in ind_members.columns:
            ind_members = ind_members.sort_values("rs_percentile", ascending=False)
        keep_n = max(10, math.ceil(len(ind_members) * INDUSTRY_LEADER_FRAC))
        members = ind_members.head(keep_n)["ticker"].tolist()
        if not members:
            continue
        sub = rs_full[rs_full["ticker"].isin(members)].copy()
        if cutoff is not None:
            sub = sub[sub["date"] >= cutoff]
        sub = sub.dropna(subset=["rs_line"])
        if sub.empty:
            continue
        # 멤버별 첫날 기준 정규화 후 date별 중앙값
        sub = sub.sort_values(["ticker", "date"])
        first_vals = sub.groupby("ticker")["rs_line"].transform("first")
        sub["rs_line_norm"] = sub["rs_line"] / first_vals
        line = sub.groupby("date")["rs_line_norm"].median().reset_index()
        line["industry"] = ind
        frames.append(line)

    if not frames:
        return None

    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        plot_df,
        x="date",
        y="rs_line_norm",
        color="industry",
        title=title,
        labels={"date": "날짜", "rs_line_norm": "RS Line (상위 절반 주도주 정규화 중앙값, 시작=1.0)"},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="기준선 1.0", annotation_position="right")
    fig.update_layout(
        title=dict(y=0.97, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=40, l=0, r=0, b=120),
        height=550,
    )
    return fig


def _show_industry_tab(screener: pd.DataFrame, rs_full: pd.DataFrame) -> None:
    st.subheader("🏭 산업(Industry) RS 분석")
    st.caption("개별 종목의 industry 라벨을 기준으로 산업별 RS 강도를 집계 "
               "(각 산업 RS 백분위 상위 절반 종목 = 주도주 평균 기준).")

    if screener.empty or "industry" not in screener.columns:
        st.info("industry 라벨이 있는 스크리너 데이터가 없습니다.")
        return

    # ── 컨트롤 ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 3])
    with c1:
        min_count = st.slider(
            "산업당 최소 종목 수",
            min_value=1, max_value=30, value=5,
            key="industry_min_count",
            help="멤버 종목이 적은 산업은 노이즈가 커 제외합니다.",
        )
    with c2:
        exclude_junk = st.checkbox(
            "정크 라벨 제외 (Unknown / Other / Shell Companies)",
            value=True, key="industry_exclude_junk",
        )

    summary = _build_industry_summary(screener, min_count=min_count, exclude_junk=exclude_junk)
    if summary.empty:
        st.warning("조건을 만족하는 산업이 없습니다. 최소 종목 수를 낮춰보세요.")
        return

    st.caption(f"집계된 산업: {len(summary)}개")

    # ── 섹션 A: 산업 히트맵 ──────────────────────────────────────────────────
    with st.container():
        st.markdown("#### 산업 RS 히트맵")
        st.plotly_chart(_make_industry_treemap(summary), use_container_width=True, key="industry_treemap")

    # ── 섹션 B: 순위표 + RS Line 추이 ────────────────────────────────────────
    with st.container():
        st.markdown("#### 산업 RS 순위표 & RS Line 추이")
        col1, col2 = st.columns([4, 6])

        with col1:
            disp_candidate = [
                "rs_rank", "industry", "stock_count",
                "rs_percentile", "rs_1m_pct", "rs_3m_pct",
            ]
            disp = summary[[c for c in disp_candidate if c in summary.columns]].copy()
            st.dataframe(
                disp,
                use_container_width=True, hide_index=True, height=550,
                column_config={
                    "rs_rank":       st.column_config.NumberColumn("순위",    format="%d"),
                    "industry":      st.column_config.TextColumn("산업"),
                    "stock_count":   st.column_config.NumberColumn("종목수",  format="%d"),
                    "rs_percentile": st.column_config.ProgressColumn(
                        "RS 백분위(평균)", min_value=0, max_value=100, format="%.1f%%"),
                    "rs_1m_pct":     st.column_config.ProgressColumn(
                        "1개월 RS 백분위(평균)", min_value=0, max_value=100, format="%.1f%%"),
                    "rs_3m_pct":     st.column_config.ProgressColumn(
                        "3개월 RS 백분위(평균)", min_value=0, max_value=100, format="%.1f%%"),
                },
            )

        with col2:
            options = summary["industry"].tolist()
            default_ind = summary.sort_values("rs_rank").head(3)["industry"].tolist()
            ctrl_left, ctrl_right = st.columns([6, 4])
            with ctrl_left:
                chosen = st.multiselect(
                    "표시할 산업 선택",
                    options=options,
                    default=default_ind,
                    key="industry_multiselect",
                )
            with ctrl_right:
                period_options = {"1개월": 1, "3개월": 3, "6개월": 6, "12개월": 12}
                period_label = st.radio(
                    "기간",
                    options=list(period_options.keys()),
                    index=2,
                    horizontal=True,
                    key="industry_chart_period",
                )
            if chosen:
                fig_line = _make_industry_trend_chart(
                    rs_full, screener, chosen,
                    f"산업 RS Line 추이 (최근 {period_label}, 멤버 정규화 중앙값)",
                    months=period_options[period_label],
                )
                if fig_line:
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("선택한 산업의 RS Line 데이터가 없습니다.")
            else:
                st.info("산업을 1개 이상 선택하세요.")

    # ── 섹션 C: 산업 드릴다운 (멤버 종목) ────────────────────────────────────
    with st.container():
        st.markdown("#### 산업별 구성 종목 드릴다운")
        drill = st.selectbox(
            "산업 선택",
            options=summary["industry"].tolist(),
            key="industry_drill_select",
        )
        members = screener[screener["industry"].astype(str).str.strip() == drill].copy()
        member_cols = [c for c in [
            "ticker", "rs_rank", "rs_percentile", "rs_score", "rs_slope",
            "price_chg_pct", "rs_1m", "rs_3m", "rs_line_phase",
            "latest_price", "market_cap", "parent_etf",
        ] if c in members.columns]
        members = members.sort_values("rs_percentile", ascending=False)

        st.caption(f"{drill} — 구성 종목 {len(members)}개")
        st.dataframe(
            members[member_cols].reset_index(drop=True),
            use_container_width=True, hide_index=True, height=320,
            column_config={
                "ticker":        st.column_config.TextColumn("종목"),
                "rs_rank":       st.column_config.NumberColumn("RS 순위",   format="%d위"),
                "rs_percentile": st.column_config.ProgressColumn(
                    "RS 백분위", min_value=0, max_value=100, format="%.1f%%"),
                "rs_score":      st.column_config.NumberColumn("RS Score",  format="%.3f"),
                "rs_slope":      st.column_config.NumberColumn("RS Slope",  format="%.4f"),
                "price_chg_pct": st.column_config.NumberColumn("1D 등락률", format="%.2f%%"),
                "rs_1m":         st.column_config.NumberColumn("1개월 RS",  format="%.3f"),
                "rs_3m":         st.column_config.NumberColumn("3개월 RS",  format="%.3f"),
                "rs_line_phase": st.column_config.TextColumn("RS Phase"),
                "latest_price":  st.column_config.NumberColumn("현재가",    format="$%.2f"),
                "market_cap":    st.column_config.NumberColumn("시가총액",   format="$%.2e"),
                "parent_etf":    st.column_config.TextColumn("편입 ETF"),
            },
        )


# ─── Tab 2: 테마 분석 ─────────────────────────────────────────────────────────
def _show_theme_tab(theme_summary: pd.DataFrame, rs_full: pd.DataFrame) -> None:
    st.subheader("테마 ETF RS 분석")

    if theme_summary.empty:
        st.info("테마 ETF 데이터가 없습니다.")
        return

    all_themes = sorted(theme_summary["theme"].dropna().unique().tolist())
    selected_themes = st.multiselect(
        "테마 필터 (비워두면 전체 표시)",
        options=all_themes,
        default=[],
        key="theme_filter",
    )
    df = (
        theme_summary[theme_summary["theme"].isin(selected_themes)].copy()
        if selected_themes
        else theme_summary.copy()
    )

    # 섹션 C: 테마 ETF 히트맵
    with st.container():
        st.markdown("#### 테마 ETF RS 히트맵")
        fig_theme_map = _make_treemap(df, "ticker", "ticker", "테마 ETF RS")
        st.plotly_chart(fig_theme_map, use_container_width=True, key="theme_treemap")

    # 섹션 D: 순위표 + RS Line 추이
    with st.container():
        st.markdown("#### 테마 ETF RS 순위표 & RS Line 추이")
        col1, col2 = st.columns([4, 6])

        with col1:
            _render_ranking_table(df, height=550)

        with col2:
            theme_tickers = df["ticker"].unique().tolist()
            default_t = (
                df.sort_values("rs_rank").head(3)["ticker"].tolist()
                if "rs_rank" in df.columns
                else theme_tickers[:3]
            )
            chosen_themes = st.multiselect(
                "표시할 테마 ETF 선택",
                options=theme_tickers,
                default=default_t,
                key="theme_multiselect",
            )
            ctrl_left, ctrl_right = st.columns([6, 4])
            with ctrl_left:
                pass
            with ctrl_right:
                period_options = {"1개월": 1, "3개월": 3, "6개월": 6, "12개월": 12}
                period_label = st.radio(
                    "기간",
                    options=list(period_options.keys()),
                    index=2,
                    horizontal=True,
                    key="theme_chart_period",
                )
            if chosen_themes:
                fig_theme_line = _make_trend_chart(
                    rs_full, chosen_themes,
                    f"테마 ETF RS Line 추이 (최근 {period_label}, 정규화)",
                    months=period_options[period_label],
                    normalize=True,
                )
                if fig_theme_line:
                    st.plotly_chart(fig_theme_line, use_container_width=True)
                else:
                    st.info("RS Line 데이터가 없습니다.")


# ─── Tab 3: 종목 스크리너 ─────────────────────────────────────────────────────
def _show_screener_tab(screener: pd.DataFrame, tech_indicators: pd.DataFrame | None = None) -> None:
    st.subheader("종목 스크리너")

    # 세션 첫 진입 시 저장된 설정 로드 → session_state 초기화
    if "screener_cfg_loaded" not in st.session_state:
        saved = _load_screener_config()
        for key, val in saved.items():
            if key in _SCREENER_CFG_KEYS and key not in st.session_state:
                st.session_state[key] = val
        st.session_state["screener_cfg_loaded"] = True

    if screener.empty:
        st.info("스크리너 데이터가 없습니다.")
        return

    # atr_pct 보강 — screener에 없으면 tech_indicators 최신 날짜 기준으로 join
    if "atr_pct" not in screener.columns and tech_indicators is not None and not tech_indicators.empty:
        if "atr_pct" in tech_indicators.columns:
            tech_latest = tech_indicators["date"].max()
            atr_today = (
                tech_indicators[tech_indicators["date"] == tech_latest]
                [["ticker", "atr_pct"]]
            )
            screener = screener.merge(atr_today, on="ticker", how="left")

    has_price   = "latest_price"      in screener.columns
    has_adv     = "avg_dollar_volume" in screener.columns
    has_mktcap  = "market_cap"        in screener.columns
    has_atr_pct = "atr_pct"           in screener.columns

    # ── 필터 컨트롤 행 1 ─────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([3, 2, 2, 2, 3])

    with f1:
        pct_min = st.slider(
            "RS 백분위 최솟값",
            min_value=0, max_value=99, value=80,
            key="screener_pct_slider",
        )
        rs_3m_pct_min = st.slider(
            "3m RS 백분위 최솟값",
            min_value=0, max_value=99, value=0,
            key="screener_3m_rs_pct_min",
        )
        rs_1m_pct_min = st.slider(
            "1m RS 백분위 최솟값",
            min_value=0, max_value=99, value=0,
            key="screener_1m_rs_pct_min",
        )
        rs_pct_filter_mode = st.radio(
            "RS 백분위 필터 조합",
            options=["AND", "OR"],
            index=0,
            key="screener_rs_pct_filter_mode",
            horizontal=True,
            help="AND: 세 조건 모두 충족. OR: 세 조건 중 하나 이상 충족.",
        )
    with f2:
        slope_only = st.checkbox(
            "RS Slope 양수만", value=True, key="screener_slope_chk"
        )
    with f3:
        selected_phase = st.multiselect(
            "RS Line Phase 필터",
            options=["강세", "약세"],
            default=["강세"],
            key="screener_phase_filter",
            help="강세: EMA 위 또는 이탈 3일 이내. 약세: EMA 아래 4일 이상. 비워두면 전체 표시.",
        )
    with f4:
        etf_only = st.checkbox(
            "ETF 편입 종목만", value=False, key="screener_etf_only"
        )
    with f5:
        ticker_search = st.text_input(
            "티커 검색",
            placeholder="예: NVDA",
            key="screener_ticker_search",
        )

    # ── 필터 컨트롤 행 2 (Price / ADV / 시가총액 / ATR% / 50일선 이격도 / 업종 순위) ──────
    g1, g2, g3, g4, g5, g6 = st.columns([3, 3, 3, 3, 3, 3])

    with g1:
        price_min = st.number_input(
            "Price 최솟값 ($)",
            min_value=0.0, value=7.0, step=1.0,
            key="screener_price_min",
        )
    with g2:
        adv_min_m = st.number_input(
            "Avg Dollar Volume 최솟값 (백만$, M)",
            min_value=0.0, value=5.0, step=1.0,
            key="screener_adv_min",
        )
    with g3:
        mktcap_min_b = st.number_input(
            "시가총액 최솟값 (B, 십억$)",
            min_value=0.0, value=0.0, step=0.5,
            key="screener_mktcap_min",
        )
    with g4:
        atr_pct_min = st.number_input(
            "ATR% 최솟값",
            min_value=0.0, value=3.5, step=0.5,
            key="screener_atr_pct_min",
            help="14일 ATR ÷ 종가 × 100. 변동성이 일정 수준 이상인 종목만 표시.",
        )
    with g5:
        atr_multiple_max = st.number_input(
            "50일선 이격도 최댓값",
            min_value=0.0, value=6.0, step=0.5,
            key="screener_atr_multiple_max",
            help="50일선으로부터 ATR 몇 배 이내 종목만 표시. 0이면 필터 미적용.",
        )
    with g6:
        ind_rank_max = st.number_input(
            "업종 순위 최솟값",
            min_value=0, value=0, step=1,
            key="screener_ind_rank_max",
            help="해당 순위 이내 업종에 속한 종목만 표시. 0이면 필터 미적용.",
        )

    # ── 정렬 기준 선택 + 설정 저장 ───────────────────────────────────────────
    sort_col1, sort_col2, sort_col3 = st.columns([3, 5, 2])
    with sort_col1:
        sort_options = {
            "1개월 RS (단기)":    "rs_1m",
            "3개월 RS (중기)":    "rs_3m",
            "RS Percentile":       "rs_percentile",

            "RS Slope":            "rs_slope",
            "EMA 위 비중":        "rs_line_above_ema_pct",
        }
        sort_label = st.selectbox(
            "정렬 기준",
            options=list(sort_options.keys()),
            index=0,
            key="screener_sort_by",
        )
        sort_col = sort_options[sort_label]
    with sort_col3:
        st.write("")  # 레이블 높이 맞춤
        if st.button("💾 기본값 저장", key="screener_save_btn", use_container_width=True):
            _save_screener_config()
            st.toast("필터 기본값이 저장되었습니다.", icon="✅")

    # ── 필터 전 사전 계산 (ind_rs_rank, rs_1m_pct, rs_3m_pct) ──────────────
    screener = screener.copy()
    if "industry" in screener.columns:
        _ind_summary = _build_industry_summary(screener, min_count=5, exclude_junk=True)
        if not _ind_summary.empty:
            _ind_rank_map = _ind_summary.set_index("industry")["rs_rank"].to_dict()
            screener["ind_rs_rank"] = screener["industry"].map(_ind_rank_map)
    for _col in ("rs_1m", "rs_3m"):
        if _col in screener.columns:
            _pct_map = dict(zip(screener["ticker"], (screener[_col].rank(pct=True) * 100).round(1)))
            screener[f"{_col}_pct"] = screener["ticker"].map(_pct_map)

    # ── 필터 적용 ─────────────────────────────────────────────────────────────
    filtered = screener.copy()
    # ── RS 백분위 3종 필터 (AND / OR) ────────────────────────────────────────
    _pct_conditions: list = []
    if pct_min > 0 and "rs_percentile" in filtered.columns:
        _pct_conditions.append(filtered["rs_percentile"] >= pct_min)
    if rs_3m_pct_min > 0 and "rs_3m_pct" in filtered.columns:
        _pct_conditions.append(filtered["rs_3m_pct"].fillna(0) >= rs_3m_pct_min)
    if rs_1m_pct_min > 0 and "rs_1m_pct" in filtered.columns:
        _pct_conditions.append(filtered["rs_1m_pct"].fillna(0) >= rs_1m_pct_min)
    if _pct_conditions:
        combine = functools.reduce(operator.and_ if rs_pct_filter_mode == "AND" else operator.or_, _pct_conditions)
        filtered = filtered[combine]
    if slope_only and "rs_slope" in filtered.columns:
        filtered = filtered[filtered["rs_slope"] > 0]
    if selected_phase and "rs_line_phase" in filtered.columns:
        filtered = filtered[filtered["rs_line_phase"].isin(selected_phase)]
    if etf_only and "parent_etf" in filtered.columns:
        filtered = filtered[filtered["parent_etf"] != ""]
    if ticker_search.strip():
        filtered = filtered[
            filtered["ticker"].str.contains(ticker_search.strip().upper(), na=False)
        ]
    if has_price and price_min > 0:
        filtered = filtered[filtered["latest_price"].fillna(0) >= price_min]
    if has_adv and adv_min_m > 0:
        adv_min_raw = adv_min_m * 1_000_000
        filtered = filtered[filtered["avg_dollar_volume"].fillna(0) >= adv_min_raw]
    if has_mktcap and mktcap_min_b > 0:
        mktcap_min_raw = mktcap_min_b * 1e9
        filtered = filtered[filtered["market_cap"].fillna(0) >= mktcap_min_raw]
    if has_atr_pct and atr_pct_min > 0:
        filtered = filtered[filtered["atr_pct"].fillna(0) >= atr_pct_min]
    if atr_multiple_max > 0 and "atr_multiple" in filtered.columns:
        filtered = filtered[filtered["atr_multiple"].fillna(0) <= atr_multiple_max]
    if ind_rank_max > 0 and "ind_rs_rank" in filtered.columns:
        filtered = filtered[filtered["ind_rs_rank"].fillna(999) <= ind_rank_max]

    # ── 정렬 ──────────────────────────────────────────────────────────────────
    sort_target = sort_col if sort_col in filtered.columns else "rs_percentile"
    filtered = filtered.sort_values(sort_target, ascending=False).reset_index(drop=True)

    # 표시용 단위 변환 컬럼 추가
    if has_adv:
        filtered["adv_m"] = filtered["avg_dollar_volume"] / 1_000_000
    if has_mktcap:
        filtered["mktcap_b"] = filtered["market_cap"] / 1e9
    st.caption(f"총 {len(filtered):,}개 종목 (전체 {len(screener):,}개 중)")
    if selected_phase:
        st.caption(
            f"Phase 필터: {', '.join(selected_phase)} → {sort_label} 기준 내림차순 정렬"
        )

    display_cols = [c for c in [
        "ticker", "industry", "ind_rs_rank",
        "price_chg_pct", "price_chg_5d_pct",
        "price_chg_1m_pct", "price_chg_3m_pct", "price_chg_6m_pct", "price_chg_1y_pct",
        "rs_percentile", "rs_1m_pct", "rs_3m_pct",
        "rs_line_phase", "atr_multiple",
        "latest_price", "adv_m", "mktcap_b", "atr_pct",
        "parent_etf", "theme",
    ] if c in filtered.columns]

    col_config = {
        "rs_rank":       st.column_config.NumberColumn("RS 순위",     format="%d"),
        "rs_percentile": st.column_config.ProgressColumn(
            "RS 백분위", min_value=0, max_value=100, format="%.1f%%"
        ),
        "rs_score":      st.column_config.NumberColumn("RS Score",    format="%.3f"),
        "rs_slope":      st.column_config.NumberColumn("RS Slope",    format="%.4f"),
        "latest_price":  st.column_config.NumberColumn("Price ($)",   format="$%.2f"),
        "adv_m":         st.column_config.NumberColumn("Avg DV (M$)", format="$%.1fM"),
        "mktcap_b":      st.column_config.NumberColumn("시가총액 (B)",  format="$%.2fB"),
        "price_chg_pct":    st.column_config.NumberColumn("1D 등락률", format="%.2f%%"),
        "price_chg_5d_pct": st.column_config.NumberColumn("1W 등락률", format="%.2f%%"),
        "price_chg_1m_pct": st.column_config.NumberColumn("1M 등락률", format="%.2f%%"),
        "price_chg_3m_pct": st.column_config.NumberColumn("3M 등락률", format="%.2f%%"),
        "price_chg_6m_pct": st.column_config.NumberColumn("6M 등락률", format="%.2f%%"),
        "price_chg_1y_pct": st.column_config.NumberColumn("1Y 등락률", format="%.2f%%"),

        "atr_pct":          st.column_config.NumberColumn("ATR%",              format="%.2f%%"),
        "atr_multiple":     st.column_config.NumberColumn("50일선 이격도(ATR)",  format="%.2f"),
        "ind_rs_rank":      st.column_config.NumberColumn("업종 순위",           format="%d"),
        "rs_1m_pct":     st.column_config.ProgressColumn(
            "1m RS 백분위", min_value=0, max_value=100, format="%.1f%%"),
        "rs_3m_pct":     st.column_config.ProgressColumn(
            "3m RS 백분위", min_value=0, max_value=100, format="%.1f%%"),
        "rs_line_phase": st.column_config.TextColumn("RS Phase"),
    }

    def _phase_row_color(row) -> list[str]:
        phase = row.get("rs_line_phase", "")
        if phase == "강세":
            return ["background-color: rgba(29,158,117,0.15)"] * len(row)
        elif phase == "약세":
            return ["background-color: rgba(226,75,74,0.06)"] * len(row)
        return [""] * len(row)

    if "rs_line_phase" in filtered.columns:
        styled_filtered = filtered[display_cols].style.apply(_phase_row_color, axis=1)
        st.dataframe(
            styled_filtered,
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
            height=600,
        )
    else:
        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
            height=600,
        )


# ─── Tab 4: Market Breadth ────────────────────────────────────────────────────
def _show_breadth_tab(breadth_df: pd.DataFrame) -> None:
    st.subheader("Market Breadth — 시장 내부 건강도")

    if breadth_df.empty:
        st.info("Breadth 데이터가 없습니다. rs_engine.py를 먼저 실행하세요.")
        return

    # 기간 필터
    period_map = {"1개월": 21, "3개월": 63, "6개월": 126, "1년": 252}
    period_label = st.radio(
        "표시 기간",
        options=list(period_map.keys()),
        index=2,
        horizontal=True,
        key="breadth_period",
    )
    days = period_map[period_label]
    df = breadth_df.sort_values("date").tail(days).copy()

    # 요약 지표 카드
    latest = df.iloc[-1]
    score = int(latest.get("breadth_score", 0))
    score_label, _ = BREADTH_SCORE_LABELS.get(score, ("N/A", "secondary"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net New High",   f"{int(latest.get('net_new_high', 0)):+d}")
    c2.metric("50MA 위 비율",   f"{latest.get('pct_above_50ma', 0):.1f}%")
    c3.metric("200MA 위 비율",  f"{latest.get('pct_above_200ma', 0):.1f}%")
    c4.metric("A/D Line",       f"{int(latest.get('ad_line', 0)):,}")
    c5.metric("Breadth Score",  f"{score} / 5  {score_label}")

    st.divider()

    # 섹션 A: 52W Net New High
    with st.container():
        st.markdown("#### 52주 Net New High")

        df["nnh_ma20"] = df["net_new_high"].rolling(20).mean()
        colors = df["net_new_high"].apply(lambda v: "#1D9E75" if v >= 0 else "#E24B4A")

        fig_nnh = go.Figure()
        fig_nnh.add_trace(go.Bar(
            x=df["date"],
            y=df["net_new_high"],
            marker_color=colors,
            name="Net New High",
            hovertemplate="%{x|%Y-%m-%d}<br>Net New High: %{y:+d}",
        ))
        fig_nnh.add_trace(go.Scatter(
            x=df["date"], y=df["nnh_ma20"],
            line=dict(color="#EF9F27", width=1.5),
            name="20MA",
            hovertemplate="%{x|%Y-%m-%d}<br>20MA: %{y:.1f}",
        ))
        fig_nnh.add_hline(y=0, line_dash="dot", line_color="gray", line_width=0.8)
        fig_nnh.update_layout(
            height=280,
            margin=dict(t=10, b=40, l=0, r=0),
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
            bargap=0.1,
        )
        st.plotly_chart(fig_nnh, use_container_width=True)

        nnh_slope = df["nnh_ma20"].diff(5).iloc[-1]
        if nnh_slope < 0 and latest.get("net_new_high", 0) < df["net_new_high"].quantile(0.3):
            st.warning(
                "⚠️ 다이버전스 감지: Net New High가 감소 추세입니다. "
                "지수 상승 지속 여부를 주의 깊게 모니터링하세요."
            )

    st.divider()

    # 섹션 B: MA 위 종목 비율
    with st.container():
        st.markdown("#### 이동평균선 위 종목 비율")

        fig_ma = go.Figure()

        for y0, y1, color, label in [
            (0,  20,  "rgba(226,75,74,0.07)",  "약세"),
            (20, 50,  "rgba(239,159,39,0.07)", "주의"),
            (50, 80,  "rgba(29,158,117,0.07)", "강세"),
            (80, 100, "rgba(29,158,117,0.13)", "과열"),
        ]:
            fig_ma.add_hrect(
                y0=y0, y1=y1,
                fillcolor=color, line_width=0,
                annotation_text=label,
                annotation_position="right",
                annotation_font_size=10,
                annotation_font_color="gray",
            )

        fig_ma.add_trace(go.Scatter(
            x=df["date"], y=df["pct_above_50ma"],
            line=dict(color="#378ADD", width=2),
            name="50MA 위 %",
            hovertemplate="%{x|%Y-%m-%d}<br>50MA 위: %{y:.1f}%",
        ))
        fig_ma.add_trace(go.Scatter(
            x=df["date"], y=df["pct_above_200ma"],
            line=dict(color="#7F77DD", width=1.5, dash="dash"),
            name="200MA 위 %",
            hovertemplate="%{x|%Y-%m-%d}<br>200MA 위: %{y:.1f}%",
        ))

        for y_val, color, label in [
            (50, "gray",    "50%"),
            (80, "#1D9E75", "80%"),
            (20, "#E24B4A", "20%"),
        ]:
            fig_ma.add_hline(
                y=y_val, line_dash="dot", line_color=color,
                line_width=0.8,
                annotation_text=label,
                annotation_position="right",
            )

        fig_ma.update_layout(
            height=300,
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            margin=dict(t=10, b=40, l=0, r=60),
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    st.divider()

    # 섹션 C: A/D Line
    with st.container():
        st.markdown("#### Advance / Decline Line")

        col_left, col_right = st.columns([6, 4])

        with col_left:
            fig_ad = go.Figure()
            fig_ad.add_trace(go.Scatter(
                x=df["date"], y=df["ad_line"],
                fill="tozeroy",
                fillcolor="rgba(55,138,221,0.1)",
                line=dict(color="#378ADD", width=1.8),
                name="A/D Line",
                hovertemplate="%{x|%Y-%m-%d}<br>A/D Line: %{y:,}",
            ))
            fig_ad.update_layout(
                height=250,
                margin=dict(t=10, b=40, l=0, r=0),
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig_ad, use_container_width=True)

        with col_right:
            st.markdown("##### 해석 가이드")
            st.markdown(
                """
- A/D Line **상승** → 시장 내부 건강
- 지수 신고가 + A/D Line **하락** → 천장 다이버전스 경고
- A/D Line **급락** → 광범위한 매도
                """
            )
            ad_change = df["ad_line"].iloc[-1] - df["ad_line"].iloc[max(-20, -len(df))]
            trend_label = "상승 중" if ad_change > 0 else "하락 중"
            trend_color = "normal" if ad_change > 0 else "inverse"
            st.metric(
                "A/D Line 20일 변화",
                f"{int(ad_change):+,}",
                delta=trend_label,
                delta_color=trend_color,
            )

    st.divider()

    # 섹션 D: Breadth Score 히스토리
    with st.container():
        st.markdown("#### Breadth Score 히스토리")

        color_map = {5: "#1D9E75", 4: "#5DCAA5", 3: "#EF9F27",
                     2: "#F0997B", 1: "#E24B4A", 0: "#A32D2D"}
        fig_score = go.Figure(go.Bar(
            x=df["date"],
            y=df["breadth_score"],
            marker_color=df["breadth_score"].map(color_map).fillna("gray"),
            hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y}/5",
        ))
        fig_score.update_layout(
            height=200,
            yaxis=dict(range=[0, 5.5], dtick=1),
            margin=dict(t=10, b=40, l=0, r=0),
            bargap=0.1,
        )
        st.plotly_chart(fig_score, use_container_width=True)


# ─── Tab 5: 포트폴리오 ────────────────────────────────────────────────────────

def _show_portfolio_tab(
    portfolio_snapshot: pd.DataFrame,
    trade_history: pd.DataFrame,
    rs_full: pd.DataFrame,
) -> None:
    st.subheader("💼 포트폴리오")

    if portfolio_snapshot.empty:
        st.info(
            "포트폴리오 데이터가 없습니다.\n\n"
            "터미널에서 아래 명령어를 실행해 데이터를 수집하세요:\n"
            "```bash\npython kis_collector.py --mode real\n```"
        )
        return

    latest_date = portfolio_snapshot["date"].max()
    today_snap  = portfolio_snapshot[portfolio_snapshot["date"] == latest_date].copy()
    st.caption(f"기준일: {latest_date.strftime('%Y-%m-%d')} 마감 기준")

    today_snap = _join_rs_to_portfolio(today_snap, rs_full)

    # ── 섹션 1: 계좌 요약 헤더 ────────────────────────────────────────────
    with st.container():
        st.markdown("#### 💰 계좌 요약")

        total_eval    = float(today_snap["total_eval_usd"].iloc[0]) if "total_eval_usd" in today_snap.columns and not today_snap.empty else 0.0
        cash_usd      = float(today_snap["cash_usd"].iloc[0])       if "cash_usd"       in today_snap.columns and not today_snap.empty else 0.0
        total_pnl     = float(today_snap["pnl_usd"].sum())          if "pnl_usd"        in today_snap.columns else 0.0
        total_cost    = float((today_snap["eval_amt_usd"].sum() - total_pnl)) if "eval_amt_usd" in today_snap.columns else 0.0
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 평가금액", f"${total_eval:,.0f}")
        c2.metric("총 평가손익", f"${total_pnl:+,.0f}")
        c3.metric(
            "총 수익률", f"{total_pnl_pct:+.2f}%",
            delta=f"{total_pnl_pct:+.2f}%",
            delta_color="normal" if total_pnl_pct >= 0 else "inverse",
        )
        c4.metric("가용 예수금", f"${cash_usd:,.0f}")

    st.divider()

    # ── 섹션 2: 보유 포지션 테이블 + RS 연동 ─────────────────────────────
    with st.container():
        st.markdown("#### 📋 보유 포지션")

        display_cols = [c for c in PORTFOLIO_COLS_DISPLAY if c in today_snap.columns]
        display_df   = today_snap[display_cols].sort_values(
            "eval_amt_usd", ascending=False
        ) if "eval_amt_usd" in display_cols else today_snap[display_cols]

        def _row_style(row: pd.Series) -> list[str]:
            if row.get("rs_signal", "") == "🔴 RS약세":
                return ["background-color: rgba(226,75,74,0.08)"] * len(row)
            return [""] * len(row)

        styled = display_df.style.apply(_row_style, axis=1)

        col_config = {
            "ticker":            st.column_config.TextColumn("종목"),
            "qty":               st.column_config.NumberColumn("수량",     format="%.0f"),
            "avg_price_usd":     st.column_config.NumberColumn("평균단가", format="$%.2f"),
            "current_price_usd": st.column_config.NumberColumn("현재가",   format="$%.2f"),
            "pnl_pct":           st.column_config.NumberColumn("수익률",   format="%+.2f%%"),
            "eval_amt_usd":      st.column_config.NumberColumn("평가금액", format="$%.0f"),
            "rs_rank":           st.column_config.NumberColumn("RS 순위",  format="%d위"),
            "rs_score":          st.column_config.NumberColumn("RS Score", format="%.3f"),
            "rs_slope":          st.column_config.NumberColumn("RS Slope", format="%.4f"),
            "rs_signal":         st.column_config.TextColumn("RS 신호"),
        }

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
            height=400,
        )

    st.divider()

    # ── 섹션 3: 자산 추이 차트 ───────────────────────────────────────────
    with st.container():
        st.markdown("#### 📈 자산 추이")

        period_map   = {"1주": 7, "1개월": 30, "3개월": 90, "6개월": 180, "YTD": None}
        period_label = st.radio(
            "기간", options=list(period_map.keys()),
            index=2, horizontal=True, key="portfolio_period",
        )
        days = period_map[period_label]

        daily_total = (
            portfolio_snapshot
            .groupby("date")["eval_amt_usd"]
            .sum()
            .reset_index()
            .rename(columns={"eval_amt_usd": "total_usd"})
        ) if "eval_amt_usd" in portfolio_snapshot.columns else pd.DataFrame()

        if not daily_total.empty:
            max_date = daily_total["date"].max()
            if days:
                cutoff = max_date - pd.Timedelta(days=days)
            else:
                cutoff = pd.Timestamp(f"{max_date.year}-01-01")
            daily_total = daily_total[daily_total["date"] >= cutoff]

        if daily_total.empty or len(daily_total) < 2:
            st.info("자산 추이 데이터를 누적 중입니다. 최소 2일치 수집 후 차트가 표시됩니다.")
        else:
            first_val = float(daily_total["total_usd"].iloc[0]) or 1.0
            daily_total["cum_pct"] = (daily_total["total_usd"] / first_val - 1) * 100

            start_d = daily_total["date"].min()
            end_d   = daily_total["date"].max()
            spy_ret = _get_spy_cumulative_return(rs_full, start_d, end_d)

            fig_asset = go.Figure()
            fig_asset.add_trace(go.Scatter(
                x=daily_total["date"], y=daily_total["cum_pct"],
                name="내 계좌",
                line=dict(color="#378ADD", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>내 계좌: %{y:+.2f}%<extra></extra>",
            ))

            if spy_ret is not None and not spy_ret.empty:
                fig_asset.add_trace(go.Scatter(
                    x=spy_ret.index, y=spy_ret.values,
                    name="SPY",
                    line=dict(color="#EF9F27", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>SPY: %{y:+.2f}%<extra></extra>",
                ))

            fig_asset.add_hline(y=0, line_dash="dot", line_color="gray", line_width=0.8)
            fig_asset.update_layout(
                height=320,
                yaxis=dict(ticksuffix="%"),
                margin=dict(t=10, b=60, l=0, r=0),
                legend=dict(orientation="h", y=-0.2),
                hovermode="x unified",
            )
            st.plotly_chart(fig_asset, use_container_width=True)

    st.divider()

    # ── 섹션 4: 거래 내역 + 실현손익 ─────────────────────────────────────
    with st.container():
        st.markdown("#### 📒 거래 내역 & 실현손익")

        col_hist, col_pnl = st.columns([6, 4])

        with col_hist:
            trade_period = st.selectbox(
                "조회 기간",
                ["최근 30일", "최근 90일", "전체"],
                key="trade_period_select",
            )
            trade_df = trade_history.copy()
            if not trade_df.empty:
                if trade_period == "최근 30일":
                    trade_df = trade_df[trade_df["date"] >= trade_df["date"].max() - pd.Timedelta(days=30)]
                elif trade_period == "최근 90일":
                    trade_df = trade_df[trade_df["date"] >= trade_df["date"].max() - pd.Timedelta(days=90)]
                trade_df = trade_df.sort_values("date", ascending=False)

            hist_cols = [c for c in ["date", "ticker", "side", "qty", "price_usd", "amount_usd", "realized_pnl_usd"] if c in trade_df.columns]
            st.dataframe(
                trade_df[hist_cols] if not trade_df.empty else pd.DataFrame(columns=hist_cols),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date":             st.column_config.DateColumn("날짜",    format="YYYY-MM-DD"),
                    "ticker":           st.column_config.TextColumn("종목"),
                    "side":             st.column_config.TextColumn("구분"),
                    "qty":              st.column_config.NumberColumn("수량",    format="%.0f"),
                    "price_usd":        st.column_config.NumberColumn("단가",    format="$%.2f"),
                    "amount_usd":       st.column_config.NumberColumn("체결금액", format="$%.0f"),
                    "realized_pnl_usd": st.column_config.NumberColumn("실현손익", format="$%+.0f"),
                },
                height=350,
            )

        with col_pnl:
            st.markdown("**종목별 누적 실현손익**")
            pnl_df = _calc_realized_pnl_by_ticker(trade_history)

            if pnl_df.empty:
                st.info("실현손익 데이터가 없습니다.")
            else:
                pnl_df = pnl_df.sort_values("total_realized_pnl_usd", ascending=True)
                fig_pnl = go.Figure(go.Bar(
                    x=pnl_df["total_realized_pnl_usd"],
                    y=pnl_df["ticker"],
                    orientation="h",
                    marker_color=pnl_df["total_realized_pnl_usd"].apply(
                        lambda v: "#1D9E75" if v >= 0 else "#E24B4A"
                    ),
                    text=pnl_df["total_realized_pnl_usd"].apply(lambda v: f"${v:+,.0f}"),
                    textposition="outside",
                    hovertemplate="%{y}<br>실현손익: $%{x:+,.0f}<extra></extra>",
                ))
                fig_pnl.add_vline(x=0, line_color="gray", line_width=0.8)
                fig_pnl.update_layout(
                    height=350,
                    margin=dict(t=10, b=10, l=0, r=80),
                    showlegend=False,
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

    st.divider()

    # ── 섹션 5: RS 경고 패널 ─────────────────────────────────────────────
    with st.container():
        st.markdown("#### ⚡ RS 경고 패널")

        warnings = _get_rs_warnings(today_snap, rs_full)

        if not warnings:
            st.success("보유 종목 중 특이 RS 신호 없음")
        else:
            danger_list  = [w for w in warnings if w["type"] == "danger"]
            warning_list = [w for w in warnings if w["type"] == "warning"]
            success_list = [w for w in warnings if w["type"] == "success"]

            col_d, col_w, col_s = st.columns(3)

            with col_d:
                st.markdown("**🔴 RS 급락 경고**")
                if danger_list:
                    for w in danger_list:
                        st.error(f"**{w['ticker']}** — {w['msg']}")
                else:
                    st.info("해당 없음")

            with col_w:
                st.markdown("**🟡 모멘텀 약화**")
                if warning_list:
                    for w in warning_list:
                        st.warning(f"**{w['ticker']}** — {w['msg']}")
                else:
                    st.info("해당 없음")

            with col_s:
                st.markdown("**🟢 RS 강세 유지**")
                if success_list:
                    for w in success_list:
                        st.success(f"**{w['ticker']}** — {w['msg']}")
                else:
                    st.info("해당 없음")


# ─── Tab 6: Watch List ────────────────────────────────────────────────────────

def _show_watchlist_tab(
    rs_full: pd.DataFrame,
    sector_summary: pd.DataFrame,
    theme_summary: pd.DataFrame,
    screener: pd.DataFrame,
    tech_indicators: pd.DataFrame,
) -> None:
    st.subheader("👀 Watch List 분석")
    st.caption("오늘 매수 후보 종목을 입력하고 RS 기반 우선순위를 확인하세요.")

    # ── session_state 초기화 ──────────────────────────────────────────────
    if "watchlist_tickers" not in st.session_state:
        st.session_state["watchlist_tickers"] = _load_watchlist()

    # ── 섹션 1: Watch List 입력 & 관리 ──────────────────────────────────
    with st.container():
        st.markdown("#### 📝 Watch List 입력")

        col_input, col_btn, col_clear = st.columns([5, 1, 1])

        with col_input:
            ticker_input = st.text_input(
                "티커 입력 (콤마로 여러 개 입력 가능)",
                placeholder="예: NVDA, MSFT, AMD",
                key="watchlist_input",
                label_visibility="collapsed",
            )
        with col_btn:
            if st.button("➕ 추가", use_container_width=True, key="watchlist_add_btn"):
                if ticker_input.strip():
                    new_tickers = [
                        t.strip().upper()
                        for t in ticker_input.replace(",", " ").split()
                        if t.strip()
                    ]
                    existing = st.session_state["watchlist_tickers"]
                    added = [t for t in new_tickers if t not in existing]
                    st.session_state["watchlist_tickers"] = existing + added
                    _save_watchlist(st.session_state["watchlist_tickers"])
                    st.rerun()
        with col_clear:
            if st.button("🗑️ 전체삭제", use_container_width=True, key="watchlist_clear_btn"):
                st.session_state["watchlist_tickers"] = []
                _save_watchlist([])
                st.rerun()

        tickers = st.session_state["watchlist_tickers"]
        if tickers:
            st.markdown("**현재 Watch List:**")
            cols = st.columns(min(len(tickers), 10))
            for i, ticker in enumerate(tickers):
                with cols[i % 10]:
                    if st.button(f"{ticker} ✕", key=f"wl_del_{ticker}"):
                        st.session_state["watchlist_tickers"].remove(ticker)
                        _save_watchlist(st.session_state["watchlist_tickers"])
                        st.rerun()
        else:
            st.info("Watch List가 비어있습니다. 티커를 입력해 추가하세요.")
            return

    if not tickers:
        return

    # 데이터 빌드
    wl_df = _build_watchlist_data(
        tickers, rs_full, sector_summary, theme_summary, screener, tech_indicators
    )
    if wl_df.empty:
        st.warning("rs_full에서 Watch List 종목 데이터를 찾을 수 없습니다.")
        return

    # rs_1m_pct / rs_3m_pct — 전체 스크리너 기준 백분위
    for _col in ("rs_1m", "rs_3m"):
        if _col in screener.columns:
            _pct_map = dict(zip(screener["ticker"], (screener[_col].rank(pct=True) * 100).round(1)))
            wl_df[f"{_col}_pct"] = wl_df["ticker"].map(_pct_map)

    st.divider()

    # ── 섹션 2: 매수 우선순위 종합 점수 카드 ─────────────────────────────
    with st.container():
        st.markdown("#### 🏆 매수 우선순위")

        sorted_df = wl_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        n = len(sorted_df)
        cols = st.columns(min(n, 6))

        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if i >= 6:
                break
            score  = row.get("priority_score", 0)
            ticker = row["ticker"]
            with cols[i]:
                if i == 0:
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(135deg,#F5A623,#EF9F27);
                                    border-radius:12px;padding:16px 10px;text-align:center;
                                    color:white;box-shadow:0 4px 14px rgba(239,159,39,0.45);">
                          <div style="font-size:13px;font-weight:700;opacity:0.95;">🏆 1순위</div>
                          <div style="font-size:24px;font-weight:800;margin:4px 0;letter-spacing:0.5px;">{ticker}</div>
                          <div style="font-size:34px;font-weight:800;line-height:1;">{score:.0f}</div>
                          <div style="font-size:11px;opacity:0.9;">/ 100점</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #d0d7de;border-radius:12px;
                                    padding:16px 10px;text-align:center;background:#ffffff;">
                          <div style="font-size:13px;font-weight:600;color:#57606a;">{i+1}순위</div>
                          <div style="font-size:22px;font-weight:700;margin:4px 0;color:#24292f;">{ticker}</div>
                          <div style="font-size:32px;font-weight:800;line-height:1;color:#378ADD;">{score:.0f}</div>
                          <div style="font-size:11px;color:#8b949e;">/ 100점</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # 점수 breakdown — 1순위 종목 기준
        top_row = sorted_df.iloc[0]

        # ④ 산업 강도 상세
        _ind_strength = top_row.get("industry_rs_percentile", float("nan"))
        _ind_rank     = top_row.get("industry_rank", float("nan"))
        _ind_pts      = _calc_priority_score_industry(_ind_strength)
        if pd.isna(_ind_strength):
            _ind_detail = f"{top_row.get('industry', 'N/A')} (강도 데이터 없음)"
        else:
            _rank_disp = f", 산업순위 {int(_ind_rank)}위" if pd.notna(_ind_rank) else ""
            _ind_detail = f"{top_row.get('industry', 'N/A')} — 주도주 RS 백분위 {_ind_strength:.1f}%{_rank_disp}"

        # ③ RS Line 근접도 상세
        _today_val = top_row.get("rs_line_today", float("nan"))
        _high_val  = top_row.get("rs_line_52w_high", float("nan"))
        _ratio = (_today_val / _high_val) if (
            not pd.isna(_today_val) and not pd.isna(_high_val) and _high_val != 0
        ) else 0.0
        _rs_line_pts = _calc_priority_score_rs_line(_today_val, _high_val)
        _rs_line_detail = (
            f"신고가 비율 {_ratio*100:.1f}% "
            f"({'✅ 신고가 경신' if _ratio >= 1.0 else f'{(1-_ratio)*100:.1f}% 이내'})"
        )

        # ⑥ Extension 상세
        _atr_mult = top_row.get("atr_multiple", float("nan"))
        _ext_pts  = _calc_priority_score_extension(_atr_mult)
        if pd.isna(_atr_mult):
            _ext_detail = "데이터 없음 (중립 5점)"
        elif _atr_mult < 0:
            _ext_detail = f"50SMA 아래 ({_atr_mult:.2f}배)"
        else:
            _ext_detail = (
                f"ATR Multiple = {_atr_mult:.2f}배 "
                f"({'이상적 구간' if _atr_mult <= 1.0 else '약간 extended' if _atr_mult <= 2.0 else 'Extended' if _atr_mult <= 3.0 else '과도 extended'})"
            )

        with st.expander(
            f"점수 구성 상세 — {top_row['ticker']} ({top_row.get('priority_score', 0):.0f}점)",
            expanded=False,
        ):
            breakdown_items = [
                ("① RS Percentile", 20, top_row.get("rs_percentile", 0) / 100 * 20 if pd.notna(top_row.get("rs_percentile")) else 0.0,
                 f"RS 백분위 {top_row.get('rs_percentile', 0):.1f}%" if pd.notna(top_row.get("rs_percentile")) else "데이터 없음"),
                ("② RS Line Phase", 20, _calc_priority_score_phase(top_row.get("rs_line_phase")),
                 str(top_row.get("rs_line_phase") or "데이터 없음")),
                ("③ RS Line 근접도", 20, _rs_line_pts, _rs_line_detail),
                ("④ 산업 강도", 20, _ind_pts, _ind_detail),
                ("⑤ Extension 페널티", 20, _ext_pts, _ext_detail),
            ]
            for label, maxpts, pts, detail in breakdown_items:
                pct_bar = min(pts / maxpts * 100, 100) if maxpts else 0
                st.markdown(
                    f"""
                    <div style="margin:8px 0;">
                      <div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:3px;">
                        <span style="font-weight:600;">{label} ({maxpts}점)</span>
                        <span style="color:#57606a;">{pts:.0f}점 · {detail}</span>
                      </div>
                      <div style="background:#eaeef2;border-radius:6px;height:10px;overflow:hidden;">
                        <div style="background:#378ADD;width:{pct_bar:.0f}%;height:100%;"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── 섹션 3+4: RS 비교 테이블 + RS Line 차트 ──────────────────────────
    with st.container():
        st.markdown("#### 📊 RS 상세 비교")
        col_table, col_chart = st.columns([4, 6])

        with col_table:
            display_cols = [
                c for c in [
                    "ticker",
                    "rs_percentile", "rs_3m_pct", "rs_1m_pct",
                    "rs_line_phase", "rs_line_newhigh",
                    "industry", "industry_rs_percentile", "industry_rank",
                    "atr_multiple",
                ] if c in sorted_df.columns
            ]
            st.dataframe(
                sorted_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=350,
                column_config={
                    "ticker":          st.column_config.TextColumn("종목"),
                    "rs_percentile":   st.column_config.ProgressColumn(
                        "RS 백분위", min_value=0, max_value=100, format="%.1f%%"),
                    "rs_3m_pct":       st.column_config.ProgressColumn(
                        "3m RS 백분위", min_value=0, max_value=100, format="%.1f%%"),
                    "rs_1m_pct":       st.column_config.ProgressColumn(
                        "1m RS 백분위", min_value=0, max_value=100, format="%.1f%%"),
                    "rs_line_phase":   st.column_config.TextColumn("RS Phase"),
                    "rs_line_newhigh": st.column_config.CheckboxColumn("RS 신고가"),
                    "industry":               st.column_config.TextColumn("산업"),
                    "industry_rs_percentile": st.column_config.ProgressColumn(
                        "산업 강도(주도주)", min_value=0, max_value=100, format="%.1f%%"),
                    "industry_rank":   st.column_config.NumberColumn("산업 순위", format="%d위"),
                    "atr_multiple":    st.column_config.NumberColumn("ATR Multiple", format="%.2f배"),
                },
            )

        with col_chart:
            period_map = {"1주": 1, "1개월": 1, "3개월": 3, "6개월": 6}
            period_label = st.radio(
                "RS Line 기간",
                options=list(period_map.keys()),
                index=1,
                horizontal=True,
                key="wl_rs_line_period",
            )
            fig_rs = _make_trend_chart(
                rs_full,
                tickers=sorted_df["ticker"].tolist(),
                title="RS Line 비교 (정규화)",
                months=period_map[period_label],
                normalize=True,
            )
            if fig_rs:
                top_ticker = sorted_df.iloc[0]["ticker"]
                for trace in fig_rs.data:
                    if hasattr(trace, "line"):
                        if getattr(trace, "name", None) == top_ticker:
                            trace.line.width = 3
                            trace.line.dash  = "solid"
                        else:
                            trace.line.width = 1.2
                st.plotly_chart(fig_rs, use_container_width=True)
            else:
                st.info("RS Line 데이터를 찾을 수 없습니다.")

    st.divider()

    # ── 섹션 5: 산업 Strength 비교 ──────────────────────────────────────────
    with st.container():
        st.markdown("#### 🏭 산업 Strength")

        # 산업 RS 강도 바 차트
        st.markdown("**소속 산업 RS 강도 (상위 절반 주도주 평균 백분위)**")
        ind_plot = (
            sorted_df[["ticker", "industry", "industry_rs_percentile"]]
            .dropna(subset=["industry_rs_percentile"])
            .sort_values("industry_rs_percentile", ascending=True)
        )
        if not ind_plot.empty:
            bar_height = max(200, len(ind_plot) * 28)
            fig_ind = go.Figure(go.Bar(
                x=ind_plot["industry_rs_percentile"],
                y=ind_plot["industry"].fillna("N/A") + " (" + ind_plot["ticker"] + ")",
                orientation="h",
                marker_color=ind_plot["industry_rs_percentile"].apply(
                    lambda v: "#1D9E75" if v >= 70 else "#EF9F27" if v >= 50 else "#E24B4A"
                ),
                text=ind_plot["industry_rs_percentile"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                hovertemplate="%{y}<br>산업 RS 백분위: %{x:.1f}%<extra></extra>",
            ))
            fig_ind.update_layout(
                height=bar_height,
                margin=dict(t=10, b=10, l=0, r=60),
                xaxis=dict(title="산업 RS 백분위 (%)", range=[0, 110]),
                showlegend=False,
            )
            st.plotly_chart(fig_ind, use_container_width=True)
        else:
            st.info("산업 데이터 없음")

        # Strength 매트릭스 버블 차트
        st.markdown("**Strength 매트릭스** — 우상단 + 큰 버블 = 확신 매수존")
        bubble_df = wl_df.dropna(subset=["rs_percentile", "industry_rs_percentile"]).copy()

        if len(bubble_df) >= 2:
            slope_min = bubble_df["rs_slope"].min()
            slope_max = bubble_df["rs_slope"].max()
            slope_range = slope_max - slope_min if slope_max != slope_min else 1
            bubble_df["bubble_size"] = (
                (bubble_df["rs_slope"] - slope_min) / slope_range * 32 + 8
            )

            top_ticker = sorted_df.iloc[0]["ticker"]
            fig_bubble = go.Figure()
            for _, row in bubble_df.iterrows():
                is_top = row["ticker"] == top_ticker
                color  = "#EF9F27" if is_top else ("#1D9E75" if row.get("rs_line_newhigh") else "#378ADD")
                ind_name = row.get("industry", "N/A") or "N/A"
                ind_pct  = row["industry_rs_percentile"]
                fig_bubble.add_trace(go.Scatter(
                    x=[row["rs_percentile"]],
                    y=[ind_pct],
                    mode="markers+text",
                    marker=dict(
                        size=row["bubble_size"],
                        color=color,
                        opacity=0.85,
                        line=dict(width=2 if is_top else 0, color="white"),
                    ),
                    text=[row["ticker"]],
                    textposition="middle center",
                    textfont=dict(size=11, color="white", family="monospace"),
                    hovertemplate=(
                        f"{row['ticker']}<br>"
                        f"RS 백분위: {row['rs_percentile']:.1f}%<br>"
                        f"산업: {ind_name}<br>"
                        f"산업 RS 백분위: {ind_pct:.1f}%<br>"
                        f"RS Slope: {row['rs_slope']:.4f}<br>"
                        f"종합 점수: {row['priority_score']:.0f}점"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))

            fig_bubble.add_shape(
                type="rect", x0=70, x1=100, y0=60, y1=103,
                fillcolor="rgba(29,158,117,0.07)", line_width=0,
            )
            fig_bubble.add_annotation(
                x=85, y=101,
                text="확신 매수존", font=dict(size=10, color="#1D9E75"), showarrow=False,
            )
            fig_bubble.update_layout(
                height=350,
                xaxis=dict(title="종목 RS 백분위 (%)", range=[0, 105]),
                yaxis=dict(title="산업 RS 강도 (%)", range=[0, 105]),
                margin=dict(t=30, b=40, l=50, r=20),
                hovermode="closest",
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("버블 차트는 2개 이상 종목이 필요합니다.")


# ─── Tab 0: Daily Summary ─────────────────────────────────────────────────────
def _show_daily_summary_tab(
    rs_full: pd.DataFrame,
    sector_summary: pd.DataFrame,
    theme_summary: pd.DataFrame,
    breadth_df: pd.DataFrame,
    screener: pd.DataFrame | None = None,
) -> None:
    st.subheader("📅 Daily Summary")
    st.caption(f"기준일: {pd.to_datetime(rs_full['date']).max().strftime('%Y-%m-%d')} 마감 기준")

    latest_date   = pd.to_datetime(rs_full["date"]).max()
    today_rs      = rs_full[rs_full["date"] == latest_date].copy()
    breadth_today = breadth_df.sort_values("date").iloc[-1] if not breadth_df.empty else pd.Series(dtype=float)

    # ── 개별 종목 필터: 주가 ≥ $7 / 평균거래대금 ≥ $5M / 시총 ≥ $2B ─────────
    PRICE_MIN  = 7
    ADV_MIN    = 5_000_000
    MKTCAP_MIN = 2_000_000_000
    if screener is not None and not screener.empty:
        _mask = pd.Series(True, index=screener.index)
        if "latest_price"     in screener.columns: _mask &= screener["latest_price"]     >= PRICE_MIN
        if "avg_dollar_volume" in screener.columns: _mask &= screener["avg_dollar_volume"] >= ADV_MIN
        if "market_cap"       in screener.columns: _mask &= screener["market_cap"]       >= MKTCAP_MIN
        filtered_tickers: set[str] = set(screener.loc[_mask, "ticker"].tolist())
    else:
        filtered_tickers = None  # 필터 없이 전체 표시

    # ── 섹션 1: 시장 요약 헤더 ──────────────────────────────────────────────
    with st.container():
        st.markdown("#### 📊 오늘의 시장 요약")

        market_ret = _calc_daily_return(rs_full, MARKET_TICKERS + SECTOR_TICKERS)

        col_spy, col_qqq, col_iwm, col_nnh, col_score = st.columns(5)
        for col, ticker in zip([col_spy, col_qqq, col_iwm], MARKET_TICKERS):
            name = MARKET_TICKER_NAMES[ticker]
            row  = market_ret[market_ret["ticker"] == ticker]
            pct  = float(row["daily_pct"].values[0]) if not row.empty and pd.notna(row["daily_pct"].values[0]) else 0.0
            col.metric(
                label=name,
                value=f"{pct:+.2f}%",
                delta=f"{pct:+.2f}%",
                delta_color="normal" if pct >= 0 else "inverse",
            )

        nnh   = int(breadth_today.get("net_new_high",  0)) if not breadth_today.empty else 0
        score = int(breadth_today.get("breadth_score", 0)) if not breadth_today.empty else 0
        score_label, _ = BREADTH_SCORE_LABELS.get(score, ("N/A", "secondary"))
        col_nnh.metric("Net New High",   f"{nnh:+d}")
        col_score.metric("Breadth Score", f"{score}/5 {score_label}")

        spy_ret = market_ret[market_ret["ticker"] == "SPY"]["daily_pct"]
        qqq_ret = market_ret[market_ret["ticker"] == "QQQ"]["daily_pct"]
        iwm_ret = market_ret[market_ret["ticker"] == "IWM"]["daily_pct"]
        spy_pct = float(spy_ret.values[0]) if not spy_ret.empty and pd.notna(spy_ret.values[0]) else 0.0
        qqq_pct = float(qqq_ret.values[0]) if not qqq_ret.empty and pd.notna(qqq_ret.values[0]) else 0.0
        iwm_pct = float(iwm_ret.values[0]) if not iwm_ret.empty and pd.notna(iwm_ret.values[0]) else 0.0

        char_df = pd.DataFrame({
            "ticker":    ["SPY", "QQQ", "IWM"],
            "daily_pct": [spy_pct, qqq_pct, iwm_pct],
        })
        char_text, char_color = _get_market_character(char_df, breadth_today)

        if char_color == "success":
            st.success(char_text)
        elif char_color == "warning":
            st.warning(char_text)
        elif char_color == "danger":
            st.error(char_text)
        else:
            st.info(char_text)

    st.divider()

    # ── 섹션 2 + 3: 섹터 등락 & 테마 ETF Top/Bottom ──────────────────────
    col_sec, col_theme = st.columns(2)

    with col_sec:
        st.markdown("#### 섹터 당일 등락")

        sector_ret = market_ret[market_ret["ticker"].isin(SECTOR_TICKERS)].copy()
        sector_ret = sector_ret.merge(
            sector_summary[["ticker", "sector_name", "rs_rank"]],
            on="ticker", how="left",
        ).dropna(subset=["daily_pct"]).sort_values("daily_pct", ascending=False)

        fig_sec = go.Figure(go.Bar(
            x=sector_ret["daily_pct"],
            y=sector_ret["sector_name"].fillna(sector_ret["ticker"]),
            orientation="h",
            marker_color=sector_ret["daily_pct"].apply(lambda v: "#1D9E75" if v >= 0 else "#E24B4A"),
            text=sector_ret["daily_pct"].apply(lambda v: f"{v:+.2f}%"),
            textposition="outside",
            hovertemplate="%{y}<br>등락: %{x:+.2f}%<extra></extra>",
        ))
        fig_sec.add_vline(x=0, line_color="gray", line_width=0.8)
        fig_sec.update_layout(
            height=380,
            margin=dict(t=10, b=10, l=0, r=60),
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    with col_theme:
        st.markdown("#### 테마 ETF 강세 Top 5 / 약세 Bottom 5")

        theme_tickers = theme_summary["ticker"].unique().tolist()
        theme_ret = _calc_daily_return(rs_full, theme_tickers)
        theme_ret = theme_ret.merge(
            theme_summary[["ticker", "theme", "rs_rank"]].drop_duplicates("ticker"),
            on="ticker", how="left",
        ).dropna(subset=["daily_pct"])

        top5    = theme_ret.nlargest(5,  "daily_pct")
        bottom5 = theme_ret.nsmallest(5, "daily_pct")

        rename_map = {"ticker": "ETF", "theme": "테마", "daily_pct": "오늘 등락(%)", "rs_rank": "RS 순위"}
        pct_cfg    = {"오늘 등락(%)": st.column_config.NumberColumn(format="%+.2f%%")}

        st.markdown("**🚀 강세 Top 5**")
        st.dataframe(
            top5[["ticker", "theme", "daily_pct", "rs_rank"]].rename(columns=rename_map),
            use_container_width=True, hide_index=True, column_config=pct_cfg,
        )
        st.markdown("**📉 약세 Bottom 5**")
        st.dataframe(
            bottom5[["ticker", "theme", "daily_pct", "rs_rank"]].rename(columns=rename_map),
            use_container_width=True, hide_index=True, column_config=pct_cfg,
        )

    st.divider()

    # ── 섹션 4: RS 순위 변동 ─────────────────────────────────────────────
    with st.container():
        st.markdown("#### 🔄 오늘 RS 순위 변동 (전일 대비)")

        rank_chg = _calc_rank_change(
            rs_full,
            tickers=list(filtered_tickers) if filtered_tickers is not None else None,
        )

        col_up, col_down = st.columns(2)
        rename_rank = {"ticker": "종목", "rank_today": "오늘 순위", "rank_prev": "전일 순위", "rank_change": "변동"}

        with col_up:
            st.markdown("**🚀 RS 순위 급등 Top 10**")
            st.dataframe(
                rank_chg[rank_chg["rank_change"] > 0].head(10).rename(columns=rename_rank),
                use_container_width=True, hide_index=True,
                column_config={"변동": st.column_config.NumberColumn(format="+%d")},
            )

        with col_down:
            st.markdown("**📉 RS 순위 급락 Top 10**")
            st.dataframe(
                rank_chg[rank_chg["rank_change"] < 0].head(10).rename(columns=rename_rank),
                use_container_width=True, hide_index=True,
                column_config={"변동": st.column_config.NumberColumn(format="%d")},
            )

    st.divider()

    # ── 섹션 5: 오늘의 주목 신호 ─────────────────────────────────────────
    with st.container():
        st.markdown("#### ⚡ 오늘의 주목 신호")

        col_sig1, col_sig2 = st.columns(2)

        with col_sig1:
            st.markdown("**📈 RS Line 52W 신고가 종목**")
            nh_tickers = _get_rs_line_new_highs(rs_full)
            if filtered_tickers is not None:
                nh_tickers = [t for t in nh_tickers if t in filtered_tickers]
            if nh_tickers:
                nh_df = (
                    today_rs[today_rs["ticker"].isin(nh_tickers)][["ticker", "rs_score", "rs_percentile"]]
                    .sort_values("rs_score", ascending=False)
                    .head(10)
                )
                st.dataframe(
                    nh_df.rename(columns={"ticker": "종목", "rs_score": "RS Score", "rs_percentile": "RS 백분위"}),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "RS Score":  st.column_config.NumberColumn(format="%.3f"),
                        "RS 백분위": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                    },
                )
            else:
                st.info("오늘 RS Line 52W 신고가 종목 없음")

            st.markdown("**🔄 섹터 로테이션 신호 (3일 연속)**")
            rotation = _get_sector_rotation_signals(rs_full, SECTOR_TICKERS, days=3)
            if rotation:
                rot_df = pd.DataFrame(rotation)
                rot_df["방향"] = rot_df["direction"].map({"up": "🟢 RS 상승 중", "down": "🔴 RS 하락 중"})
                st.dataframe(
                    rot_df.rename(columns={"ticker": "섹터", "days": "연속일", "rank_today": "현재 RS 순위"})[
                        ["섹터", "방향", "연속일", "현재 RS 순위"]
                    ],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("3일 연속 방향성 섹터 없음")

        with col_sig2:
            st.markdown("**🩺 오늘 Breadth 신호**")
            if not breadth_today.empty:
                b_items = [
                    ("Net New High",  f"{int(breadth_today.get('net_new_high',  0)):+d}", breadth_today.get("net_new_high",  0) > 0),
                    ("50MA 위 비율",  f"{breadth_today.get('pct_above_50ma',  0):.1f}%", breadth_today.get("pct_above_50ma",  0) >= 50),
                    ("200MA 위 비율", f"{breadth_today.get('pct_above_200ma', 0):.1f}%", breadth_today.get("pct_above_200ma", 0) >= 50),
                    ("Breadth Score", f"{int(breadth_today.get('breadth_score', 0))}/5",  breadth_today.get("breadth_score", 0) >= 3),
                ]
                for label, val, is_ok in b_items:
                    st.markdown(f"{'🟢' if is_ok else '🔴'} **{label}**: {val}")
            else:
                st.info("Breadth 데이터 없음")

            st.markdown("**👀 내일 주목 후보**")
            st.caption("RS 순위 급등 + RS Line 신고가 + RS Percentile 상위 10%")

            rank_top = set(rank_chg[rank_chg["rank_change"] >= 5].head(50)["ticker"].tolist())
            nh_set   = set(nh_tickers)
            _pct_src = today_rs if filtered_tickers is None else today_rs[today_rs["ticker"].isin(filtered_tickers)]
            pct_top  = set(_pct_src[_pct_src["rs_percentile"] >= 90]["ticker"].tolist())

            two_cond   = (rank_top & nh_set) | (rank_top & pct_top) | (nh_set & pct_top)
            three_cond = rank_top & nh_set & pct_top

            if three_cond:
                st.success(f"3개 조건 모두 충족: {', '.join(sorted(three_cond))}")
            if two_cond - three_cond:
                st.info(f"2개 조건 충족: {', '.join(sorted(two_cond - three_cond))}")
            if not two_cond:
                st.info("교차 조건 충족 종목 없음")


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="RS Dashboard", page_icon="📊", layout="wide")

    sector_summary = load_sector_summary()
    theme_summary  = load_theme_summary()
    screener       = load_screener()
    rs_full        = load_rs_full()

    render_sidebar(rs_full)

    st.title("📊 RS Dashboard")

    tab1, tab2, tab_ind, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📅 Daily Summary",
        "📊 섹터 분석",
        "🏭 산업 분석",
        "🎯 테마 분석",
        "🔍 종목 스크리너",
        "🩺 Market Breadth",
        "💼 포트폴리오",
        "👀 Watch List",
    ])

    with tab1:
        breadth_df = load_breadth_metrics()
        _show_daily_summary_tab(
            rs_full=rs_full,
            sector_summary=sector_summary,
            theme_summary=theme_summary,
            breadth_df=breadth_df,
            screener=screener,
        )

    with tab2:
        _show_sector_tab(sector_summary, rs_full)

    with tab_ind:
        _show_industry_tab(screener, rs_full)

    with tab3:
        _show_theme_tab(theme_summary, rs_full)

    with tab4:
        _show_screener_tab(screener, load_technical_indicators())

    with tab5:
        breadth_df = load_breadth_metrics()
        _show_breadth_tab(breadth_df)

    with tab6:
        portfolio_snapshot = load_portfolio_snapshot()
        trade_history      = load_trade_history()
        _show_portfolio_tab(portfolio_snapshot, trade_history, rs_full)

    with tab7:
        tech_indicators = load_technical_indicators()
        _show_watchlist_tab(
            rs_full        = rs_full,
            sector_summary = sector_summary,
            theme_summary  = theme_summary,
            screener       = screener,
            tech_indicators = tech_indicators,
        )


if __name__ == "__main__":
    main()
