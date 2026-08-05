# 실행: streamlit run app.py

from __future__ import annotations

import datetime
import functools
import html
import json
import math
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── 상수 ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
# rs_engine이 소비하는 파이프라인 중간 산출물. ETF 종가/고가/메타데이터는
# output/이 아니라 여기에만 있다 (Performance Table의 52주·YTD 계산에 필요).
INTERMEDIATE_DIR = Path(".intermediate")
DASHBOARD_CSS_PATH = Path("dashboard.css")
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

# RS Line 추이 차트의 기간 선택지 (라벨 → 개월 수).
# 섹터/산업/테마/Watch List 탭이 공유한다 — 탭마다 dict를 따로 두면 값이 어긋나
# 실제로 Watch List의 "1주"가 "1개월"과 같은 차트를 그리는 버그가 있었다.
TREND_PERIOD_MONTHS: dict[str, int] = {"1개월": 1, "3개월": 3, "6개월": 6, "12개월": 12}

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

# 섹터 분석 탭 '시장 RS 순위표'용 지수 ETF (시총가중 vs 동일가중, 대형/중형/소형 비교)
MARKET_INDEX_NAME_MAP: dict[str, str] = {
    "SPY":  "S&P 500 (대형)",
    "RSP":  "S&P 500 동일가중",
    "QQQ":  "NASDAQ 100",
    "QQQE": "NASDAQ 100 동일가중",
    "MDY":  "S&P 400 (중형)",
    "IWM":  "러셀 2000 (소형)",
}
MARKET_INDEX_TICKERS: list[str] = list(MARKET_INDEX_NAME_MAP.keys())

MARKET_TICKER_NAMES: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
}

# Watch List 탭 상수
WATCHLIST_PATH = OUTPUT_DIR / "watchlist.parquet"

# 종합 점수는 _calc_priority_score()가 5개 항목 × 20점으로 계산한다.
# 각 항목의 배점은 해당 _calc_priority_score_*() 함수 안에 있으므로 그쪽을 참조할 것.

# 포트폴리오 탭 상수
RS_SIGNAL_THRESHOLDS: dict[str, dict] = {
    "strong":  {"min_percentile": 80, "slope_min":  0.01},
    "neutral": {"min_percentile": 50, "slope_min": -0.01},
}

PORTFOLIO_COLS_DISPLAY: list[str] = [
    "ticker", "name", "qty", "avg_price_usd", "current_price_usd",
    "pnl_pct", "pnl_usd", "eval_amt_usd", "weight_pct", "holding_days",
    "daily_pnl_pct", "rs_rank", "rs_percentile", "rs_slope", "rs_signal",
    "industry", "dist_pct_50sma", "stop_price_usd",
]

BENCHMARK_TICKERS: set[str] = {"SPY", "QQQ", "IWM"}

# 자산 추이 백필용 종가 프레임 (날짜 × 티커). rs_full(220만 행)을 피벗하는 것보다 싸다.
CLOSE_PRICES_PATH = INTERMEDIATE_DIR / "close.parquet"

# ATR 기반 손절가 = 현재가 - (ATR14 × 이 배수). Minervini류 변동성 손절 관례를 따른다.
ATR_STOP_MULTIPLE = 2.0

# 소수점 잔량(dust) 임계값. 전량 매도 후에도 브로커 정산 과정에서 $1 미만의
# 소수 주식이 남는 일이 있는데(실측: AAPL 0.000594주 = $0.18), 이걸 정상 포지션으로
# 취급하면 보유 테이블을 어지럽히고 RS·손절 경고를 독차지해 진짜 신호를 묻어버린다.
# 평가금액이 이 값 미만이면 대시보드 전반에서 제외한다.
DUST_POSITION_USD = 1.0

# 차트 색상 — 탭 전체에서 일관되게 쓴다.
COLOR_EQUITY = "#378ADD"   # 내 계좌
COLOR_BENCH  = "#EF9F27"   # SPY
COLOR_UP     = "#1D9E75"   # 수익
COLOR_DOWN   = "#E24B4A"   # 손실
COLOR_CASH   = "#9AA7B4"   # 예수금


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
def load_close_prices() -> pd.DataFrame:
    """일별 종가 프레임 (날짜 인덱스 × 티커 컬럼). 개별주 + ETF 통합.

    포트폴리오 자산 추이 백필용 — 토스 API가 과거 잔고를 주지 않으므로
    주문 내역을 이 종가에 리플레이해 과거 equity curve를 복원한다.

    ⚠️ close.parquet에는 개별주만 있고 ETF 종가는 etf_close.parquet에 따로 있다.
    둘을 합치지 않으면 ETF 보유 구간의 평가금액이 통째로 빠져서, 실제로는 없던
    급락이 곡선에 찍히고 MDD가 크게 왜곡된다(XBI 1일 보유로 MDD가 -2.6% 대신
    -18.1%로 나온 사례가 있다).
    """
    try:
        df = pd.read_parquet(CLOSE_PRICES_PATH)
        df.index = pd.to_datetime(df.index)
    except Exception:
        return pd.DataFrame()

    try:
        etf = pd.read_parquet(INTERMEDIATE_DIR / "etf_close.parquet")
        etf.index = pd.to_datetime(etf.index)
        # 개별주 프레임에 이미 있는 티커는 그대로 두고 ETF 전용 컬럼만 덧붙인다.
        new_cols = [c for c in etf.columns if c not in df.columns]
        if new_cols:
            df = df.join(etf[new_cols], how="outer")
    except Exception:
        pass  # ETF 프레임이 없어도 개별주만으로 동작한다

    return df.sort_index()


@st.cache_data
def load_technical_indicators() -> pd.DataFrame:
    try:
        df = pd.read_parquet(OUTPUT_DIR / "technical_indicators.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"technical_indicators.parquet 로드 실패: {e}")
        st.stop()


@st.cache_data
def load_etf_info() -> pd.DataFrame:
    """ETF 메타데이터(Ticker 인덱스 + Category / Theme / Name).

    Performance Table의 섹션 구분(Market / Sector / Theme)과 표시명 출처.
    없으면 빈 DataFrame → 호출부가 안내 메시지를 띄우고 넘어간다.
    """
    try:
        return pd.read_parquet(INTERMEDIATE_DIR / "etf_info.parquet")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_etf_price_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(종가, 고가) 프레임 — 날짜 인덱스 × ETF 티커 컬럼.

    52주 고점 대비/YTD 계산용. technical_indicators(226만 행)를 쓰지 않고
    ETF 전용 프레임(340행 × 59열)을 읽어 비용을 낮춘다.
    """
    try:
        close = pd.read_parquet(INTERMEDIATE_DIR / "etf_close.parquet")
        high = pd.read_parquet(INTERMEDIATE_DIR / "etf_high.parquet")
        close.index = pd.to_datetime(close.index)
        high.index = pd.to_datetime(high.index)
        return close, high
    except Exception:
        return pd.DataFrame(), pd.DataFrame()



# ─── 유틸 함수 ────────────────────────────────────────────────────────────────
def _slope_color(val: object) -> str:
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "background-color: #2ecc71; color: #000000"
    return "background-color: #e74c3c; color: #ffffff"


def _pct_color(val: object, cap: float) -> str:
    """등락률(%) 값을 부호·크기에 따라 초록/빨강 그라데이션으로 강조. cap(해당 컬럼 내 최대 절대값) 이상은 최대 채도."""
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if pd.isna(v) or v == 0 or not cap:
        return ""

    intensity = min(abs(v) / cap, 1.0)
    if v > 0:
        base = (46, 204, 113)   # 초록
        text = "#000000"
    else:
        base = (231, 76, 60)    # 빨강
        text = "#000000" if intensity < 0.5 else "#ffffff"
    r, g, b = (int(255 + (c - 255) * intensity) for c in base)
    return f"background-color: rgb({r},{g},{b}); color: {text}"


def _pct_color_column(col: pd.Series) -> list[str]:
    """컬럼별 최대 절대값을 기준으로 독립적인 색상 범위를 적용."""
    cap = col.abs().max()
    return [_pct_color(v, cap) for v in col]


# ─── Performance Table 컴포넌트 ───────────────────────────────────────────────
# 스타일 대부분은 dashboard.css에 있다. 여기서는 "값에 따라 달라지는" 조건부
# 서식(데이터 바 / 그라데이션 배경)만 인라인 style 문자열로 만든다.

PERF_COLUMNS: list[str] = ["Category", "Daily", "P5D", "Off52High", "YTD"]
PERF_HEADERS: list[str] = [
    "Category", "% Daily Change", "% P5D Change",
    "% Off 52-Wk High", "% Gain/Loss YTD",
]

_BAR_POS = "rgba(29,158,117,0.55)"   # 데이터 바 — 양수(초록)
_BAR_NEG = "rgba(226,75,74,0.55)"    # 데이터 바 — 음수(빨강)

# % Off 52-Wk High 배경 앵커 — 0에 가까울수록 좋다.
# (기준값, RGB) 쌍 사이를 선형 보간한다. 스펙의 -0.2/-2/-6/-12 구간을 따른다.
_OFF52_ANCHORS: list[tuple[float, tuple[int, int, int]]] = [
    (0.0,   (26, 132, 86)),    # 진초록
    (-2.0,  (150, 214, 178)),  # 연초록
    (-4.0,  (255, 255, 255)),  # 흰색 (중립)
    (-6.0,  (247, 190, 190)),  # 연분홍
    (-12.0, (192, 57, 43)),    # 진빨강
]

# % Gain/Loss YTD 배경 채도 상한. 테이블별 max로 정규화하면 값이 모두 작은
# 테이블이 과장돼 보이므로 고정 스케일을 쓴다.
_YTD_CAP = 30.0


def _lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """두 RGB 사이를 t(0~1)로 선형 보간."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))  # type: ignore[return-value]


def _bar_cell_style(v: float, vmax: float) -> str:
    """0을 중앙에 둔 Excel 스타일 데이터 바 (셀 전체 채색이 아님).

    vmax는 같은 테이블 같은 열의 최대 절대값. 양수는 중앙에서 오른쪽으로,
    음수는 중앙에서 왼쪽으로 뻗는다.
    """
    if pd.isna(v) or not vmax:
        return ""
    half = min(abs(float(v)) / vmax, 1.0) * 50.0  # 반폭 기준 최대 50%
    if v >= 0:
        start, end, color = 50.0, 50.0 + half, _BAR_POS
    else:
        start, end, color = 50.0 - half, 50.0, _BAR_NEG
    # background 단축 속성 대신 background-image를 쓴다 — 단축 속성은
    # dashboard.css가 지정한 background-clip을 리셋해 바가 테두리를 덮는다.
    return (
        "background-image:linear-gradient(90deg,"
        f"transparent {start:.1f}%,{color} {start:.1f}%,"
        f"{color} {end:.1f}%,transparent {end:.1f}%)"
    )


def _diverging_bg(v: float) -> str:
    """% Off 52-Wk High 배경 — 0 근처 진초록 → 흰색 → 진빨강."""
    if pd.isna(v):
        return ""
    x = float(v)
    anchors = _OFF52_ANCHORS
    if x >= anchors[0][0]:
        rgb = anchors[0][1]
    elif x <= anchors[-1][0]:
        rgb = anchors[-1][1]
    else:
        rgb = anchors[-1][1]
        for (v1, c1), (v2, c2) in zip(anchors, anchors[1:]):
            if v1 >= x >= v2:
                t = (v1 - x) / (v1 - v2)
                rgb = _lerp_rgb(c1, c2, t)
                break
    return f"background-color:rgb{rgb};color:{_readable_text(rgb)}"


def _signed_bg(v: float, cap: float = _YTD_CAP) -> str:
    """% Gain/Loss YTD 배경 — 양수 초록 / 음수 빨강, |v|가 클수록 진하게."""
    if pd.isna(v):
        return ""
    intensity = min(abs(float(v)) / cap, 1.0) if cap else 0.0
    base = (26, 132, 86) if v >= 0 else (192, 57, 43)
    rgb = _lerp_rgb((255, 255, 255), base, intensity)
    return f"background-color:rgb{rgb};color:{_readable_text(rgb)}"


def _readable_text(rgb: tuple[int, int, int]) -> str:
    """배경 밝기에 따라 검정/흰색 글자를 고른다 (진한 배경에서 가독성 확보)."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if luminance < 140 else "#1f2328"


@st.cache_data
def _read_css(path: str, mtime: float) -> str:
    """mtime을 캐시 키에 넣어 CSS를 고친 뒤 새로고침하면 바로 반영되게 한다."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_dashboard_css() -> str:
    """dashboard.css 내용. 파일이 없어도 앱은 계속 동작한다(스타일만 빠짐)."""
    try:
        mtime = DASHBOARD_CSS_PATH.stat().st_mtime
    except OSError:
        return ""
    return _read_css(str(DASHBOARD_CSS_PATH), mtime)


def _inject_dashboard_css() -> None:
    """dashboard.css를 페이지에 주입한다. main()에서 실행당 1회 호출한다.

    session_state로 "한 번만" 가드하면 안 된다 — Streamlit은 리렌더마다
    엘리먼트 트리를 새로 만들므로, 두 번째 실행부터 <style>이 빠져 스타일이
    통째로 사라진다.
    """
    css = _load_dashboard_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _fmt_pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{float(v):+.2f}"


def render_performance_table(title: str, df: pd.DataFrame) -> None:
    """카드 + Navy 헤더 + Excel 스타일 조건부 서식 테이블을 렌더링한다.

    Args:
        title: 섹션 헤더 문구 (예: "S&P 500 Sub-Market Performance")
        df:    PERF_COLUMNS(Category, Daily, P5D, Off52High, YTD) 5열 DataFrame.
               값은 % 단위 숫자이며 NaN은 "–"로 표시되고 배경색이 붙지 않는다.
               선택 열 "Ticker"가 있으면 Category 앞에 티커 배지로 함께 표시한다.

    같은 함수로 모든 Performance Table을 만든다 — 섹션을 추가하려면
    적절한 5열 DataFrame을 만들어 한 번 더 호출하면 된다.
    스타일은 main()이 주입한 dashboard.css에 의존한다.
    """
    safe_title = html.escape(str(title))
    if df is None or df.empty:
        st.markdown(
            f'<div class="pf-card"><div class="pf-title">{safe_title}</div>'
            f'<div class="pf-wrap"><p class="pf-na">표시할 데이터가 없습니다.</p></div></div>',
            unsafe_allow_html=True,
        )
        return

    has_ticker = "Ticker" in df.columns
    data = df.reindex(columns=PERF_COLUMNS + (["Ticker"] if has_ticker else [])).copy()
    for col in ("Daily", "P5D", "Off52High", "YTD"):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # 데이터 바 스케일은 테이블·열 단위로 독립 산정
    bar_max = {c: float(np.nanmax(np.abs(data[c].to_numpy(dtype=float)))
                        if data[c].notna().any() else 0.0)
               for c in ("Daily", "P5D")}

    head = "".join(f"<th>{html.escape(h)}</th>" for h in PERF_HEADERS)
    rows: list[str] = []
    for _, r in data.iterrows():
        label = html.escape(str(r["Category"]))
        if has_ticker and pd.notna(r["Ticker"]):
            # 티커는 이름 뒤에 "이름: TICKER" 형태로. 콜론까지 span 안에 넣어야
            # CSS가 없을 때도 이름과 티커가 붙어 보이지 않는다.
            label += f'<span class="pf-tkr">: {html.escape(str(r["Ticker"]))}</span>'
        cells = [f'<td class="pf-cat">{label}</td>']
        for col in ("Daily", "P5D"):
            cells.append(_perf_cell(r[col], _bar_cell_style(r[col], bar_max[col])))
        cells.append(_perf_cell(r["Off52High"], _diverging_bg(r["Off52High"])))
        cells.append(_perf_cell(r["YTD"], _signed_bg(r["YTD"])))
        rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f'<div class="pf-card">'
        f'<div class="pf-title">{safe_title}</div>'
        f'<div class="pf-wrap"><table class="pf-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"
        f"</table></div></div>",
        unsafe_allow_html=True,
    )


def _perf_cell(value: float, style: str) -> str:
    cls = "pf-num pf-na" if pd.isna(value) else "pf-num"
    style_attr = f' style="{style}"' if style else ""
    return f'<td class="{cls}"{style_attr}>{_fmt_pct(value)}</td>'


PERF_52W_WINDOW = 252  # 52주 = 약 252 거래일


@st.cache_data
def _build_etf_performance(etf_summary: pd.DataFrame) -> pd.DataFrame:
    """curated ETF 전체의 Performance Table 지표를 계산한다.

    입력 etf_summary는 theme_summary.parquet — 이름과 달리 Theme만이 아니라
    curated ETF 59종 전부(Market/Sector/Industry/Theme/…)를 담고 있다.
    Daily·P5D는 여기서 그대로 가져오고, 52주 고점 대비와 YTD는 ETF 종가/고가
    프레임에서 계산한다(파이프라인 산출물에 없는 지표).

    Returns:
        ticker 인덱스 + Category, Name, Daily, P5D, Off52High, YTD
        (계산 불가 시 빈 DataFrame)
    """
    if etf_summary is None or etf_summary.empty:
        return pd.DataFrame()

    info = load_etf_info()
    close, high = load_etf_price_frames()
    if info.empty or close.empty:
        return pd.DataFrame()

    base = (
        etf_summary.drop_duplicates("ticker")
        .set_index("ticker")
        .reindex(columns=["price_chg_pct", "price_chg_5d_pct"])
        .rename(columns={"price_chg_pct": "Daily", "price_chg_5d_pct": "P5D"})
    )

    latest = close.index.max()
    last_close = close.loc[latest]

    # 52주 고점은 인트라데이 고가 기준 (rs_engine의 52W 계산과 동일한 관례).
    # 고가 프레임이 없으면 종가로 대체한다.
    high_src = high if not high.empty else close
    w52_high = high_src.tail(PERF_52W_WINDOW).max()
    off52 = (last_close / w52_high - 1) * 100

    # YTD 기준선 = 전년도 마지막 거래일 종가. 올해 상장한 ETF는 기준선이 없어
    # NaN이 되고, 렌더러가 "–"로 표시한다.
    prev_year = close[close.index < pd.Timestamp(year=latest.year, month=1, day=1)]
    ytd = (
        (last_close / prev_year.iloc[-1] - 1) * 100
        if not prev_year.empty
        else pd.Series(float("nan"), index=close.columns)
    )

    out = base.join(info[["Category", "Theme", "Name"]], how="left")
    out["Off52High"] = off52.reindex(out.index)
    out["YTD"] = ytd.reindex(out.index)
    # 표시명은 Theme 라벨(예: "Semiconductors")을 쓰고, 없으면 티커로 폴백
    out["Name"] = out["Theme"].fillna(pd.Series(out.index, index=out.index))
    return out[["Category", "Name", "Daily", "P5D", "Off52High", "YTD"]]


def _perf_section_df(
    perf: pd.DataFrame,
    category: str,
    *,
    name_map: dict[str, str] | None = None,
    top_n: int | None = None,
    sort_by: str = "P5D",
) -> pd.DataFrame:
    """_build_etf_performance 결과를 render_performance_table 입력 형태로 변환.

    모든 섹션을 sort_by 열(기본 5일 등락) 내림차순으로 정렬하고, top_n을 주면
    상위 N개만 남긴다. name_map을 주면 ETF 이름 대신 그 라벨을 Category에 쓴다.
    티커는 Ticker 열로 함께 넘겨 렌더러가 배지로 표시한다.
    """
    if perf.empty:
        return pd.DataFrame()
    sub = perf[perf["Category"] == category].copy()
    if sub.empty:
        return pd.DataFrame()

    if name_map:
        sub["Category"] = sub.index.map(lambda t: name_map.get(t, t))
    else:
        sub["Category"] = sub["Name"]
    sub["Ticker"] = sub.index.astype(str)

    sub = sub.sort_values(sort_by, ascending=False, na_position="last")
    if top_n is not None:
        sub = sub.head(top_n)
    return sub[PERF_COLUMNS + ["Ticker"]]


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


def _render_ranking_table(df: pd.DataFrame, height: int = 500, exclude_cols: list[str] | None = None) -> None:
    """rs_slope 색상 강조 + column_config가 적용된 순위표."""
    candidate_cols = [
        "rs_rank", "ticker", "sector_name", "theme", "rs_score", "rs_slope",
        "price_chg_pct", "price_chg_5d_pct",
        "price_chg_1m_pct", "price_chg_3m_pct", "price_chg_6m_pct", "price_chg_1y_pct",
        "rs_1m", "rs_3m", "rs_line_phase",
    ]
    if exclude_cols:
        candidate_cols = [c for c in candidate_cols if c not in exclude_cols]
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

    pct_cols = [
        c for c in (
            "price_chg_pct", "price_chg_5d_pct",
            "price_chg_1m_pct", "price_chg_3m_pct", "price_chg_6m_pct", "price_chg_1y_pct",
        )
        if c in display_df.columns
    ]

    styled = display_df.style
    if "rs_slope" in display_df.columns:
        styled = styled.map(_slope_color, subset=["rs_slope"])
    if pct_cols:
        styled = styled.apply(_pct_color_column, subset=pct_cols, axis=0)  # type: ignore[arg-type]

    if "rs_slope" in display_df.columns or pct_cols:
        st.dataframe(styled, width='stretch', hide_index=True, column_config=col_config, height=height)
    else:
        st.dataframe(display_df, width='stretch', hide_index=True, column_config=col_config, height=height)


# ─── Daily Summary 헬퍼 함수 ──────────────────────────────────────────────────
def _calc_daily_return(rs_full: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """최신 거래일의 실제 주가 등락률(price_chg_pct)을 daily_pct로 반환."""
    df = rs_full[rs_full["ticker"].isin(tickers)]
    if df.empty or "price_chg_pct" not in df.columns:
        return pd.DataFrame(columns=["ticker", "daily_pct"])

    today = df[df["date"] == df["date"].max()]
    return (
        today[["ticker", "price_chg_pct"]]
        .rename(columns={"price_chg_pct": "daily_pct"})
        .reset_index(drop=True)
    )


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


@st.cache_data(show_spinner=False)
def _get_rs_line_new_highs(rs_full: pd.DataFrame, lookback: int = 252) -> list[str]:
    """오늘 RS Line이 최근 lookback일 신고가인 종목 리스트 (SPY 제외, rs_score 내림차순).

    st.tabs는 보이지 않는 탭의 본문도 매 리렌더마다 실행하므로, 어느 탭에서 위젯을
    하나 건드려도 이 함수가 226만 행을 다시 훑는다(실측 0.93초). 입력이 rs_full뿐인
    순수 함수라 캐시하면 해시 비용(0.10초)만 남는다.
    """
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


# ─── 포트폴리오 탭 헬퍼 함수 ──────────────────────────────────────────────────
# 이 섹션의 함수는 전부 @st.cache_data로 감싼다. st.tabs가 숨은 탭 본문도 매
# 리렌더마다 실행하므로, 캐시가 없으면 다른 탭의 슬라이더 하나만 움직여도
# 220만 행짜리 rs_full 필터링과 FIFO 매칭이 통째로 다시 돈다.


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


_CLOSED_TRADE_COLS = [
    "ticker", "open_date", "close_date", "holding_days", "qty",
    "entry_price_usd", "exit_price_usd", "pnl_usd", "pnl_pct", "fees_usd",
]
_OPEN_POSITION_COLS = [
    "ticker", "position_open_date", "holding_days", "qty", "avg_cost_usd",
]


@st.cache_data(show_spinner=False)
def _fifo_match_trades(
    trade_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    체결 내역을 종목별 FIFO로 매칭해 청산 거래와 현재 보유 포지션을 산출.

    토스 Open API는 실현손익도 보유기간도 주지 않는다. 둘 다 여기서 만든다.
    수수료·세금은 매수분은 진입 원가에, 매도분은 청산 대금에서 차감해 반영한다.

    Args:
        trade_history: order_id, date, ticker, side, qty, price_usd,
                       amount_usd, commission_usd, tax_usd 컬럼을 가진 체결 내역

    Returns:
        (closed_trades, open_positions)
        closed_trades:  청산된 매매 1건 = 1행 (매도 체결 × 소진된 매수 로트)
        open_positions: 종목별 현재 잔여 포지션. position_open_date는 수량이
                        0에서 양수로 바뀐 시점 — 중간 추가매수로 리셋되지 않는다.
    """
    empty = (
        pd.DataFrame(columns=_CLOSED_TRADE_COLS),
        pd.DataFrame(columns=_OPEN_POSITION_COLS),
    )
    if trade_history.empty or "side" not in trade_history.columns:
        return empty

    df = trade_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["qty", "price_usd", "amount_usd", "commission_usd", "tax_usd"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 같은 날 여러 건이면 매수를 먼저 처리해야 당일 매매가 음수 재고를 만들지 않는다.
    df["_side_rank"] = (df["side"] == "SELL").astype(int)
    df = df.sort_values(["ticker", "date", "_side_rank"], kind="stable")

    closed: list[dict] = []
    open_rows: list[dict] = []

    for ticker, group in df.groupby("ticker", sort=False):
        # lots: [수량, 주당 원가(수수료 포함), 매수일] 큐
        lots: list[list] = []
        position_open_date: pd.Timestamp | None = None

        for _, row in group.iterrows():
            qty = float(row["qty"])
            if qty <= 0:
                continue
            fees = float(row["commission_usd"]) + float(row["tax_usd"])
            price = float(row["price_usd"])

            if row["side"] == "BUY":
                if not lots:
                    position_open_date = row["date"]
                # 매수 수수료를 주당 원가에 얹는다.
                lots.append([qty, price + (fees / qty if qty else 0.0), row["date"]])
                continue

            # SELL — 오래된 로트부터 소진
            remaining = qty
            # 매도 수수료/세금은 이번 매도 수량에 비례 배분한다.
            fee_per_share = fees / qty if qty else 0.0
            while remaining > 1e-9 and lots:
                lot_qty, lot_cost, lot_date = lots[0]
                matched = min(remaining, lot_qty)

                entry = lot_cost
                exit_ = price - fee_per_share
                pnl = (exit_ - entry) * matched

                closed.append({
                    "ticker":          ticker,
                    "open_date":       lot_date,
                    "close_date":      row["date"],
                    "holding_days":    int((row["date"] - lot_date).days),
                    "qty":             matched,
                    "entry_price_usd": entry,
                    "exit_price_usd":  exit_,
                    "pnl_usd":         pnl,
                    "pnl_pct":         ((exit_ / entry - 1) * 100) if entry else 0.0,
                    "fees_usd":        fee_per_share * matched,
                })

                remaining -= matched
                if lot_qty - matched <= 1e-9:
                    lots.pop(0)
                else:
                    lots[0][0] = lot_qty - matched

            if not lots:
                position_open_date = None

        if lots:
            total_qty = sum(l[0] for l in lots)
            total_cost = sum(l[0] * l[1] for l in lots)
            open_rows.append({
                "ticker":             ticker,
                "position_open_date": position_open_date or lots[0][2],
                "holding_days":       int(
                    (pd.Timestamp.today().normalize() - (position_open_date or lots[0][2])).days
                ),
                "qty":                total_qty,
                "avg_cost_usd":       (total_cost / total_qty) if total_qty else 0.0,
            })

    closed_df = pd.DataFrame(closed, columns=_CLOSED_TRADE_COLS)
    open_df = pd.DataFrame(open_rows, columns=_OPEN_POSITION_COLS)
    return closed_df, open_df


@st.cache_data(show_spinner=False)
def _calc_realized_pnl_by_ticker(closed_trades: pd.DataFrame) -> pd.DataFrame:
    """
    종목별 누적 실현손익 집계 (_fifo_match_trades 결과 기준).

    Returns:
        DataFrame with columns: ticker, total_realized_pnl_usd
    """
    if closed_trades.empty:
        return pd.DataFrame(columns=["ticker", "total_realized_pnl_usd"])

    return (
        closed_trades.groupby("ticker")["pnl_usd"]
        .sum()
        .reset_index()
        .rename(columns={"pnl_usd": "total_realized_pnl_usd"})
    )


@st.cache_data(show_spinner=False)
def _calc_trade_stats(closed_trades: pd.DataFrame) -> dict:
    """
    청산 거래 기반 매매 통계.

    모멘텀 매매에서 '손실은 짧게, 수익은 길게'가 실제로 지켜지는지 보는 지표다.
    승자/패자 평균 보유일이 뒤집혀 있으면 수익을 일찍 자르고 손실을 끌고 있다는 뜻.

    Returns:
        승률/평균손익/손익비/기대값/최대연속손실/평균보유일을 담은 dict.
        거래가 없으면 빈 dict.
    """
    if closed_trades.empty:
        return {}

    wins = closed_trades[closed_trades["pnl_usd"] > 0]
    losses = closed_trades[closed_trades["pnl_usd"] <= 0]

    n = len(closed_trades)
    win_rate = len(wins) / n * 100 if n else 0.0
    avg_win_pct = float(wins["pnl_pct"].mean()) if not wins.empty else 0.0
    avg_loss_pct = float(losses["pnl_pct"].mean()) if not losses.empty else 0.0
    payoff = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct else float("nan")

    # 기대값(%) = 승률×평균수익 - 패률×평균손실
    expectancy = (win_rate / 100) * avg_win_pct + (1 - win_rate / 100) * avg_loss_pct

    # 최대 연속 손실
    max_streak = streak = 0
    for pnl in closed_trades.sort_values("close_date")["pnl_usd"]:
        streak = streak + 1 if pnl <= 0 else 0
        max_streak = max(max_streak, streak)

    return {
        "trade_count":      n,
        "win_rate":         win_rate,
        "avg_win_pct":      avg_win_pct,
        "avg_loss_pct":     avg_loss_pct,
        "payoff":           payoff,
        "expectancy_pct":   expectancy,
        "total_pnl_usd":    float(closed_trades["pnl_usd"].sum()),
        "total_fees_usd":   float(closed_trades["fees_usd"].sum()),
        "max_loss_streak":  max_streak,
        "avg_win_days":     float(wins["holding_days"].mean()) if not wins.empty else 0.0,
        "avg_loss_days":    float(losses["holding_days"].mean()) if not losses.empty else 0.0,
        "best_pct":         float(closed_trades["pnl_pct"].max()),
        "worst_pct":        float(closed_trades["pnl_pct"].min()),
    }


@st.cache_data(show_spinner=False)
def _build_equity_curve(
    trade_history: pd.DataFrame,
    portfolio_snapshot: pd.DataFrame,
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    예수금을 포함한 일별 자산 추이를 만든다.

    토스 Open API에는 과거 잔고 엔드포인트가 없다. 그래서 두 소스를 잇는다:
      - backfill: 주문 내역을 로컬 종가(close.parquet)에 리플레이해 과거를 복원
      - snapshot: 수집기가 매일 저장한 실측 잔고 (있는 날짜는 이쪽이 우선)

    ⚠️ 백필 구간의 한계: 입출금·배당·환전이 반영되지 않는다(API 미제공). 현금은
    '오늘 예수금에서 이후 매매 현금흐름을 거꾸로 되돌린' 추정치다. 스냅샷이
    쌓일수록 실측 구간이 늘어나 정확해진다.

    Returns:
        date, market_value_usd, cash_usd, equity_usd, source 컬럼의 DataFrame
    """
    cols = ["date", "market_value_usd", "cash_usd", "equity_usd", "source"]
    if close_df.empty or trade_history.empty:
        return _snapshot_only_curve(portfolio_snapshot, cols)

    trades = trade_history.copy()
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    for col in ["qty", "amount_usd", "commission_usd", "tax_usd"]:
        if col not in trades.columns:
            trades[col] = 0.0
        trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0.0)

    tickers = [t for t in trades["ticker"].unique() if t in close_df.columns]
    if not tickers:
        return _snapshot_only_curve(portfolio_snapshot, cols)

    start = trades["date"].min()
    prices = close_df.loc[close_df.index >= start, tickers].sort_index()
    if prices.empty:
        return _snapshot_only_curve(portfolio_snapshot, cols)

    # ── 1. 일별 보유수량 = 해당일까지 체결의 부호합 누적
    trades["signed_qty"] = trades["qty"] * trades["side"].map({"BUY": 1, "SELL": -1}).fillna(0)
    qty_by_day = (
        trades[trades["ticker"].isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="signed_qty", aggfunc="sum")
        .reindex(columns=tickers)
    )
    qty_daily = (
        qty_by_day.reindex(prices.index.union(qty_by_day.index))
        .fillna(0.0)
        .cumsum()
        .reindex(prices.index)
        .clip(lower=0)  # 데이터 누락으로 음수 재고가 생겨도 평가금액을 왜곡하지 않게
    )

    market_value = (qty_daily * prices.ffill()).sum(axis=1)

    # ── 2. 현금 역산: cash(t) = cash_today - (t 이후 발생한 순현금흐름)
    #    매수는 현금 유출(-), 매도는 유입(+), 수수료·세금은 항상 유출.
    trades["cash_flow"] = (
        trades["amount_usd"] * trades["side"].map({"BUY": -1, "SELL": 1}).fillna(0)
        - trades["commission_usd"] - trades["tax_usd"]
    )
    flow_by_day = trades.groupby("date")["cash_flow"].sum().reindex(prices.index).fillna(0.0)

    cash_today = 0.0
    if not portfolio_snapshot.empty and "cash_usd" in portfolio_snapshot.columns:
        latest_snap = portfolio_snapshot[
            portfolio_snapshot["date"] == portfolio_snapshot["date"].max()
        ]
        if not latest_snap.empty:
            cash_today = float(latest_snap["cash_usd"].iloc[0])

    # 누적 유입액에서 "그날까지의 누적"을 빼면 그 시점의 현금이 나온다.
    cum_flow = flow_by_day.cumsum()
    cash_series = cash_today - (cum_flow.iloc[-1] - cum_flow)

    curve = pd.DataFrame({
        "date":             prices.index,
        "market_value_usd": market_value.values,
        "cash_usd":         cash_series.values,
        "source":           "backfill",
    })
    curve["equity_usd"] = curve["market_value_usd"] + curve["cash_usd"]

    # ── 3. 스냅샷 실측값이 있는 날짜는 덮어쓴다
    snap_daily = _snapshot_daily_totals(portfolio_snapshot)
    if not snap_daily.empty:
        curve = curve[~curve["date"].isin(snap_daily["date"])]
        curve = pd.concat([curve, snap_daily], ignore_index=True)

    return curve.sort_values("date")[cols].reset_index(drop=True)


def _drop_dust_positions(df: pd.DataFrame) -> pd.DataFrame:
    """평가금액 $1 미만의 소수점 잔량 포지션을 제거한다. (DUST_POSITION_USD 참조)"""
    if df.empty or "eval_amt_usd" not in df.columns:
        return df
    return df[df["eval_amt_usd"] >= DUST_POSITION_USD].copy()


def _snapshot_daily_totals(portfolio_snapshot: pd.DataFrame) -> pd.DataFrame:
    """스냅샷에서 일별 (평가금액, 예수금, 총자산)을 뽑아낸다.

    dust를 여기서도 빼야 백필 구간과 기준이 맞는다. 백필은 주문 내역 FIFO로
    수량을 계산하므로 잔량이 0으로 떨어져 애초에 dust가 잡히지 않는데, 스냅샷만
    포함시키면 이어붙이는 지점에서 곡선이 미세하게 튄다.
    """
    cols = ["date", "market_value_usd", "cash_usd", "equity_usd", "source"]
    if portfolio_snapshot.empty or "eval_amt_usd" not in portfolio_snapshot.columns:
        return pd.DataFrame(columns=cols)

    # 예수금은 계좌 단위 값이라 dust 제거로 사라지면 안 된다 — 먼저 날짜별로 확보한다.
    cash_by_date = portfolio_snapshot.groupby("date")["cash_usd"].first() \
        if "cash_usd" in portfolio_snapshot.columns else pd.Series(dtype=float)

    filtered = _drop_dust_positions(portfolio_snapshot)
    if filtered.empty:
        # 보유가 전부 dust인 날도 예수금만으로 총자산을 남긴다.
        out = cash_by_date.reset_index().rename(columns={"cash_usd": "cash_usd"})
        out["market_value_usd"] = 0.0
        out["cash_usd"] = out["cash_usd"].fillna(0.0)
        out["equity_usd"] = out["cash_usd"]
        out["source"] = "snapshot"
        return out[cols]

    grouped = filtered.groupby("date").agg(
        market_value_usd=("eval_amt_usd", "sum"),
        # cash_usd는 모든 행에 같은 계좌 단위 값이 반복 저장돼 있다.
        cash_usd=("cash_usd", "first"),
    ).reset_index()
    grouped["cash_usd"] = grouped["cash_usd"].fillna(0.0)
    grouped["equity_usd"] = grouped["market_value_usd"] + grouped["cash_usd"]
    grouped["source"] = "snapshot"
    return grouped[cols]


def _snapshot_only_curve(portfolio_snapshot: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """백필 재료가 없을 때 스냅샷만으로 곡선을 만든다."""
    snap = _snapshot_daily_totals(portfolio_snapshot)
    return snap.sort_values("date").reset_index(drop=True) if not snap.empty \
        else pd.DataFrame(columns=cols)


def _calc_drawdown(equity: pd.Series) -> tuple[pd.Series, float]:
    """
    자산 곡선의 낙폭(%) 시계열과 최대낙폭(MDD, %)을 반환.
    """
    if equity.empty:
        return pd.Series(dtype=float), 0.0
    running_max = equity.cummax().replace(0, float("nan"))
    dd = (equity / running_max - 1) * 100
    dd = dd.fillna(0.0)
    return dd, float(dd.min())


@st.cache_data(show_spinner=False)
def _get_rs_warnings(
    portfolio_with_rs: pd.DataFrame,
    rs_full: pd.DataFrame,
) -> list[dict]:
    """
    보유 종목 중 RS·손절 관련 경고/기회 신호 감지.

    Args:
        portfolio_with_rs: _join_rs_to_portfolio() 결과에 기술적 지표까지 합쳐진 DataFrame
        rs_full:           전체 RS 이력 (5거래일 순위 비교용)

    Returns:
        list of {"type": "danger"|"warning"|"success", "ticker": str, "msg": str}
    """
    warnings: list[dict] = []
    if portfolio_with_rs.empty:
        return warnings

    tickers = set(portfolio_with_rs["ticker"])
    df_hist = rs_full[rs_full["ticker"].isin(tickers)][["date", "ticker", "rs_rank"]].copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    dates = sorted(df_hist["date"].unique())

    # 5 거래일 전 순위와 비교하기 위한 두 날짜
    latest_date = dates[-1] if dates else None
    prev_date = dates[-6] if len(dates) >= 6 else (dates[0] if dates else None)

    rank_today = pd.Series(dtype=float)
    rank_prev = pd.Series(dtype=float)
    if latest_date is not None:
        rank_today = df_hist[df_hist["date"] == latest_date].set_index("ticker")["rs_rank"]
    if prev_date is not None:
        rank_prev = df_hist[df_hist["date"] == prev_date].set_index("ticker")["rs_rank"]

    rank_drop = (rank_today - rank_prev).dropna()

    for row in portfolio_with_rs.itertuples(index=False):
        ticker = row.ticker
        rs_slope = getattr(row, "rs_slope", float("nan"))
        rs_pct = getattr(row, "rs_percentile", float("nan"))
        pnl_pct = getattr(row, "pnl_pct", 0.0)
        dist_50 = getattr(row, "dist_pct_50sma", float("nan"))
        price = getattr(row, "current_price_usd", float("nan"))
        stop = getattr(row, "stop_price_usd", float("nan"))

        # 조건 1: RS 순위 급락 (5거래일 전 대비 20위 이상 하락)
        drop = rank_drop.get(ticker)
        if drop is not None and drop >= 20:
            warnings.append({
                "type":   "danger",
                "ticker": ticker,
                "msg":    f"RS 순위 {int(rank_today[ticker])}위, 5일 전 대비 -{int(drop)} 급락",
            })

        # 조건 2: 50일선 이탈 — 추세 매매의 1차 방어선이 무너진 상태
        if pd.notna(dist_50) and dist_50 < 0:
            warnings.append({
                "type":   "danger",
                "ticker": ticker,
                "msg":    f"50일선 {dist_50:+.1f}% 이탈, 추세 훼손",
            })

        # 조건 3: ATR 손절가 근접 (손절선까지 3% 이내)
        if pd.notna(price) and pd.notna(stop) and stop > 0 and price > 0:
            buffer_pct = (price / stop - 1) * 100
            if 0 <= buffer_pct <= 3:
                warnings.append({
                    "type":   "warning",
                    "ticker": ticker,
                    "msg":    f"ATR 손절가 ${stop:,.2f} 근접 (여유 {buffer_pct:.1f}%)",
                })

        # 조건 4: RS Slope 음전환 + 수익 중
        if pd.notna(rs_slope) and rs_slope < 0 and pd.notna(pnl_pct) and pnl_pct > 0:
            warnings.append({
                "type":   "warning",
                "ticker": ticker,
                "msg":    "수익 중이나 RS Slope 음전환, 모멘텀 약화",
            })

        # 조건 5: RS 상위권 + Slope 가속
        if pd.notna(rs_pct) and rs_pct >= 95 and pd.notna(rs_slope) and rs_slope > 0.02:
            warnings.append({
                "type":   "success",
                "ticker": ticker,
                "msg":    "RS 상위권 + Slope 가속, 홀딩 or 추가 매수 검토",
            })

    return warnings


@st.cache_data(show_spinner=False)
def _enrich_holdings(
    today_snap: pd.DataFrame,
    rs_full: pd.DataFrame,
    screener: pd.DataFrame,
    tech_indicators: pd.DataFrame,
    open_positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    보유 종목 테이블에 RS·산업·기술적 지표·보유기간·비중을 모두 붙인다.
    소수점 잔량(dust) 포지션은 여기서 걸러지므로 테이블·KPI·경고에서 모두 빠진다.

    Returns:
        PORTFOLIO_COLS_DISPLAY의 컬럼을 갖춘 DataFrame
    """
    df = _join_rs_to_portfolio(_drop_dust_positions(today_snap), rs_full)
    if df.empty:
        return df

    # 산업/테마 — 스크리너에서 가져온다
    if not screener.empty and "ticker" in screener.columns:
        meta_cols = [c for c in ["ticker", "industry", "theme", "parent_etf"]
                     if c in screener.columns]
        df = df.merge(screener[meta_cols].drop_duplicates("ticker"), on="ticker", how="left")

    # 기술적 지표 최신일 — 50일선 이격도와 ATR 손절가
    if not tech_indicators.empty and "ticker" in tech_indicators.columns:
        tech = tech_indicators.copy()
        tech["date"] = pd.to_datetime(tech["date"])
        latest_tech = tech[tech["date"] == tech["date"].max()]
        tech_cols = [c for c in ["ticker", "sma_50", "atr_14", "dist_pct_50sma"]
                     if c in latest_tech.columns]
        df = df.merge(latest_tech[tech_cols].drop_duplicates("ticker"), on="ticker", how="left")

    if "atr_14" in df.columns:
        df["stop_price_usd"] = df["current_price_usd"] - df["atr_14"] * ATR_STOP_MULTIPLE
    else:
        df["stop_price_usd"] = float("nan")

    # 보유기간 — FIFO로 산출한 현재 포지션 진입일 기준
    if not open_positions.empty:
        df = df.merge(
            open_positions[["ticker", "holding_days", "position_open_date"]],
            on="ticker", how="left",
        )
    else:
        df["holding_days"] = float("nan")
        df["position_open_date"] = pd.NaT

    # 비중 — 주식 평가금액 합계 대비
    total_eval = df["eval_amt_usd"].sum() if "eval_amt_usd" in df.columns else 0.0
    df["weight_pct"] = (df["eval_amt_usd"] / total_eval * 100) if total_eval else 0.0

    return df


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

    industry_strength = 해당 산업의 주도주(상위 10~20종목) 평균 RS 백분위 (0~100).
    백분위가 높을수록(주도 산업) 높은 점수.

    Args:
        industry_strength (float): 산업 주도주(상위 10~20종목) 평균 RS 백분위

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

    점수 구성 (각 20점, 합계 100점):
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

    # ④ 산업 강도 (20점) — 소속 산업 주도주(상위 10~20종목) 평균 RS 백분위
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
    screener: pd.DataFrame,
    tech_indicators: pd.DataFrame,
) -> pd.DataFrame:
    """
    Watch List 종목들의 RS 데이터 + 산업 강도 + 기술적 지표를 종합한 분석 DataFrame 생성.

    rs_full에 없는 ticker는 NaN으로 처리하고 st.warning으로 안내.

    Returns:
        ticker, rs_rank, rs_score, rs_percentile, rs_slope, rs_1m, rs_3m,
        rs_line_phase, rs_line_newhigh, rs_line_today, rs_line_52w_high,
        industry, industry_rs_percentile, industry_rank,
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

    # 종목별 industry 매핑
    screener_idx = (
        screener.drop_duplicates("ticker").set_index("ticker")
        if "ticker" in screener.columns else pd.DataFrame()
    )

    # 산업 강도 lookup: 각 산업 주도주 평균 RS 백분위 (앱 공통 순위 기준 모집단)
    industry_summary = _build_industry_summary(screener)  # 앱 공통 순위 기준
    industry_strength_map: dict = {}
    industry_rank_map: dict = {}
    if not industry_summary.empty:
        industry_strength_map = industry_summary.set_index("industry")["rs_percentile"].to_dict()
        industry_rank_map = industry_summary.set_index("industry")["rs_rank"].to_dict()

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

        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 기술적 지표 join (최신 날짜 기준) — Extension 점수용 atr_multiple
    if tech_indicators is not None and not tech_indicators.empty:
        tech_latest = tech_indicators["date"].max()
        tech_today = tech_indicators[tech_indicators["date"] == tech_latest][
            ["ticker", "atr_multiple"]
        ]
        df = df.merge(tech_today, on="ticker", how="left")
    else:
        df["atr_multiple"] = float("nan")

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
        if st.button("🔄 데이터 새로고침", width='stretch'):
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

    # 섹션 A: 시장 RS 순위표 (주요 지수 ETF 비교)
    with st.container():
        st.markdown("#### 시장 RS 순위표")
        st.caption("주요 지수 ETF 비교 — 대형/중형/소형, 시총가중 vs 동일가중")
        latest_date = pd.to_datetime(rs_full["date"]).max()
        mkt = rs_full[
            (pd.to_datetime(rs_full["date"]) == latest_date)
            & (rs_full["ticker"].isin(MARKET_INDEX_TICKERS))
        ].drop_duplicates("ticker").copy()
        if mkt.empty:
            st.info("시장 지수 ETF 데이터가 없습니다.")
        else:
            mkt["sector_name"] = mkt["ticker"].map(MARKET_INDEX_NAME_MAP).fillna(mkt["ticker"])
            mkt = mkt.sort_values("rs_score", ascending=False).reset_index(drop=True)
            mkt["rs_rank"] = range(1, len(mkt) + 1)
            _render_ranking_table(
                mkt, height=250,
                exclude_cols=["rs_score", "rs_slope", "rs_1m", "rs_3m", "rs_line_phase"],
            )

    # 섹션 B: 섹터 RS 순위표
    with st.container():
        st.markdown("#### 섹터 RS 순위표")
        _render_ranking_table(
            df, height=550,
            exclude_cols=["rs_score", "rs_slope", "rs_1m", "rs_3m", "rs_line_phase"],
        )

    # 섹션 C: RS Line 추이
    with st.container():
        st.markdown("#### 섹터 RS Line 추이")
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
            period_label = st.radio(
                "기간",
                options=list(TREND_PERIOD_MONTHS.keys()),
                index=2,
                horizontal=True,
                key="sector_chart_period",
            )
        selected_months = TREND_PERIOD_MONTHS[period_label]

        if chosen:
            fig_line = _make_trend_chart(
                rs_full, chosen,
                f"섹터 RS Line 추이 (최근 {period_label}, 정규화)",
                months=selected_months,
                normalize=True,
            )
            if fig_line:
                st.plotly_chart(fig_line, width='stretch')
            else:
                st.info("RS Line 데이터가 없습니다.")


# ─── Tab 1.5: 산업 분석 ───────────────────────────────────────────────────────
INDUSTRY_JUNK_LABELS: set[str] = {"", "Unknown", "Other", "Shell Companies", "N/A"}

# ── 산업 '주도주' 표본 규칙 (순위표·추이 차트 공통) ──────────────────────────
# 각 산업에서 RS 백분위 상위 몇 종목을 주도주로 보고 평균을 낼지 정한다.
# 기본은 상위 절반이되, 양끝을 [MIN, MAX]로 자른다:
#   - 하한 10: 표본이 너무 작으면 평균이 출렁인다.
#   - 상한 20: 상한이 없으면 대형 산업은 수백 종목을 평균해 '주도주'가 희석된다.
# 결과적으로 모든 산업이 10~20종목이라는 비슷한 크기의 주도주 집단으로 비교된다.
# (대형 산업일수록 더 엄선된 상위권만 표본이 되므로 순위에서 유리해진다 — 의도된 동작:
#  "주도주를 많이 거느린 산업이 곧 주도 산업"이라는 관점을 따른다.)
INDUSTRY_LEADER_FRAC: float = 0.5
INDUSTRY_LEADER_MIN: int = 10
INDUSTRY_LEADER_MAX: int = 20
# 순위 기준 모집단: 종목 수가 이보다 적은 산업은 주도주 10종목을 채울 수 없어
# 순위에서 제외한다. 앱 전체(산업 탭·스크리너·Watch List)가 이 기준을 공유한다.
INDUSTRY_RANK_MIN_COUNT: int = 10


def _industry_leader_count(n: int) -> int:
    """산업 크기 n에 대한 주도주 표본 개수 — 상위 절반을 [MIN, MAX]로 제한."""
    return min(max(INDUSTRY_LEADER_MIN, math.ceil(n * INDUSTRY_LEADER_FRAC)), INDUSTRY_LEADER_MAX)


@st.cache_data(show_spinner=False)
def _build_industry_summary(
    screener: pd.DataFrame,
    min_count: int = INDUSTRY_RANK_MIN_COUNT,
    exclude_junk: bool = True,
) -> pd.DataFrame:
    """
    개별 종목의 industry 라벨을 기준으로 산업별 RS 지표 집계.

    **기본 인자가 곧 앱 전체의 단일 순위 기준이다.** 산업 탭·스크리너·Watch List가
    모두 인자 없이 호출해 동일한 rs_rank를 얻는다. 인자를 바꿔 호출하면 모집단이
    달라져 rs_rank 숫자의 의미도 달라지므로, 표시 범위를 좁히고 싶을 때는 반환된
    DataFrame을 필터링할 것(rs_rank는 기준값 그대로 유지된다).

    RS 지표 평균은 각 산업의 RS 백분위 상위 _industry_leader_count()개 종목
    (= 주도주)만으로 계산해 하위 종목이 평균을 끌어내리지 않게 한다.
    stock_count는 선별 전 산업 전체 종목 수.

    리렌더마다 여러 번 호출되므로 캐시한다(st.tabs는 숨은 탭 본문도 매번 실행).

    Returns:
        DataFrame[industry, stock_count, rs_score, rs_percentile, rs_slope,
                  pct_positive_slope, rs_rank]  (rs_percentile 내림차순)
    """
    if screener.empty or "industry" not in screener.columns:
        return pd.DataFrame()

    df = screener.copy()
    df = df[df["industry"].notna()]
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

    # 각 산업의 RS 백분위 상위 주도주만 선별 (NaN 백분위는 정렬상 자동 후순위).
    # keep_n이 실제 종목 수보다 커도 within_rank 비교가 전체를 취하므로 안전.
    df_sorted = df.sort_values("rs_percentile", ascending=False)
    grp = df_sorted.groupby("industry")
    within_rank = grp.cumcount()  # 0-based: 백분위 내림차순 순위
    keep_n = grp["ticker"].transform("size").apply(_industry_leader_count)
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
    # 주도주 중 RS Slope 양수 비율
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


def _make_industry_trend_chart(
    rs_full: pd.DataFrame,
    screener: pd.DataFrame,
    industries: list[str],
    title: str,
    months: int | None = None,
) -> go.Figure | None:
    """산업별 RS Line 추이 — 주도주 rs_line을 정규화 후 date별 median 집계.

    순위표와 동일한 _industry_leader_count() 규칙으로 주도주를 선별해, 표와 차트가
    같은 종목 집합을 근거로 삼도록 한다.
    """
    if not industries or "rs_line" not in rs_full.columns:
        return None

    cutoff = rs_full["date"].max() - pd.DateOffset(months=months) if months else None
    frames = []
    for ind in industries:
        # 순위표와 동일한 주도주 선별 (RS 백분위 내림차순)
        ind_members = screener[screener["industry"] == ind]
        if ind_members.empty:
            continue
        if "rs_percentile" in ind_members.columns:
            ind_members = ind_members.sort_values("rs_percentile", ascending=False)
        members = ind_members.head(_industry_leader_count(len(ind_members)))["ticker"].tolist()
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
        labels={"date": "날짜", "rs_line_norm": "RS Line (주도주 정규화 중앙값, 시작=1.0)"},
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
    st.caption(f"개별 종목의 industry 라벨을 기준으로 산업별 RS 강도를 집계 "
               f"(각 산업 RS 백분위 상위 {INDUSTRY_LEADER_MIN}~{INDUSTRY_LEADER_MAX}종목 "
               f"= 주도주 평균 기준).")

    if screener.empty or "industry" not in screener.columns:
        st.info("industry 라벨이 있는 스크리너 데이터가 없습니다.")
        return

    # 순위는 앱 전체 공통 기준으로 한 번만 산정하고, 슬라이더는 표시 범위만 좁힌다.
    # (슬라이더가 집계 모집단을 바꾸면 rs_rank가 재부여돼 "5위"의 의미가 흔들린다)
    summary = _build_industry_summary(screener)
    if summary.empty:
        st.info("집계할 산업 데이터가 없습니다.")
        return

    # ── 컨트롤 ───────────────────────────────────────────────────────────────
    min_count = st.slider(
        "산업당 최소 종목 수",
        min_value=INDUSTRY_RANK_MIN_COUNT, max_value=30, value=INDUSTRY_RANK_MIN_COUNT,
        key="industry_min_count",
        help="표시할 산업만 좁힙니다. 순위는 전체 기준으로 고정되어 바뀌지 않습니다.",
    )

    disp_summary = summary[summary["stock_count"] >= min_count]
    if disp_summary.empty:
        st.warning("조건을 만족하는 산업이 없습니다. 최소 종목 수를 낮춰보세요.")
        return

    st.caption(f"표시 중인 산업: {len(disp_summary)}개 (전체 순위 기준 {len(summary)}개 중)")

    # ── 산업 RS 순위표 ───────────────────────────────────────────────────────
    with st.container():
        st.markdown("#### 산업 RS 순위표")
        disp_candidate = [
            "rs_rank", "industry", "stock_count",
            "rs_percentile", "rs_1m_pct", "rs_3m_pct",
        ]
        disp = disp_summary[[c for c in disp_candidate if c in disp_summary.columns]].copy()
        st.dataframe(
            disp,
            width='stretch', hide_index=True, height=550,
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

    # ── 산업 RS Line 추이 ────────────────────────────────────────────────────
    with st.container():
        st.markdown("#### 산업 RS Line 추이")
        options = disp_summary["industry"].tolist()
        default_ind = disp_summary.sort_values("rs_rank").head(3)["industry"].tolist()
        ctrl_left, ctrl_right = st.columns([6, 4])
        with ctrl_left:
            chosen = st.multiselect(
                "표시할 산업 선택",
                options=options,
                default=default_ind,
                key="industry_multiselect",
            )
        with ctrl_right:
            period_label = st.radio(
                "기간",
                options=list(TREND_PERIOD_MONTHS.keys()),
                index=2,
                horizontal=True,
                key="industry_chart_period",
            )
        if chosen:
            fig_line = _make_industry_trend_chart(
                rs_full, screener, chosen,
                f"산업 RS Line 추이 (최근 {period_label}, 멤버 정규화 중앙값)",
                months=TREND_PERIOD_MONTHS[period_label],
            )
            if fig_line:
                st.plotly_chart(fig_line, width='stretch')
            else:
                st.info("선택한 산업의 RS Line 데이터가 없습니다.")
        else:
            st.info("산업을 1개 이상 선택하세요.")

    # ── 섹션 C: 산업 드릴다운 (멤버 종목) ────────────────────────────────────
    with st.container():
        st.markdown("#### 산업별 구성 종목 드릴다운")
        drill = st.selectbox(
            "산업 선택",
            options=disp_summary["industry"].tolist(),
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
            width='stretch', hide_index=True, height=320,
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

    # 섹션 C: 테마 ETF RS 순위표
    with st.container():
        st.markdown("#### 테마 ETF RS 순위표")
        _render_ranking_table(
            df, height=550,
            exclude_cols=["rs_score", "rs_slope", "rs_1m", "rs_3m", "rs_line_phase"],
        )

    # 섹션 D: RS Line 추이
    with st.container():
        st.markdown("#### 테마 ETF RS Line 추이")
        theme_tickers = df["ticker"].unique().tolist()
        default_t = (
            df.sort_values("rs_rank").head(3)["ticker"].tolist()
            if "rs_rank" in df.columns
            else theme_tickers[:3]
        )
        ctrl_left, ctrl_right = st.columns([6, 4])
        with ctrl_left:
            chosen_themes = st.multiselect(
                "표시할 테마 ETF 선택",
                options=theme_tickers,
                default=default_t,
                key="theme_multiselect",
            )
        with ctrl_right:
            period_label = st.radio(
                "기간",
                options=list(TREND_PERIOD_MONTHS.keys()),
                index=2,
                horizontal=True,
                key="theme_chart_period",
            )
        if chosen_themes:
            fig_theme_line = _make_trend_chart(
                rs_full, chosen_themes,
                f"테마 ETF RS Line 추이 (최근 {period_label}, 정규화)",
                months=TREND_PERIOD_MONTHS[period_label],
                normalize=True,
            )
            if fig_theme_line:
                st.plotly_chart(fig_theme_line, width='stretch')
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
            "업종 순위 상위 N위 이내",
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
        if st.button("💾 기본값 저장", key="screener_save_btn", width='stretch'):
            _save_screener_config()
            st.toast("필터 기본값이 저장되었습니다.", icon="✅")

    # ── 필터 전 사전 계산 (ind_rs_rank, rs_1m_pct, rs_3m_pct) ──────────────
    screener = screener.copy()
    if "industry" in screener.columns:
        _ind_summary = _build_industry_summary(screener)  # 앱 공통 순위 기준
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
            width='stretch',
            hide_index=True,
            column_config=col_config,
            height=600,
        )
    else:
        st.dataframe(
            filtered[display_cols],
            width='stretch',
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
        st.plotly_chart(fig_nnh, width='stretch')

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
        st.plotly_chart(fig_ma, width='stretch')

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
            st.plotly_chart(fig_ad, width='stretch')

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
        st.plotly_chart(fig_score, width='stretch')


# ─── Tab 5: 포트폴리오 ────────────────────────────────────────────────────────

def _show_portfolio_tab(
    portfolio_snapshot: pd.DataFrame,
    trade_history: pd.DataFrame,
    rs_full: pd.DataFrame,
    screener: pd.DataFrame,
    tech_indicators: pd.DataFrame,
    close_df: pd.DataFrame,
) -> None:
    st.subheader("💼 포트폴리오")

    if portfolio_snapshot.empty and trade_history.empty:
        st.info(
            "포트폴리오 데이터가 없습니다.\n\n"
            "`.env`에 토스증권 Open API 자격증명을 넣고 아래 명령어를 실행하세요:\n"
            "```bash\npython toss_collector.py\n```"
        )
        return

    # FIFO 매칭은 실현손익·보유기간·매매통계·산점도가 모두 의존하는 공통 재료다.
    closed_trades, open_positions = _fifo_match_trades(trade_history)
    equity_curve = _build_equity_curve(trade_history, portfolio_snapshot, close_df)

    if portfolio_snapshot.empty:
        st.warning(
            "보유 잔고 스냅샷이 없어 거래 내역만 표시합니다. "
            "`python toss_collector.py`를 실행하면 잔고가 채워집니다."
        )
        today_snap = pd.DataFrame()
    else:
        latest_date = portfolio_snapshot["date"].max()
        today_snap = portfolio_snapshot[portfolio_snapshot["date"] == latest_date].copy()
        st.caption(f"기준일: {latest_date.strftime('%Y-%m-%d')} · 토스증권 Open API")

    holdings = (
        _enrich_holdings(today_snap, rs_full, screener, tech_indicators, open_positions)
        if not today_snap.empty else pd.DataFrame()
    )

    # ── 섹션 1: 계좌 요약 ────────────────────────────────────────────────
    with st.container():
        st.markdown("#### 💰 계좌 요약")

        stock_value = float(holdings["eval_amt_usd"].sum()) if not holdings.empty else 0.0
        cash_usd = (
            float(today_snap["cash_usd"].iloc[0])
            if not today_snap.empty and "cash_usd" in today_snap.columns else 0.0
        )
        total_asset = stock_value + cash_usd
        cash_weight = (cash_usd / total_asset * 100) if total_asset else 0.0

        unreal_pnl = float(holdings["pnl_usd"].sum()) if not holdings.empty else 0.0
        cost_basis = stock_value - unreal_pnl
        unreal_pct = (unreal_pnl / cost_basis * 100) if cost_basis else 0.0

        daily_pnl = (
            float(holdings["daily_pnl_usd"].sum())
            if not holdings.empty and "daily_pnl_usd" in holdings.columns else 0.0
        )
        realized_pnl = float(closed_trades["pnl_usd"].sum()) if not closed_trades.empty else 0.0

        mdd = 0.0
        if not equity_curve.empty:
            _, mdd = _calc_drawdown(equity_curve.set_index("date")["equity_usd"])

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("총 자산", f"${total_asset:,.0f}")
        c2.metric("주식 평가금액", f"${stock_value:,.0f}")
        c3.metric("예수금", f"${cash_usd:,.0f}", delta=f"비중 {cash_weight:.1f}%",
                  delta_color="off")
        c4.metric("평가손익", f"${unreal_pnl:+,.0f}", delta=f"{unreal_pct:+.2f}%")
        c5.metric("실현손익 누계", f"${realized_pnl:+,.0f}",
                  delta=f"{len(closed_trades)}건 청산", delta_color="off")
        c6.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%",
                  delta=f"일간 ${daily_pnl:+,.0f}", delta_color="off")

    st.divider()

    # ── 섹션 2: 보유 포지션 ──────────────────────────────────────────────
    with st.container():
        st.markdown("#### 📋 보유 포지션")

        if holdings.empty:
            st.info("보유 중인 종목이 없습니다.")
        else:
            display_cols = [c for c in PORTFOLIO_COLS_DISPLAY if c in holdings.columns]
            display_df = holdings[display_cols].sort_values("eval_amt_usd", ascending=False)

            def _row_style(row: pd.Series) -> list[str]:
                if row.get("rs_signal", "") == "🔴 RS약세":
                    return ["background-color: rgba(226,75,74,0.08)"] * len(row)
                return [""] * len(row)

            styled = display_df.style.apply(_row_style, axis=1)
            pct_cols = [c for c in ["pnl_pct", "daily_pnl_pct"] if c in display_df.columns]
            if pct_cols:
                styled = styled.apply(_pct_color_column, subset=pct_cols)

            st.dataframe(
                styled,
                width='stretch',
                hide_index=True,
                column_config={
                    "ticker":            st.column_config.TextColumn("종목"),
                    "name":              st.column_config.TextColumn("종목명"),
                    "qty":               st.column_config.NumberColumn("수량",      format="%.0f"),
                    "avg_price_usd":     st.column_config.NumberColumn("평균단가",  format="$%.2f"),
                    "current_price_usd": st.column_config.NumberColumn("현재가",    format="$%.2f"),
                    "pnl_pct":           st.column_config.NumberColumn("수익률",    format="%+.2f%%"),
                    "pnl_usd":           st.column_config.NumberColumn("평가손익",  format="$%+.0f"),
                    "eval_amt_usd":      st.column_config.NumberColumn("평가금액",  format="$%.0f"),
                    "weight_pct":        st.column_config.NumberColumn("비중",      format="%.1f%%"),
                    "holding_days":      st.column_config.NumberColumn("보유기간",  format="%.0f일"),
                    "daily_pnl_pct":     st.column_config.NumberColumn("일간",      format="%+.2f%%"),
                    "rs_rank":           st.column_config.NumberColumn("RS 순위",   format="%d위"),
                    "rs_percentile":     st.column_config.NumberColumn("RS %ile",   format="%.0f"),
                    "rs_slope":          st.column_config.NumberColumn("RS Slope",  format="%.4f"),
                    "rs_signal":         st.column_config.TextColumn("RS 신호"),
                    "industry":          st.column_config.TextColumn("산업"),
                    "dist_pct_50sma":    st.column_config.NumberColumn("50일선 이격", format="%+.1f%%"),
                    "stop_price_usd":    st.column_config.NumberColumn("ATR 손절가", format="$%.2f"),
                },
                height=min(400, 60 + 35 * len(display_df)),
            )
            caption = (
                f"보유기간은 주문 내역 FIFO 기준 현재 포지션 진입일부터입니다. "
                f"ATR 손절가 = 현재가 − ATR(14) × {ATR_STOP_MULTIPLE:g}."
            )
            # 종목이 조용히 사라진 것처럼 보이지 않도록 숨긴 dust를 밝힌다.
            dust = today_snap[today_snap["eval_amt_usd"] < DUST_POSITION_USD]
            if not dust.empty:
                names = ", ".join(dust["ticker"].astype(str))
                caption += (
                    f" · 소수점 잔량 {len(dust)}종목({names}, 합계 "
                    f"${dust['eval_amt_usd'].sum():,.2f})은 숨김."
                )
            st.caption(caption)

    st.divider()

    # ── 섹션 3: 자산 추이 (예수금 포함) ──────────────────────────────────
    with st.container():
        st.markdown("#### 📈 자산 추이")

        if equity_curve.empty or len(equity_curve) < 2:
            st.info("자산 추이를 계산할 데이터가 부족합니다. 거래 내역이 수집되면 표시됩니다.")
        else:
            col_p, col_m = st.columns([3, 2])
            with col_p:
                period_map = {"1개월": 30, "3개월": 90, "6개월": 180, "YTD": None, "전체": -1}
                period_label = st.radio(
                    "기간", options=list(period_map.keys()),
                    index=1, horizontal=True, key="portfolio_period",
                )
            with col_m:
                view_mode = st.radio(
                    "표시", options=["수익률 (%)", "금액 ($)"],
                    index=0, horizontal=True, key="portfolio_view_mode",
                )

            days = period_map[period_label]
            curve = equity_curve.copy()
            max_date = curve["date"].max()
            if days == -1:
                pass
            elif days is None:
                curve = curve[curve["date"] >= pd.Timestamp(f"{max_date.year}-01-01")]
            else:
                curve = curve[curve["date"] >= max_date - pd.Timedelta(days=days)]

            if len(curve) < 2:
                st.info("선택한 기간에 표시할 데이터가 부족합니다.")
            else:
                _render_equity_chart(curve, rs_full, view_mode)

                # 백필 구간이 섞여 있으면 한계를 반드시 알린다.
                if (curve["source"] == "backfill").any():
                    st.caption(
                        "⚠️ 음영 구간은 주문 내역을 과거 종가에 리플레이해 복원한 추정치입니다 "
                        "(토스 API가 과거 잔고를 제공하지 않음). 입출금·배당·환전은 반영되지 "
                        "않으며, 일별 스냅샷이 쌓일수록 실측 구간으로 대체됩니다."
                    )

    st.divider()

    # ── 섹션 4: 포트폴리오 구성 ──────────────────────────────────────────
    with st.container():
        st.markdown("#### 🧩 포트폴리오 구성")

        if holdings.empty:
            st.info("보유 종목이 없어 구성을 표시할 수 없습니다.")
        else:
            col_w, col_i, col_c = st.columns([4, 4, 2])

            with col_w:
                st.markdown("**종목별 비중**")
                wt = holdings[["ticker", "weight_pct"]].sort_values("weight_pct")
                fig_w = go.Figure(go.Bar(
                    x=wt["weight_pct"], y=wt["ticker"], orientation="h",
                    marker_color=COLOR_EQUITY,
                    text=wt["weight_pct"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                    hovertemplate="%{y}<br>비중: %{x:.2f}%<extra></extra>",
                ))
                fig_w.update_layout(
                    height=max(220, 28 * len(wt)),
                    margin=dict(t=10, b=10, l=0, r=50),
                    showlegend=False,
                    xaxis=dict(ticksuffix="%", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                )
                st.plotly_chart(fig_w, width='stretch')

            with col_i:
                st.markdown("**산업별 노출**")
                if "industry" in holdings.columns and holdings["industry"].notna().any():
                    ind = (
                        holdings.groupby(holdings["industry"].fillna("미분류"))["weight_pct"]
                        .sum().reset_index(name="weight_pct")
                        .sort_values("weight_pct")
                    )
                    ind.columns = ["industry", "weight_pct"]
                    fig_i = go.Figure(go.Bar(
                        x=ind["weight_pct"], y=ind["industry"], orientation="h",
                        marker_color=COLOR_EQUITY,
                        text=ind["weight_pct"].apply(lambda v: f"{v:.1f}%"),
                        textposition="outside",
                        hovertemplate="%{y}<br>비중: %{x:.2f}%<extra></extra>",
                    ))
                    fig_i.update_layout(
                        height=max(220, 28 * len(ind)),
                        margin=dict(t=10, b=10, l=0, r=50),
                        showlegend=False,
                        xaxis=dict(ticksuffix="%", showgrid=True,
                                   gridcolor="rgba(128,128,128,0.15)"),
                    )
                    st.plotly_chart(fig_i, width='stretch')
                else:
                    st.info("산업 정보가 없습니다.")

            with col_c:
                st.markdown("**집중도**")
                top3 = holdings.nlargest(3, "weight_pct")["weight_pct"].sum()
                top1 = holdings["weight_pct"].max()
                st.metric("최대 종목 비중", f"{top1:.1f}%")
                st.metric("상위 3종목 합", f"{top3:.1f}%")
                st.metric("보유 종목 수", f"{len(holdings)}개")
                st.metric("현금 비중", f"{cash_weight:.1f}%")

    st.divider()

    # ── 섹션 5: 거래 내역 & 실현손익 ─────────────────────────────────────
    with st.container():
        st.markdown("#### 📒 거래 내역 & 실현손익")

        stats = _calc_trade_stats(closed_trades)
        if stats:
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.metric("청산 거래", f"{stats['trade_count']}건")
            s2.metric("승률", f"{stats['win_rate']:.1f}%")
            s3.metric("평균 수익 / 손실",
                      f"{stats['avg_win_pct']:+.1f}% / {stats['avg_loss_pct']:+.1f}%")
            payoff = stats["payoff"]
            s4.metric("손익비", f"{payoff:.2f}" if pd.notna(payoff) else "—")
            s5.metric("기대값", f"{stats['expectancy_pct']:+.2f}%")
            s6.metric("평균 보유일 (승/패)",
                      f"{stats['avg_win_days']:.0f} / {stats['avg_loss_days']:.0f}일")
            if stats["avg_win_days"] < stats["avg_loss_days"]:
                st.caption(
                    "⚠️ 이긴 거래를 진 거래보다 짧게 들고 있습니다 — "
                    "수익은 일찍 자르고 손실은 오래 끄는 패턴입니다."
                )

        col_hist, col_pnl = st.columns([6, 4])

        with col_hist:
            trade_period = st.selectbox(
                "조회 기간", ["최근 30일", "최근 90일", "전체"], key="trade_period_select",
            )
            trade_df = trade_history.copy()
            if not trade_df.empty:
                trade_df["date"] = pd.to_datetime(trade_df["date"])
                if trade_period != "전체":
                    span = 30 if trade_period == "최근 30일" else 90
                    trade_df = trade_df[
                        trade_df["date"] >= trade_df["date"].max() - pd.Timedelta(days=span)
                    ]
                trade_df = trade_df.sort_values("date", ascending=False)

            hist_cols = [c for c in ["date", "ticker", "side", "qty", "price_usd",
                                     "amount_usd", "commission_usd", "tax_usd"]
                         if c in trade_df.columns]
            st.dataframe(
                trade_df[hist_cols] if not trade_df.empty else pd.DataFrame(columns=hist_cols),
                width='stretch',
                hide_index=True,
                column_config={
                    "date":           st.column_config.DateColumn("날짜",    format="YYYY-MM-DD"),
                    "ticker":         st.column_config.TextColumn("종목"),
                    "side":           st.column_config.TextColumn("구분"),
                    "qty":            st.column_config.NumberColumn("수량",     format="%.0f"),
                    "price_usd":      st.column_config.NumberColumn("단가",     format="$%.2f"),
                    "amount_usd":     st.column_config.NumberColumn("체결금액", format="$%.0f"),
                    "commission_usd": st.column_config.NumberColumn("수수료",   format="$%.2f"),
                    "tax_usd":        st.column_config.NumberColumn("세금",     format="$%.2f"),
                },
                height=350,
            )

        with col_pnl:
            st.markdown("**종목별 누적 실현손익**")
            pnl_df = _calc_realized_pnl_by_ticker(closed_trades)

            if pnl_df.empty:
                st.info("청산된 거래가 없습니다.")
            else:
                pnl_df = pnl_df.sort_values("total_realized_pnl_usd", ascending=True)
                fig_pnl = go.Figure(go.Bar(
                    x=pnl_df["total_realized_pnl_usd"],
                    y=pnl_df["ticker"],
                    orientation="h",
                    marker_color=pnl_df["total_realized_pnl_usd"].apply(
                        lambda v: COLOR_UP if v >= 0 else COLOR_DOWN
                    ),
                    # 부호는 0선 기준 방향과 텍스트 라벨로도 드러난다 (색상 단독 의존 금지).
                    text=pnl_df["total_realized_pnl_usd"].apply(lambda v: f"${v:+,.0f}"),
                    textposition="outside",
                    hovertemplate="%{y}<br>실현손익: $%{x:+,.0f}<extra></extra>",
                ))
                fig_pnl.add_vline(x=0, line_color="gray", line_width=0.8)
                fig_pnl.update_layout(
                    height=350,
                    margin=dict(t=10, b=10, l=0, r=80),
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                )
                st.plotly_chart(fig_pnl, width='stretch')

            if stats:
                st.caption(
                    f"누적 거래비용(수수료·세금) ${stats['total_fees_usd']:,.2f} · "
                    f"최대 연속 손실 {stats['max_loss_streak']}회"
                )

    st.divider()

    # ── 섹션 6: 보유기간 × 수익률 ────────────────────────────────────────
    with st.container():
        st.markdown("#### ⏳ 보유기간 × 수익률")

        if closed_trades.empty and holdings.empty:
            st.info("표시할 포지션이 없습니다.")
        else:
            _render_holding_scatter(closed_trades, holdings)
            st.caption(
                "오른쪽 위(오래 들고 크게 번 거래)가 많아야 추세추종이 제대로 작동한 것입니다. "
                "왼쪽 위에 몰려 있으면 수익을 너무 빨리 실현하고 있다는 신호입니다."
            )

    st.divider()

    # ── 섹션 7: RS·손절 경고 패널 ────────────────────────────────────────
    with st.container():
        st.markdown("#### ⚡ RS·손절 경고 패널")

        warnings = _get_rs_warnings(holdings, rs_full) if not holdings.empty else []

        if not warnings:
            st.success("보유 종목 중 특이 신호 없음")
        else:
            danger_list = [w for w in warnings if w["type"] == "danger"]
            warning_list = [w for w in warnings if w["type"] == "warning"]
            success_list = [w for w in warnings if w["type"] == "success"]

            col_d, col_w, col_s = st.columns(3)

            with col_d:
                st.markdown("**🔴 추세 훼손 / RS 급락**")
                if danger_list:
                    for w in danger_list:
                        st.error(f"**{w['ticker']}** — {w['msg']}")
                else:
                    st.info("해당 없음")

            with col_w:
                st.markdown("**🟡 모멘텀 약화 / 손절 근접**")
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


def _render_equity_chart(curve: pd.DataFrame, rs_full: pd.DataFrame, view_mode: str) -> None:
    """자산 추이 본 차트 + 낙폭 서브차트를 그린다.

    수익률 모드에서만 SPY를 겹친다. 금액 모드에 SPY를 얹으면 축이 두 개가 되는데,
    이중축은 두 계열의 교차점을 임의로 만들어내므로 쓰지 않는다.
    """
    equity = curve.set_index("date")["equity_usd"]
    dd_series, mdd = _calc_drawdown(equity)

    fig = go.Figure()

    grid = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")

    if view_mode.startswith("수익률"):
        base = float(equity.iloc[0]) or 1.0
        cum_pct = (equity / base - 1) * 100

        fig.add_trace(go.Scatter(
            x=cum_pct.index, y=cum_pct.values, name="내 계좌",
            line=dict(color=COLOR_EQUITY, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>내 계좌: %{y:+.2f}%<extra></extra>",
        ))

        spy_ret = _get_spy_cumulative_return(
            rs_full, curve["date"].min(), curve["date"].max()
        )
        if spy_ret is not None and not spy_ret.empty:
            fig.add_trace(go.Scatter(
                x=spy_ret.index, y=spy_ret.values, name="SPY",
                line=dict(color=COLOR_BENCH, width=1.5, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>SPY: %{y:+.2f}%<extra></extra>",
            ))
            excess = float(cum_pct.iloc[-1]) - float(spy_ret.iloc[-1])
        else:
            excess = None

        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=0.8)
        fig.update_layout(yaxis=dict(ticksuffix="%", **grid))

        m1, m2, m3 = st.columns(3)
        m1.metric("기간 수익률", f"{float(cum_pct.iloc[-1]):+.2f}%")
        m2.metric("최대 낙폭", f"{mdd:.2f}%")
        m3.metric("SPY 대비", f"{excess:+.2f}%p" if excess is not None else "—")
    else:
        # 좌축 총자산 + 우축 포지션 비율(주식/총자산).
        # 우축은 데이터에 맞춰 자동 스케일하지 않고 0~100%로 고정한다. 자동 스케일
        # 이중축은 두 선의 교차점을 임의로 만들어 없는 상관관계를 시사하는데,
        # 비율 축을 절대 범위로 못박으면 그 왜곡이 사라진다.
        total = curve["equity_usd"].replace(0, float("nan"))
        position_pct = (curve["market_value_usd"] / total * 100).fillna(0.0).clip(0, 100)

        fig.add_trace(go.Scatter(
            x=curve["date"], y=curve["equity_usd"], name="총자산",
            mode="lines", line=dict(color=COLOR_EQUITY, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>총자산: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=curve["date"], y=position_pct, name="포지션 비율",
            mode="lines", yaxis="y2",
            line=dict(color=COLOR_CASH, width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>포지션 비율: %{y:.1f}%<extra></extra>",
        ))

        fig.update_layout(
            yaxis=dict(title="총자산", tickprefix="$", separatethousands=True, **grid),
            yaxis2=dict(
                title="포지션 비율", overlaying="y", side="right",
                range=[0, 100], ticksuffix="%", showgrid=False,
            ),
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재 총자산", f"${float(equity.iloc[-1]):,.0f}")
        m2.metric("기간 증감", f"${float(equity.iloc[-1] - equity.iloc[0]):+,.0f}")
        m3.metric("포지션 비율", f"{float(position_pct.iloc[-1]):.1f}%",
                  delta=f"현금 {100 - float(position_pct.iloc[-1]):.1f}%",
                  delta_color="off")
        m4.metric("최대 낙폭", f"{mdd:.2f}%")

    _shade_backfill(fig, curve)
    fig.update_layout(
        height=330,
        margin=dict(t=10, b=50, l=0, r=0),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, width='stretch')

    # 낙폭(underwater) 서브차트 — 손실 구간의 깊이와 길이를 같은 x축으로 본다.
    fig_dd = go.Figure(go.Scatter(
        x=dd_series.index, y=dd_series.values, name="낙폭",
        fill="tozeroy", mode="lines",
        line=dict(color=COLOR_DOWN, width=1),
        fillcolor="rgba(226,75,74,0.18)",
        hovertemplate="%{x|%Y-%m-%d}<br>낙폭: %{y:.2f}%<extra></extra>",
    ))
    _shade_backfill(fig_dd, curve)
    fig_dd.update_layout(
        height=150,
        margin=dict(t=5, b=10, l=0, r=0),
        showlegend=False,
        hovermode="x unified",
        yaxis=dict(ticksuffix="%", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_dd, width='stretch')


def _shade_backfill(fig: go.Figure, curve: pd.DataFrame) -> None:
    """백필(추정) 구간에 옅은 음영을 깔아 실측 구간과 구분한다."""
    backfill = curve[curve["source"] == "backfill"]
    if backfill.empty or (curve["source"] == "snapshot").sum() == 0:
        return
    fig.add_vrect(
        x0=backfill["date"].min(), x1=backfill["date"].max(),
        fillcolor="rgba(128,128,128,0.10)", line_width=0,
        annotation_text="추정 구간", annotation_position="top left",
        annotation_font_size=10,
    )


def _render_holding_scatter(closed_trades: pd.DataFrame, holdings: pd.DataFrame) -> None:
    """보유기간(x) × 수익률(y) 산점도. 청산 거래와 보유 중 포지션을 함께 본다."""
    fig = go.Figure()

    if not closed_trades.empty:
        wins = closed_trades[closed_trades["pnl_usd"] > 0]
        losses = closed_trades[closed_trades["pnl_usd"] <= 0]
        # 색뿐 아니라 마커 모양으로도 승/패를 구분한다.
        for subset, name, color, symbol in (
            (wins,   "청산 (수익)", COLOR_UP,   "triangle-up"),
            (losses, "청산 (손실)", COLOR_DOWN, "triangle-down"),
        ):
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["holding_days"], y=subset["pnl_pct"],
                mode="markers", name=name,
                marker=dict(color=color, size=10, symbol=symbol,
                            line=dict(width=1, color="rgba(255,255,255,0.7)")),
                customdata=subset[["ticker", "pnl_usd"]],
                hovertemplate=("%{customdata[0]}<br>보유 %{x:.0f}일<br>"
                               "수익률 %{y:+.2f}%<br>손익 $%{customdata[1]:+,.0f}"
                               "<extra></extra>"),
            ))

    if not holdings.empty and "holding_days" in holdings.columns:
        cur = holdings.dropna(subset=["holding_days", "pnl_pct"])
        if not cur.empty:
            sizes = cur["eval_amt_usd"]
            max_size = float(sizes.max()) or 1.0
            fig.add_trace(go.Scatter(
                x=cur["holding_days"], y=cur["pnl_pct"],
                mode="markers+text", name="보유 중",
                text=cur["ticker"], textposition="top center",
                textfont=dict(size=10),
                marker=dict(
                    color=COLOR_EQUITY, symbol="circle",
                    size=(sizes / max_size * 26 + 10).tolist(),
                    line=dict(width=2, color="rgba(255,255,255,0.8)"),
                ),
                customdata=cur[["ticker", "eval_amt_usd"]],
                hovertemplate=("%{customdata[0]}<br>보유 %{x:.0f}일<br>"
                               "수익률 %{y:+.2f}%<br>평가금액 $%{customdata[1]:,.0f}"
                               "<extra></extra>"),
            ))

    if not fig.data:
        st.info("표시할 포지션이 없습니다.")
        return

    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=0.8)
    fig.update_layout(
        height=380,
        margin=dict(t=10, b=50, l=0, r=0),
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(title="보유기간 (일)", showgrid=True,
                   gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title="수익률 (%)", ticksuffix="%", showgrid=True,
                   gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(fig, width='stretch')


# ─── Tab 6: Watch List ────────────────────────────────────────────────────────

def _show_watchlist_tab(
    rs_full: pd.DataFrame,
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
            if st.button("➕ 추가", width='stretch', key="watchlist_add_btn"):
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
            if st.button("🗑️ 전체삭제", width='stretch', key="watchlist_clear_btn"):
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

    # 데이터 빌드
    wl_df = _build_watchlist_data(tickers, rs_full, screener, tech_indicators)
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
                width='stretch',
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
            period_label = st.radio(
                "RS Line 기간",
                options=list(TREND_PERIOD_MONTHS.keys()),
                index=0,  # 기존 기본값(1개월) 유지
                horizontal=True,
                key="wl_rs_line_period",
            )
            fig_rs = _make_trend_chart(
                rs_full,
                tickers=sorted_df["ticker"].tolist(),
                title="RS Line 비교 (정규화)",
                months=TREND_PERIOD_MONTHS[period_label],
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
                st.plotly_chart(fig_rs, width='stretch')
            else:
                st.info("RS Line 데이터를 찾을 수 없습니다.")

    st.divider()

    # ── 섹션 5: 산업 Strength 비교 ──────────────────────────────────────────
    with st.container():
        st.markdown("#### 🏭 산업 Strength")

        # 산업 RS 강도 바 차트
        st.markdown("**소속 산업 RS 강도 (주도주 평균 백분위)**")
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
            st.plotly_chart(fig_ind, width='stretch')
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
            st.plotly_chart(fig_bubble, width='stretch')
        else:
            st.info("버블 차트는 2개 이상 종목이 필요합니다.")


# ─── Tab 0: Daily Summary ─────────────────────────────────────────────────────
def _show_daily_summary_tab(
    rs_full: pd.DataFrame,
    theme_summary: pd.DataFrame,
    breadth_df: pd.DataFrame,
) -> None:
    st.subheader("📅 Daily Summary")
    st.caption(f"기준일: {pd.to_datetime(rs_full['date']).max().strftime('%Y-%m-%d')} 마감 기준")

    breadth_today = breadth_df.sort_values("date").iloc[-1] if not breadth_df.empty else pd.Series(dtype=float)

    # ── 섹션 1: 시장 요약 헤더 ──────────────────────────────────────────────
    with st.container():
        st.markdown("#### 📊 오늘의 시장 요약")

        market_ret = _calc_daily_return(rs_full, MARKET_TICKERS)

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

    # ── 섹션 2: Performance Tables ─────────────────────────────────────────
    # 세 섹션 모두 동일한 render_performance_table() 컴포넌트를 사용한다.
    # ETF를 추가하면 _perf_section_df(...) 한 줄로 섹션을 더 붙일 수 있다.
    perf = _build_etf_performance(theme_summary)
    if perf.empty:
        st.info(
            "ETF 성과 데이터를 불러오지 못했습니다. "
            "`python data_downloader.py`로 중간 산출물(.intermediate)을 먼저 생성하세요."
        )
        return

    render_performance_table(
        "S&P 500 Sub-Market Performance",
        _perf_section_df(perf, "Market", name_map=MARKET_INDEX_NAME_MAP),
    )
    render_performance_table(
        "S&P 500 Sub-Sector Performance",
        _perf_section_df(perf, "Sector"),
    )
    render_performance_table(
        "Top 10 Thematic Sectors",
        _perf_section_df(perf, "Theme", top_n=10),
    )


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="RS Dashboard", page_icon="📊", layout="wide")
    _inject_dashboard_css()

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
            theme_summary=theme_summary,
            breadth_df=breadth_df,
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
        _show_portfolio_tab(
            portfolio_snapshot = load_portfolio_snapshot(),
            trade_history      = load_trade_history(),
            rs_full            = rs_full,
            screener           = screener,
            tech_indicators    = load_technical_indicators(),
            close_df           = load_close_prices(),
        )

    with tab7:
        _show_watchlist_tab(
            rs_full         = rs_full,
            screener        = screener,
            tech_indicators = load_technical_indicators(),
        )


if __name__ == "__main__":
    main()
