import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from scipy.stats import linregress

# ─── 섹터 ETF 고정 목록 (SPY, QQQ + SPDR 11종) ────────────────────────────────
SECTOR_TICKERS: List[str] = [
    "SPY", "QQQ",
    "XLK", "XLV", "XLF", "XLI", "XLE",
    "XLY", "XLB", "XLRE", "XLP", "XLU", "XLC",
]

SECTOR_NAME_MAP: Dict[str, str] = {
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RSEngine")


def load_prices(parquet_path: str) -> pd.DataFrame:
    """
    parquet 파일을 로드하고 결측치를 처리하여 date x ticker 형태의 pivot table 반환.
    
    Args:
        parquet_path (str): 입력 parquet 파일 경로
        
    Returns:
        pd.DataFrame: 인덱스가 DatetimeIndex이고, 컬럼이 ticker인 close 가격 데이터프레임
    """
    logger.info(f"가격 데이터 로드 중: {parquet_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    
    # Long format (date, ticker, close) 인 경우 Pivot 처리
    if 'close' in df.columns and 'ticker' in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        df = df.pivot(columns='ticker', values='close')
    else:
        # 인덱스가 date인지 확인
        if df.index.name != 'date' and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
        # 컬럼이 MultiIndex인 경우 (yfinance 형식 등)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                df = df.xs('Close', level=0, axis=1)
            elif 'Close' in df.columns.levels[1]:
                df = df.xs('Close', level=1, axis=1)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # 날짜 오름차순 정렬
    df = df.sort_index()
        
    # 0은 yfinance 데이터 오류 — ffill 전에 NaN으로 변환해 inf 전파 차단
    df = df.replace(0, np.nan).ffill().dropna(how='all')
    # 유효 데이터가 전혀 없는 티커(다운로드 실패) 제거 — NaN 컬럼이 rs_score NaN 유발
    df = df.dropna(axis=1, how='all')
    
    logger.info(f"데이터 로드 완료: {df.shape[0]}일, {df.shape[1]}종목")
    return df


def calc_rs(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """
    IBD 방식 RS Score 계산 (SPY 대비 상대값).

    데이터가 없는 구간(신규 상장 등)은 해당 기간의 SPY 수익률로 대체.
    → "상장 전에는 시장 평균만큼 움직였다"는 중립 가정.
    → 모든 종목이 동일한 가중치 구조(2+1+1+1=5)로 비교 가능.

    최소 조건: c63 데이터가 있어야 점수 부여 (63일 미만 신규 상장 제외).
    """
    logger.info("IBD 방식 RS Score 계산 중 (SPY 대체 방식)...")

    if benchmark not in prices.columns:
        raise ValueError(f"벤치마크 '{benchmark}'가 prices에 없습니다.")

    spy = prices[benchmark]

    # ── SPY 가격 비율 (각 기간별 스칼라 Series) ──────────────────
    spy_c63  = spy / spy.shift(63)
    spy_c126 = spy / spy.shift(126)
    spy_c189 = spy / spy.shift(189)
    spy_c252 = spy / spy.shift(252)

    # ── 종목 가격 비율: 없는 구간은 SPY 값으로 대체 ─────────────
    # fillna(spy_cN): 해당 날짜의 SPY 비율로 NaN을 채움
    c63  = (prices / prices.shift(63)  ).apply(lambda col: col.fillna(spy_c63))
    c126 = (prices / prices.shift(126) ).apply(lambda col: col.fillna(spy_c126))
    c189 = (prices / prices.shift(189) ).apply(lambda col: col.fillna(spy_c189))
    c252 = (prices / prices.shift(252) ).apply(lambda col: col.fillna(spy_c252))

    # ── 단기 RS Score 계산 (SPY 대비 정규화) ──────────────────────────────
    c21     = prices / prices.shift(21)
    spy_c21 = spy / spy.shift(21)
    rs_1m   = c21.div(spy_c21, axis=0).where(c21.notna()).replace([np.inf, -np.inf], np.nan)

    # 3개월 RS: pre-fillna orig_c63 / spy_c63 (spy_c63은 위에서 이미 계산됨)
    orig_c63 = prices / prices.shift(63)
    rs_3m    = orig_c63.div(spy_c63, axis=0).where(orig_c63.notna()).replace([np.inf, -np.inf], np.nan)

    # ── 가중 합산 (모든 종목이 동일한 5점 구조) ──────────────────
    stock_raw = 2*c63 + c126 + c189 + c252   # 합계 가중치 = 5

    # ── SPY raw (동일 공식 적용) ──────────────────────────────────
    spy_raw = (2*spy_c63 + spy_c126 + spy_c189 + spy_c252)

    # ── SPY 대비 RS Raw Score ─────────────────────────────────────
    rs_raw = stock_raw.div(spy_raw, axis=0)

    # ── 최소 조건: 원본 c63이 있어야 점수 부여 ───────────────────
    original_c63 = prices / prices.shift(63)
    rs_raw = rs_raw.where(original_c63.notna())
    rs_raw = rs_raw.replace([np.inf, -np.inf], np.nan)

    # ── Long format 변환 ─────────────────────────────────────────
    reset = rs_raw.reset_index()
    reset.rename(columns={reset.columns[0]: 'date'}, inplace=True)
    result = reset.melt(id_vars='date', var_name='ticker', value_name='rs_score')

    def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
        r = wide.reset_index()
        r.rename(columns={r.columns[0]: 'date'}, inplace=True)
        return r.melt(id_vars='date', var_name='ticker', value_name=value_name)

    result = result.merge(_to_long(rs_1m, 'rs_1m'), on=['date', 'ticker'], how='left')
    result = result.merge(_to_long(rs_3m, 'rs_3m'), on=['date', 'ticker'], how='left')

    logger.info(f"RS Score 계산 완료: {result['ticker'].nunique()}개 종목")
    return result


def calc_rs_line(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    """
    RS Line 계산: 종목 종가 / 벤치마크 종가.

    RS Score(순위용)와 달리 RS Line은 추이 차트 시각화 전용.
    값이 상승하면 벤치마크 대비 아웃퍼폼, 하락하면 언더퍼폼.
    IBD에서는 주가 신고가 이전에 RS Line이 먼저 신고가를 치는 것을
    강력한 매수 신호로 해석함.

    Args:
        prices (pd.DataFrame): date x ticker pivot 테이블
        benchmark (str): 기준 티커 (기본값 "SPY")

    Returns:
        pd.DataFrame: Long format (date, ticker, rs_line)
    """
    logger.info(f"RS Line 계산 중 (벤치마크: {benchmark})")
    if benchmark not in prices.columns:
        raise ValueError(f"벤치마크 종목 '{benchmark}'이(가) 데이터에 존재하지 않습니다.")

    rs_line = prices.div(prices[benchmark], axis=0)

    reset = rs_line.reset_index()
    reset.rename(columns={reset.columns[0]: 'date'}, inplace=True)
    return reset.melt(id_vars='date', var_name='ticker', value_name='rs_line')


def calc_rs_line_phase(
    rs_full: pd.DataFrame,
    ema_span: int = 21,
) -> pd.DataFrame:
    """
    RS Line의 21EMA 기반 강세/약세 판단.

    강세 조건 (둘 중 하나):
      1. RS Line > EMA21 (현재 EMA 위)
      2. RS Line <= EMA21 이지만 이탈한 기간이 3일 이하
    약세: EMA 아래로 이탈한 기간이 4일 이상
    데이터 없음: NaN

    Args:
        rs_full (pd.DataFrame): Long format (date, ticker, rs_line 포함)
        ema_span (int): EMA 기간 (기본값 21)

    Returns:
        pd.DataFrame: Long format
        컬럼: date, ticker, rs_line_ema21, rs_line_phase
    """
    if "rs_line" not in rs_full.columns:
        raise ValueError("rs_full에 rs_line 컬럼이 없습니다.")

    logger.info("RS Line Phase 계산 중 (강세/약세, EMA 이탈 3일 유예)...")

    df = rs_full[["date", "ticker", "rs_line"]].copy().sort_values(["ticker", "date"])

    df["rs_line_ema21"] = df.groupby("ticker")["rs_line"].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean()
    )

    df["_above"] = df["rs_line"] > df["rs_line_ema21"]

    def _consec_below(s: pd.Series) -> pd.Series:
        count = 0
        out = []
        for v in s:
            count = 0 if v else count + 1
            out.append(count)
        return pd.Series(out, index=s.index)

    df["_days_below"] = df.groupby("ticker")["_above"].transform(_consec_below)

    def _classify(above, days_below):
        if pd.isna(above) or pd.isna(days_below):
            return "데이터 없음"
        if above or days_below <= 3:
            return "강세"
        return "약세"

    df["rs_line_phase"] = [
        _classify(a, d) for a, d in zip(df["_above"], df["_days_below"])
    ]
    df.drop(columns=["_above", "_days_below"], inplace=True)

    return df[["date", "ticker", "rs_line_ema21", "rs_line_phase"]]


def calc_rs_line_ma_stack(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    short_window: int = 20,
    long_window: int = 50,
) -> pd.DataFrame:
    """
    RS Line의 20/50일 SMA 정배열 판정.

    RS Line 자체에 이동평균을 입혀 단기(20일)·중기(50일) 추세가 모두
    같은 방향으로 정렬됐는지 확인한다. rs_line_phase(21EMA, 3일 유예)가
    최근 며칠만 보는 것과 달리, 최소 수주~수개월간 지속적으로 시장을
    아웃퍼폼하는 추세적 강세주를 선별하는 용도.

    판정 기준:
      정배열:      rs_line > SMA20 > SMA50
      역배열:      rs_line < SMA20 < SMA50
      혼조:        그 외
      데이터 없음: SMA50 산출 불가(상장 50일 미만) 또는 rs_line NaN

    rs_line_stack_days: 연속 정배열 유지 일수 (정배열 아니면 0, 판정 불가 시 NaN)

    Args:
        prices (pd.DataFrame): date x ticker pivot 테이블 (load_prices 출력물)
        benchmark (str): 기준 티커 (기본값 "SPY")
        short_window (int): 단기 SMA 기간 (기본값 20)
        long_window (int): 중기 SMA 기간 (기본값 50)

    Returns:
        pd.DataFrame: Long format
        컬럼: date, ticker, rs_line_sma20, rs_line_sma50,
               rs_line_ma_stack, rs_line_stack_days
    """
    logger.info(
        f"RS Line MA Stack(정배열) 계산 중 "
        f"(SMA{short_window}/{long_window}, 벤치마크: {benchmark})"
    )
    if benchmark not in prices.columns:
        raise ValueError(f"벤치마크 종목 '{benchmark}'이(가) 데이터에 존재하지 않습니다.")

    # wide 연산: 전 종목 동시 rolling — long format groupby.transform 대비 훨씬 빠름
    rs_line = prices.div(prices[benchmark], axis=0)
    sma_s = rs_line.rolling(short_window).mean()   # min_periods 기본값 = window
    sma_l = rs_line.rolling(long_window).mean()

    aligned  = (rs_line > sma_s) & (sma_s > sma_l)
    inverted = (rs_line < sma_s) & (sma_s < sma_l)
    valid    = sma_l.notna() & rs_line.notna()

    # 연속 정배열 일수 (컬럼별 run-length, 벡터 연산)
    # 직전 비정배열 시점의 cumsum을 빼면 현재 연속 구간 길이가 남는다
    _cum   = aligned.cumsum()
    _reset = _cum.where(~aligned).ffill().fillna(0)
    stack_days = (_cum - _reset).where(aligned, 0).where(valid)

    # np.select는 앞 조건 우선 → ~valid를 먼저 두어 NaN이 '혼조'로 새지 않게 한다
    label = np.select(
        [(~valid).to_numpy(), aligned.to_numpy(), inverted.to_numpy()],
        ["데이터 없음", "정배열", "역배열"],
        default="혼조",
    )
    ma_stack = pd.DataFrame(label, index=rs_line.index, columns=rs_line.columns)

    def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
        r = wide.reset_index()
        r.rename(columns={r.columns[0]: 'date'}, inplace=True)
        return r.melt(id_vars='date', var_name='ticker', value_name=value_name)

    result = _to_long(sma_s, 'rs_line_sma20')
    for wide, name in [
        (sma_l,      'rs_line_sma50'),
        (ma_stack,   'rs_line_ma_stack'),
        (stack_days, 'rs_line_stack_days'),
    ]:
        result = result.merge(_to_long(wide, name), on=['date', 'ticker'], how='left')

    return result


def calc_rs_line_nh(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    lookback: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RS Line의 50거래일(약 3개월) 신고가(New High) 여부를 일별 및 최신 스냅샷 기준으로 계산.

    IBD 방법론:
      주가 신고가 이전에 RS Line이 먼저 신고가를 기록하는 종목은 시장 주도주(Leader)의
      가장 핵심적인 선행 신호 중 하나임.

    계산 방식:
      1. wide matrix: rs_line = prices / prices[benchmark]
      2. 50일 롤링 최고치: rolling_max = rs_line.rolling(lookback, min_periods=min(10, lookback)).max()
      3. 당일 신고가 판정: is_nh = (rs_line >= rolling_max * 0.9999) & rs_line.notna()
      4. 3개월(50거래일) 내 신고가 달성 여부: is_nh.rolling(lookback, min_periods=1).max().astype(bool)
      5. 최신 기준 경과일수(days_ago): 최근 50거래일 내 가장 최근 신고가 달성일로부터 경과 일수
         - 0: 당일 신고가
         - 1: 어제(1거래일 전) 신고가
         - ...
         - NaN: 최근 50거래일 내 신고가 기록 없음

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
          - hist_df: Long format (date, ticker, rs_line_nh, rs_line_nh_in_3m) - rs_full 병합용
          - latest_snap_df: DataFrame(ticker, rs_line_nh_today, rs_line_nh_1m, rs_line_nh_3m, rs_line_nh_days_ago) - stock_screener 병합용
    """
    logger.info(f"RS Line New High(50거래일 신고가) 일별 집계 중 (벤치마크: {benchmark}, lookback={lookback})")
    if benchmark not in prices.columns:
        raise ValueError(f"벤치마크 종목 '{benchmark}'이(가) 데이터에 존재하지 않습니다.")

    rs_line = prices.div(prices[benchmark], axis=0)
    rolling_max = rs_line.rolling(lookback, min_periods=min(10, lookback)).max()
    is_nh = (rs_line >= rolling_max * 0.9999) & rs_line.notna()
    nh_in_3m = is_nh.rolling(lookback, min_periods=1).max().astype(bool)

    # Long format 변환
    def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
        r = wide.reset_index()
        r.rename(columns={r.columns[0]: 'date'}, inplace=True)
        return r.melt(id_vars='date', var_name='ticker', value_name=value_name)

    hist_df = _to_long(is_nh, 'rs_line_nh')
    hist_df = hist_df.merge(_to_long(nh_in_3m, 'rs_line_nh_in_3m'), on=['date', 'ticker'], how='left')

    # 최신 스냅샷 지표 계산 (screener용)
    window = min(lookback, len(is_nh))
    sub_nh = is_nh.iloc[-window:].to_numpy()
    dist = np.arange(window - 1, -1, -1)[:, None]
    dist_masked = np.where(sub_nh, dist, 9999)
    min_dist = np.min(dist_masked, axis=0)
    days_ago = pd.Series(np.where(min_dist < 9999, min_dist, np.nan), index=is_nh.columns)

    latest_is_nh = is_nh.iloc[-1] if len(is_nh) > 0 else pd.Series(False, index=is_nh.columns)
    latest_snap_df = pd.DataFrame({
        'ticker': is_nh.columns,
        'rs_line_nh_today': latest_is_nh.values,
        'rs_line_nh_1m': (days_ago <= 21).values,
        'rs_line_nh_3m': days_ago.notna().values,
        'rs_line_nh_days_ago': days_ago.values,
    })

    return hist_df, latest_snap_df


def calc_rs_slope(rs_series: pd.Series, window: int = 20) -> pd.Series:
    """
    최근 window일간의 RS Line 값에 대해 선형회귀 기울기(slope)를 계산 (벡터화 최적화).

    Args:
        rs_series (pd.Series): 단일 종목의 RS Line 시계열 데이터
        window (int): 롤링 윈도우 크기
        
    Returns:
        pd.Series: 계산된 롤링 기울기 시계열
    """
    arr = rs_series.to_numpy().astype(float)
    if len(arr) < window:
        return pd.Series(np.nan, index=rs_series.index)
        
    x = np.arange(window)
    x_dev = x - x.mean()
    weights = x_dev / np.sum(x_dev**2)
    
    padded = np.pad(arr, (window - 1, 0), mode='constant', constant_values=np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(padded, window_shape=window)
    
    with np.errstate(invalid='ignore'):
        slopes = np.dot(windows, weights)
        
    return pd.Series(slopes, index=rs_series.index)


def rank_rs(rs_df: pd.DataFrame, period_col: str = "rs_score") -> pd.DataFrame:
    """
    지정된 RS 컬럼을 기준으로 내림차순 순위 및 백분위 순위를 부여.

    Args:
        rs_df (pd.DataFrame): RS 값들이 포함된 데이터프레임
        period_col (str): 순위를 매길 기준 컬럼명 (기본값: "rs_score")
        
    Returns:
        pd.DataFrame: rs_rank, rs_percentile 컬럼이 추가된 데이터프레임
    """
    logger.info(f"RS 순위 계산 (기준: {period_col})")
    if period_col not in rs_df.columns:
        raise ValueError(f"기준 컬럼 '{period_col}'이(가) 데이터에 존재하지 않습니다.")
        
    df = rs_df.copy()
    
    # 같은 날짜 내에서 순위 평가 (값이 클수록 높은 순위/백분위)
    df['rs_rank'] = df.groupby('date')[period_col].rank(ascending=False, method='min')
    df['rs_percentile'] = df.groupby('date')[period_col].rank(pct=True) * 100

    # 단기 RS 백분위 사전 계산 (rs_percentile과 동일 기준: 같은 날짜 전체 유니버스)
    # → app.py에서 매 렌더마다 .rank(pct=True) 하던 것을 파이프라인에서 1회 계산으로 대체
    for _col in ("rs_1m", "rs_3m"):
        if _col in df.columns:
            df[f"{_col}_pct"] = df.groupby('date')[_col].rank(pct=True) * 100

    return df


def build_sector_summary(rs_df: pd.DataFrame, sector_tickers: List[str], sector_name_map: Dict[str, str] = None) -> pd.DataFrame:
    """
    최신 날짜 기준 섹터 ETF들의 RS 순위 요약 반환.
    
    Args:
        rs_df (pd.DataFrame): RS 전체 데이터
        sector_tickers (List[str]): 필터링할 섹터 ETF 리스트
        sector_name_map (Dict[str, str], optional): 섹터 티커 -> 이름 매핑
        
    Returns:
        pd.DataFrame: 섹터 요약 데이터 (ticker, sector_name, rs_20, rs_60, rs_slope, rs_rank)
    """
    logger.info("섹터 ETF 요약 생성 중")
    latest_date = rs_df['date'].max()
    latest_df = rs_df[rs_df['date'] == latest_date]
    
    sector_df = latest_df[latest_df['ticker'].isin(sector_tickers)].copy()
    if sector_name_map:
        sector_df['sector_name'] = sector_df['ticker'].map(sector_name_map).fillna(sector_df['ticker'])
    else:
        sector_df['sector_name'] = sector_df['ticker']
    
    cols = ['ticker', 'sector_name', 'rs_score', 'rs_slope', 'rs_rank',
            'rs_1m', 'rs_3m', 'rs_1m_pct', 'rs_3m_pct', 'rs_line_phase',
            'price_chg_pct', 'price_chg_5d_pct',
            'price_chg_1m_pct', 'price_chg_3m_pct', 'price_chg_6m_pct', 'price_chg_1y_pct']
    cols = [c for c in cols if c in sector_df.columns]

    return sector_df[cols].sort_values('rs_rank')


def build_theme_summary(rs_df: pd.DataFrame, theme_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    테마별 ETF들의 RS 요약 반환.
    
    Args:
        rs_df (pd.DataFrame): RS 전체 데이터
        theme_map (Dict[str, List[str]]): 테마명 -> 해당 ETF 티커 리스트 매핑
        
    Returns:
        pd.DataFrame: 테마 요약 데이터 (ticker, theme, rs_20, rs_60, rs_slope, rs_rank)
    """
    logger.info("테마 ETF 요약 생성 중")
    latest_date = rs_df['date'].max()
    latest_df = rs_df[rs_df['date'] == latest_date]
    
    records = []
    for theme, tickers in theme_map.items():
        theme_df = latest_df[latest_df['ticker'].isin(tickers)]
        for _, row in theme_df.iterrows():
            record = row.to_dict()
            record['theme'] = theme
            records.append(record)
            
    if not records:
        empty_cols = ['ticker', 'theme', 'rs_20', 'rs_60', 'rs_slope', 'rs_rank']
        return pd.DataFrame(columns=[c for c in empty_cols if c in latest_df.columns or c == 'theme'])
        
    result_df = pd.DataFrame(records)
    cols = ['ticker', 'theme', 'rs_score', 'rs_slope', 'rs_rank',
            'rs_1m', 'rs_3m', 'rs_1m_pct', 'rs_3m_pct', 'rs_line_phase',
            'price_chg_pct', 'price_chg_5d_pct',
            'price_chg_1m_pct', 'price_chg_3m_pct', 'price_chg_6m_pct', 'price_chg_1y_pct']
    cols = [c for c in cols if c in result_df.columns]

    return result_df[cols].sort_values('rs_rank')


def build_stock_screener(
    rs_df: pd.DataFrame,
    etf_holdings: Dict[str, List[str]],
    etf_to_theme: Dict[str, str] = None,
    etf_tickers: set = None,
    prices_df: pd.DataFrame = None,
    volume_df: pd.DataFrame = None,
    market_cap_s: pd.Series = None,
    sector_info_df: pd.DataFrame = None,
    avg_dv_days: int = 20,
    rs_line_nh_snap_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    전체 유니버스 개별 종목의 최신 RS 지표 스냅샷.

    ETF(섹터·테마 포함)는 etf_tickers로 제외하고 순수 개별 종목만 반환.
    ETF 보유 종목 매핑(etf_holdings)과 테마(etf_to_theme)는 선택적 enrichment.
    prices_df / volume_df가 주어지면 latest_price, avg_dollar_volume 컬럼 추가.
    market_cap_s(ticker→시가총액 $)가 주어지면 market_cap 컬럼 추가.
    rs_line_nh_snap_df가 주어지면 rs_line_nh_today, rs_line_nh_1m, rs_line_nh_3m, rs_line_nh_days_ago 추가.
    사전 필터링 없이 전체를 저장 — 필터링은 대시보드(app.py)에서 수행.

    Returns:
        pd.DataFrame: ticker, rs_rank, rs_percentile, rs_score, rs_slope,
                      latest_price, avg_dollar_volume, market_cap, parent_etf, theme,
                      rs_line_ma_stack, rs_line_stack_days,
                      rs_line_nh_today, rs_line_nh_1m, rs_line_nh_3m, rs_line_nh_days_ago
                      (rs_percentile 내림차순)
    """
    logger.info("종목 스크리너 생성 (전체 유니버스 기준)")
    latest_date = rs_df['date'].max()
    stocks_df = rs_df[rs_df['date'] == latest_date].copy()

    # ETF 티커 제외 (섹터 ETF + 테마 ETF 전체)
    if etf_tickers:
        stocks_df = stocks_df[~stocks_df['ticker'].isin(etf_tickers)]

    if stocks_df.empty:
        return pd.DataFrame(
            columns=['ticker', 'rs_rank', 'rs_percentile', 'rs_score', 'rs_slope',
                     'latest_price', 'avg_dollar_volume', 'market_cap', 'parent_etf', 'theme']
        )

    # ── 가격 / 거래대금 enrichment ────────────────────────────────────────────
    if prices_df is not None and not prices_df.empty:
        latest_p: pd.Series = prices_df.iloc[-1]
        stocks_df['latest_price'] = stocks_df['ticker'].map(latest_p)

        if volume_df is not None and not volume_df.empty:
            n = min(avg_dv_days, len(prices_df), len(volume_df))
            common_cols = prices_df.columns.intersection(volume_df.columns)
            p_tail = prices_df[common_cols].tail(n)
            v_tail = volume_df[common_cols].tail(n).replace(0, np.nan)  # 비거래일 제외
            avg_dv: pd.Series = (p_tail * v_tail).mean(skipna=True)
            stocks_df['avg_dollar_volume'] = stocks_df['ticker'].map(avg_dv)
        else:
            stocks_df['avg_dollar_volume'] = np.nan
    else:
        stocks_df['latest_price'] = np.nan
        stocks_df['avg_dollar_volume'] = np.nan

    # ── 시가총액 enrichment ───────────────────────────────────────────────────
    if market_cap_s is not None and not market_cap_s.empty:
        stocks_df['market_cap'] = stocks_df['ticker'].map(market_cap_s)
    else:
        stocks_df['market_cap'] = np.nan

    # ── Industry enrichment ──────────────────────────────────────────────────
    if sector_info_df is not None and not sector_info_df.empty and 'industry' in sector_info_df.columns:
        stocks_df['industry'] = stocks_df['ticker'].map(sector_info_df['industry'])
    else:
        stocks_df['industry'] = np.nan

    # ── ETF 보유 종목 역매핑 (enrichment) ────────────────────────────────────
    ticker_to_etf: Dict[str, str] = {}
    for etf, holdings in etf_holdings.items():
        for t in holdings:
            if t in ticker_to_etf:
                if etf not in ticker_to_etf[t].split(', '):
                    ticker_to_etf[t] += f", {etf}"
            else:
                ticker_to_etf[t] = etf

    stocks_df['parent_etf'] = stocks_df['ticker'].map(ticker_to_etf).fillna("")

    if etf_to_theme:
        def _get_themes(etfs_str: str) -> str:
            if not etfs_str:
                return ""
            themes = list(dict.fromkeys(
                etf_to_theme.get(e, "") for e in etfs_str.split(", ")
                if etf_to_theme.get(e, "")
            ))
            return ", ".join(themes)
        stocks_df['theme'] = stocks_df['parent_etf'].apply(_get_themes)
    else:
        stocks_df['theme'] = ""

    # ── RS Line New High 최신 스냅샷 enrichment ─────────────────────────────
    if rs_line_nh_snap_df is not None and not rs_line_nh_snap_df.empty:
        stocks_df = stocks_df.merge(rs_line_nh_snap_df, on='ticker', how='left')

    cols = ['ticker', 'rs_rank', 'rs_percentile', 'rs_score', 'rs_slope',
            'latest_price', 'avg_dollar_volume', 'market_cap', 'industry', 'parent_etf', 'theme',
            'rs_line_ema21', 'rs_line_phase', 'rs_line_3m_pct', 'rs_line_1m_pct',
            'rs_line_ma_stack', 'rs_line_stack_days', 'rs_line_sma20', 'rs_line_sma50',
            'rs_line_nh', 'rs_line_nh_in_3m',
            'rs_line_nh_today', 'rs_line_nh_1m', 'rs_line_nh_3m', 'rs_line_nh_days_ago',
            'rs_1m', 'rs_3m', 'rs_1m_pct', 'rs_3m_pct',
            'price_chg_pct', 'price_chg_5d_pct',
            'price_chg_1m_pct', 'price_chg_3m_pct', 'price_chg_6m_pct', 'price_chg_1y_pct']
    cols = [c for c in cols if c in stocks_df.columns]

    return stocks_df[cols].sort_values('rs_percentile', ascending=False).reset_index(drop=True)


def build_breadth_metrics(
    prices: pd.DataFrame,
    universe_tickers: List[str],
    ma_windows: List[int] = [50, 200],
    high_df: pd.DataFrame = None,
    low_df: pd.DataFrame = None,
    nh_lookback: int = 260,
) -> pd.DataFrame:
    """
    NYSE / NASDAQ / AMEX 상장 종목 기반 Market Breadth 지표 일별 계산.

    high_df / low_df(인트라데이 고가·저가)가 제공되면 Finviz와 동일한 방식으로
    52W 신고가·신저가를 계산. 없으면 종가 기반 fallback.
    nh_lookback: 52 캘린더 주 ≈ 260 거래일 (기본값)

    컬럼: date, new_high_count, new_low_count, net_new_high,
           pct_above_50ma, pct_above_200ma,
           advances, declines, ad_diff, ad_line, breadth_score
    """
    available = [t for t in universe_tickers if t in prices.columns]
    missing = len(universe_tickers) - len(available)
    if missing > 0:
        logger.warning(f"breadth_tickers 중 {missing}개 종목이 prices에 없어 스킵됩니다.")
    if not available:
        logger.warning("사용 가능한 Market Breadth 종목이 없습니다.")
        return pd.DataFrame()

    p = prices[available].copy()

    # ── 전처리: 0 → NaN 치환 후 전체 NaN 컬럼 제거 ──────────────────────────
    p = p.replace(0, np.nan)
    p = p.dropna(axis=1, how="all")
    logger.info(
        f"Breadth 유니버스: {len(available)}개 → 0 제거 후 유효 {p.shape[1]}개 종목"
    )
    if p.shape[1] == 0:
        logger.warning("전처리 후 유효 종목이 없습니다.")
        return pd.DataFrame()

    result = pd.DataFrame(index=p.index)

    # ── 1. 52W 신고가/신저가 ──────────────────────────────────────────────────
    logger.info("52W 신고가/신저가 계산 중...")
    use_intraday = (
        high_df is not None and low_df is not None
        and not high_df.empty and not low_df.empty
    )
    if use_intraday:
        # Finviz 방식: 인트라데이 high/low 기반, 260 거래일(≈52 캘린더 주) lookback
        h_cols = [t for t in p.columns if t in high_df.columns]
        l_cols = [t for t in p.columns if t in low_df.columns]
        h = high_df[h_cols].reindex(p.index).replace(0, np.nan)
        l = low_df[l_cols].reindex(p.index).replace(0, np.nan)
        high_roll = h.rolling(nh_lookback, min_periods=nh_lookback).max()
        low_roll  = l.rolling(nh_lookback, min_periods=nh_lookback).min()
        new_high_count = (h >= high_roll * 0.9999).sum(axis=1)
        new_low_count  = (l <= low_roll  * 1.0001).sum(axis=1)
        logger.info(f"인트라데이 high/low 기반 52W 계산 (lookback={nh_lookback}일, "
                    f"high {len(h_cols)}종목 / low {len(l_cols)}종목)")
    else:
        # fallback: 종가 기반
        high_roll = p.rolling(nh_lookback, min_periods=nh_lookback).max()
        low_roll  = p.rolling(nh_lookback, min_periods=nh_lookback).min()
        new_high_count = (p >= high_roll * 0.9999).sum(axis=1)
        new_low_count  = (p <= low_roll  * 1.0001).sum(axis=1)
        logger.info(f"종가 기반 52W 계산 (fallback, lookback={nh_lookback}일)")
    result["new_high_count"] = new_high_count
    result["new_low_count"]  = new_low_count
    result["net_new_high"]   = new_high_count - new_low_count

    # ── 2. MA 위 종목 비율 ────────────────────────────────────────────────────
    # 분모를 p.notna()가 아닌 ma.notna()로 산정
    # → 가격은 있지만 MA 산출 이력이 부족한 신규 상장 종목을 분모에서 제외
    # → p > ma: ma가 NaN이면 비교 결과 False → above 자동 제외
    for window in ma_windows:
        logger.info(f"{window}일 이동평균 위 종목 비율 계산 중...")
        ma    = p.rolling(window, min_periods=window).mean()
        above = (p > ma).sum(axis=1)
        total = ma.notna().sum(axis=1).replace(0, np.nan)   # 유효 MA 보유 종목만 분모
        result[f"pct_above_{window}ma"] = (above / total * 100).round(1)

    # ── 3. Advance / Decline Line ─────────────────────────────────────────────
    # diff() 결과 NaN(첫날·데이터 공백): > 0 / < 0 모두 False → 집계 자동 제외
    # 0 → NaN 치환 후이므로 전일 대비 0% 변동 종목이 advances에 합산되지 않음
    logger.info("Advance/Decline Line 계산 중...")
    daily_chg = p.diff()
    advances  = (daily_chg > 0).sum(axis=1)
    declines  = (daily_chg < 0).sum(axis=1)
    ad_diff   = advances - declines
    result["advances"] = advances
    result["declines"] = declines
    result["ad_diff"]  = ad_diff
    result["ad_line"]  = ad_diff.cumsum()

    # ── 4. Breadth Score (0~5점) ──────────────────────────────────────────────
    logger.info("Breadth Score 계산 중...")
    score = pd.Series(0.0, index=p.index)
    score += (result["net_new_high"]    >  0).astype(float)
    score += (result["pct_above_50ma"]  >= 50).astype(float)
    score += (result["pct_above_200ma"] >= 50).astype(float)
    score += (result["ad_line"].diff(20) > 0).astype(float)
    score += (result["pct_above_50ma"]  >= 70).astype(float)   # 강세 보너스
    result["breadth_score"] = score

    result = result.reset_index()
    result.rename(columns={result.columns[0]: "date"}, inplace=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def build_sector_breadth(
    prices: pd.DataFrame,
    sector_holdings: Dict[str, List[str]],
    ma_window: int = 50,
) -> pd.DataFrame:
    """
    섹터별 MA 위 종목 비율 계산 (최신 날짜 기준 스냅샷).

    컬럼: sector, total_stocks, above_ma, pct_above_ma, date
    """
    latest_date = prices.index.max()
    records = []

    for sector, tickers in sector_holdings.items():
        logger.info(f"섹터 Breadth 계산: {sector}")
        available = [t for t in tickers if t in prices.columns]
        if not available:
            continue

        p_sec      = prices[available]
        ma         = p_sec.rolling(ma_window, min_periods=ma_window).mean()
        latest_p   = p_sec.loc[latest_date]
        latest_ma  = ma.loc[latest_date]

        above = int((latest_p > latest_ma).sum())
        total = int(latest_p.notna().sum())
        pct   = round(above / total * 100, 1) if total > 0 else 0.0

        records.append({
            "sector":      sector,
            "total_stocks": total,
            "above_ma":    above,
            "pct_above_ma": pct,
            "date":        pd.Timestamp(latest_date),
        })

    return pd.DataFrame(records)


def build_technical_indicators(
    prices: pd.DataFrame,
    high_df: pd.DataFrame = None,
    low_df: pd.DataFrame = None,
    ma_window: int = 50,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    종목별 기술적 지표 계산.

    계산 지표:
    1. 50SMA: close의 50일 단순이동평균
    2. ATR(14): True Range의 14일 평균
       - high_df/low_df 제공 시: True Range = max(H-L, |H-Cₚ|, |L-Cₚ|)
       - 미제공 시(fallback): |close - close.shift(1)| 근사
    3. ATR_pct: ATR / close * 100  (변동성 %)
    4. 50SMA_dist_pct: (close - 50SMA) / 50SMA * 100  (거리 %)
    5. atr_multiple: 50SMA_dist_pct / ATR_pct
       (50SMA로부터 몇 배 ATR만큼 떨어져 있는지)

    Args:
        prices (pd.DataFrame): date x ticker close pivot (load_prices 출력물)
        high_df (pd.DataFrame): date x ticker 고가 pivot (True Range용, 선택)
        low_df (pd.DataFrame): date x ticker 저가 pivot (True Range용, 선택)
        ma_window (int): 이동평균 기간 (기본값 50)
        atr_window (int): ATR 기간 (기본값 14)

    Returns:
        pd.DataFrame: Long format
        컬럼: date, ticker, close, sma_50, atr_14, atr_pct,
               dist_pct_50sma, atr_multiple
    """
    logger.info("기술적 지표 계산 중 (50SMA, ATR14)...")

    # 1. 50SMA
    sma_50 = prices.rolling(ma_window).mean()

    # 2. ATR(14): True Range 기반 (high/low 있으면) 또는 종가 근사 (fallback)
    use_tr = (
        high_df is not None and low_df is not None
        and not high_df.empty and not low_df.empty
    )
    if use_tr:
        # high/low를 prices의 인덱스·컬럼에 정렬 (없는 값은 NaN)
        h = high_df.reindex(index=prices.index, columns=prices.columns).replace(0, np.nan)
        l = low_df.reindex(index=prices.index, columns=prices.columns).replace(0, np.nan)
        prev_close = prices.shift(1)
        tr1 = h - l                      # 당일 고저 폭
        tr2 = (h - prev_close).abs()     # 전일 종가 대비 고가 갭
        tr3 = (l - prev_close).abs()     # 전일 종가 대비 저가 갭
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_14 = true_range.rolling(atr_window).mean()
        logger.info(f"ATR True Range 기반 계산 (high/low {h.shape[1]}종목)")
    else:
        # fallback: 종가 기반 근사
        daily_range = prices.diff().abs()
        atr_14 = daily_range.rolling(atr_window).mean()
        logger.info("ATR 종가 기반 근사 계산 (fallback, high/low 없음)")

    # 3. ATR_pct (변동성 %)
    atr_pct = atr_14 / prices * 100

    # 4. 50SMA로부터 거리 (%)
    dist_pct_50sma = (prices - sma_50) / sma_50 * 100

    # 5. ATR Multiple (ATR_pct=0이면 ZeroDivision 방지 위해 NaN 처리)
    atr_pct_safe = atr_pct.replace(0, np.nan)
    atr_multiple = dist_pct_50sma / atr_pct_safe

    # inf/-inf → NaN 처리
    atr_pct        = atr_pct.replace([np.inf, -np.inf], np.nan)
    dist_pct_50sma = dist_pct_50sma.replace([np.inf, -np.inf], np.nan)
    atr_multiple   = atr_multiple.replace([np.inf, -np.inf], np.nan)

    # ── Long format 변환: 각 지표를 melt 후 merge ────────────────────────────
    def _melt(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
        reset = df_wide.reset_index()
        reset.rename(columns={reset.columns[0]: "date"}, inplace=True)
        return reset.melt(id_vars="date", var_name="ticker", value_name=value_name)

    result = _melt(prices, "close")
    for df_wide, name in [
        (sma_50,         "sma_50"),
        (atr_14,         "atr_14"),
        (atr_pct,        "atr_pct"),
        (dist_pct_50sma, "dist_pct_50sma"),
        (atr_multiple,   "atr_multiple"),
    ]:
        result = pd.merge(result, _melt(df_wide, name), on=["date", "ticker"], how="left")

    result["date"] = pd.to_datetime(result["date"])
    logger.info(f"기술적 지표 계산 완료: {result['ticker'].nunique()}개 종목")
    return result


def run_pipeline(config: Dict[str, Any]) -> None:
    """
    전체 전처리 파이프라인 실행. (로드 -> RS/기울기/랭크 계산 -> 요약 추출 -> 저장)
    
    Args:
        config (Dict[str, Any]): 설정값 및 메타데이터 딕셔너리
    """
    logger.info("=== RS 엔진 전처리 시작 ===")
    
    close_path = config.get("close_path", ".intermediate/close.parquet")
    etf_close_path = config.get("etf_close_path", ".intermediate/etf_close.parquet")
    etf_info_path = config.get("etf_info_path", ".intermediate/etf_info.parquet")
    etf_holdings_path = config.get("etf_holdings_path", ".intermediate/etf_top_holdings.parquet")
    spy_path = config.get("spy_close_path", ".intermediate/spy_close.parquet")
    
    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark = config.get("benchmark", "SPY")

    # 1. 메타데이터 로드
    logger.info("메타데이터 로드 중")
    etf_info = pd.read_parquet(etf_info_path) if os.path.exists(etf_info_path) else pd.DataFrame()
    etf_holdings_df = pd.read_parquet(etf_holdings_path) if os.path.exists(etf_holdings_path) else pd.DataFrame()
    
    # 섹터 ETF: 고정 13개 (SPY, QQQ + SPDR 11종) — etf_info 내용과 무관하게 고정
    sector_etfs = SECTOR_TICKERS
    sector_name_map = SECTOR_NAME_MAP

    # 테마 맵: etf_info에서 동적 로드
    if not etf_info.empty:
        etf_to_theme = etf_info['Theme'].to_dict()
        theme_map = {}
        for theme, group in etf_info.groupby('Theme'):
            theme_map[theme] = group.index.tolist()
    else:
        theme_map = config.get("theme_map", {})
        etf_to_theme = {}

    etf_holdings = {}
    if not etf_holdings_df.empty:
        for etf, group in etf_holdings_df.groupby('ETF'):
            etf_holdings[etf] = group['HoldingTicker'].tolist()
    if not etf_holdings:
        etf_holdings = config.get("etf_holdings", {})

    # 2. 가격 데이터 로드 및 병합
    logger.info("가격 데이터 로드 및 병합 중")
    prices_stock = load_prices(close_path)
    if os.path.exists(etf_close_path):
        prices_etf = load_prices(etf_close_path)
        overlapping = set(prices_stock.columns).intersection(set(prices_etf.columns))
        prices_etf = prices_etf.drop(columns=list(overlapping))
        prices = pd.concat([prices_stock, prices_etf], axis=1).sort_index().ffill().dropna(how='all')
    else:
        prices = prices_stock

    if benchmark not in prices.columns and os.path.exists(spy_path):
        spy_df = pd.read_parquet(spy_path)
        if 'spy_close' in spy_df.columns:
            prices[benchmark] = spy_df['spy_close']
        else:
            prices[benchmark] = spy_df.iloc[:, 0]

    # 3. RS 계산
    rs_df = calc_rs(prices, benchmark=benchmark)

    # RS Line 계산 및 병합
    logger.info("RS Line 계산 및 병합 중")
    rs_line_df = calc_rs_line(prices, benchmark=benchmark)
    rs_df = pd.merge(rs_df, rs_line_df, on=['date', 'ticker'], how='left')

    # RS Line 단기 백분위 계산 (3개월 63일, 1개월 21일)
    logger.info("RS Line 단기 백분위 계산 중 (3개월·1개월)")
    rs_df = rs_df.sort_values(['ticker', 'date'])
    def _rs_line_range_pct(window: int):
        def _inner(x: pd.Series) -> pd.Series:
            lo = x.rolling(window, min_periods=window).min()
            hi = x.rolling(window, min_periods=window).max()
            return (x - lo) / (hi - lo).replace(0, np.nan) * 100
        return _inner
    rs_df['rs_line_3m_pct'] = rs_df.groupby('ticker')['rs_line'].transform(_rs_line_range_pct(63))
    rs_df['rs_line_1m_pct'] = rs_df.groupby('ticker')['rs_line'].transform(_rs_line_range_pct(21))

    # RS Line Phase 계산 및 rs_full에 merge
    logger.info("RS Line Phase 계산 및 병합 중")
    phase_df = calc_rs_line_phase(rs_df)
    rs_df = pd.merge(
        rs_df,
        phase_df,
        on=["date", "ticker"],
        how="left"
    )

    # RS Line 20/50 SMA 정배열 계산 및 병합
    # (merge로 정렬이 흐트러지지만 바로 아래 4번에서 재정렬됨)
    logger.info("RS Line MA Stack(정배열) 계산 및 병합 중")
    ma_stack_df = calc_rs_line_ma_stack(prices, benchmark=benchmark)
    rs_df = pd.merge(rs_df, ma_stack_df, on=["date", "ticker"], how="left")

    # RS Line New High(50거래일 신고가) 계산 및 병합
    logger.info("RS Line New High(50거래일 신고가) 계산 및 병합 중")
    nh_hist_df, nh_snap_df = calc_rs_line_nh(prices, benchmark=benchmark, lookback=50)
    rs_df = pd.merge(rs_df, nh_hist_df, on=["date", "ticker"], how="left")

    # 4. RS 기울기 계산
    logger.info("RS Line Slope(기울기) 계산 중 (기간: 20일, 기준: RS Line)")
    rs_df = rs_df.sort_values(['ticker', 'date'])
    rs_df['rs_slope'] = rs_df.groupby('ticker')['rs_line'].transform(lambda x: calc_rs_slope(x, window=20))

    # 5. 순위 및 백분위 계산
    rs_df = rank_rs(rs_df, period_col="rs_score")

    # 5-B. 당일 / 주간 실제 주가 등락률 추가
    logger.info("당일·주간 주가 등락률 계산 중")

    def _chg_to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
        r = wide.reset_index()
        r.rename(columns={r.columns[0]: "date"}, inplace=True)
        return r.melt(id_vars="date", var_name="ticker", value_name=value_name)

    rs_df = pd.merge(rs_df, _chg_to_long((prices / prices.shift(1)   - 1) * 100, "price_chg_pct"),
                     on=["date", "ticker"], how="left")
    rs_df = pd.merge(rs_df, _chg_to_long((prices / prices.shift(5)   - 1) * 100, "price_chg_5d_pct"),
                     on=["date", "ticker"], how="left")
    rs_df = pd.merge(rs_df, _chg_to_long((prices / prices.shift(21)  - 1) * 100, "price_chg_1m_pct"),
                     on=["date", "ticker"], how="left")
    rs_df = pd.merge(rs_df, _chg_to_long((prices / prices.shift(63)  - 1) * 100, "price_chg_3m_pct"),
                     on=["date", "ticker"], how="left")
    rs_df = pd.merge(rs_df, _chg_to_long((prices / prices.shift(126) - 1) * 100, "price_chg_6m_pct"),
                     on=["date", "ticker"], how="left")
    rs_df = pd.merge(rs_df, _chg_to_long((prices / prices.shift(252) - 1) * 100, "price_chg_1y_pct"),
                     on=["date", "ticker"], how="left")

    # 6. 요약 및 스크리닝
    sector_summary = build_sector_summary(rs_df, sector_etfs, sector_name_map)
    theme_summary = build_theme_summary(rs_df, theme_map)
    # 제외할 ETF 티커: 섹터 ETF + etf_info 전체 (테마 ETF 포함)
    etf_tickers_all = set(sector_etfs) | (set(etf_info.index.tolist()) if not etf_info.empty else set())

    # 거래량 로드 (latest_price / avg_dollar_volume 계산용)
    volume_path = config.get("volume_path", ".intermediate/volume.parquet")
    vol_df = None
    if os.path.exists(volume_path):
        try:
            vol_df = pd.read_parquet(volume_path)
            vol_df.index = pd.to_datetime(vol_df.index)
            vol_df = vol_df.sort_index()
            logger.info(f"거래량 데이터 로드 완료: {vol_df.shape}")
        except Exception as e:
            logger.warning(f"거래량 데이터 로드 실패, avg_dollar_volume 생략: {e}")

    # 시가총액 로드
    market_cap_path = config.get("market_cap_path", ".intermediate/market_caps.parquet")
    market_cap_s = None
    if os.path.exists(market_cap_path):
        try:
            mc_df = pd.read_parquet(market_cap_path)
            # _save_series 로 저장 시 단일 컬럼 DataFrame → Series로 변환
            market_cap_s = mc_df.iloc[:, 0]
            logger.info(f"시가총액 데이터 로드 완료: {len(market_cap_s)}개 종목")
        except Exception as e:
            logger.warning(f"시가총액 데이터 로드 실패, market_cap 생략: {e}")

    # sector_info 로드 (industry 컬럼용)
    sector_info_path = config.get("sector_info_path", ".intermediate/sector_info.parquet")
    sector_info_df = None
    if os.path.exists(sector_info_path):
        try:
            sector_info_df = pd.read_parquet(sector_info_path)
            logger.info(f"sector_info 로드 완료: {len(sector_info_df)}개 종목")
        except Exception as e:
            logger.warning(f"sector_info 로드 실패, industry 생략: {e}")

    screener = build_stock_screener(
        rs_df, etf_holdings,
        etf_to_theme=etf_to_theme,
        etf_tickers=etf_tickers_all,
        prices_df=prices_stock,
        volume_df=vol_df,
        market_cap_s=market_cap_s,
        sector_info_df=sector_info_df,
        rs_line_nh_snap_df=nh_snap_df,
    )

    # 6-B. Breadth 지표 계산 (NYSE / NASDAQ / AMEX 종목 기반)
    breadth_tickers = config.get("breadth_tickers", [])
    sector_holdings = config.get("etf_holdings", {})

    if breadth_tickers:
        # 인트라데이 high / low 로드 (Finviz 방식 52W 계산용)
        breadth_high_df = None
        breadth_low_df  = None
        for _key, _var_name in [("high_path", "high"), ("low_path", "low")]:
            _path = config.get(_key, f".intermediate/{_var_name}.parquet")
            if os.path.exists(_path):
                try:
                    _df = pd.read_parquet(_path)
                    _df.index = pd.to_datetime(_df.index)
                    _df = _df.sort_index().replace(0, np.nan)
                    if _var_name == "high":
                        breadth_high_df = _df
                    else:
                        breadth_low_df = _df
                    logger.info(f"{_var_name}.parquet 로드 완료: {_df.shape}")
                except Exception as _e:
                    logger.warning(f"{_var_name}.parquet 로드 실패, fallback 사용: {_e}")

        breadth_df        = build_breadth_metrics(
            prices, breadth_tickers,
            high_df=breadth_high_df,
            low_df=breadth_low_df,
        )
        sector_breadth_df = build_sector_breadth(prices, sector_holdings)
        breadth_df.to_parquet(output_dir / "breadth_metrics.parquet", index=False)
        sector_breadth_df.to_parquet(output_dir / "sector_breadth.parquet", index=False)
        logger.info("Breadth 지표 저장 완료")
    else:
        logger.warning("breadth_tickers가 비어있어 Breadth 계산을 건너뜁니다.")

    # 6-C. 기술적 지표 계산 (50SMA, ATR14 → Extension 점수용)
    logger.info("기술적 지표 계산 중...")
    # 인트라데이 high / low 로드 (True Range 기반 ATR용)
    tech_high_df = None
    tech_low_df  = None
    for _key, _var_name in [("high_path", "high"), ("low_path", "low")]:
        _path = config.get(_key, f".intermediate/{_var_name}.parquet")
        if os.path.exists(_path):
            try:
                _df = pd.read_parquet(_path)
                _df.index = pd.to_datetime(_df.index)
                _df = _df.sort_index()
                if _var_name == "high":
                    tech_high_df = _df
                else:
                    tech_low_df = _df
            except Exception as _e:
                logger.warning(f"{_var_name}.parquet 로드 실패, ATR 종가 근사 사용: {_e}")
    tech_df = build_technical_indicators(prices, high_df=tech_high_df, low_df=tech_low_df)
    # 저장 컬럼: date, ticker, close, sma_50, atr_14,
    #            atr_pct, dist_pct_50sma, atr_multiple
    tech_df.to_parquet(output_dir / "technical_indicators.parquet", index=False)
    logger.info("technical_indicators.parquet 저장 완료")

    # atr_multiple → screener에 최신일 기준 merge
    _snap_date = rs_df['date'].max()
    _atr_snap  = (
        tech_df[tech_df['date'] == _snap_date][['ticker', 'atr_multiple']]
        .drop_duplicates('ticker')
    )
    screener = pd.merge(screener, _atr_snap, on='ticker', how='left')

    # 7. 저장
    logger.info("결과물 파켓(parquet) 저장 중")
    sector_summary.to_parquet(output_dir / "sector_summary.parquet", index=False)
    theme_summary.to_parquet(output_dir / "theme_summary.parquet", index=False)
    screener.to_parquet(output_dir / "stock_screener.parquet", index=False)
    # rs_full 저장 컬럼:
    # date, ticker, rs_score, rs_line,
    # rs_slope, rs_rank, rs_percentile,
    # rs_line_ema21, rs_line_phase,
    # rs_line_sma20, rs_line_sma50, rs_line_ma_stack, rs_line_stack_days,
    # rs_line_nh, rs_line_nh_in_3m,
    # rs_1m, rs_3m, rs_1m_pct, rs_3m_pct, price_chg_pct, price_chg_5d_pct
    rs_df.to_parquet(output_dir / "rs_full.parquet", index=False)
    
    logger.info("=== RS 엔진 전처리 완료 ===")

if __name__ == "__main__":
    # NYSE / NASDAQ / AMEX 티커 로드 (data_downloader.py가 저장한 universe 활용)
    # otherlisted.txt 거래소 코드: N=NYSE, A=NYSE MKT(AMEX), P=NYSE Arca
    _NYSE_NASDAQ_AMEX = {"NASDAQ", "N", "A", "P"}
    _universe_path = ".intermediate/universe.parquet"
    _breadth_tickers: List[str] = []
    if os.path.exists(_universe_path):
        try:
            _uni = pd.read_parquet(_universe_path)
            if "Exchange" in _uni.columns:
                _uni = _uni[_uni["Exchange"].isin(_NYSE_NASDAQ_AMEX)]
            _breadth_tickers = _uni.index.tolist()
            logger.info(
                f"Market Breadth 티커 로드: {len(_breadth_tickers):,}개 "
                f"(NYSE/NASDAQ/AMEX, 거래소 필터 적용)"
            )
        except Exception as _e:
            logger.warning(f"Universe 파일 로드 실패, Breadth 계산을 건너뜁니다: {_e}")

    config = {
        "close_path": ".intermediate/close.parquet",
        "etf_close_path": ".intermediate/etf_close.parquet",
        "etf_info_path": ".intermediate/etf_info.parquet",
        "etf_holdings_path": ".intermediate/etf_top_holdings.parquet",
        "spy_close_path": ".intermediate/spy_close.parquet",
        "output_dir": "output",
        "benchmark": "SPY",
        "breadth_tickers": _breadth_tickers,  # NYSE/NASDAQ/AMEX Market Breadth 계산용
    }

    if os.path.exists(config["close_path"]):
        run_pipeline(config)
    else:
        logger.warning(f"입력 파일({config['close_path']})이 존재하지 않아 실행을 건너뜁니다.")
        logger.info("데이터 다운로더를 먼저 실행하여 데이터를 준비하세요.")
