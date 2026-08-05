"""
Leader Stock Discovery Engine - Data Downloader.

데이터 다운로드만 전담하는 독립 실행 모듈입니다.
다운로드 결과는 중간 파일(parquet)로 저장되며,
data_processor.py가 이를 읽어 전처리/지표 연산을 수행합니다.

실행: python data_downloader.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from data_loader import (
    UNIVERSE_MIN_DOLLAR_VOLUME,
    calculate_market_caps_from_ohlcv,
    download_ohlcv,
    download_spy_close,
    fetch_etf_top_holdings,
    fetch_filtered_universe,
    fetch_sector_industry_map,
    fetch_shares_outstanding,
    normalize_ticker,
)

DOWNLOAD_PERIOD = "16mo"
ETF_TOP_HOLDINGS_LIMIT = 10
INTERMEDIATE_DIR = Path(__file__).parent / ".intermediate"


def _spdr_holdings_url(ticker: str) -> str:
    return (
        "https://www.ssga.com/us/en/intermediary/etfs/library-content/"
        f"products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"
    )


CURATED_ETFS = [
    # ── Sector (SPDR 11개) ────────────────────────────────────────────────────
    {"Ticker": "XLB",  "Name": "Materials Select Sector SPDR Fund",               "Category": "Sector",        "Theme": "Materials",               "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLB")},
    {"Ticker": "XLC",  "Name": "Communication Services Select Sector SPDR Fund",  "Category": "Sector",        "Theme": "Communication Services",  "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLC")},
    {"Ticker": "XLE",  "Name": "Energy Select Sector SPDR Fund",                  "Category": "Sector",        "Theme": "Energy",                  "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLE")},
    {"Ticker": "XLF",  "Name": "Financial Select Sector SPDR Fund",               "Category": "Sector",        "Theme": "Financials",              "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLF")},
    {"Ticker": "XLI",  "Name": "Industrial Select Sector SPDR Fund",              "Category": "Sector",        "Theme": "Industrials",             "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLI")},
    {"Ticker": "XLK",  "Name": "Technology Select Sector SPDR Fund",              "Category": "Sector",        "Theme": "Technology",              "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLK")},
    {"Ticker": "XLP",  "Name": "Consumer Staples Select Sector SPDR Fund",        "Category": "Sector",        "Theme": "Consumer Staples",        "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLP")},
    {"Ticker": "XLRE", "Name": "Real Estate Select Sector SPDR Fund",             "Category": "Sector",        "Theme": "Real Estate",             "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLRE")},
    {"Ticker": "XLU",  "Name": "Utilities Select Sector SPDR Fund",               "Category": "Sector",        "Theme": "Utilities",               "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLU")},
    {"Ticker": "XLV",  "Name": "Health Care Select Sector SPDR Fund",             "Category": "Sector",        "Theme": "Health Care",             "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLV")},
    {"Ticker": "XLY",  "Name": "Consumer Discretionary Select Sector SPDR Fund",  "Category": "Sector",        "Theme": "Consumer Discretionary",  "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XLY")},
 
    # ── Market Benchmark ──────────────────────────────────────────────────────
    {"Ticker": "SPY",  "Name": "SPDR S&P 500 ETF Trust",                          "Category": "Market",        "Theme": "S&P 500",                 "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("SPY")},
    {"Ticker": "RSP",  "Name": "Invesco S&P 500 Equal Weight ETF",                "Category": "Market",        "Theme": "S&P 500 Equal Weight",    "Provider": "Invesco",    "HoldingsUrl": ""},
    {"Ticker": "QQQ",  "Name": "Invesco QQQ Trust",                               "Category": "Market",        "Theme": "Nasdaq 100",              "Provider": "Invesco",    "HoldingsUrl": ""},
    {"Ticker": "QQQE", "Name": "Direxion NASDAQ-100 Equal Weighted Index ETF",    "Category": "Market",        "Theme": "Nasdaq 100 Equal Weight", "Provider": "Direxion",   "HoldingsUrl": ""},
    {"Ticker": "MDY",  "Name": "SPDR S&P MidCap 400 ETF Trust",                   "Category": "Market",        "Theme": "Mid Caps",                "Provider": "SPDR",       "HoldingsUrl": ""},
    {"Ticker": "IWM",  "Name": "iShares Russell 2000 ETF",                        "Category": "Market",        "Theme": "Small Caps",              "Provider": "iShares",    "HoldingsUrl": ""},
 
    # ── Industry (SPDR S&P 서브인더스트리) ────────────────────────────────────
    {"Ticker": "XBI",  "Name": "SPDR S&P Biotech ETF",                            "Category": "Industry",      "Theme": "Biotech",                 "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XBI")},
    {"Ticker": "XHB",  "Name": "SPDR S&P Homebuilders ETF",                       "Category": "Industry",      "Theme": "Homebuilders",            "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XHB")},
    {"Ticker": "XOP",  "Name": "SPDR S&P Oil & Gas Exploration & Production ETF", "Category": "Industry",      "Theme": "Oil & Gas E&P",           "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XOP")},
    {"Ticker": "XME",  "Name": "SPDR S&P Metals & Mining ETF",                    "Category": "Industry",      "Theme": "Metals & Mining",         "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XME")},
    {"Ticker": "XRT",  "Name": "SPDR S&P Retail ETF",                             "Category": "Industry",      "Theme": "Retail",                  "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("XRT")},
    {"Ticker": "KBE",  "Name": "SPDR S&P Bank ETF",                               "Category": "Industry",      "Theme": "Banks",                   "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("KBE")},
    {"Ticker": "KRE",  "Name": "SPDR S&P Regional Banking ETF",                   "Category": "Industry",      "Theme": "Regional Banks",          "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("KRE")},
    {"Ticker": "KIE",  "Name": "SPDR S&P Insurance ETF",                          "Category": "Industry",      "Theme": "Insurance",               "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("KIE")},
    {"Ticker": "KOMP", "Name": "SPDR S&P Kensho New Economies Composite ETF",     "Category": "Theme",         "Theme": "New Economies",           "Provider": "SPDR",       "HoldingsUrl": _spdr_holdings_url("KOMP")},
 
    # ── Semiconductors ────────────────────────────────────────────────────────
    {"Ticker": "SMH",  "Name": "VanEck Semiconductor ETF",                        "Category": "Theme",         "Theme": "Semiconductors",          "Provider": "VanEck",     "HoldingsUrl": ""},
    {"Ticker": "SOXX", "Name": "iShares Semiconductor ETF",                       "Category": "Theme",         "Theme": "Semiconductors",          "Provider": "iShares",    "HoldingsUrl": "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf/1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund"},
    {"Ticker": "DRAM", "Name": "Roundhill Memory ETF",                            "Category": "Theme",         "Theme": "Memory Chips",            "Provider": "Roundhill",  "HoldingsUrl": ""},  # ★ 신규
 
    # ── AI / Robotics ─────────────────────────────────────────────────────────
    {"Ticker": "AIQ",  "Name": "Global X Artificial Intelligence & Technology ETF","Category": "Theme",        "Theme": "AI & Technology",         "Provider": "Global X",   "HoldingsUrl": ""},
    {"Ticker": "BOTZ", "Name": "Global X Robotics & Artificial Intelligence ETF", "Category": "Theme",         "Theme": "Robotics & AI",           "Provider": "Global X",   "HoldingsUrl": ""},
    {"Ticker": "ROBT", "Name": "First Trust Nasdaq AI and Robotics ETF",          "Category": "Theme",         "Theme": "AI & Robotics",           "Provider": "First Trust","HoldingsUrl": ""},  # ★ 신규 추천
 
    # ── Software / Cloud ──────────────────────────────────────────────────────
    {"Ticker": "IGV",  "Name": "iShares Expanded Tech-Software Sector ETF",       "Category": "Theme",         "Theme": "Software",                "Provider": "iShares",    "HoldingsUrl": ""},
    {"Ticker": "WCLD", "Name": "WisdomTree Cloud Computing ETF",                  "Category": "Theme",         "Theme": "Cloud Computing",         "Provider": "WisdomTree", "HoldingsUrl": ""},  # ★ 신규 추천
 
    # ── Cybersecurity ─────────────────────────────────────────────────────────
    {"Ticker": "HACK", "Name": "Amplify Cybersecurity ETF",                       "Category": "Theme",         "Theme": "Cybersecurity",           "Provider": "Amplify",    "HoldingsUrl": ""},
    {"Ticker": "CIBR", "Name": "First Trust NASDAQ Cybersecurity ETF",            "Category": "Theme",         "Theme": "Cybersecurity",           "Provider": "First Trust","HoldingsUrl": ""},  # ★ 신규 추천
 
    # ── Smart Grid / Infrastructure ───────────────────────────────────────────
    {"Ticker": "GRID", "Name": "First Trust NASDAQ Clean Edge Smart Grid & Energy Index Fund", "Category": "Theme", "Theme": "Smart Grid",         "Provider": "First Trust","HoldingsUrl": ""},
    {"Ticker": "PAVE", "Name": "Global X U.S. Infrastructure Development ETF",   "Category": "Theme",         "Theme": "Infrastructure",          "Provider": "Global X",   "HoldingsUrl": ""},
 
    # ── Innovation / Growth ───────────────────────────────────────────────────
    {"Ticker": "ARKK", "Name": "ARK Innovation ETF",                              "Category": "Theme",         "Theme": "Innovation",              "Provider": "ARK",        "HoldingsUrl": ""},
    {"Ticker": "ARKQ", "Name": "ARK Autonomous Technology & Robotics ETF",        "Category": "Theme",         "Theme": "Robotics",                "Provider": "ARK",        "HoldingsUrl": ""},
 
    # ── Healthcare ────────────────────────────────────────────────────────────
    {"Ticker": "IBB",  "Name": "iShares Biotechnology ETF",                       "Category": "Theme",         "Theme": "Biotech",                 "Provider": "iShares",    "HoldingsUrl": ""},
 
    # ── Defense & Space ───────────────────────────────────────────────────────
    {"Ticker": "ITA",  "Name": "iShares U.S. Aerospace & Defense ETF",            "Category": "Theme",         "Theme": "Aerospace & Defense",     "Provider": "iShares",    "HoldingsUrl": ""},
    {"Ticker": "UFO",  "Name": "Procure Space ETF",                               "Category": "Theme",         "Theme": "Space",                   "Provider": "ProcureAM",  "HoldingsUrl": ""},
 
    # ── Nuclear / Uranium ─────────────────────────────────────────────────────
    {"Ticker": "URA",  "Name": "Sprott Uranium Miners ETF",                       "Category": "Theme",         "Theme": "Uranium",                 "Provider": "Sprott",     "HoldingsUrl": ""},
    {"Ticker": "NLR",  "Name": "VanEck Uranium and Nuclear ETF",                  "Category": "Theme",         "Theme": "Nuclear Energy",          "Provider": "VanEck",     "HoldingsUrl": ""},
 
    # ── Energy ────────────────────────────────────────────────────────────────
    {"Ticker": "TAN",  "Name": "Invesco Solar ETF",                               "Category": "Theme",         "Theme": "Solar",                   "Provider": "Invesco",    "HoldingsUrl": ""},
    {"Ticker": "ICLN", "Name": "iShares Global Clean Energy ETF",                 "Category": "Theme",         "Theme": "Clean Energy",            "Provider": "iShares",    "HoldingsUrl": ""},
    {"Ticker": "COAL", "Name": "Range Global Coal Index ETF",                     "Category": "Theme",         "Theme": "Coal",                    "Provider": "Range",      "HoldingsUrl": ""},
 
    # ── Materials / Mining ────────────────────────────────────────────────────
    {"Ticker": "COPX", "Name": "Global X Copper Miners ETF",                      "Category": "Theme",         "Theme": "Copper Miners",           "Provider": "Global X",   "HoldingsUrl": ""},
 
    # ── Real Estate ───────────────────────────────────────────────────────────
    {"Ticker": "VNQ",  "Name": "Vanguard Real Estate ETF",                        "Category": "Sector",        "Theme": "Real Estate",             "Provider": "Vanguard",   "HoldingsUrl": ""},
    {"Ticker": "IYR",  "Name": "iShares U.S. Real Estate ETF",                    "Category": "Sector",        "Theme": "Real Estate",             "Provider": "iShares",    "HoldingsUrl": ""},
 
    # ── Commodities ───────────────────────────────────────────────────────────
    {"Ticker": "GLD",  "Name": "SPDR Gold Shares",                                "Category": "Commodity",     "Theme": "Gold",                    "Provider": "SPDR",       "HoldingsUrl": ""},
    {"Ticker": "SLV",  "Name": "iShares Silver Trust",                            "Category": "Commodity",     "Theme": "Silver",                  "Provider": "iShares",    "HoldingsUrl": ""},
    {"Ticker": "USO",  "Name": "United States Oil Fund",                          "Category": "Commodity",     "Theme": "Oil",                     "Provider": "USCF",       "HoldingsUrl": ""},
 
    # ── International ─────────────────────────────────────────────────────────
    {"Ticker": "EFA",  "Name": "iShares MSCI EAFE ETF",                           "Category": "International", "Theme": "Developed Markets ex-US", "Provider": "iShares",    "HoldingsUrl": ""},
    {"Ticker": "EEM",  "Name": "iShares MSCI Emerging Markets ETF",               "Category": "International", "Theme": "Emerging Markets",        "Provider": "iShares",    "HoldingsUrl": ""},
    {"Ticker": "EWY",  "Name": "iShares MSCI South Korea ETF",                    "Category": "International", "Theme": "South Korea",             "Provider": "iShares",    "HoldingsUrl": ""},  # ★ 신규
    {"Ticker": "KWEB", "Name": "KraneShares CSI China Internet ETF",              "Category": "International", "Theme": "China Internet",          "Provider": "KraneShares","HoldingsUrl": ""},  # ★ 신규 추천

    # ── Crypto / Digital Assets ───────────────────────────────────────────────
    {"Ticker": "WGMI", "Name": "Valkyrie Bitcoin Miners ETF",                    "Category": "Theme",         "Theme": "Bitcoin Mining",          "Provider": "Valkyrie",   "HoldingsUrl": ""},
]

def _save_intermediate(df: pd.DataFrame, name: str) -> None:
    """중간 파일을 .intermediate 디렉토리에 저장합니다.

    빈 결과(레이트리밋 등 수집 실패)가 기존 정상 파일을 덮어쓰지 않도록 보호한다.
    """
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    path = INTERMEDIATE_DIR / f"{name}.parquet"
    if df.empty and path.exists():
        print(f"  [Save-Skip] {name}: 새 데이터가 비어 있어 기존 파일을 보존합니다.")
        return
    df.to_parquet(path, index=True)
    print(f"  [Save] {name} -> {path} ({len(df)}행)")


def _save_series(series: pd.Series, name: str) -> None:
    """시리즈를 중간 파일로 저장합니다. (빈 결과는 기존 파일 보존)"""
    INTERMEDIATE_DIR.mkdir(exist_ok=True)
    path = INTERMEDIATE_DIR / f"{name}.parquet"
    if series.empty and path.exists():
        print(f"  [Save-Skip] {name}: 새 데이터가 비어 있어 기존 파일을 보존합니다.")
        return
    df = series.to_frame(name=name)
    df.to_parquet(path, index=True)
    print(f"  [Save] {name} -> {path} ({len(df)}행)")


def run_download() -> None:
    """
    전체 다운로드 파이프라인을 실행하고 결과를 중간 파일로 저장합니다.
    """
    print("=" * 60)
    print("Data Downloader - 데이터 다운로드 전용")
    print("=" * 60)

    # [Step 1] 필터링된 유니버스 로드
    print("[Step 1/7] 유니버스 로드 중...")
    universe = fetch_filtered_universe()
    _save_intermediate(universe.set_index("Ticker"), "universe")
    print(f"  -> {len(universe):,}개 종목")

    # [Step 2] 섹터/산업 메타데이터 수집
    print("[Step 2/7] 섹터/산업 메타데이터 수집 중...")
    sector_info = fetch_sector_industry_map(universe["Ticker"].tolist())
    sector_df = pd.DataFrame.from_dict(sector_info, orient="index")
    _save_intermediate(sector_df, "sector_info")
    print(f"  -> {len(sector_df):,}개 종목 메타데이터")

    # [Step 3] OHLCV 시계열 데이터 다운로드
    yf_tickers = [normalize_ticker(t) for t in universe["Ticker"]]
    print(f"[Step 3/7] OHLCV 다운로드 ({len(yf_tickers):,}개 종목, period={DOWNLOAD_PERIOD})...")
    close_df, high_df, low_df, volume_df = download_ohlcv(
        yf_tickers, period=DOWNLOAD_PERIOD,
        min_dollar_volume=UNIVERSE_MIN_DOLLAR_VOLUME,
    )
    _save_intermediate(close_df, "close")
    _save_intermediate(high_df, "high")
    _save_intermediate(low_df, "low")
    _save_intermediate(volume_df, "volume")

    # [Step 4] SPY 벤치마크 다운로드
    print(f"[Step 4/7] SPY 벤치마크 다운로드...")
    spy_close = download_spy_close(DOWNLOAD_PERIOD)
    _save_series(spy_close, "spy_close")

    # [Step 5] ETF 메타데이터 및 OHLCV 다운로드
    print(f"[Step 5/7] ETF 메타데이터 및 OHLCV 다운로드...")
    etf_info = pd.DataFrame(CURATED_ETFS).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    _save_intermediate(etf_info.set_index("Ticker"), "etf_info")

    etf_tickers = [normalize_ticker(t) for t in etf_info["Ticker"]]
    etf_close_df, etf_high_df, etf_low_df, etf_volume_df = download_ohlcv(
        etf_tickers,
        period=DOWNLOAD_PERIOD,
        cache_name="etf_ohlcv_master_cache.parquet",
    )
    _save_intermediate(etf_close_df, "etf_close")
    _save_intermediate(etf_high_df, "etf_high")
    _save_intermediate(etf_low_df, "etf_low")
    _save_intermediate(etf_volume_df, "etf_volume")

    # [Step 6] ETF 상위 보유종목 다운로드
    print(f"[Step 6/7] ETF 상위 보유종목 다운로드 (top {ETF_TOP_HOLDINGS_LIMIT})...")
    etf_top_holdings = fetch_etf_top_holdings(etf_info, limit=ETF_TOP_HOLDINGS_LIMIT)
    # 레이트리밋 등으로 일부 ETF 수집이 실패해도 이전 파일의 해당 ETF 행을 보존해,
    # 부분 실패가 기존 정상 데이터를 지우지 않게 한다 (보유종목은 천천히 변하므로 안전).
    prev_path = INTERMEDIATE_DIR / "etf_top_holdings.parquet"
    if prev_path.exists() and not etf_top_holdings.empty:
        try:
            prev = pd.read_parquet(prev_path)
            fetched_ok = set(etf_top_holdings["ETF"].unique())
            keep = prev[~prev["ETF"].isin(fetched_ok)]
            if not keep.empty:
                etf_top_holdings = pd.concat([etf_top_holdings, keep], ignore_index=True)
                print(f"  [ETF Holdings] 이번 수집 실패 {keep['ETF'].nunique()}개 ETF는 "
                      f"이전 데이터 보존 ({len(keep)}행)")
        except Exception as e:
            print(f"  [Warning] 이전 ETF 보유종목 병합 실패: {e}")
    _save_intermediate(etf_top_holdings, "etf_top_holdings")

    # [Step 7] 시가총액 데이터 수집 및 우회 계산 (수정된 핵심 파트)
    print(f"[Step 7/7] 발행 주식 수 확인 및 시가총액 우회 계산 중...")
    
    # 1. 발행 주식 수 마스터 로드 (최초 1회만 API 호출, 이후엔 0초만에 캐시 로드)
    #    유동성 하한에 걸린 휴면 티커는 close_df에서 이미 빠졌다. 시가총액을 낼 대상이
    #    아니므로 여기서도 제외해, 실행당 메타데이터 상한을 활성 종목에만 쓴다.
    active_tickers = [t for t in yf_tickers if t in close_df.columns]
    shares_outstanding = fetch_shares_outstanding(active_tickers, force_update=False)
    
    # 2. 다운로드된 종가(close_df)와 주식 수를 곱해 시가총액 자체 연산
    market_caps = calculate_market_caps_from_ohlcv(close_df, shares_outstanding)
    
    # 3. 중간 결과 저장 (.intermediate/market_caps.parquet)
    _save_series(market_caps, "market_caps")

    print("=" * 60)
    print("데이터 다운로드 및 시가총액 계산 완료!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_download()
    except Exception as exc:
        print(f"[Error] 데이터 다운로드 실패: {exc}", file=sys.stderr)
        sys.exit(1)
