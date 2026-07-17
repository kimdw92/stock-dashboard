import ftplib
import json
import random
import time
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import urllib.request

DOWNLOAD_CHUNK_SIZE = 1000
DOWNLOAD_MAX_RETRIES = 3       # 배치 다운로드 실패 시 재시도 횟수
DOWNLOAD_RETRY_BACKOFF = 3.0  # 재시도 간 기본 대기(초), 시도마다 배수 증가

def _cache_path(filename: str) -> Path:
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / filename

def normalize_ticker(ticker: str) -> str:
    return str(ticker).replace('.', '-')

def display_ticker(ticker: str) -> str:
    return str(ticker).replace('-', '.')


# NASDAQ Trader가 매 거래일 갱신하는 상장 종목 목록 파일 (익명 FTP)
NASDAQ_FTP_HOST = "ftp.nasdaqtrader.com"
NASDAQ_FTP_DIR = "SymbolDirectory"
UNIVERSE_FILES = ("nasdaqlisted.txt", "otherlisted.txt")
UNIVERSE_MAX_AGE_DAYS = 7


def _download_nasdaq_symbol_file(filename: str, dest_path: Path) -> None:
    """NASDAQ Trader FTP에서 심볼 파일 하나를 받아 원자적으로 저장한다."""
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with ftplib.FTP(NASDAQ_FTP_HOST, timeout=30) as ftp:
        ftp.login()  # 익명 로그인
        ftp.cwd(NASDAQ_FTP_DIR)
        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write)
    tmp_path.replace(dest_path)  # 성공 시에만 교체 (부분 다운로드로 기존 캐시 훼손 방지)


def ensure_universe_files(max_age_days: int = UNIVERSE_MAX_AGE_DAYS, force: bool = False) -> None:
    """
    유니버스 목록 파일(nasdaqlisted / otherlisted)이 없거나 오래되면 자동으로 갱신한다.
    다운로드 실패 시 기존 캐시가 있으면 그대로 사용하고, 없으면 경고만 남긴다.
    """
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    for filename in UNIVERSE_FILES:
        dest = cache_dir / filename
        is_fresh = (
            dest.exists()
            and (time.time() - dest.stat().st_mtime) < max_age_days * 86400
        )
        if is_fresh and not force:
            print(f"  [Universe] {filename} 최신 캐시 사용 (갱신 불필요)")
            continue
        try:
            print(f"  [Universe] {filename} 다운로드 중 (NASDAQ Trader FTP)...")
            _download_nasdaq_symbol_file(filename, dest)
            print(f"  [Universe] {filename} 갱신 완료 ({dest.stat().st_size:,} bytes)")
        except Exception as e:
            if dest.exists():
                print(f"  [Warning] {filename} 다운로드 실패, 기존 캐시 사용: {e}")
            else:
                print(f"  [Error] {filename} 다운로드 실패, 캐시도 없음: {e}")


def fetch_filtered_universe() -> pd.DataFrame:
    cache_dir = Path(__file__).parent / ".cache"
    nasdaq_path = cache_dir / "nasdaqlisted.txt"
    other_path = cache_dir / "otherlisted.txt"

    # 목록 파일이 없거나 오래되면 자동 갱신 시도
    ensure_universe_files()
    
    def clean_tickers(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if df.empty or col_name not in df.columns:
            return df
        
        # 1. 알파벳(A-Z)으로만 이루어진 티커만 남김 (하이픈, 마침표, 숫자 제거)
        is_pure = df[col_name].str.match(r'^[A-Za-z]+$', na=False)
        df = df[is_pure]
        
        # 2. [추가] 보통 미국 보통주는 1~4글자입니다. 5글자이면서 맨 뒤가 'W'로 끝나는 워런트 종목 제거
        # 예: MYXXW, ITHAW 등을 저격해서 제거합니다.
        is_warrant = df[col_name].str.match(r'^[A-Za-z]{4}[Ww]$', na=False)
        
        return df[~is_warrant]

    dfs = []
    if nasdaq_path.exists():
        try:
            nasdaq = pd.read_csv(nasdaq_path, sep='|')
            nasdaq = nasdaq.dropna(subset=['Symbol'])
            nasdaq = nasdaq[nasdaq['Symbol'] != '']
            nasdaq = nasdaq[~nasdaq['Symbol'].str.contains('File Creation', na=False)]
            
            if 'ETF' in nasdaq.columns:
                nasdaq = nasdaq[nasdaq['ETF'] == 'N']
            if 'Test Issue' in nasdaq.columns:
                nasdaq = nasdaq[nasdaq['Test Issue'] == 'N']
            
            # --- 여기에 clean_tickers 적용 ---
            nasdaq = clean_tickers(nasdaq, 'Symbol')
            
            df_n = pd.DataFrame({'Ticker': nasdaq['Symbol'], 'Exchange': 'NASDAQ'})
            dfs.append(df_n)
        except Exception as e:
            print(f"[Error] Error loading nasdaq: {e}")
            
    if other_path.exists():
        try:
            other = pd.read_csv(other_path, sep='|')
            other = other.dropna(subset=['ACT Symbol'])
            other = other[other['ACT Symbol'] != '']
            other = other[~other['ACT Symbol'].str.contains('File Creation', na=False)]
            
            if 'ETF' in other.columns:
                other = other[other['ETF'] == 'N']
            if 'Test Issue' in other.columns:
                other = other[other['Test Issue'] == 'N']
                
            # --- 여기에 clean_tickers 적용 ---
            other = clean_tickers(other, 'ACT Symbol')
                
            df_o = pd.DataFrame({'Ticker': other['ACT Symbol'], 'Exchange': other['Exchange']})
            dfs.append(df_o)
        except Exception as e:
            print(f"[Error] Error loading other: {e}")
            
    if dfs:
        res = pd.concat(dfs, ignore_index=True)
        return res
    else:
        print("[Warning] No universe files found. Falling back to default tickers.")
        return pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOGL", "NVDA"], "Exchange": ["NASDAQ"]*4})

def fetch_sector_industry_map(tickers: list[str]) -> dict[str, dict[str, str]]:
    cache_path = Path(__file__).parent / ".cache" / "sector_industry_map.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"[Warning] Failed to load sector_industry_map: {e}")
    return {}

def _extract_field(raw: pd.DataFrame, field: str, chunk: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.levels[0]:
            return raw.xs(field, level=0, axis=1)
    else:
        if field in raw.columns:
            df = raw[[field]].copy()
            df.columns = [chunk[0]]
            return df
    return pd.DataFrame()

def download_ohlcv(
    yf_tickers: list[str],
    period: str = "16mo",
    cache_name: str = "ohlcv_master_cache.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    OHLCV 데이터를 거대 청크 단위로 병렬(Multi-threading) 다운로드하여 속도를 극대화합니다.
    """
    cache_path = _cache_path(cache_name)
    
    cached_close, cached_high, cached_low, cached_volume = None, None, None, None
    is_incremental = False
    
    if cache_path.exists():
        try:
            cached_master = pd.read_parquet(cache_path)
            cached_master.index = pd.to_datetime(cached_master.index, level=0) if isinstance(cached_master.index, pd.MultiIndex) else pd.to_datetime(cached_master.index)
            
            if isinstance(cached_master.index, pd.MultiIndex):
                cached_close = cached_master.xs('Close', level=1, drop_level=True) if 'Close' in cached_master.columns else cached_master
                cached_high = cached_master.xs('High', level=1, drop_level=True) if 'High' in cached_master.columns else cached_master
                cached_low = cached_master.xs('Low', level=1, drop_level=True) if 'Low' in cached_master.columns else cached_master
                cached_volume = cached_master.xs('Volume', level=1, drop_level=True) if 'Volume' in cached_master.columns else cached_master
            else:
                if isinstance(cached_master.columns, pd.MultiIndex):
                    try:
                        cached_close = cached_master.xs('Close', level=1, axis=1)
                        cached_high = cached_master.xs('High', level=1, axis=1)
                        cached_low = cached_master.xs('Low', level=1, axis=1)
                        cached_volume = cached_master.xs('Volume', level=1, axis=1)
                    except Exception:
                        cached_close, cached_high, cached_low, cached_volume = cached_master, cached_master, cached_master, cached_master
                else:
                    cached_close, cached_high, cached_low, cached_volume = cached_master, cached_master, cached_master, cached_master
            
            latest_cached_date = cached_close.index.max() if hasattr(cached_close.index, 'max') else cached_master.index.max()
            is_incremental = True
            print(f"  [Cache] 마스터 캐시 로드 완료 (최신 날짜: {latest_cached_date.date()})")
            print(f"  [Cache] 증분 업데이트 모드로 전환 (캐시 이후 데이터만 요청)")
        except Exception as exc:
            print(f"  [Warning] 마스터 캐시 로드 실패: {exc}. 전체 기간 재다운로드합니다.")
            cached_close = None
    
    # ── 신규/데이터 부족 티커 감지 ─────────────────────────────────────────────
    # 증분 모드에서 캐시에 없거나 유효 행이 30개 미만인 티커는 전체 기간 재다운로드
    FULL_DL_THRESHOLD = 30  # 유효 종가 행 수 기준
    need_full: set[str] = set()
    if is_incremental and cached_close is not None:
        for t in yf_tickers:
            if t not in cached_close.columns:
                need_full.add(t)
            else:
                valid_count = int(cached_close[t].replace(0, float("nan")).notna().sum())
                if valid_count < FULL_DL_THRESHOLD:
                    need_full.add(t)
        if need_full:
            print(f"  [Detect] 신규/데이터 부족 {len(need_full)}개 티커 → 전체 기간({period}) 재다운로드")

    close_frames, high_frames, low_frames, volume_frames = [], [], [], []

    def _run_chunks(
        ticker_list: list[str],
        label: str,
        *,
        dl_period: str | None = None,
        start: str | None = None,
    ) -> None:
        """ticker_list를 청크 단위로 다운로드해 *_frames 리스트에 누적.

        dl_period 또는 start 중 하나만 지정한다. start를 쓰면 해당 날짜부터
        오늘까지 전부 요청하므로, 캐시 갱신 간격이 1주일을 넘어도 결측 구간이
        생기지 않는다(고정 period="1wk" 사용 시 발생하던 문제).
        """
        chunk_list = [
            ticker_list[i : i + DOWNLOAD_CHUNK_SIZE]
            for i in range(0, len(ticker_list), DOWNLOAD_CHUNK_SIZE)
        ]
        desc = f"start={start}" if start else dl_period
        for idx, chunk in enumerate(chunk_list, start=1):
            print(f"  [Download] {label} 배치 {idx}/{len(chunk_list)} ({len(chunk)}종목, {desc})...")
            # 배치 단위 재시도: 일시적 네트워크 오류/레이트리밋으로 한 배치가 실패해도
            # 전체 파이프라인을 중단시키지 않고, 재시도 후에도 실패하면 해당 배치만 건너뛴다.
            raw = None
            for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
                try:
                    if start:
                        raw = yf.download(chunk, start=start, threads=True, progress=False, auto_adjust=True)
                    else:
                        raw = yf.download(chunk, period=dl_period, threads=True, progress=False, auto_adjust=True)
                    break
                except Exception as exc:
                    if attempt < DOWNLOAD_MAX_RETRIES:
                        wait = DOWNLOAD_RETRY_BACKOFF * attempt + random.uniform(0, 1.0)
                        print(f"  [Retry] {label} 배치 {idx} 실패({type(exc).__name__}), "
                              f"{attempt}/{DOWNLOAD_MAX_RETRIES} 재시도까지 {wait:.1f}초 대기...")
                        time.sleep(wait)
                    else:
                        print(f"  [Skip] {label} 배치 {idx} {DOWNLOAD_MAX_RETRIES}회 실패, "
                              f"이 배치({len(chunk)}종목)를 건너뜁니다: {exc}")
            time.sleep(random.uniform(1.0, 2.0))
            if raw is None or raw.empty:
                continue
            c = _extract_field(raw, "Close", chunk)
            h = _extract_field(raw, "High", chunk)
            l = _extract_field(raw, "Low", chunk)
            v = _extract_field(raw, "Volume", chunk)
            if not c.empty: close_frames.append(c)
            if not h.empty: high_frames.append(h)
            if not l.empty: low_frames.append(l)
            if not v.empty: volume_frames.append(v)

    if is_incremental:
        # 전체 기간: 신규 또는 데이터 부족 티커
        full_list = [t for t in yf_tickers if t in need_full]
        if full_list:
            _run_chunks(full_list, "전체기간", dl_period=period)
        # 증분: 캐시에 충분한 데이터가 있는 기존 티커 → 캐시 마지막 날짜 다음날부터 요청
        incr_list = [t for t in yf_tickers if t not in need_full]
        if incr_list:
            incr_start = (latest_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if incr_start <= pd.Timestamp.now().strftime("%Y-%m-%d"):
                _run_chunks(incr_list, "증분", start=incr_start)
            else:
                print(f"  [Skip] 증분 대상 {len(incr_list)}개 티커는 이미 최신 상태입니다.")
    else:
        # 캐시 없음: 전체 종목 전체 기간 다운로드
        _run_chunks(yf_tickers, "전체기간", dl_period=period)

    if not close_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    new_close_df = pd.concat(close_frames, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
    new_high_df = pd.concat(high_frames, axis=1).loc[:, lambda df: ~df.columns.duplicated()] if high_frames else pd.DataFrame()
    new_low_df = pd.concat(low_frames, axis=1).loc[:, lambda df: ~df.columns.duplicated()] if low_frames else pd.DataFrame()
    new_volume_df = pd.concat(volume_frames, axis=1).loc[:, lambda df: ~df.columns.duplicated()] if volume_frames else pd.DataFrame()
    
    if cached_close is not None and not new_close_df.empty:
        # ── 컬럼 그룹별 병합 ──────────────────────────────────────────────────────
        # axis=0 전체 concat + keep='last'를 사용하면 중복 날짜에서 new_close_df 행
        # 전체가 캐시를 덮어씌워 증분 티커(1wk 다운로드)의 역사 데이터가 NaN이 됨.
        # 따라서 컬럼을 세 그룹으로 분리해 병합한다:
        #   1) unchanged_cols : 이번 다운로드에 없는 티커 → 캐시 그대로
        #   2) full_cols      : need_full 티커 → 새 전체기간 데이터로 교체
        #   3) incr_cols      : 증분(1wk) 티커 → 새 날짜 행만 캐시에 append

        def _merge_field(
            cached: pd.DataFrame,
            new_df: pd.DataFrame,
        ) -> pd.DataFrame:
            full_cols = [t for t in new_df.columns if t in need_full]
            incr_cols = [t for t in new_df.columns if t not in need_full]

            parts: list[pd.DataFrame] = []

            # 1) 캐시에만 있고 이번 다운로드에 없는 컬럼 → 그대로 유지
            unchanged = [c for c in cached.columns if c not in new_df.columns]
            if unchanged:
                parts.append(cached[unchanged])

            # 2) need_full 티커 → 새 데이터로 컬럼 전체 교체
            if full_cols:
                parts.append(new_df[full_cols])

            # 3) 증분 티커 → 캐시 행 유지 + 새 날짜 행만 추가
            if incr_cols:
                cached_incr = [c for c in incr_cols if c in cached.columns]
                new_incr    = [c for c in incr_cols if c not in cached.columns]
                if cached_incr:
                    incr_cached_df = cached[cached_incr]
                    # dropna(how='all'): need_full 티커(WGMI 등)가 full download 기간
                    # 전체 날짜를 new_df 인덱스에 추가해, incr 티커 컬럼이 과거 구간에
                    # NaN 행을 갖게 됨. 이 NaN 행을 제거하지 않으면 keep='last'로 인해
                    # 캐시의 실제 과거 데이터가 NaN으로 덮어씌워지는 문제 발생.
                    incr_new_df    = (
                        new_df[[c for c in cached_incr if c in new_df.columns]]
                        .dropna(how="all")  # incr 티커에 실제 데이터가 없는 날짜 제거
                    )
                    incr_merged = pd.concat([incr_cached_df, incr_new_df], axis=0)
                    incr_merged = incr_merged[~incr_merged.index.duplicated(keep="last")].sort_index()
                    parts.append(incr_merged)
                if new_incr:  # 드문 엣지 케이스: incr로 분류됐으나 캐시에 없는 티커
                    parts.append(new_df[[c for c in new_incr if c in new_df.columns]])

            merged = pd.concat(parts, axis=1)
            return merged.sort_index()

        close_df   = _merge_field(cached_close,  new_close_df)
        high_df    = _merge_field(cached_high,   new_high_df)  if not new_high_df.empty   else cached_high
        low_df     = _merge_field(cached_low,    new_low_df)   if not new_low_df.empty    else cached_low
        volume_df  = _merge_field(cached_volume, new_volume_df) if not new_volume_df.empty else cached_volume

        print(f"  [Merge] 캐시({len(cached_close)}행) + 신규({len(new_close_df)}행) -> 최종({len(close_df)}행)")
    else:
        close_df, high_df, low_df, volume_df = new_close_df, new_high_df, new_low_df, new_volume_df
        print(f"  [Download] 신규 다운로드 완료 (캐시 없음)")
    
    # 결측치 보정: ffill만 수행, fillna(0) 금지
    # 0으로 채우면 다운로드 실패 티커가 rs_engine에서 inf/NaN을 유발함
    close_df = close_df.ffill()
    high_df = high_df.ffill() if not high_df.empty else high_df
    low_df = low_df.ffill() if not low_df.empty else low_df
    volume_df = volume_df.ffill() if not volume_df.empty else volume_df
    
    try:
        close_df_tagged = close_df.copy()
        high_df_tagged = high_df.copy()
        low_df_tagged = low_df.copy()
        volume_df_tagged = volume_df.copy()
        
        close_df_tagged.columns = pd.MultiIndex.from_product([close_df_tagged.columns, ['Close']])
        high_df_tagged.columns = pd.MultiIndex.from_product([high_df_tagged.columns, ['High']])
        low_df_tagged.columns = pd.MultiIndex.from_product([low_df_tagged.columns, ['Low']])
        volume_df_tagged.columns = pd.MultiIndex.from_product([volume_df_tagged.columns, ['Volume']])
        
        master_cache = pd.concat([close_df_tagged, high_df_tagged, low_df_tagged, volume_df_tagged], axis=1)
        master_cache = master_cache.sort_index(axis=1)
        
        master_cache.to_parquet(cache_path)
        print(f"  [Cache] 마스터 캐시 업데이트: {cache_path} ({len(master_cache)}행)")
    except Exception as exc:
        print(f"  [Warning] 마스터 캐시 저장 실패: {exc}")
    
    min_rows = 300
    tail_rows = min(len(close_df), 340)
    
    close_df = close_df.tail(tail_rows) if len(close_df) > min_rows else close_df
    high_df = high_df.tail(tail_rows) if not high_df.empty and len(high_df) > min_rows else high_df
    low_df = low_df.tail(tail_rows) if not low_df.empty and len(low_df) > min_rows else low_df
    volume_df = volume_df.tail(tail_rows) if not volume_df.empty and len(volume_df) > min_rows else volume_df
    
    print(f"  [Return] 최종 슬라이싱: {close_df.shape}")
    return close_df, high_df, low_df, volume_df

def download_spy_close(period: str = "16mo") -> pd.Series:
    print(f"  [Download] SPY 벤치마크 다운로드 (period={period})...")
    raw = yf.download("SPY", period=period, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"] if isinstance(raw, pd.DataFrame) else raw
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def _download_url_content(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(index - 1, 0)


def _read_xlsx_first_sheet(content: bytes) -> pd.DataFrame:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(BytesIO(content)) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(".//x:si", namespace):
                texts = [node.text or "" for node in item.findall(".//x:t", namespace)]
                shared_strings.append("".join(texts))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[object]] = []
        for row in sheet_root.findall(".//x:sheetData/x:row", namespace):
            values: list[object] = []
            for cell in row.findall("x:c", namespace):
                col_idx = _xlsx_column_index(cell.attrib.get("r", "A1"))
                while len(values) <= col_idx:
                    values.append(None)

                cell_type = cell.attrib.get("t")
                value_node = cell.find("x:v", namespace)
                if cell_type == "inlineStr":
                    texts = [node.text or "" for node in cell.findall(".//x:t", namespace)]
                    value = "".join(texts)
                elif value_node is None:
                    value = None
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text or 0)]
                else:
                    value = value_node.text
                values[col_idx] = value
            rows.append(values)
    return pd.DataFrame(rows)


def _find_holdings_header_row(raw: pd.DataFrame) -> int | None:
    for idx in range(min(len(raw), 50)):
        values = [str(v).strip().lower() for v in raw.iloc[idx].tolist()]
        normalized = [re.sub(r"[^a-z0-9]", "", v) for v in values]
        has_ticker = any(v in {"ticker", "symbol", "tickersymbol"} for v in normalized)
        has_name = any(v in {"name", "holdingname", "securityname", "companyname"} for v in normalized)
        has_weight = any("weight" in v or v in {"percent", "marketvalueweight"} for v in normalized)
        if has_ticker and (has_name or has_weight):
            return idx
    return None


def _column_by_alias(columns: pd.Index, aliases: set[str], contains: str | None = None) -> str | None:
    for col in columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(col).strip().lower())
        if normalized in aliases or (contains and contains in normalized):
            return col
    return None


def _parse_weight(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text:
        return None
    try:
        weight = float(text)
    except ValueError:
        return None
    if weight > 1:
        weight = weight / 100
    return weight


def _screenable_ticker(ticker: object) -> str:
    if pd.isna(ticker):
        return ""
    value = str(ticker).strip().upper()
    if not value:
        return ""
    blocked_terms = {
        "CASH",
        "USD",
        "US DOLLAR",
        "U.S. DOLLAR",
        "TREASURY",
        "FUTURE",
        "FUTURES",
        "OPTION",
        "OPTIONS",
        "SWAP",
        "REPO",
        "COLLATERAL",
    }
    if any(term in value for term in blocked_terms):
        return ""
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,7}$", value):
        return ""
    return normalize_ticker(value)


def _records_from_holdings_table(
    table: pd.DataFrame,
    etf_ticker: str,
    source: str,
    status: str,
    limit: int,
) -> list[dict[str, object]]:
    ticker_col = _column_by_alias(table.columns, {"ticker", "symbol", "tickersymbol"})
    name_col = _column_by_alias(table.columns, {"name", "holdingname", "securityname", "companyname"}, "name")
    weight_col = _column_by_alias(table.columns, {"weight", "weightpct", "weightpercent"}, "weight")
    if weight_col is None:
        weight_col = _column_by_alias(table.columns, {"percent", "holdingpercent"}, "percent")

    if ticker_col is None:
        return []

    records: list[dict[str, object]] = []
    for _, row in table.iterrows():
        holding_ticker = str(row.get(ticker_col, "")).strip().upper()
        if not holding_ticker or holding_ticker.lower() == "nan":
            continue
        weight = _parse_weight(row.get(weight_col)) if weight_col is not None else None
        records.append(
            {
                "ETF": etf_ticker,
                "HoldingTicker": holding_ticker,
                "HoldingName": str(row.get(name_col, "")).strip() if name_col is not None else "",
                "Weight": weight,
                "Rank": len(records) + 1,
                "Source": source,
                "SourceStatus": status,
                "AsOfDate": pd.Timestamp.utcnow().normalize().date().isoformat(),
                "ScreenableTicker": _screenable_ticker(holding_ticker),
            }
        )
        if len(records) >= limit:
            break
    return records


def _fetch_official_etf_holdings(record: dict[str, object], limit: int) -> list[dict[str, object]]:
    etf_ticker = str(record.get("Ticker", "")).strip().upper()
    url = str(record.get("HoldingsUrl", "") or "").strip()
    if not url:
        return []

    content = _download_url_content(url)
    lower_url = url.lower()
    if lower_url.endswith(".xlsx") or content[:2] == b"PK":
        raw = _read_xlsx_first_sheet(content)
    else:
        text = content.decode("utf-8-sig", errors="replace")
        raw = pd.read_csv(StringIO(text), header=None, dtype=str, engine="python", on_bad_lines="skip")

    header_row = _find_holdings_header_row(raw)
    if header_row is None:
        return []

    table = raw.iloc[header_row + 1 :].copy()
    table.columns = raw.iloc[header_row].astype(str).str.strip()
    table = table.dropna(how="all")
    return _records_from_holdings_table(table, etf_ticker, "official", "official", limit)


def _fetch_yfinance_etf_holdings(etf_ticker: str, limit: int, status: str) -> list[dict[str, object]]:
    try:
        top_holdings = yf.Ticker(etf_ticker).funds_data.top_holdings
    except Exception as exc:
        print(f"  [ETF Holdings] {etf_ticker}: yfinance fallback failed ({exc})")
        return []

    if top_holdings is None or top_holdings.empty:
        return []

    table = top_holdings.reset_index()
    return _records_from_holdings_table(table, etf_ticker, "yfinance", status, limit)


def fetch_etf_top_holdings(etfs: pd.DataFrame | list[dict[str, object]] | list[str], limit: int = 10) -> pd.DataFrame:
    """
    ETF별 상위 보유종목을 다운로드합니다.

    공식 holdings 파일이 지정된 ETF는 공식 파일을 먼저 시도하고, 실패하면 yfinance의
    top_holdings를 사용합니다.
    """
    if isinstance(etfs, pd.DataFrame):
        records = etfs.reset_index(drop=False).to_dict("records")
    else:
        records = [{"Ticker": etf} if isinstance(etf, str) else etf for etf in etfs]

    all_records: list[dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        etf_ticker = str(record.get("Ticker", "")).strip().upper()
        if not etf_ticker:
            continue

        print(f"  [ETF Holdings] {idx}/{len(records)} {etf_ticker} 상위 보유종목 수집...")
        rows: list[dict[str, object]] = []
        official_url = str(record.get("HoldingsUrl", "") or "").strip()
        fallback_status = "yfinance"
        try:
            rows = _fetch_official_etf_holdings(record, limit)
            if rows:
                print(f"  [ETF Holdings] {etf_ticker}: official {len(rows)}개")
            elif official_url:
                fallback_status = "official_empty"
        except Exception as exc:
            fallback_status = f"official_failed:{type(exc).__name__}"
            print(f"  [ETF Holdings] {etf_ticker}: official failed, yfinance fallback ({exc})")

        if not rows:
            rows = _fetch_yfinance_etf_holdings(etf_ticker, limit, fallback_status)
            if rows:
                print(f"  [ETF Holdings] {etf_ticker}: yfinance {len(rows)}개")
            else:
                print(f"  [ETF Holdings] {etf_ticker}: no holdings found")

        all_records.extend(rows[:limit])
        time.sleep(random.uniform(0.2, 0.5))

    columns = [
        "ETF",
        "HoldingTicker",
        "HoldingName",
        "Weight",
        "Rank",
        "Source",
        "SourceStatus",
        "AsOfDate",
        "ScreenableTicker",
    ]
    return pd.DataFrame(all_records, columns=columns)


def _fetch_shares_batch(tickers: list[str]) -> dict[str, float | None]:
    """주어진 티커들의 발행 주식 수를 멀티스레드로 수집해 dict로 반환한다."""
    result: dict[str, float | None] = {}
    total = len(tickers)

    def _fetch_single_shares(ticker: str) -> tuple[str, float | None]:
        try:
            time.sleep(random.uniform(0.1, 0.3))  # 디도스 방어 우회용 미세 지연
            tkr = yf.Ticker(ticker)

            # yfinance에서 가장 가볍고 차단 안 당하는 full_shares 사용
            shares_series = tkr.get_shares_full()
            if shares_series is not None and not shares_series.empty:
                # 가장 최근 시점의 주식 수 반환
                return ticker, float(shares_series.iloc[-1])

            # 백업용 fast_info
            if hasattr(tkr, 'fast_info') and 'sharesOutstanding' in tkr.fast_info:
                val = tkr.fast_info['sharesOutstanding']
                return ticker, float(val) if val is not None else None
        except Exception:
            pass
        return ticker, None

    # 안정적인 크롤링을 위해 스레드는 4~5개로 제한
    max_workers = 5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_fetch_single_shares, t): t for t in tickers}
        for idx, future in enumerate(as_completed(future_to_ticker), start=1):
            ticker, shares = future.result()
            result[ticker] = shares
            if idx % 200 == 0 or idx == total:
                print(f"  [Shares] 진행률: {idx}/{total} 완료...")
    return result


def fetch_shares_outstanding(
    tickers: list[str],
    force_update: bool = False,
    max_age_days: int = 30,
) -> pd.Series:
    """
    모든 종목의 발행 주식 수(Shares Outstanding)를 가져와 로컬 캐시에 저장합니다.

    야후 파이낸스 차단을 피하기 위해 전체 재수집은 피하되, 다음 두 경우는 증분 수집한다:
      1) 캐시에 없는 신규 티커 (유니버스 편입 종목)
      2) 마지막 갱신일이 max_age_days를 초과한 오래된 티커
    force_update=True면 전체를 강제 재수집한다.
    """
    CACHE_PATH = Path(__file__).parent / ".cache" / "shares_outstanding_cache.parquet"
    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)

    today = pd.Timestamp.now().normalize()

    # 1. 기존 캐시 로드 (SharesOut + UpdatedAt 컬럼)
    cached_df = pd.DataFrame(columns=["SharesOut", "UpdatedAt"])
    if CACHE_PATH.exists():
        try:
            cached_df = pd.read_parquet(CACHE_PATH)
            if "UpdatedAt" not in cached_df.columns:
                # 구버전 캐시(주식수만 저장) 호환: 갱신일 미상으로 취급
                cached_df["UpdatedAt"] = pd.NaT
            print(f"  [Shares] 발행 주식 수 캐시 로드 ({len(cached_df):,}개 종목)")
        except Exception as e:
            print(f"  [Warning] 주식 수 캐시 로드 실패, 새로 수집합니다: {e}")
            cached_df = pd.DataFrame(columns=["SharesOut", "UpdatedAt"])

    cached_df.index = cached_df.index.astype(str)

    # 2. 수집 대상 선정
    if force_update:
        to_fetch = list(dict.fromkeys(tickers))
        print(f"  [Shares] force_update=True → 전체 {len(to_fetch):,}개 종목 재수집")
    else:
        stale_cutoff = today - pd.Timedelta(days=max_age_days)
        updated_at = pd.to_datetime(cached_df.get("UpdatedAt"), errors="coerce")
        to_fetch = []
        for t in dict.fromkeys(tickers):  # 순서 유지 + 중복 제거
            if t not in cached_df.index:
                to_fetch.append(t)                       # (1) 신규 티커
            elif pd.isna(updated_at.get(t)) or updated_at.get(t) < stale_cutoff:
                to_fetch.append(t)                       # (2) 오래된 티커
        if to_fetch:
            print(f"  [Shares] 증분 수집 대상 {len(to_fetch):,}개 "
                  f"(신규/{max_age_days}일 초과), 캐시 재사용 {len(cached_df) - len([t for t in to_fetch if t in cached_df.index]):,}개")
        else:
            print(f"  [Shares] 갱신 대상 없음 → 캐시 그대로 사용 ({len(cached_df):,}개)")

    # 3. 증분 수집 실행 및 캐시 병합
    if to_fetch:
        fetched = _fetch_shares_batch(to_fetch)
        for t, shares in fetched.items():
            # 이번 수집에서 실패(None)했지만 기존 캐시에 유효값이 있으면 캐시를 보존한다.
            if shares is None and t in cached_df.index and pd.notna(cached_df.at[t, "SharesOut"]):
                continue
            cached_df.loc[t, "SharesOut"] = shares
            cached_df.loc[t, "UpdatedAt"] = today

        # 저장
        try:
            cached_df.to_parquet(CACHE_PATH)
            print(f"  [Shares] 마스터 캐시 저장 완료 -> {CACHE_PATH} ({len(cached_df):,}개)")
        except Exception as e:
            print(f"  [Warning] 주식 수 캐시 저장 실패: {e}")

    # 4. 이번 요청 티커에 해당하는 시리즈 반환 (NaN 유지 — downstream에서 필터링)
    series = pd.to_numeric(cached_df["SharesOut"], errors="coerce")
    series = series.reindex(list(dict.fromkeys(tickers)))
    series.index.name = "Ticker"
    return series

def calculate_market_caps_from_ohlcv(close_df: pd.DataFrame, shares_series: pd.Series) -> pd.Series:
    """
    [핵심 우회 로직]
    다운로드 성공한 OHLCV의 최신 종가(Close)와 캐시된 발행 주식 수를 곱해
    실시간 시가총액을 계산합니다. (API 호출 0회)
    """
    print("  [MarketCap] 최신 종가 데이터 기반 시가총액 자체 연산 중...")
    
    # 1. OHLCV 데이터프레임의 가장 마지막 행(가장 최신 날짜)의 종가 가져오기
    # 데이터가 슬라이싱되어 있으므로 마지막 행이 오늘 혹은 직전 거래일입니다.
    latest_close = close_df.iloc[-1] 
    
    # 2. 주식 수 데이터와 티커 축을 맞춤 (인덱스 동기화)
    # yfinance 다운로드 시 하이픈(-)이나 마침표(.) 변환이 있을 수 있으므로 인덱스를 매핑합니다.
    latest_close.index.name = "Ticker"
    
    # 3. 시가총액 계산: 종가 * 발행 주식 수
    # 두 시리즈의 인덱스(Ticker)가 자동으로 매칭되어 곱해집니다.
    market_caps = latest_close * shares_series
    
    # 결측치 처리 (주식 수가 없는 경우 등)
    market_caps = market_caps.fillna(0)
    
    print(f"  [MarketCap] 연산 완료: {market_caps.notna().sum():,}개 종목 시총 확보")
    return market_caps
