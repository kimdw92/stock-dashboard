import ftplib
import json
import logging
import random
import time
from datetime import datetime, time as dt_time
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf
import re
import urllib.request

# yfinance의 티커별 실패 로그 억제 ("$XXX: possibly delisted...", "N Failed downloads" 등).
# 초저유동성 SPAC 유닛·정크 티커가 매 실행 수십 줄의 노이즈를 만들지만 실제 문제가 아니며,
# 레이트리밋 등 실질 장애는 성공률 기반 감지(_run_chunks)가 자체 [RateLimit]/[Abort]
# 로그로 알리므로 진단 능력은 유지된다.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# yf.download는 티커당 HTTP 요청 1건을 보낸다(배치 엔드포인트가 아님). 즉 배치 크기
# = 요청 수이며, 야후의 IP당 한도는 "총 요청 수 × 순간 동시성"으로 소진된다.
# threads=True는 min(len(chunk), cpu_count()*2)까지 올라가(이 머신 기준 44) 초당
# 수십 건을 쏘므로 3천 건 언저리에서 429가 터진다. 동시성을 명시적으로 낮게 고정한다.
DOWNLOAD_CHUNK_SIZE = 300  # 배치당 요청 버스트를 낮춰 레이트리밋 확률 감소
DOWNLOAD_THREADS = 8       # yf.download 동시 요청 수 (True로 두면 cpu_count()*2까지 상승)
DOWNLOAD_CHUNK_PAUSE = (5.0, 9.0)  # 청크 간 대기(초) 범위 — 분당 요청 속도 상한 역할
DOWNLOAD_MAX_RETRIES = 3       # 배치 다운로드 실패 시 재시도 횟수
DOWNLOAD_RETRY_BACKOFF = 3.0  # 재시도 간 기본 대기(초), 시도마다 배수 증가
RATE_LIMIT_MIN_SUCCESS = 0.3  # 배치 성공률이 이 값 미만이면 레이트리밋으로 판정
# 레이트리밋 재시도: 야후의 차단 창은 분 단위라 90초 대기로는 거의 풀리지 않는다.
# 길게 한 번만 기다려 보고 그래도 막혀 있으면 즉시 중단한다 — 남은 티커는 다음
# 실행이 "오래된 순" 정렬 덕분에 이어받으므로, 여기서 15분씩 붙잡고 있을 이유가 없다.
RATE_LIMIT_COOLDOWN = 240.0   # 레이트리밋 판정 시 재시도 전 대기(초)
RATE_LIMIT_MAX_RETRIES = 2    # 레이트리밋 상황에서의 최대 시도 횟수(초과 시 중단)
# 증분 시작일을 과거로 되돌릴 수 있는 최대 일수. 장기 미갱신 티커 하나가 전체 배치의
# 요청 구간을 몇 달치로 늘리는 것을 막는다(그런 티커는 need_full 경로가 처리).
INCR_MAX_LOOKBACK_DAYS = 45
# 요청했는데도 야후가 새 봉을 주지 않은 횟수가 이 값에 도달하면 "사실상 데이터 없음"으로
# 보고 우선순위 큐 뒤로 보낸다. 상장폐지·거래정지 티커는 몇 번을 물어도 응답이 없는데,
# 이들을 계속 앞에 두면 정말 밀린 티커의 자리를 뺏고 요청 구간만 늘어난다.
STALE_MISS_LIMIT = 3
# ── 미조정 스플릿(액면분할/병합) 감지 ────────────────────────────────────────
# 마스터 캐시는 증분 append 구조라, 스플릿 이전에 저장된 과거 행은 옛 가격 스케일로
# 남는다. yfinance의 auto_adjust는 *그 요청으로 받아온 구간*만 소급 조정하므로,
# 새 날짜만 이어붙이면 한 시계열 안에 조정 전/후 스케일이 섞인다. RS Score는 가격
# '비율'이라 이 이음매를 그대로 수익률로 읽어, 스플릿 배수만큼 점수가 부풀거나 쪼그라든다.
# (2026-07-31 캐시 실측: 1:10 리버스 스플릿을 맞은 PSQH가 rs_score 3.68 = 백분위 99.5로
#  전체 31위. 보정하면 0.41 = 백분위 7.0. RS 상위 100종목 중 23개가 같은 유형이었다.)
#
# 구분 기준은 거래량이다. 리버스 스플릿은 주식 수를 기계적으로 나누므로 가격이 N배
# 뛰는 대신 거래량이 ~1/N로 줄지만, 진짜 급등은 거래량이 함께 폭발한다.
# (같은 실측: 상승 점프 173건의 거래량비 중앙값은 12.6배. 스플릿 의심군은 1.0 미만에
#  몰려 있고 그 사이 구간이 비어 있어, 1.0이 두 분포의 골에 해당한다.)
#
# 감지 대상은 상승 점프(리버스 스플릿)로 한정한다. 정방향 분할도 원리상 RS를 왜곡하지만
# (가격 1/N → 폭락처럼 보임), 이 캐시에는 표본이 없어 임계값을 맞출 수가 없다. 실측한
# 하락 점프 153건은 전부 달러 거래량이 함께 붕괴(비율 0.01~0.6)한 마이크로캡 실제 폭락으로,
# 달러 거래량이 보존(≈1)되는 분할 서명과 다르다. 근거 없는 규칙을 넣으면 죽어가는 종목만
# 매번 재다운로드하게 된다. 피해 규모도 비대칭이다 — 리버스 스플릿은 점수를 부풀려 상위
# 순위를 오염시키지만, 정방향 분할은 좋은 종목 하나를 가릴 뿐 남의 순위를 밀어내지 않는다.
SPLIT_JUMP_RATIO = 3.0       # 하루 만에 이 배수 이상 오르면 점프 후보
SPLIT_VOL_RATIO_MAX = 1.0    # 점프 후/전 거래량 중앙값 비율이 이 값 미만이면 스플릿 판정
SPLIT_VOL_WINDOW = 10        # 거래량 비교 구간(거래일)
SPLIT_SCAN_LOOKBACK = 252    # 스캔 범위 = RS 최장 창(252일). 그보다 과거의 스플릿은
                             # 계산 구간이 전부 조정 후 스케일이라 왜곡을 일으키지 않는다.
# ── 선제적 쿨다운(중간 휴식) ─────────────────────────────────────────────────
# 야후의 제한은 "하루 총량 쿼터"가 아니라 짧은 윈도우의 속도 제한으로 보인다.
# 근거(2026-07-29 실측): 누적 ~3,300건에서 차단됐으나 247초 대기 후 추가 2,100건이
# 문제없이 통과했다. 총량 쿼터였다면 4분 만에 그만큼 더 들어갈 수 없다.
#
# 따라서 "실행당 총량을 제한하고 남은 티커를 다음 실행으로 넘기는" 방식은 맞지 않는다.
# 그렇게 하면 1회 실행이 유니버스의 절반만 갱신하고, 나머지 절반은 ffill 때문에
# 전일 종가가 그대로 남아(등락률 0%) RS 계산이 어긋난다. 총 소요 시간도 더 길다.
#
# 대신 차단당하기 *전에* 미리 쉰다: REQUESTS_BEFORE_PAUSE건을 보낼 때마다
# DOWNLOAD_PAUSE_SECONDS만큼 쉬고 카운터를 리셋한 뒤 계속 진행한다. 1회 실행으로
# 전체 유니버스를 커버하는 것이 목표다.
# 임계값 이력: 콜드 스타트 차단 지점(~3,300건) 기준으로 3,000을 잡았으나, 2026-08-03
# 실행에서 *휴식 이후* 구간이 2,837건에서 차단됐다. 300초 휴식이 야후 카운터를 완전히
# 되돌리지 못한다는 뜻이므로, 콜드 스타트가 아니라 이 2,837을 기준으로 15% 마진을 둔다.
# (5,618건 기준 휴식 1회 → 2회로 늘지만, 240초 레이트리밋 쿨다운과 재시도 낭비가 사라져
#  총 대기시간은 비슷하고 마지막 배치가 통째로 날아갈 위험이 없어진다.)
# 그래도 막히면 아래쪽의 성공률 감지(RATE_LIMIT_*)가 240초 쿨다운 → 재시도 → 중단 순으로
# 처리하고, 중단되더라도 "오래된 순" 정렬 덕분에 다음 실행이 밀린 티커부터 이어받는다.
# 재시도도 실제 요청을 소모하므로 카운터는 시도 단위로 가산한다.
REQUESTS_BEFORE_PAUSE = 2400
DOWNLOAD_PAUSE_SECONDS = 300.0
# ── 유동성 하한(휴면 티커) ───────────────────────────────────────────────────
# 유니버스 필터는 구조적 조건(티커 형식·워런트·우선주)만 걸러서, 일평균 거래대금이
# $0인 종목까지 매일 요청하고 있었다. 이들은 스크리너 기준(ADV $10M)의 수십분의 1이라
# 결과에 등장할 수 없으면서 요청 예산과 RS 백분위 모집단만 차지한다.
# (2026-08-03 실측: 6,594종목 중 1,875개가 ADV $250k 미만. 반대로 현재 스크리너를
#  통과하는 309종목 중 ADV $500k 미만은 0개 — 하한을 걸어도 실사용 영향이 없다.)
#
# 하한 미달 티커는 '휴면'으로 두고 반환 프레임에서 제외한다. 다운로드만 건너뛰고 프레임에
# 남기면 ffill로 전일 종가가 굳어(등락률 0%) RS가 오히려 더 나빠지므로, 빼려면 계산에서도
# 완전히 빼야 한다. 캐시 행 자체는 지우지 않는다(다시 유동성이 붙으면 이어서 쓴다).
UNIVERSE_MIN_DOLLAR_VOLUME = 250_000.0
UNIVERSE_ADV_WINDOW = 60     # 거래대금 중앙값 산출 구간(거래일)
# 휴면 티커도 갱신을 아예 멈추면 ADV가 과거 값에 얼어붙어 영원히 복귀하지 못한다.
# 매 실행 가장 오래된 일부만 새로고침해 복귀 판정이 계속 살아 있게 한다(요청 예산 상한).
# 250은 요청 예산에서 역산한 값이다: 활성 4,513 + 250 = 4,763건이면 REQUESTS_BEFORE_PAUSE
# 기준 휴식이 1회로 끝난다(300건만 더 늘어도 2회가 되어 300초를 더 쓴다). 이 속도면
# 휴면 전체를 약 4~5회 실행(일간 실행 기준 1주일)에 한 바퀴 돌아 복귀 감지에 충분하다.
DORMANT_REFRESH_PER_RUN = 250
# yf.Ticker 계열(.info / get_shares_full 등) 메타데이터 수집의 실행당 최대 요청 수.
# 캐시 만료가 한꺼번에 도래해도(예: 90일/30일 주기) 하루 요청량이 폭증하지 않도록
# 상한을 두고, 초과분은 다음 실행으로 이월한다. OHLCV(~6.5천 건)와 합산해도
# IP 차단 임계 아래에 머물게 하는 것이 목적.
INFO_FETCH_PER_RUN_CAP = 1500


# yfinance 요청에 재사용할 curl_cffi 브라우저 임퍼소네이션 세션(모듈 전역 1개).
# 커넥션 풀 재사용 + 브라우저 지문으로 야후 차단(429)을 줄여 재시도/대기를 감소시킨다.
# 전역 1개로 충분한 이유: yfinance는 넘겨받은 세션을 싱글턴 YfData에 심어
# yf.download의 워커 스레드 전부가 공유하고(multi.py의 `YfData(session=session)`),
# 이 모듈의 호출자는 모두 메인 스레드다. 동시성은 DOWNLOAD_THREADS로 별도 제한.
# curl_cffi 미설치 등 실패 시 None → yfinance 기본 세션으로 폴백.
_YF_SESSION = None
_YF_SESSION_INIT = False


def _get_yf_session():
    """공유 curl_cffi 세션을 반환한다(최초 1회 생성, 실패 시 None → 기본 세션)."""
    global _YF_SESSION, _YF_SESSION_INIT
    if _YF_SESSION_INIT:
        return _YF_SESSION
    _YF_SESSION_INIT = True
    try:
        from curl_cffi import requests as _cffi_requests
        _YF_SESSION = _cffi_requests.Session(impersonate="chrome")
        print("  [Session] curl_cffi 브라우저 임퍼소네이션 세션 활성화")
    except Exception as exc:  # 미설치/호환 문제 → 기본 세션 폴백
        print(f"  [Session] curl_cffi 세션 비활성화(기본 세션 사용): {exc}")
        _YF_SESSION = None
    return _YF_SESSION

def _cache_path(filename: str) -> Path:
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / filename

def normalize_ticker(ticker: str) -> str:
    return str(ticker).replace('.', '-')


def _read_json(path: Path) -> dict | None:
    """JSON 파일을 읽어 dict 반환. 없거나 파싱 실패 시 None (호출부는 캐시 없음으로 취급)."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json_atomic(path: Path, data: dict, warn_label: str) -> bool:
    """원자적(tmp→replace) JSON 저장. 실패는 경고만 남기고 False 반환(파이프라인 계속)."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)
        return True
    except Exception as exc:
        print(f"  [Warning] {warn_label} 저장 실패: {exc}")
        return False


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


# 다운로드 유니버스에서 제외할 비정상 증권 유형 (Security Name 기반).
# 유닛/권리/워런트/우선주/채권/SPAC — RS 스크리닝 대상이 아니며, 데이터도 희소해
# 다운로드·메타데이터 요청만 낭비한다. 단어 경계(\b)를 써서 오탐을 방지한다:
#   - "United Airlines"의 unit, "Bright..."의 right 등은 매칭되지 않음
#   - ADR("American Depositary Shares")는 정상 종목이므로 depositary는 패턴에 넣지 않음
#     (우선주 예탁증서는 preferred 키워드가 잡아줌)
UNIVERSE_JUNK_NAME_PAT = re.compile(
    r"\bunits?\b|\brights?\b|\bwarrants?\b|\bpreferred\b|\bnotes? due\b"
    r"|acquisition corp|acquisition co\b",
    re.IGNORECASE,
)


def _exclude_shell_industry(universe: pd.DataFrame) -> pd.DataFrame:
    """sector 캐시의 industry='Shell Companies' 라벨(SPAC)을 유니버스에서 제외.

    캐시가 없으면 그대로 반환(best-effort). 이름 패턴이 못 잡는 SPAC 보통주를 걸러낸다.
    """
    if universe.empty:
        return universe
    sec = _read_json(Path(__file__).parent / ".cache" / "sector_industry_map.json")
    if not sec:
        return universe
    is_shell = universe["Ticker"].map(
        lambda t: sec.get(t, {}).get("industry", "") == "Shell Companies"
    )
    n_shell = int(is_shell.sum())
    if n_shell:
        print(f"  [Universe] Shell Companies(SPAC) {n_shell}개 제외")
    return universe[~is_shell]


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
        df = df[~is_warrant]

        # 3. Security Name 기반 비정상 증권 제외 (유닛/권리/워런트/우선주/채권/SPAC)
        if "Security Name" in df.columns:
            junk = df["Security Name"].astype(str).str.contains(
                UNIVERSE_JUNK_NAME_PAT, na=False, regex=True
            )
            if junk.any():
                print(f"  [Universe] 비정상 증권(유닛/워런트/우선주 등) {int(junk.sum())}개 제외")
            df = df[~junk]

        return df

    # (파일, 심볼 컬럼, 거래소 컬럼) — 거래소 컬럼이 None이면 NASDAQ 고정
    file_specs = [
        (nasdaq_path, "Symbol", None),
        (other_path, "ACT Symbol", "Exchange"),
    ]
    dfs = []
    for path, sym_col, exch_col in file_specs:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep='|')
            df = df.dropna(subset=[sym_col])
            df = df[df[sym_col] != '']
            df = df[~df[sym_col].str.contains('File Creation', na=False)]  # 푸터 행 제거
            if 'ETF' in df.columns:
                df = df[df['ETF'] == 'N']
            if 'Test Issue' in df.columns:
                df = df[df['Test Issue'] == 'N']
            df = clean_tickers(df, sym_col)
            dfs.append(pd.DataFrame({
                'Ticker': df[sym_col],
                'Exchange': df[exch_col] if exch_col else 'NASDAQ',
            }))
        except Exception as e:
            print(f"[Error] {path.name} 로드 실패: {e}")

    if dfs:
        res = pd.concat(dfs, ignore_index=True)
        # SPAC 보통주 제외 (이름 패턴이 못 잡는 케이스를 industry 라벨로 보완)
        res = _exclude_shell_industry(res)
        return res
    else:
        print("[Warning] No universe files found. Falling back to default tickers.")
        return pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOGL", "NVDA"], "Exchange": ["NASDAQ"]*4})

def _sequential_fetch(tickers: list[str], fetch_one, label: str, is_ok=None) -> dict:
    """티커별 메타데이터를 순차 수집해 {티커: 결과} dict로 반환한다 (진행률 로그 포함).

    yf.Ticker 계열(.info / get_shares_full 등) 엔드포인트는 야후 crumb 인증 때문에
    동시 요청 시 HTTP 401("Invalid Crumb")이 대량 발생한다(실측: 5스레드 성공률 ~16%,
    순차 100%). 따라서 병렬을 쓰지 않는다.
    """
    if is_ok is None:
        is_ok = lambda v: v is not None
    result: dict = {}
    total = len(tickers)
    ok = 0
    for idx, ticker in enumerate(tickers, start=1):
        val = fetch_one(ticker)
        result[ticker] = val
        if is_ok(val):
            ok += 1
        if idx % 200 == 0 or idx == total:
            print(f"  [{label}] 진행률: {idx}/{total} (확보 {ok})")
    return result


def _cap_fetch_list(primary: list[str], secondary: list[str], label: str,
                    detail: str, total_requested: int) -> list[str]:
    """수집 대상을 실행당 상한(INFO_FETCH_PER_RUN_CAP)으로 자르고 로그를 남긴다.

    primary가 secondary보다 우선하며, 초과분은 다음 실행으로 이월된다.
    빈 리스트 반환 시 "갱신 대상 없음" 로그는 호출부 몫(캐시 크기 표기가 달라서).
    """
    to_fetch = (primary + secondary)[:INFO_FETCH_PER_RUN_CAP]
    deferred = len(primary) + len(secondary) - len(to_fetch)
    if to_fetch:
        print(f"  [{label}] 증분 수집 대상 {len(to_fetch):,}개 ({detail}"
              f"{f', 상한 초과 {deferred:,}건은 다음 실행으로 이월' if deferred else ''}), "
              f"캐시 재사용 {total_requested - len(to_fetch):,}개")
    return to_fetch


def _fetch_single_sector_industry(ticker: str, retries: int = 2) -> dict[str, str] | None:
    """단일 티커의 sector/industry를 yfinance에서 조회한다.

    반환값 구분:
      - 정상 응답: {"sector", "industry", "status": "ok"}
        (sector/industry가 실제로 비어 있어도 status="ok" — Yahoo에 데이터가 없는 종목)
      - 차단/오류(429·빈 응답 등): None
        → 호출부에서 status="failed"로 저장돼 다음 실행 때 짧은 주기로 재시도된다.

    차단 완화를 위해 요청 간 지연을 늘리고, 실패 시 백오프 재시도한다.
    """
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(0.3, 0.8))  # 차단 완화용 지연
            info = yf.Ticker(ticker, session=_get_yf_session()).info
            if not info:
                raise ValueError("empty info")  # 빈 응답은 차단으로 간주 → 재시도
            return {"sector": str(info.get("sector") or ""),
                    "industry": str(info.get("industry") or ""),
                    "status": "ok"}
        except Exception:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1.0))  # 지수 백오프
    return None


def fetch_sector_industry_map(
    tickers: list[str],
    force_update: bool = False,
    max_age_days: int = 90,
    empty_retry_days: int = 3,
) -> dict[str, dict[str, str]]:
    """
    모든 종목의 sector/industry 메타데이터를 yfinance에서 수집해 로컬 캐시에 저장한다.

    야후 파이낸스 차단을 피하기 위해 전체 재수집은 피하되, 다음 경우는 증분 수집한다:
      1) 캐시에 없는 신규 티커 (유니버스 편입 종목)
      2) 정상 수집된 티커의 갱신일이 max_age_days를 초과한 경우
      3) 차단/오류로 값이 비어 있는 티커(status != "ok")의 갱신일이
         empty_retry_days를 초과한 경우 — 차단으로 빈 값이 영구화되지 않도록
         짧은 주기로 재시도한다. (status="ok"인 진짜 데이터 없는 종목은 재시도 안 함)
    force_update=True면 전체를 강제 재수집한다.

    반환: {ticker: {"sector": str, "industry": str}} (요청 티커에 한정)
    """
    cache_path = Path(__file__).parent / ".cache" / "sector_industry_map.json"
    cache_path.parent.mkdir(exist_ok=True, parents=True)

    today_str = pd.Timestamp.now().normalize().date().isoformat()

    # 1. 기존 캐시 로드 (각 항목: {"sector", "industry", "updated_at"}) — 실패 시 재수집
    cache: dict[str, dict[str, str]] = _read_json(cache_path) or {}
    if cache:
        print(f"  [Sector] sector/industry 캐시 로드 ({len(cache):,}개 종목)")

    # 2. 수집 대상 선정
    unique_tickers = list(dict.fromkeys(tickers))  # 순서 유지 + 중복 제거
    if force_update:
        to_fetch = unique_tickers
        print(f"  [Sector] force_update=True → 전체 {len(to_fetch):,}개 종목 재수집")
    else:
        now = pd.Timestamp.now().normalize()
        stale_cutoff = now - pd.Timedelta(days=max_age_days)
        empty_retry_cutoff = now - pd.Timedelta(days=empty_retry_days)
        primary = []      # (1) 신규 + (2) 만료 — 우선 수집
        retry_empty = []  # (3) 빈 값 재시도 — 남는 예산으로 수집
        for t in unique_tickers:
            entry = cache.get(t)
            if entry is None:
                primary.append(t)                        # (1) 신규 티커
                continue
            updated = pd.to_datetime(entry.get("updated_at"), errors="coerce")
            has_value = bool(entry.get("sector") or entry.get("industry"))
            if has_value or entry.get("status") == "ok":
                # 값이 있거나, 정상 수집됐지만 진짜 데이터가 없는 종목 → 장기 주기
                if pd.isna(updated) or updated < stale_cutoff:
                    primary.append(t)                    # (2) 오래된 티커
            else:
                # 차단/오류로 빈 값(status="failed" 또는 레거시 무status) → 단기 재시도
                if pd.isna(updated) or updated < empty_retry_cutoff:
                    retry_empty.append(t)                # (3) 빈 값 재시도
        to_fetch = _cap_fetch_list(
            primary, retry_empty, "Sector",
            f"신규/만료 {len(primary):,} + 빈값 재시도 {len(retry_empty):,}",
            len(unique_tickers),
        )
        if not to_fetch:
            print(f"  [Sector] 갱신 대상 없음 → 캐시 그대로 사용 ({len(cache):,}개)")

    # 3. 증분 수집 실행 및 캐시 병합
    if to_fetch:
        fetched = _sequential_fetch(
            to_fetch, _fetch_single_sector_industry, "Sector",
            is_ok=lambda v: bool(v and (v.get("sector") or v.get("industry"))),
        )
        for t, info in fetched.items():
            # 이번 수집에서 실패(None)했지만 기존 캐시에 유효값이 있으면 캐시를 보존한다.
            if info is None:
                if t in cache and (cache[t].get("sector") or cache[t].get("industry")):
                    continue
                # 차단/오류 → status="failed"로 표시해 다음 실행 때 단기 재시도되게 한다.
                cache[t] = {"sector": "", "industry": "", "status": "failed",
                            "updated_at": today_str}
            else:
                cache[t] = {**info, "updated_at": today_str}  # info에 status="ok" 포함

        if _write_json_atomic(cache_path, cache, "sector/industry 캐시"):
            print(f"  [Sector] 마스터 캐시 저장 완료 -> {cache_path} ({len(cache):,}개)")

    # 4. 요청 티커에 해당하는 sector/industry만 반환 (내부 메타데이터 updated_at 제외)
    result: dict[str, dict[str, str]] = {}
    for t in unique_tickers:
        entry = cache.get(t)
        if entry and (entry.get("sector") or entry.get("industry")):
            result[t] = {"sector": entry.get("sector", ""), "industry": entry.get("industry", "")}
    return result

def _extract_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """yf.download 결과에서 필드 하나를 (날짜 × 티커) 프레임으로 추출.

    티커를 리스트로 넘기면 결과는 항상 (필드, 티커) MultiIndex 컬럼이다.
    """
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()
    if field not in raw.columns.levels[0]:
        return pd.DataFrame()
    out = raw.xs(field, level=0, axis=1)
    return out if isinstance(out, pd.DataFrame) else out.to_frame()

def _has_new_trading_day(start: str, probe_ticker: str = "SPY") -> bool:
    """[start, today] 구간에 실제 거래일(신규 데이터)이 있는지 확인.

    프로브 게이트: 대량(수천 티커) 증분 배치를 돌리기 전에 새 거래일 유무를
    판별한다. 주말·공휴일·같은 날 재실행이면 False → 배치 전체를 생략.

    1차(로컬, 요청 0건): 구간에 평일이 하나도 없으면(주말 실행) 즉시 False.
      레이트리밋 등 네트워크 상태와 무관하게 확정적으로 동작한다.
    2차(프로브, 요청 1건): 평일이 있으면 SPY로 실데이터 유무 확인(공휴일 걸러냄).
      프로브 자체가 실패하면(네트워크 등) 안전하게 True를 반환해 정상 경로로 진행한다.
    """
    # 1차: 주말 로컬 판정 — [start, 오늘]에 평일이 없으면 거래일도 없다
    if len(pd.bdate_range(start, pd.Timestamp.now().normalize())) == 0:
        return False
    try:
        # 2차: start보다 10일 이전부터 요청해 서비스 정상 여부와 신규 거래일을 동시 판별.
        # 이 확장 구간은 평일을 반드시 포함하므로, 서비스가 정상이면 SPY 데이터가
        # 반드시 존재한다. 따라서 빈 응답 = 레이트리밋/서비스 이상이며, 이때 False를
        # 반환하면 거래일인데도 업데이트를 조용히 건너뛰는 오판이 되므로 True로
        # 진행한다(이후 배치의 성공률 감지가 실제 차단 여부를 처리).
        ext_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        probe = yf.download(
            [probe_ticker], start=ext_start, threads=False, progress=False,
            auto_adjust=True, session=_get_yf_session(),
        )
        valid = probe.dropna(how="all") if probe is not None else pd.DataFrame()
        if valid.empty:
            print("  [Probe] SPY 응답 없음(레이트리밋/서비스 이상 의심) → 증분 진행")
            return True
        idx = pd.to_datetime(valid.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return bool((idx >= pd.Timestamp(start)).any())
    except Exception as exc:
        print(f"  [Probe] 거래일 확인 실패({type(exc).__name__}) → 증분 진행: {exc}")
        return True


# ── 잠정(intraday) 봉 마커 ─────────────────────────────────────────────────────
# 장중 실행 시 저장된 그날의 봉은 최종 종가가 아닌 잠정 스냅샷이다. 캐시 최신일이
# 잠정 상태임을 sidecar JSON에 기록해, 이후(장 마감 후) 실행이 그 날짜를 다시
# 요청해 최종 종가로 교체하도록 한다. (병합은 combine_first로 새 값이 우선이므로
# 재요청만 하면 잠정 봉이 최종 종가로 정확히 덮어써진다.)
US_MARKET_CLOSE_ET = dt_time(16, 15)  # 16:00 정규장 마감 + 15분 정산 버퍼


def _bar_is_finalized(bar_date, now_et: datetime | None = None) -> bool:
    """`bar_date`(거래일)의 봉이 최종 확정됐는지(미 동부시각 장 마감 통과) 판정.

    - 과거 날짜: 항상 확정.
    - 오늘(ET) 날짜: 마감(16:15 ET)이 지났으면 확정, 아니면 잠정.
    반환 True=최종 확정 / False=잠정(장중 캡처 가능성).
    반휴장일(13:00 마감)은 드물어 별도 처리하지 않는다 — 잠정 오판 시에도 다음
    실행에서 동일 최종값으로 1회 더 덮어쓸 뿐 무해하다.
    """
    d = pd.Timestamp(bar_date).date()
    if now_et is None:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.date() > d:
        return True
    if now_et.date() < d:
        return False
    return now_et.time() >= US_MARKET_CLOSE_ET


def _provisional_path(cache_path: Path) -> Path:
    """마스터 캐시 옆 잠정 봉 마커 파일 경로 (주식/ETF 캐시별로 분리)."""
    return cache_path.with_name(cache_path.stem + ".provisional.json")


def _read_provisional(cache_path: Path):
    """잠정 마커의 날짜(pd.Timestamp)를 반환. 없거나 읽기 실패 시 None."""
    d = (_read_json(_provisional_path(cache_path)) or {}).get("provisional_date")
    try:
        return pd.Timestamp(d) if d else None
    except Exception:
        return None


def _write_provisional(cache_path: Path, bar_date) -> None:
    """최신 봉이 잠정 상태임을 기록 (원자적 쓰기, 실패는 무해하게 무시)."""
    _write_json_atomic(
        _provisional_path(cache_path),
        {"provisional_date": pd.Timestamp(bar_date).strftime("%Y-%m-%d")},
        "잠정 마커",
    )


def _clear_provisional(cache_path: Path) -> None:
    """최신 봉이 확정됐으므로 잠정 마커 제거."""
    p = _provisional_path(cache_path)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


# ── 연속 미수신 카운터 ────────────────────────────────────────────────────────
# "캐시가 밀렸다"는 사실만으로는 레이트리밋 피해자와 상장폐지/거래정지 티커를
# 구분할 수 없다. 둘 다 최신봉이 과거에 멈춰 있기 때문이다. 그래서 데이터가 아니라
# *요청 결과*를 기록한다 — 요청했는데도 새 봉이 오지 않으면 카운트를 올리고,
# 오면 0으로 되돌린다. 몇 번을 물어도 응답이 없는 티커만 후순위로 밀려난다.

def _stale_path(cache_path: Path) -> Path:
    """마스터 캐시 옆 연속 미수신 카운터 파일 경로."""
    return cache_path.with_name(cache_path.stem + ".stale.json")


def _read_stale_misses(cache_path: Path) -> dict[str, int]:
    """{티커: 연속 미수신 횟수}. 없거나 읽기 실패 시 빈 dict."""
    data = _read_json(_stale_path(cache_path)) or {}
    try:
        return {str(k): int(v) for k, v in data.get("misses", {}).items()}
    except Exception:
        return {}


def _write_stale_misses(cache_path: Path, misses: dict[str, int]) -> None:
    """연속 미수신 카운터 저장 (원자적 쓰기, 실패는 무해하게 무시)."""
    _write_json_atomic(
        _stale_path(cache_path),
        {"updated": pd.Timestamp.now().strftime("%Y-%m-%d"), "misses": misses},
        "미수신 카운터",
    )


# ── 스플릿 보수 기록 ──────────────────────────────────────────────────────────
# 점프 하나를 어디까지 처리했는지 단계로 기록한다. {티커: {"jump": 날짜, "stage": 단계}}
#   redownloaded : 전체 기간을 다시 받아봤다 (야후가 스플릿을 알면 여기서 해결된다)
#   adjusted     : 재다운로드로도 안 고쳐져 우리가 직접 소급 보정했다
#
# 두 단계로 나누는 이유는 순서가 중요해서다. 야후가 스플릿을 인지한 경우 재다운로드가
# 정본 데이터를 주므로 항상 먼저 시도하고, 그래도 점프가 남을 때만 자체 보정으로 넘어간다.
# (2026-08-03 실측: 재다운로드한 46종목 중 23종목만 해결됐다. 나머지 23종목은 야후 원본
#  자체가 미조정이라 몇 번을 다시 받아도 같은 점프가 돌아온다.)
#
# 기록은 무한 루프 방지 역할도 겸한다. 같은 점프를 이미 처리했으면 다시 플래그하지 않으므로,
# 거래량까지 줄어든 진짜 급등이 매 실행 재다운로드를 반복하는 일이 없다.

# 재다운로드로 봉 구성이 바뀌면(빠졌던 봉이 채워지는 등) 같은 사건인데도 이음매가
# 하루이틀 밀린다. 날짜를 정확히 일치시키면 그때마다 새 점프로 오인해 재다운로드와
# 보정 사이를 오간다. (실측: DFNS는 기록 2026-07-21, 재다운로드 후 감지 2026-07-20)
SPLIT_JUMP_MATCH_DAYS = 5


def _same_jump(a: str, b: str) -> bool:
    """두 점프일이 같은 사건을 가리키는지 (SPLIT_JUMP_MATCH_DAYS 이내면 동일)."""
    if not a or not b:
        return False
    try:
        return abs((pd.Timestamp(a) - pd.Timestamp(b)).days) <= SPLIT_JUMP_MATCH_DAYS
    except Exception:
        return a == b


def _splitfix_path(cache_path: Path) -> Path:
    """마스터 캐시 옆 스플릿 보수 기록 파일 경로."""
    return cache_path.with_name(cache_path.stem + ".splitfix.json")


def _read_splitfix(cache_path: Path) -> dict[str, dict[str, str]]:
    """{티커: {"jump": 'YYYY-MM-DD', "stage": ...}}. 없거나 실패 시 빈 dict.

    구버전 포맷({티커: 날짜문자열})은 재다운로드까지 마친 상태로 읽는다.
    """
    data = _read_json(_splitfix_path(cache_path)) or {}
    out: dict[str, dict[str, str]] = {}
    try:
        for k, v in (data.get("repaired") or {}).items():
            if isinstance(v, dict):
                jump, stage = str(v.get("jump", "")), str(v.get("stage", "redownloaded"))
            else:  # 구버전: 날짜 문자열만 저장했고, 그 시점 동작은 재다운로드였다
                jump, stage = str(v), "redownloaded"
            if jump:
                out[str(k)] = {"jump": jump, "stage": stage}
    except Exception:
        return {}
    return out


def _write_splitfix(cache_path: Path, repaired: dict[str, dict[str, str]]) -> None:
    """스플릿 보수 기록 저장 (원자적 쓰기, 실패는 무해하게 무시)."""
    _write_json_atomic(
        _splitfix_path(cache_path),
        {"updated": pd.Timestamp.now().strftime("%Y-%m-%d"), "repaired": repaired},
        "스플릿 보수 기록",
    )


def _backadjust_split(
    frames: dict[str, pd.DataFrame],
    ticker: str,
    jump_date: str,
) -> float | None:
    """점프 이전 구간을 소급 보정해 시계열 스케일을 잇는다(제자리 수정).

    야후가 스플릿을 인지하지 못해 재다운로드로도 고쳐지지 않는 티커의 최후 수단이다.
    yfinance의 back-adjust와 같은 방향으로, 점프 배수 R을 과거 가격에 곱하고 과거
    거래량은 나눈다(분할 전에는 주식 수가 R배 많았으므로).

    R은 실제 분할비(1:10 등)가 아니라 이음매의 관측 배수라, 그날의 진짜 등락분까지
    함께 흡수해 하루치 수익률이 0에 가까워진다. 9배짜리 가짜 점프를 남겨두는 것에
    비하면 무시할 수 있는 오차다.

    Returns:
        적용한 배수 R. 대상 행/값이 부적합해 보정하지 않았으면 None.
    """
    close = frames["close"]
    # 중복 컬럼이면 close[ticker]가 DataFrame이 되어 아래 스칼라 접근이 깨진다.
    # 병합 경로상 나올 일이 없지만, 조용히 잘못된 배수를 적용하느니 건너뛴다.
    if ticker not in close.columns or (close.columns == ticker).sum() != 1:
        return None
    idx = close.index
    pos = idx.get_indexer([pd.Timestamp(jump_date)])[0]
    if pos < 1:  # 점프일이 프레임에 없거나 첫 행이면 이전 구간이 없다
        return None

    cur, prev = close[ticker].iat[pos], close[ticker].iat[pos - 1]
    if not (np.isfinite(cur) and np.isfinite(prev)) or prev <= 0 or cur <= 0:
        return None
    ratio = float(cur) / float(prev)
    if ratio < SPLIT_JUMP_RATIO:  # 이미 보정됐거나 점프가 아니다
        return None

    for key in ("close", "high", "low"):
        df = frames.get(key)
        if df is not None and not df.empty and ticker in df.columns:
            df.iloc[:pos, df.columns.get_loc(ticker)] *= ratio
    vol = frames.get("volume")
    if vol is not None and not vol.empty and ticker in vol.columns:
        vol.iloc[:pos, vol.columns.get_loc(ticker)] /= ratio
    return ratio


def _split_by_liquidity(
    cached_close: pd.DataFrame | None,
    cached_volume: pd.DataFrame | None,
    tickers: list[str],
    floor: float,
) -> tuple[list[str], list[str]]:
    """유동성 하한으로 티커를 (활성, 휴면)으로 가른다.

    기준은 최근 UNIVERSE_ADV_WINDOW 거래일의 일간 거래대금(종가×거래량) 중앙값이다.
    평균이 아니라 중앙값을 쓰는 이유는 단발성 급등 하루가 1년 내내 죽어 있던 종목을
    통과시키지 못하게 하기 위해서다.

    캐시에 없는 티커(신규 상장 등)는 판단 근거가 없으므로 항상 활성으로 둔다 —
    데이터가 없다고 빼버리면 영영 받지 못한다.

    Returns:
        (활성 티커, 휴면 티커) — floor<=0이거나 캐시가 없으면 전부 활성
    """
    if floor <= 0 or cached_close is None or cached_volume is None:
        return list(tickers), []
    if cached_close.empty or cached_volume.empty:
        return list(tickers), []

    known = [t for t in tickers if t in cached_close.columns and t in cached_volume.columns]
    if not known:
        return list(tickers), []

    px = cached_close[known].tail(UNIVERSE_ADV_WINDOW)
    vol = cached_volume[known].reindex(px.index)
    adv = (px * vol).median(skipna=True)

    # NaN(유효 표본 없음)은 0으로 보고 휴면 처리한다 — 60일간 거래 기록이 없다는 뜻
    dormant = set(adv.index[~(adv.fillna(0.0) >= floor)])
    active = [t for t in tickers if t not in dormant]
    return active, [t for t in tickers if t in dormant]


def _detect_split_artifacts(
    cached_close: pd.DataFrame,
    cached_volume: pd.DataFrame | None,
    tickers: list[str],
    repaired: dict[str, str],
) -> dict[str, str]:
    """캐시에서 미조정 리버스 스플릿으로 보이는 가격 점프를 찾는다.

    판정: 하루 가격비가 SPLIT_JUMP_RATIO 배 이상이면서, 점프 전후 거래량 중앙값
    비율이 SPLIT_VOL_RATIO_MAX 미만인 경우. 거래량은 스플릿과 실제 급등을 가르는
    유일한 신호라, 거래량 캐시가 없으면 판정을 포기한다(오탐 방지).

    Args:
        cached_close: 캐시 종가 (DatetimeIndex × 티커)
        cached_volume: 캐시 거래량. None/빈 프레임이면 감지를 건너뛴다.
        tickers: 이번 실행에서 다룰 티커 (이 중에서만 찾는다)
        repaired: 이미 처리한 {티커: {"jump": 날짜, ...}} — 같은 점프일이면 제외.
            날짜 비교가 아니라 일치 여부로 거른다. 최근 점프를 보정하고 나면 그보다
            과거의 미조정 스플릿이 드러나는데, 부등호로 거르면 그 층을 영영 못 잡는다.

    Returns:
        {티커: 가장 최근 스플릿 의심 점프일 'YYYY-MM-DD'}
    """
    if cached_volume is None or cached_volume.empty or cached_close.empty:
        return {}

    cols = [t for t in tickers if t in cached_close.columns and t in cached_volume.columns]
    if not cols:
        return {}

    # shift(1) 비교를 위해 스캔 범위보다 한 행 더 확보
    close = cached_close[cols].tail(SPLIT_SCAN_LOOKBACK + 1)
    volume = cached_volume[cols].reindex(close.index)
    if len(close) < SPLIT_VOL_WINDOW + 2:
        return {}

    # 0은 "거래 없음/미수신"이지 실제 가격·거래량이 아니다 → NaN으로 돌려 비율에서 제외
    # to_numpy는 읽기 전용 뷰를 줄 수 있어 copy=True로 쓰기 가능한 배열을 확보한다
    px = close.to_numpy(dtype="float64", copy=True)
    vol = volume.to_numpy(dtype="float64", copy=True)
    px[~(px > 0)] = np.nan
    vol[~(vol > 0)] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = px[1:] / px[:-1]
    hit = np.isfinite(ratio) & (ratio >= SPLIT_JUMP_RATIO)
    if not hit.any():
        return {}

    n_rows = px.shape[0]
    dates = close.index
    W = SPLIT_VOL_WINDOW
    found: dict[str, str] = {}

    # 점프는 전체의 극소수라 후보 좌표만 순회한다(6천여 티커 전수 루프 회피)
    for r, ci in zip(*np.nonzero(hit)):
        i = r + 1  # ratio[r] = px[r+1]/px[r] 이므로 점프가 찍힌 날은 종가 행 r+1
        ticker = cols[ci]
        jump_day = dates[i].strftime("%Y-%m-%d")
        # 더 최근 점프를 이미 찾았으면 갱신할 필요 없음
        if found.get(ticker, "") >= jump_day:
            continue

        before = vol[max(0, i - W):i, ci]
        after = vol[i:min(n_rows, i + W), ci]
        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]
        # 점프 직후 구간은 1개(점프 당일)만 있어도 판정한다. 방금 일어난 스플릿이
        # 가장 시급한 오염원인데, 표본을 더 요구하면 며칠간 놓치게 된다.
        if len(before) < 3 or len(after) < 1:
            continue
        base = float(np.median(before))
        if base <= 0:
            continue
        if float(np.median(after)) / base < SPLIT_VOL_RATIO_MAX:
            found[ticker] = jump_day

    # 이미 처리한 점프는 제외 (재다운로드로도 못 고쳤고 보정까지 끝난 건 = 더 볼 일 없음)
    return {t: d for t, d in found.items()
            if not _same_jump((repaired.get(t) or {}).get("jump", ""), d)}


def _tail_slice(close_df, high_df, low_df, volume_df, min_rows: int = 300, max_rows: int = 340):
    """반환 프레임을 계산에 충분한 최근 구간으로 슬라이싱 (min_rows 이하면 그대로)."""
    if len(close_df) <= min_rows:
        return close_df, high_df, low_df, volume_df
    n = min(len(close_df), max_rows)
    return (
        close_df.tail(n),
        high_df.tail(n) if high_df is not None and not high_df.empty else high_df,
        low_df.tail(n) if low_df is not None and not low_df.empty else low_df,
        volume_df.tail(n) if volume_df is not None and not volume_df.empty else volume_df,
    )


def download_ohlcv(
    yf_tickers: list[str],
    period: str = "16mo",
    cache_name: str = "ohlcv_master_cache.parquet",
    min_dollar_volume: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    OHLCV를 마스터 캐시 기반 증분 방식으로 다운로드한다.

    Args:
        min_dollar_volume: 유동성 하한(일 거래대금 중앙값). 미달 티커는 '휴면'으로
            분류해 반환 프레임에서 제외하고, 매 실행 일부만 새로고침한다.
            0이면 하한 없음 — 큐레이션된 ETF처럼 모수가 작고 고정된 경우에 쓴다.
    """
    cache_path = _cache_path(cache_name)

    cached_close, cached_high, cached_low, cached_volume = None, None, None, None
    is_incremental = False

    if cache_path.exists():
        try:
            # 캐시 포맷은 이 함수가 항상 같은 형태로 저장한다:
            # DatetimeIndex(날짜) × MultiIndex 컬럼(티커, 필드). 형식이 어긋난
            # 캐시(손상 등)는 아래 xs가 실패 → except가 전체 재다운로드로 복구.
            cached_master = pd.read_parquet(cache_path)
            cached_master.index = pd.to_datetime(cached_master.index)
            cached_close = cached_master.xs('Close', level=1, axis=1)
            cached_high = cached_master.xs('High', level=1, axis=1)
            cached_low = cached_master.xs('Low', level=1, axis=1)
            cached_volume = cached_master.xs('Volume', level=1, axis=1)
            latest_cached_date = cached_close.index.max()
            is_incremental = True
            print(f"  [Cache] 마스터 캐시 로드 완료 (최신 날짜: {latest_cached_date.date()})")
            print(f"  [Cache] 증분 업데이트 모드로 전환 (캐시 이후 데이터만 요청)")
        except Exception as exc:
            print(f"  [Warning] 마스터 캐시 로드 실패: {exc}. 전체 기간 재다운로드합니다.")
            cached_close, cached_high, cached_low, cached_volume = None, None, None, None
    
    # ── 유동성 하한: 휴면 티커 분리 ───────────────────────────────────────────
    # 휴면 티커는 반환 프레임에서 빠지므로 RS·스크리너 모집단에서도 제외된다.
    # 다만 전부 갱신을 멈추면 ADV가 얼어붙어 복귀 판정이 불가능해지므로, 가장 오래
    # 갱신되지 않은 순으로 매 실행 DORMANT_REFRESH_PER_RUN개만 새로고침한다.
    # (새로고침분도 이번 회차 반환 프레임에는 넣지 않는다 — 캐시 유지가 목적이다.)
    dormant_tickers: list[str] = []
    active_tickers, dormant_tickers = _split_by_liquidity(
        cached_close, cached_volume, yf_tickers, min_dollar_volume
    )
    if dormant_tickers:
        _stale_order = sorted(
            dormant_tickers,
            key=lambda t: (
                cached_close[t].last_valid_index()
                if cached_close is not None and t in cached_close.columns
                else pd.Timestamp.min
            ) or pd.Timestamp.min,
        )
        _refresh = _stale_order[:DORMANT_REFRESH_PER_RUN]
        yf_tickers = active_tickers + _refresh
        print(f"  [Liquidity] 거래대금 중앙값 ${min_dollar_volume:,.0f} 미만 "
              f"{len(dormant_tickers):,}개 티커를 휴면 처리 → 활성 {len(active_tickers):,}개 "
              f"(휴면 중 {len(_refresh):,}개는 캐시 유지용 갱신)")

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

    # ── 미조정 스플릿 티커 감지 ───────────────────────────────────────────────
    # 증분 append로는 스플릿 이전 행의 가격 스케일을 고칠 수 없다. 전체 기간을 다시
    # 받으면 yfinance가 시계열 전체를 조정해 주므로 need_full에 편입한다.
    split_hits: dict[str, str] = {}
    split_repaired: dict[str, dict[str, str]] = {}
    if is_incremental and cached_close is not None:
        split_repaired = _read_splitfix(cache_path)
        split_hits = _detect_split_artifacts(
            cached_close, cached_volume, yf_tickers, split_repaired
        )
        # 이미 need_full인 티커는 어차피 전체 재다운로드라 새로 셀 필요 없다
        newly = {t: d for t, d in split_hits.items() if t not in need_full}
        if newly:
            need_full.update(newly.keys())
            _preview = ", ".join(f"{t}({d})" for t, d in sorted(newly.items())[:5])
            print(f"  [Split] 미조정 스플릿 의심 {len(newly)}개 티커 → 전체 기간({period}) "
                  f"재다운로드: {_preview}{' ...' if len(newly) > 5 else ''}")

    close_frames, high_frames, low_frames, volume_frames = [], [], [], []
    rate_limit_abort = False  # 레이트리밋 지속 시 남은 배치 전체 중단 플래그
    requests_since_pause = 0  # 마지막 휴식 이후 보낸 요청 수 (시도 단위 누적)
    requests_total = 0        # 이번 실행 전체 요청 수 (로그용)
    pause_count = 0           # 선제적 휴식 횟수 (로그용)
    # 미수신 카운터 갱신용: 실제로 요청이 나간 티커 / 새 봉을 받은 티커.
    # 레이트리밋/예산 소진으로 아예 요청조차 못 한 티커는 attempted에 들어가지
    # 않으므로 미수신으로 오인되지 않는다.
    attempted: set[str] = set()
    received: set[str] = set()

    def _chunk_success_ratio(raw, chunk: list[str]) -> float:
        """다운로드 결과에서 유효 데이터(Close 1개 이상)를 받은 티커 비율."""
        if raw is None or raw.empty:
            return 0.0
        c = _extract_field(raw, "Close")
        if c.empty:
            return 0.0
        return float(c.notna().any().sum()) / max(len(chunk), 1)

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

        레이트리밋 대응은 2단이다.
        1) 예방(선제 휴식): REQUESTS_BEFORE_PAUSE건마다 DOWNLOAD_PAUSE_SECONDS만큼
           쉬고 계속 진행한다. 차단당하기 전에 속도 윈도우를 비우는 것이 목적이며,
           중단하지 않으므로 1회 실행으로 전체 티커를 커버한다.
        2) 사후(성공률 감지): yf.download는 티커별 실패(YFRateLimitError 포함)를
           예외로 던지지 않고 조용히 빈 컬럼으로 반환하므로, 배치 성공률이
           RATE_LIMIT_MIN_SUCCESS 미만이면 레이트리밋으로 판정해 장시간 대기 후
           재시도한다. 재시도 후에도 낮으면 남은 배치를 전부 중단해(추가 요청이
           차단만 연장시킴) 다음 실행이 밀린 티커부터 이어받도록 한다.
        """
        nonlocal rate_limit_abort, requests_since_pause, requests_total, pause_count
        chunk_list = [
            ticker_list[i : i + DOWNLOAD_CHUNK_SIZE]
            for i in range(0, len(ticker_list), DOWNLOAD_CHUNK_SIZE)
        ]
        desc = f"start={start}" if start else dl_period
        for idx, chunk in enumerate(chunk_list, start=1):
            if rate_limit_abort:
                print(f"  [Abort] 레이트리밋 지속 → {label} 배치 {idx}/{len(chunk_list)} 이후를 건너뜁니다.")
                return
            # 선제적 휴식: 이 배치를 보내면 차단 임계에 근접하는 경우, 429를 맞고
            # 회복을 기다리는 대신 미리 쉬어 속도 윈도우를 비운다. 중단하지 않고
            # 계속 진행하므로 1회 실행으로 전체 유니버스를 커버한다.
            if requests_since_pause + len(chunk) > REQUESTS_BEFORE_PAUSE:
                pause_count += 1
                remaining = sum(len(c) for c in chunk_list[idx - 1:])
                print(f"  [Pause] 누적 {requests_since_pause:,}건 → 차단 예방을 위해 "
                      f"{DOWNLOAD_PAUSE_SECONDS:.0f}초 휴식 후 계속합니다 "
                      f"({label} 배치 {idx}/{len(chunk_list)}, 남은 {remaining:,}종목).")
                time.sleep(DOWNLOAD_PAUSE_SECONDS)
                requests_since_pause = 0
            print(f"  [Download] {label} 배치 {idx}/{len(chunk_list)} ({len(chunk)}종목, {desc})...")
            attempted.update(chunk)
            # 배치 단위 재시도: 일시적 네트워크 오류/레이트리밋으로 한 배치가 실패해도
            # 전체 파이프라인을 중단시키지 않고, 재시도 후에도 실패하면 해당 배치만 건너뛴다.
            raw = None
            session = _get_yf_session()
            attempt = 0
            while True:
                attempt += 1
                # 재시도도 실제 요청을 소모하므로 시도마다 가산
                requests_since_pause += len(chunk)
                requests_total += len(chunk)
                candidate = None
                err_name = None
                try:
                    kwargs = {"start": start} if start else {"period": dl_period}
                    candidate = yf.download(
                        chunk, threads=DOWNLOAD_THREADS, progress=False,
                        auto_adjust=True, session=session, **kwargs,
                    )
                except Exception as exc:
                    err_name = type(exc).__name__

                ratio = _chunk_success_ratio(candidate, chunk)
                if err_name is None and ratio >= RATE_LIMIT_MIN_SUCCESS:
                    raw = candidate
                    break

                # 예외 있음 = 일시적 네트워크 오류 / 예외 없이 대량 실패 = 레이트리밋 의심.
                # 둘은 원인도 회복 시간도 달라 재시도 예산을 따로 둔다.
                budget = DOWNLOAD_MAX_RETRIES if err_name is not None else RATE_LIMIT_MAX_RETRIES
                if attempt < budget:
                    if err_name is not None:
                        wait = DOWNLOAD_RETRY_BACKOFF * attempt + random.uniform(0, 1.0)
                        print(f"  [Retry] {label} 배치 {idx} 실패({err_name}), "
                              f"{attempt}/{budget} 재시도까지 {wait:.1f}초 대기...")
                    else:
                        wait = RATE_LIMIT_COOLDOWN * attempt + random.uniform(0, 10.0)
                        print(f"  [RateLimit] {label} 배치 {idx} 성공률 {ratio:.0%} → "
                              f"레이트리밋 의심, {wait:.0f}초 대기 후 재시도 "
                              f"({attempt}/{budget})...")
                    time.sleep(wait)
                    continue

                if err_name is not None:
                    print(f"  [Skip] {label} 배치 {idx} {budget}회 실패, "
                          f"이 배치({len(chunk)}종목)를 건너뜁니다: {err_name}")
                else:
                    # 부분 데이터라도 병합은 안전(성공 티커만 캐시에 반영됨)
                    raw = candidate
                    rate_limit_abort = True
                    print(f"  [Abort] {label} 배치 {idx} 재시도 후에도 성공률 {ratio:.0%} → "
                          f"레이트리밋 지속으로 판단, 남은 배치를 중단합니다. "
                          f"30~60분 후 다시 실행하세요 (캐시는 안전하며, 재실행 시 "
                          f"미갱신 티커부터 우선 처리해 이어받습니다).")
                break
            time.sleep(random.uniform(*DOWNLOAD_CHUNK_PAUSE))  # 청크 간 대기: 분당 요청 속도 완화
            if raw is None or raw.empty:
                continue
            c = _extract_field(raw, "Close")
            h = _extract_field(raw, "High")
            l = _extract_field(raw, "Low")
            v = _extract_field(raw, "Volume")
            if not c.empty:
                # 유효 종가가 하나라도 온 티커만 "수신"으로 인정
                received.update(c.columns[c.notna().any()].tolist())
            if not c.empty: close_frames.append(c)
            if not h.empty: high_frames.append(h)
            if not l.empty: low_frames.append(l)
            if not v.empty: volume_frames.append(v)

    if is_incremental:
        # 전체 기간: 신규 또는 데이터 부족 티커
        full_list = [t for t in yf_tickers if t in need_full]
        if full_list:
            _run_chunks(full_list, "전체기간", dl_period=period)
        # 증분: 캐시에 충분한 데이터가 있는 기존 티커
        incr_list = [t for t in yf_tickers if t not in need_full]
        if incr_list:
            # ── 자가 치유 + 수렴: 티커별 최신 유효봉을 기준으로 순서와 시작일 결정 ──
            # 레이트리밋으로 중단된 실행은 앞쪽 배치만 갱신하고 끝난다. 이때
            #  (a) 전역 최신일 다음날부터 요청하면 뒤처진 티커의 빠진 며칠이 영영 누락되고,
            #  (b) 매 실행 같은 유니버스 순서로 돌면 이미 최신인 앞쪽 티커만 반복 갱신하다
            #      같은 지점에서 다시 막혀, 뒤쪽 티커는 영원히 갱신되지 않는다.
            # 그래서 시작일은 "가장 뒤처진 티커" 기준으로 잡고, 요청 순서도 뒤처진
            # 티커부터로 정렬한다. 중단되더라도 매 실행이 가장 밀린 쪽부터 전진해 수렴한다.
            # (셀 단위 combine_first 병합이라 겹치는 날짜 재요청은 덮어쓰기 안전)
            _cached_incr = [t for t in incr_list if t in cached_close.columns]
            _last_valid: dict[str, pd.Timestamp] = {}
            if _cached_incr:
                # 컬럼별 마지막 유효봉 날짜. 유효봉이 하나도 없는 티커는 제외되어
                # 정렬 키에서 pd.Timestamp.min을 받아 맨 앞(가장 시급)으로 간다.
                _sub = cached_close[_cached_incr].replace(0, float("nan"))
                for _t in _cached_incr:
                    _ts = _sub[_t].last_valid_index()
                    if _ts is not None:
                        _last_valid[_t] = pd.Timestamp(_ts)

            # 물어봐도 계속 응답이 없던 티커(상장폐지/거래정지 등)는 후순위로.
            # 이들을 빼지 않으면 우선순위 큐 앞자리를 영구히 점거해, 정말 밀린
            # 티커가 매번 뒤로 밀리고 요청 구간만 최대치로 늘어난다.
            _misses = _read_stale_misses(cache_path)
            _dead = {t for t, n in _misses.items() if n >= STALE_MISS_LIMIT}
            if _dead & set(incr_list):
                print(f"  [Stale] {len(_dead & set(incr_list))}개 티커는 {STALE_MISS_LIMIT}회 연속 "
                      f"미수신 → 후순위로 요청")

            # 정렬 키: (사실상 데이터 없음, 최신봉 날짜, 티커).
            # 캐시에 없거나 유효봉이 아예 없는 티커는 pd.Timestamp.min을 받아 맨 앞으로 간다.
            incr_list = sorted(
                incr_list,
                key=lambda t: (t in _dead, _last_valid.get(t, pd.Timestamp.min), t),
            )

            # 시작일 = 뒤처진 티커 중 가장 오래된 최신봉의 다음날.
            # 상장폐지/거래정지로 몇 달째 멈춘 티커가 요청 구간을 과도하게 늘리지
            # 않도록 INCR_MAX_LOOKBACK_DAYS 이내 + 미수신 누적이 적은 티커만 기준으로 삼는다.
            _floor = pd.Timestamp.now().normalize() - pd.Timedelta(days=INCR_MAX_LOOKBACK_DAYS)
            _recent = [
                ts for t, ts in _last_valid.items() if ts >= _floor and t not in _dead
            ]
            _resume_ts = min(_recent) if _recent else latest_cached_date
            _resume_ts = min(_resume_ts, latest_cached_date)
            incr_start_ts = _resume_ts + pd.Timedelta(days=1)
            if _resume_ts < latest_cached_date:
                _behind = sum(
                    1 for t, ts in _last_valid.items()
                    if ts < latest_cached_date and t not in _dead
                )
                print(f"  [Heal] 미갱신 티커 {_behind}개(최고 {(latest_cached_date - _resume_ts).days}일 지연) "
                      f"→ {incr_start_ts.date()}부터 재요청해 누락 복구")
            # 잠정 봉 갱신: 직전 실행이 장중이라 최신일이 잠정 종가로 저장됐다면,
            # 그 날짜를 다시 요청해 최종 종가로 교체한다(마감 후 실행 시).
            _prov = _read_provisional(cache_path)
            if _prov is not None and _prov.normalize() == latest_cached_date.normalize():
                if incr_start_ts > latest_cached_date:
                    incr_start_ts = latest_cached_date
                print(f"  [Refresh] 잠정 봉({latest_cached_date.date()}) → 최종 종가로 재요청")
            incr_start = incr_start_ts.strftime("%Y-%m-%d")
            if incr_start > pd.Timestamp.now().strftime("%Y-%m-%d"):
                print(f"  [Skip] 증분 대상 {len(incr_list)}개 티커는 이미 최신 상태입니다.")
            elif not _has_new_trading_day(incr_start):
                # 프로브 게이트: SPY 한 종목으로 [incr_start, today] 새 거래일 유무만 확인.
                # 주말·공휴일·같은 날 재실행이면 새 거래일이 없어 6천여 티커 배치를 통째로 생략.
                print(f"  [Skip] {incr_start} 이후 새 거래일 없음 → 증분 배치 {len(incr_list)}개 생략")
            else:
                _run_chunks(incr_list, "증분", start=incr_start)
                # 요청이 실제로 나간 티커에 한해 미수신 카운터 갱신.
                # 받았으면 0으로 리셋, 못 받았으면 +1 → STALE_MISS_LIMIT 도달 시 후순위.
                _updated = dict(_misses)
                for t in attempted:
                    if t in received:
                        _updated.pop(t, None)
                    else:
                        _updated[t] = _updated.get(t, 0) + 1
                _write_stale_misses(cache_path, _updated)
    else:
        # 캐시 없음: 전체 종목 전체 기간 다운로드
        _run_chunks(yf_tickers, "전체기간", dl_period=period)

    if requests_total:
        print(f"  [Requests] 이번 실행 요청 {requests_total:,}건"
              f" (선제 휴식 {pause_count}회 × {DOWNLOAD_PAUSE_SECONDS:.0f}초"
              f", 휴식 간격 {REQUESTS_BEFORE_PAUSE:,}건)")

    if not close_frames:
        if cached_close is not None:
            # 새로 받은 데이터 없음(주말/공휴일 스킵, 레이트리밋 전체 실패 등).
            # 빈 프레임을 반환하면 하류(.intermediate)가 빈 파일로 덮어써져
            # RS 엔진의 섹터/테마 집계가 통째로 깨지므로, 캐시를 그대로 반환한다.
            print("  [Cache] 신규 데이터 없음 → 마스터 캐시 데이터를 그대로 반환합니다.")
            # 네 필드는 한 try 블록에서 함께 로드되므로 close가 있으면 나머지도 있다
            assert cached_high is not None and cached_low is not None and cached_volume is not None
            # 정상 경로와 동일하게 휴면 티커는 빼고 반환한다(모집단이 회차마다 달라지면 안 됨)
            _keep = [c for c in cached_close.columns if c not in set(dormant_tickers)]
            close_df = cached_close[_keep].ffill()
            high_df = (cached_high[[c for c in _keep if c in cached_high.columns]].ffill()
                       if not cached_high.empty else cached_high)
            low_df = (cached_low[[c for c in _keep if c in cached_low.columns]].ffill()
                      if not cached_low.empty else cached_low)
            volume_df = (cached_volume[[c for c in _keep if c in cached_volume.columns]].ffill()
                         if not cached_volume.empty else cached_volume)
            close_df, high_df, low_df, volume_df = _tail_slice(close_df, high_df, low_df, volume_df)
            print(f"  [Return] 최종 슬라이싱: {close_df.shape}")
            return close_df, high_df, low_df, volume_df
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
                    incr_new_df    = (
                        new_df[[c for c in cached_incr if c in new_df.columns]]
                        .dropna(how="all")  # incr 티커에 실제 데이터가 없는 날짜 제거
                    )
                    # 셀 단위 병합: 새 값이 있으면 새 값, 없으면 캐시 값 보존.
                    # (행 단위 keep='last'는 겹치는 날짜에서 이번 다운로드에 없는
                    #  티커의 기존 값을 NaN으로 덮어써 부분 실패 시 데이터가 소실됨.
                    #  [Heal] 재요청처럼 캐시와 겹치는 날짜를 받는 경우 특히 치명적.)
                    incr_merged = incr_new_df.combine_first(incr_cached_df).sort_index()
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

        # ── 재다운로드로도 안 고쳐진 스플릿 → 소급 보정 ───────────────────────
        # 병합된(=이번에 받은 정본이 반영된) 프레임에서 다시 스캔한다. 여기 남아 있다는
        # 것은 야후 원본 자체가 미조정이라는 뜻이므로, 직접 이음매를 잇는다.
        _frames = {"close": close_df, "high": high_df, "low": low_df, "volume": volume_df}
        _still = _detect_split_artifacts(close_df, volume_df, list(close_df.columns), {})
        _adjusted: dict[str, tuple[str, float]] = {}
        for _t, _d in _still.items():
            _prev = split_repaired.get(_t) or {}
            # 전체 재다운로드를 이미 거친 같은 점프만 보정 대상. 처음 보는 점프는
            # 먼저 정본 데이터를 받아볼 기회를 준다(이번 회차 need_full이 그 역할).
            if _prev.get("stage") != "redownloaded" or not _same_jump(_prev.get("jump", ""), _d):
                continue
            _r = _backadjust_split(_frames, _t, _d)
            if _r is not None:
                _adjusted[_t] = (_d, _r)
        if _adjusted:
            _ex = ", ".join(f"{t}(×{r:.1f})" for t, (_, r) in sorted(_adjusted.items())[:5])
            print(f"  [Split] 야후 미조정 {len(_adjusted)}개 티커 소급 보정: {_ex}"
                  f"{' ...' if len(_adjusted) > 5 else ''}")
    else:
        _adjusted = {}
        close_df, high_df, low_df, volume_df = new_close_df, new_high_df, new_low_df, new_volume_df
        print(f"  [Download] 신규 다운로드 완료 (캐시 없음)")
    
    # ── 마스터 캐시는 ffill 전(raw) 상태로 저장한다 ────────────────────────────
    # ffill한 프레임을 그대로 캐시에 쓰면 "받지 못한 봉"과 "가격이 그대로인 봉"이
    # 구분 불가능해진다. 그러면 (1) 레이트리밋으로 빠진 며칠이 횡보 가격으로 굳어
    # RS/모멘텀 계산을 오염시키고, (2) 어떤 티커가 밀렸는지 알 수 없어 다음 실행이
    # 이어받을 수 없다. 캐시=원본 사실, 반환 프레임=계산 편의(ffill 적용)로 분리한다.
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

        # 스플릿 의심으로 재다운로드한 티커를 기록한다(저장 성공 시에만).
        # 진짜 스플릿이었다면 조정된 시계열로 교체돼 다음 실행에서 점프 자체가 사라지고,
        # 거래량까지 줄어든 실제 급등이었다면 점프가 남지만 이 기록이 재플래그를 막는다.
        _tried = {t: {"jump": d, "stage": "redownloaded"}
                  for t, d in split_hits.items() if t in received}
        _done = {t: {"jump": d, "stage": "adjusted"} for t, (d, _) in _adjusted.items()}
        if _tried or _done:
            _kept = {t: v for t, v in split_repaired.items() if t in cached_close.columns}
            _write_splitfix(cache_path, {**_kept, **_tried, **_done})
            if _tried:
                print(f"  [Split] {len(_tried)}개 티커 전체 기간 재다운로드 완료 "
                      f"(다음 실행에서 미해결분은 소급 보정)")

        # 잠정 봉 마커 갱신: 저장된 최신일이 장중 캡처(미확정)면 마커 기록,
        # 최종 확정(마감 후/과거일)이면 마커 제거 → 다음 실행의 재요청 여부 결정.
        _latest = close_df.index.max()
        if pd.notna(_latest):
            if _bar_is_finalized(_latest):
                _clear_provisional(cache_path)
            else:
                _write_provisional(cache_path, _latest)
                print(f"  [Provisional] 최신 봉({pd.Timestamp(_latest).date()}) 장중 캡처 → "
                      f"다음 마감 후 실행에서 최종 종가로 갱신 예정")
    except Exception as exc:
        print(f"  [Warning] 마스터 캐시 저장 실패: {exc}")

    # 휴면 티커는 캐시에는 남기되 반환 프레임에서 뺀다(RS·스크리너 모집단에서 제외).
    # ffill 이전에 빼야 한다 — 남겨두면 갱신이 멈춘 종가가 그대로 굳어 등락률 0%인
    # 유령 종목이 되고, 그게 백분위 분모에 섞이면 안 빼느니만 못하다.
    if dormant_tickers:
        _drop = [t for t in dormant_tickers if t in close_df.columns]
        if _drop:
            close_df = close_df.drop(columns=_drop)
            high_df = high_df.drop(columns=_drop, errors="ignore") if not high_df.empty else high_df
            low_df = low_df.drop(columns=_drop, errors="ignore") if not low_df.empty else low_df
            volume_df = volume_df.drop(columns=_drop, errors="ignore") if not volume_df.empty else volume_df

    # 결측치 보정(반환 프레임 한정): ffill만 수행, fillna(0) 금지.
    # 0으로 채우면 다운로드 실패 티커가 rs_engine에서 inf/NaN을 유발함.
    close_df = close_df.ffill()
    high_df = high_df.ffill() if not high_df.empty else high_df
    low_df = low_df.ffill() if not low_df.empty else low_df
    volume_df = volume_df.ffill() if not volume_df.empty else volume_df

    close_df, high_df, low_df, volume_df = _tail_slice(close_df, high_df, low_df, volume_df)
    print(f"  [Return] 최종 슬라이싱: {close_df.shape}")
    return close_df, high_df, low_df, volume_df

def download_spy_close(period: str = "16mo") -> pd.Series:
    print(f"  [Download] SPY 벤치마크 다운로드 (period={period})...")
    raw = yf.download("SPY", period=period, progress=False, auto_adjust=True, session=_get_yf_session())
    if raw is None or raw.empty:
        # 레이트리밋 등 실패 시 ETF 마스터 캐시의 SPY 종가로 폴백한다.
        # 빈 시리즈를 반환하면 spy_close.parquet이 0행으로 덮어써져 RS Line
        # 벤치마크가 사라지므로, 캐시라도 반환해 하류 계산을 보호한다.
        try:
            m = pd.read_parquet(_cache_path("etf_ohlcv_master_cache.parquet"))
            spy = m.xs("Close", level=1, axis=1)["SPY"].dropna()
            if not spy.empty:
                spy.index = pd.to_datetime(spy.index)
                print(f"  [Fallback] SPY 다운로드 실패 → ETF 마스터 캐시의 SPY 사용 "
                      f"({len(spy)}행, 최신 {spy.index.max().date()})")
                return spy.sort_index()
        except Exception as exc:
            print(f"  [Warning] SPY 캐시 폴백 실패: {exc}")
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
                "AsOfDate": pd.Timestamp.now("UTC").normalize().date().isoformat(),
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
    ]
    return pd.DataFrame(all_records, columns=columns)


def _fetch_single_shares(ticker: str) -> float | None:
    """단일 티커의 발행 주식 수를 조회한다 (실패 시 None)."""
    try:
        time.sleep(random.uniform(0.3, 0.8))  # 차단 완화용 지연
        tkr = yf.Ticker(ticker, session=_get_yf_session())

        # yfinance에서 가장 가볍고 차단 안 당하는 full_shares 사용
        shares_series = tkr.get_shares_full()
        if shares_series is not None and not shares_series.empty:
            return float(shares_series.iloc[-1])  # 가장 최근 시점의 주식 수

        # 백업용 fast_info
        if 'sharesOutstanding' in tkr.fast_info:
            val = tkr.fast_info['sharesOutstanding']
            return float(val) if val is not None else None
    except Exception:
        pass
    return None


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
    unique_tickers = list(dict.fromkeys(tickers))  # 순서 유지 + 중복 제거
    if force_update:
        to_fetch = unique_tickers
        print(f"  [Shares] force_update=True → 전체 {len(to_fetch):,}개 종목 재수집")
    else:
        stale_cutoff = today - pd.Timedelta(days=max_age_days)
        updated_at = pd.to_datetime(cached_df.get("UpdatedAt"), errors="coerce")
        new_list = []    # (1) 신규 티커 — 우선 수집
        stale_list = []  # (2) 만료 티커 — 남는 예산으로 수집
        for t in unique_tickers:
            if t not in cached_df.index:
                new_list.append(t)
            elif pd.isna(updated_at.get(t)) or updated_at.get(t) < stale_cutoff:
                stale_list.append(t)
        # 실행당 상한: 30일 만료가 한꺼번에 도래해도 하루 요청량 폭증 방지 (초과분 이월)
        to_fetch = _cap_fetch_list(
            new_list, stale_list, "Shares",
            f"신규 {len(new_list):,} + {max_age_days}일 초과 {len(stale_list):,}",
            len(unique_tickers),
        )
        if not to_fetch:
            print(f"  [Shares] 갱신 대상 없음 → 캐시 그대로 사용 ({len(cached_df):,}개)")

    # 3. 증분 수집 실행 및 캐시 병합
    if to_fetch:
        fetched = _sequential_fetch(to_fetch, _fetch_single_shares, "Shares")
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
    series = series.reindex(unique_tickers)
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
