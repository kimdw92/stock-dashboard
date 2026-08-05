"""
토스증권 Open API 데이터 수집기.

미국 주식 보유 잔고·주문 내역·예수금을 수집하여 output/ 폴더에 parquet로 저장합니다.

실행:
  python toss_collector.py              # 증분 수집 (기존 거래내역 이후만)
  python toss_collector.py --full       # 전체 기간 재수집
  python toss_collector.py --dry-run    # 조회만 하고 저장하지 않음

.env 파일 (프로젝트 루트):
  TOSS_CLIENT_ID=your_client_id
  TOSS_CLIENT_SECRET=your_client_secret
  TOSS_ACCOUNT_SEQ=            # 선택. 비우면 첫 BROKERAGE 계좌 자동 선택

사전 준비:
  토스증권 WTS > 설정 > Open API 에서 client_id/secret 발급 및 IP 허용목록 등록.
  등록되지 않은 IP에서 호출하면 403 edge-blocked 오류가 납니다.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

TOSS_BASE_URL = "https://openapi.tossinvest.com"
TOKEN_CACHE_PATH = CACHE_DIR / "toss_token.json"

SNAPSHOT_PATH = OUTPUT_DIR / "portfolio_snapshot.parquet"
TRADES_PATH = OUTPUT_DIR / "trade_history.parquet"

# 미국 주식 전용 대시보드 — 국내(KR) 보유분은 RS 엔진에 대응 종목이 없어 제외한다.
TARGET_MARKET = "US"
TARGET_CURRENCY = "USD"

# Rate limit 그룹별 초당 허용 요청 수 (openapi-docs/overview.md 기준).
# ACCOUNT가 1 TPS로 가장 빡빡하므로 그룹별로 최소 호출 간격을 강제한다.
RATE_LIMIT_TPS: dict[str, float] = {
    "AUTH": 5,
    "ACCOUNT": 1,
    "ASSET": 5,
    "ORDER_HISTORY": 5,
    "ORDER_INFO": 6,
    "MARKET_INFO": 3,
}

MAX_RETRIES = 4
ORDERS_PAGE_SIZE = 100  # API 상한

_last_call_at: dict[str, float] = {}


# ─── 환경변수 로드 헬퍼 ──────────────────────────────────────────────────────

def _load_env() -> None:
    """python-dotenv로 .env 파일 로드. 파일 없으면 경고만 출력."""
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(override=True)
        if not loaded:
            logger.warning(".env 파일을 찾지 못했습니다. 환경변수가 이미 설정되어 있어야 합니다.")
    except ImportError:
        logger.warning("python-dotenv 미설치. pip install python-dotenv 후 재시도하세요.")


def _require_env(key: str) -> str:
    """
    환경변수 필수 조회. 없으면 명확한 메시지로 RuntimeError 발생.

    .env에 값을 붙여넣을 때 따옴표나 공백이 딸려오는 일이 잦은데, 그대로 보내면
    서버는 invalid_client로만 답해서 원인을 찾기 어렵다. 여기서 미리 벗겨낸다.
    """
    val = os.environ.get(key)
    if val is not None:
        val = val.strip().strip('"').strip("'").strip()
    if not val:
        raise RuntimeError(
            f"환경변수 '{key}'가 설정되지 않았습니다. "
            f".env 파일에 {key}=<값> 형태로 추가하세요 (따옴표 없이)."
        )
    return val


def _mask(secret: str) -> str:
    """자격증명을 로그에 안전하게 남기기 위한 마스킹."""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-2:]} (len={len(secret)})"


def _f(val: object) -> float:
    """API가 문자열/None으로 주는 숫자를 안전하게 float으로 변환."""
    if val is None:
        return 0.0
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


# ─── HTTP 레이어 (rate limit + 재시도) ───────────────────────────────────────

def _throttle(group: str) -> None:
    """그룹별 최소 호출 간격을 지켜 429를 예방한다."""
    tps = RATE_LIMIT_TPS.get(group, 5)
    min_interval = 1.0 / tps
    last = _last_call_at.get(group)
    if last is not None:
        elapsed = time.monotonic() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    _last_call_at[group] = time.monotonic()


def _check_response(resp: requests.Response, context: str) -> dict:
    """
    HTTP 상태 검사 후 JSON 파싱.

    오류 형식이 두 가지다:
      - 일반 API: {"error": {"code": ..., "message": ..., "requestId": ...}}
      - /oauth2/token: OAuth2 표준 {"error": "invalid_client",
                                    "error_description": ...}
    토큰 엔드포인트만 BFF 공통 envelope을 쓰지 않으므로 둘 다 받아준다.

    Raises:
        RuntimeError: 오류 응답인 경우 사람이 읽을 수 있는 메시지로 변환
    """
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(
            f"[{context}] JSON이 아닌 응답 (HTTP {resp.status_code}): {resp.text[:300]}"
        )

    if isinstance(data, dict) and data.get("error") is not None:
        err = data["error"]
        if isinstance(err, str):
            # OAuth2 표준 오류
            detail = data.get("error_description") or "(설명 없음)"
            raise RuntimeError(
                f"[{context}] 인증 오류 (HTTP {resp.status_code}, {err}): {detail}"
            )
        # 일반 API 오류
        req_id = err.get("requestId", "")
        raise RuntimeError(
            f"[{context}] 토스 API 오류 (HTTP {resp.status_code}, "
            f"code={err.get('code', '?')}): {err.get('message') or '(메시지 없음)'}"
            + (f" [requestId={req_id}]" if req_id else "")
        )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"[{context}] HTTP {resp.status_code}: {resp.text[:300]}"
        )
    return data


def _request(
    method: str,
    path: str,
    group: str,
    context: str,
    *,
    headers: dict | None = None,
    params: dict | None = None,
    data: dict | None = None,
) -> dict:
    """
    rate limit 준수 + 429/5xx 지수 백오프 재시도를 적용한 단일 요청.

    Args:
        group: RATE_LIMIT_TPS 키 (호출 간격 결정)
    """
    url = f"{TOSS_BASE_URL}{path}"
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        _throttle(group)
        try:
            resp = requests.request(
                method, url, headers=headers, params=params, data=data, timeout=20
            )
        except requests.RequestException as exc:
            last_exc = exc
            wait = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"[{context}] 네트워크 오류, {wait:.1f}s 후 재시도: {exc}")
            time.sleep(wait)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else (2 ** attempt) + random.uniform(0, 0.5)
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"[{context}] HTTP {resp.status_code}, {wait:.1f}s 후 재시도 "
                    f"({attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

        return _check_response(resp, context)

    raise RuntimeError(f"[{context}] {MAX_RETRIES}회 재시도 후에도 실패: {last_exc}")


# ─── 1. 액세스 토큰 (파일 캐시) ──────────────────────────────────────────────

def get_access_token(client_id: str, client_secret: str) -> str:
    """
    OAuth2 Client Credentials 토큰 발급. 만료 전이면 .cache/toss_token.json 재사용.

    Args:
        client_id:     토스 Open API client_id
        client_secret: 토스 Open API client_secret

    Returns:
        access_token 문자열
    """
    cached = _read_cached_token()
    if cached:
        logger.info("캐시된 액세스 토큰 재사용")
        return cached

    logger.info(f"액세스 토큰 발급 요청 중... (client_id={_mask(client_id)})")
    try:
        data = _request(
            "POST", "/oauth2/token", "AUTH", "get_access_token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type":    "client_credentials",
                "client_id":     client_id,
                "client_secret": client_secret,
            },
        )
    except RuntimeError as exc:
        if "invalid_client" in str(exc):
            raise RuntimeError(
                f"{exc}\n\n"
                "확인할 것:\n"
                "  1. .env의 TOSS_CLIENT_ID / TOSS_CLIENT_SECRET이 토스증권 WTS >\n"
                "     설정 > Open API에서 발급한 값과 정확히 일치하는지\n"
                "  2. 값에 따옴표·공백·줄바꿈이 섞여 있지 않은지\n"
                "  3. 발급한 키가 아직 활성 상태인지 (재발급 시 이전 키는 무효)\n"
                "  4. 현재 공인 IP가 Open API 설정의 허용목록에 등록돼 있는지"
            ) from exc
        raise

    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"토큰 발급 실패: {data}")

    expires_in = int(data.get("expires_in", 0) or 0)
    _write_cached_token(token, expires_in)
    logger.info(f"액세스 토큰 발급 완료 (유효 {expires_in}초)")
    return token


def _read_cached_token() -> str | None:
    """만료 60초 전까지만 유효한 것으로 간주해 캐시된 토큰을 반환."""
    try:
        with open(TOKEN_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    token = cache.get("access_token")
    expires_at = cache.get("expires_at", 0)
    if not token or time.time() >= float(expires_at) - 60:
        return None
    return str(token)


def _write_cached_token(token: str, expires_in: int) -> None:
    payload = {"access_token": token, "expires_at": time.time() + expires_in}
    try:
        with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as exc:
        logger.warning(f"토큰 캐시 저장 실패(무시하고 진행): {exc}")


def _auth_headers(token: str, account_seq: str | int | None = None) -> dict:
    """공통 인증 헤더. account_seq를 주면 계좌 컨텍스트 헤더를 추가한다."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
    }
    if account_seq is not None:
        headers["X-Tossinvest-Account"] = str(account_seq)
    return headers


# ─── 2. 계좌 조회 ────────────────────────────────────────────────────────────

def get_account_seq(token: str, override: str | None = None) -> int:
    """
    사용할 계좌의 accountSeq 결정. override가 있으면 그대로 쓰고, 없으면 첫 계좌.

    Args:
        token:    액세스 토큰
        override: .env의 TOSS_ACCOUNT_SEQ (없으면 None)

    Returns:
        accountSeq (int)
    """
    if override:
        logger.info(f"TOSS_ACCOUNT_SEQ 지정값 사용: {override}")
        return int(override)

    data = _request("GET", "/api/v1/accounts", "ACCOUNT", "get_accounts",
                    headers=_auth_headers(token))
    # getAccounts는 result가 Account 배열 그 자체다 (다른 API처럼 객체로 감싸지 않는다).
    result = data.get("result")
    accounts = result if isinstance(result, list) else []

    if not accounts:
        raise RuntimeError(
            "조회 가능한 계좌가 없습니다. 토스증권 종합매매(BROKERAGE) 계좌가 필요합니다."
        )

    acct = accounts[0]
    seq = acct.get("accountSeq")
    logger.info(
        f"계좌 선택: {acct.get('accountNo', '?')} "
        f"({acct.get('accountType', '?')}, seq={seq})"
    )
    return int(seq)


# ─── 3. 보유 주식 조회 ───────────────────────────────────────────────────────

def get_holdings(token: str, account_seq: int) -> tuple[pd.DataFrame, dict]:
    """
    보유 주식 조회 (GET /api/v1/holdings). 미국 종목만 필터링한다.

    Returns:
        (보유 종목 DataFrame, 계좌 요약 dict)
        요약 dict 키: total_eval_usd, total_purchase_usd, total_pnl_usd
    """
    logger.info("보유 주식 조회 중...")
    data = _request("GET", "/api/v1/holdings", "ASSET", "get_holdings",
                    headers=_auth_headers(token, account_seq))

    overview = data.get("result") or {}
    items = overview.get("items") or []

    rows = []
    for item in items:
        if item.get("marketCountry") != TARGET_MARKET:
            continue
        qty = _f(item.get("quantity"))
        if qty == 0:
            continue

        market_value = item.get("marketValue") or {}
        profit_loss = item.get("profitLoss") or {}
        daily_pl = item.get("dailyProfitLoss") or {}
        cost = item.get("cost") or {}

        rows.append({
            "ticker":             item.get("symbol", ""),
            "name":               item.get("name", ""),
            "qty":                qty,
            "avg_price_usd":      _f(item.get("averagePurchasePrice")),
            "current_price_usd":  _f(item.get("lastPrice")),
            "eval_amt_usd":       _f(market_value.get("amount")),
            "purchase_amt_usd":   _f(market_value.get("purchaseAmount")),
            "pnl_usd":            _f(profit_loss.get("amount")),
            # rate는 소수비율(0.1077 = 10.77%)이므로 ×100 해서 저장한다.
            "pnl_pct":            _f(profit_loss.get("rate")) * 100,
            "pnl_usd_after_cost": _f(profit_loss.get("amountAfterCost")),
            "daily_pnl_usd":      _f(daily_pl.get("amount")),
            "daily_pnl_pct":      _f(daily_pl.get("rate")) * 100,
            "commission_usd":     _f(cost.get("commission")),
            "tax_usd":            _f(cost.get("tax")),
        })

    holdings_df = pd.DataFrame(rows)

    # Price 모델은 {krw, usd} 이며 해외 종목이 없으면 usd가 null이다.
    def _usd_of(price_obj: object) -> float:
        return _f((price_obj or {}).get("usd") if isinstance(price_obj, dict) else None)

    mv = overview.get("marketValue") or {}
    pl = overview.get("profitLoss") or {}
    summary = {
        "total_eval_usd":     _usd_of(mv.get("amount")),
        "total_purchase_usd": _usd_of(overview.get("totalPurchaseAmount")),
        "total_pnl_usd":      _usd_of(pl.get("amount")),
    }

    logger.info(f"보유 주식 조회 완료: 미국 {len(holdings_df)}개 종목")
    return holdings_df, summary


# ─── 4. 예수금(매수 가능 금액) 조회 ──────────────────────────────────────────

def get_buying_power(token: str, account_seq: int) -> float:
    """
    USD 매수 가능 금액 조회 (GET /api/v1/buying-power).

    Returns:
        현금 기반 매수 가능 금액 (USD)
    """
    logger.info("USD 예수금 조회 중...")
    data = _request("GET", "/api/v1/buying-power", "ORDER_INFO", "get_buying_power",
                    headers=_auth_headers(token, account_seq),
                    params={"currency": TARGET_CURRENCY})

    result = data.get("result") or {}
    cash = _f(result.get("cashBuyingPower"))
    logger.info(f"예수금 조회 완료: 가용 USD {cash:,.2f}")
    return cash


# ─── 5. 주문(체결) 내역 조회 ─────────────────────────────────────────────────

def get_orders(
    token: str,
    account_seq: int,
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    종료된 주문 내역 조회 (GET /api/v1/orders?status=CLOSED). 커서 페이징 자동 처리.

    Args:
        from_date: 조회 시작일 YYYY-MM-DD (None이면 전체 기간)
        to_date:   조회 종료일 YYYY-MM-DD (None이면 전체 기간)

    Returns:
        체결된 주문만 담은 DataFrame.
        컬럼: order_id, date, ticker, side, qty, price_usd, amount_usd,
              commission_usd, tax_usd, status, ordered_at, filled_at
    """
    cols = ["order_id", "date", "ticker", "side", "qty", "price_usd", "amount_usd",
            "commission_usd", "tax_usd", "status", "ordered_at", "filled_at"]

    all_rows: list[dict] = []
    cursor: str | None = None
    page = 0

    span = f"{from_date or '전체'} ~ {to_date or '전체'}"
    logger.info(f"주문 내역 조회 중 ({span})...")

    while True:
        page += 1
        params: dict[str, object] = {"status": "CLOSED", "limit": ORDERS_PAGE_SIZE}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if cursor:
            params["cursor"] = cursor

        data = _request("GET", "/api/v1/orders", "ORDER_HISTORY",
                        f"get_orders page={page}",
                        headers=_auth_headers(token, account_seq), params=params)

        result = data.get("result") or {}
        orders = result.get("orders") or []

        for order in orders:
            execution = order.get("execution") or {}
            filled_qty = _f(execution.get("filledQuantity"))
            # 취소/거부된 주문은 체결수량이 0이다 — 거래 내역에서 제외.
            if filled_qty == 0:
                continue
            if order.get("currency") != TARGET_CURRENCY:
                continue

            filled_at = execution.get("filledAt") or order.get("orderedAt")
            all_rows.append({
                "order_id":       order.get("orderId", ""),
                "date":           filled_at,
                "ticker":         order.get("symbol", ""),
                "side":           order.get("side", ""),
                "qty":            filled_qty,
                "price_usd":      _f(execution.get("averageFilledPrice")),
                "amount_usd":     _f(execution.get("filledAmount")),
                "commission_usd": _f(execution.get("commission")),
                "tax_usd":        _f(execution.get("tax")),
                "status":         order.get("status", ""),
                "ordered_at":     order.get("orderedAt"),
                "filled_at":      filled_at,
            })

        if not result.get("hasNext"):
            break
        cursor = result.get("nextCursor")
        if not cursor:
            break

    if not all_rows:
        logger.info("체결된 주문 없음")
        return pd.DataFrame(columns=cols)

    trades_df = pd.DataFrame(all_rows)
    trades_df = _normalize_trades(trades_df)

    logger.info(f"주문 내역 조회 완료: {len(trades_df)}건 (총 {page}페이지)")
    return trades_df[cols]


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """체결일을 미국 거래일로, 금액 컬럼을 float으로 정규화.

    토스는 체결시각을 KST로 준다. 그런데 미국 정규장은 22:30~05:00 KST로 자정을
    넘기기 때문에, KST 달력 날짜를 그대로 쓰면 자정 이후 체결분이 하루 뒤로
    밀린다. 자산 추이 백필이 붙는 close.parquet은 미국 거래일 인덱스이므로
    미 동부시각 기준 날짜로 변환해야 보유수량과 종가가 같은 날에 정렬된다.
    """
    out = df.copy()
    parsed = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["date"] = (
        parsed.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    for col in ["qty", "price_usd", "amount_usd", "commission_usd", "tax_usd"]:
        out[col] = out[col].astype(float)
    out["side"] = out["side"].str.upper()
    return out


# ─── 6. 저장 ─────────────────────────────────────────────────────────────────

def save_snapshot(holdings_df: pd.DataFrame, summary: dict, cash_usd: float,
                  today: datetime.date) -> None:
    """
    일별 잔고 스냅샷을 누적 저장 (같은 날 재실행 시 해당 날짜를 덮어쓴다).
    """
    snap = holdings_df.copy()
    snap["date"] = pd.Timestamp(today)
    # 계좌 단위 값은 모든 행에 반복 저장한다 (앱에서 .iloc[0]으로 읽는다).
    snap["total_eval_usd"] = summary.get("total_eval_usd", 0.0)
    snap["cash_usd"] = cash_usd

    if SNAPSHOT_PATH.exists():
        existing = pd.read_parquet(SNAPSHOT_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        existing = existing[existing["date"] != pd.Timestamp(today)]
        combined = pd.concat([existing, snap], ignore_index=True)
    else:
        combined = snap

    combined = combined.sort_values(["date", "ticker"])
    combined.to_parquet(SNAPSHOT_PATH, index=False)
    logger.info(f"스냅샷 저장 완료: {SNAPSHOT_PATH} ({len(snap)}개 종목, {today})")


def save_trades(trades_df: pd.DataFrame) -> None:
    """
    체결 내역을 누적 저장. order_id 기준으로 중복 제거한다.
    """
    if trades_df.empty:
        logger.info("저장할 체결 내역 없음")
        return

    if TRADES_PATH.exists():
        existing = pd.read_parquet(TRADES_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, trades_df], ignore_index=True)
    else:
        combined = trades_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=["order_id"], keep="last")
    if before != len(combined):
        logger.info(f"중복 {before - len(combined)}건 제거")

    combined = combined.sort_values("date", ascending=False)
    combined.to_parquet(TRADES_PATH, index=False)
    logger.info(f"체결 내역 저장 완료: {TRADES_PATH} (총 {len(combined)}건)")


def _incremental_from_date() -> str | None:
    """
    기존 trade_history의 마지막 체결일 -3일을 조회 시작일로 삼는다.
    (결제 지연/정정으로 뒤늦게 확정되는 건을 다시 훑기 위한 여유분)
    파일이 없으면 None → 전체 기간 조회.
    """
    if not TRADES_PATH.exists():
        return None
    try:
        existing = pd.read_parquet(TRADES_PATH, columns=["date"])
    except Exception:
        return None
    if existing.empty:
        return None
    last = pd.to_datetime(existing["date"]).max()
    if pd.isna(last):
        return None
    return (last - pd.Timedelta(days=3)).strftime("%Y-%m-%d")


# ─── 7. 메인 파이프라인 ──────────────────────────────────────────────────────

def run_collection(dry_run: bool = False, full: bool = False) -> None:
    """
    토스증권 데이터 수집 전체 파이프라인.

    Args:
        dry_run: True면 조회 결과만 출력하고 저장하지 않는다.
        full:    True면 주문 내역을 전체 기간으로 재조회한다.
    """
    logger.info(f"=== 토스증권 데이터 수집 시작 (dry_run={dry_run}, full={full}) ===")

    _load_env()
    client_id = _require_env("TOSS_CLIENT_ID")
    client_secret = _require_env("TOSS_CLIENT_SECRET")
    account_seq_env = os.environ.get("TOSS_ACCOUNT_SEQ") or None

    token = get_access_token(client_id, client_secret)
    account_seq = get_account_seq(token, account_seq_env)

    holdings_df, summary = get_holdings(token, account_seq)
    cash_usd = get_buying_power(token, account_seq)

    from_date = None if full else _incremental_from_date()
    trades_df = get_orders(token, account_seq, from_date=from_date)

    if dry_run:
        logger.info("--- dry-run: 저장하지 않고 결과만 출력합니다 ---")
        print("\n[계좌 요약]")
        print(f"  주식 평가금액 : ${summary['total_eval_usd']:,.2f}")
        print(f"  투자 원금     : ${summary['total_purchase_usd']:,.2f}")
        print(f"  평가손익      : ${summary['total_pnl_usd']:+,.2f}")
        print(f"  예수금        : ${cash_usd:,.2f}")
        print(f"\n[보유 종목] {len(holdings_df)}건")
        if not holdings_df.empty:
            print(holdings_df.to_string(index=False))
        print(f"\n[체결 내역] {len(trades_df)}건")
        if not trades_df.empty:
            print(trades_df.head(20).to_string(index=False))
        logger.info("=== dry-run 완료 ===")
        return

    save_snapshot(holdings_df, summary, cash_usd, datetime.date.today())
    save_trades(trades_df)

    logger.info("=== 토스증권 데이터 수집 완료 ===")


# ─── __main__ ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="토스증권 Open API 데이터 수집기")
    parser.add_argument("--dry-run", action="store_true",
                        help="저장하지 않고 조회 결과만 출력")
    parser.add_argument("--full", action="store_true",
                        help="주문 내역을 전체 기간으로 재조회 (기본은 증분)")
    args = parser.parse_args()

    try:
        run_collection(dry_run=args.dry_run, full=args.full)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"예상치 못한 오류: {exc}")
        sys.exit(1)
