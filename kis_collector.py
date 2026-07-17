"""
KIS (한국투자증권) REST API 데이터 수집기.

미국 주식 잔고·체결 데이터를 수집하여 output/ 폴더에 parquet로 저장합니다.

실행:
  python kis_collector.py --mode real    # 실전
  python kis_collector.py --mode paper   # 모의

.env 파일 (프로젝트 루트):
  KIS_APP_KEY=your_app_key_here
  KIS_APP_SECRET=your_app_secret_here
  KIS_ACCOUNT_NO=12345678-01   # 계좌번호-상품코드
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
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

KIS_BASE_URL = "https://openapi.kis.or.kr"

# 실전 / 모의 TR_ID 매핑
TR_IDS: dict[str, dict[str, str]] = {
    "real":  {
        "balance": "JTTT3012R",
        "trades":  "JTTT3001R",
    },
    "paper": {
        "balance": "VTTT3012R",
        "trades":  "VTTT3001R",
    },
}


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
    """환경변수 필수 조회. 없으면 명확한 메시지로 RuntimeError 발생."""
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(
            f"환경변수 '{key}'가 설정되지 않았습니다. "
            f".env 파일에 {key}=<값> 형태로 추가하세요."
        )
    return val


def _parse_account(account_no: str) -> tuple[str, str]:
    """
    '12345678-01' 형식 계좌번호를 (CANO, ACNT_PRDT_CD)로 분리.

    Args:
        account_no: 'XXXXXXXX-YY' 형식 문자열

    Returns:
        (계좌번호 앞 8자리, 상품코드 뒤 2자리) 튜플
    """
    parts = account_no.split("-")
    if len(parts) != 2:
        raise ValueError(f"잘못된 계좌번호 형식: '{account_no}'. 예: 12345678-01")
    return parts[0].strip(), parts[1].strip()


def _check_response(resp: requests.Response, context: str) -> dict:
    """
    API 응답 상태 및 rt_cd 검사.

    Args:
        resp:    requests.Response 객체
        context: 오류 메시지에 포함할 API 호출 설명

    Returns:
        파싱된 JSON dict

    Raises:
        RuntimeError: HTTP 오류 또는 rt_cd != "0"
    """
    resp.raise_for_status()
    data = resp.json()
    rt_cd = data.get("rt_cd", "")
    if rt_cd != "0":
        msg1 = data.get("msg1", "(메시지 없음)")
        raise RuntimeError(f"[{context}] KIS API 오류 (rt_cd={rt_cd}): {msg1}")
    return data


# ─── 1. 액세스 토큰 발급 ─────────────────────────────────────────────────────

def get_access_token(app_key: str, app_secret: str) -> str:
    """
    KIS OAuth2 액세스 토큰 발급 (유효기간 12시간, 매 실행마다 재발급).

    Args:
        app_key:    KIS 앱 키
        app_secret: KIS 앱 시크릿

    Returns:
        access_token 문자열

    Raises:
        RuntimeError: 토큰 발급 실패
    """
    url = f"{KIS_BASE_URL}/oauth2/tokenP"
    payload = {
        "grant_type": "client_credentials",
        "appkey":     app_key,
        "appsecret":  app_secret,
    }
    logger.info("액세스 토큰 발급 요청 중...")
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"토큰 발급 실패: {data.get('error_description', data)}")
    logger.info("액세스 토큰 발급 완료")
    return token


# ─── 2. 해외 주식 잔고 조회 ──────────────────────────────────────────────────

def get_overseas_balance(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    mode: str = "real",
) -> tuple[pd.DataFrame, dict]:
    """
    해외 주식 잔고 조회 (TR: JTTT3012R / VTTT3012R).

    Args:
        token:      액세스 토큰
        app_key:    KIS 앱 키
        app_secret: KIS 앱 시크릿
        account_no: 계좌번호 (예: '12345678-01')
        mode:       'real' 또는 'paper'

    Returns:
        (보유 종목 DataFrame, 계좌 합계 dict) 튜플
        DataFrame 컬럼: ticker, name, qty, avg_price_usd, current_price_usd,
                        eval_amt_usd, pnl_usd, pnl_pct
        dict 키: total_pnl_usd, total_eval_usd, total_cost_usd
    """
    cano, acnt_cd = _parse_account(account_no)
    tr_id = TR_IDS[mode]["balance"]

    url = f"{KIS_BASE_URL}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers = {
        "Authorization":  f"Bearer {token}",
        "appkey":         app_key,
        "appsecret":      app_secret,
        "tr_id":          tr_id,
        "content-type":   "application/json; charset=utf-8",
    }
    params = {
        "CANO":           cano,
        "ACNT_PRDT_CD":   acnt_cd,
        "OVRS_EXCG_CD":   "NASD",
        "TR_CRCY_CD":     "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }

    logger.info("해외 주식 잔고 조회 중...")
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    data = _check_response(resp, "get_overseas_balance")

    output1 = data.get("output1", []) or []
    output2 = data.get("output2", {}) or {}
    if isinstance(output2, list):
        output2 = output2[0] if output2 else {}

    def _f(val: object) -> float:
        try:
            return float(str(val).replace(",", "")) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    rows = []
    for item in output1:
        qty = _f(item.get("ovrs_cblc_qty", 0))
        if qty == 0:
            continue
        rows.append({
            "ticker":            item.get("pdno", ""),
            "name":              item.get("prdt_name", ""),
            "qty":               qty,
            "avg_price_usd":     _f(item.get("pchs_avg_pric")),
            "current_price_usd": _f(item.get("now_pric2")),
            "eval_amt_usd":      _f(item.get("evlu_amt")),
            "pnl_usd":           _f(item.get("evlu_pfls_amt")),
            "pnl_pct":           _f(item.get("evlu_pfls_rt")),
        })

    balance_df = pd.DataFrame(rows)

    summary = {
        "total_pnl_usd":  _f(output2.get("ovrs_tot_pfls")),
        "total_eval_usd": _f(output2.get("tot_evlu_pfls_amt") or output2.get("frcr_evlu_tota")),
        "total_cost_usd": _f(output2.get("frcr_buy_amt_smtl")),
    }

    logger.info(f"잔고 조회 완료: {len(balance_df)}개 종목 보유")
    return balance_df, summary


# ─── 3. 해외 주식 체결 내역 조회 ─────────────────────────────────────────────

def get_overseas_trades(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
    from_date: str,
    to_date: str,
    mode: str = "real",
) -> pd.DataFrame:
    """
    해외 주식 체결 내역 조회 (TR: JTTT3001R / VTTT3001R). 페이지네이션 자동 처리.

    Args:
        token:      액세스 토큰
        app_key:    KIS 앱 키
        app_secret: KIS 앱 시크릿
        account_no: 계좌번호
        from_date:  조회 시작일 (YYYYMMDD)
        to_date:    조회 종료일 (YYYYMMDD)
        mode:       'real' 또는 'paper'

    Returns:
        체결 내역 DataFrame
        컬럼: date, ticker, side, qty, price_usd, amount_usd, realized_pnl_usd, exchange
    """
    cano, acnt_cd = _parse_account(account_no)
    tr_id = TR_IDS[mode]["trades"]

    url = f"{KIS_BASE_URL}/uapi/overseas-stock/v1/trading/inquire-ccnl"
    headers = {
        "Authorization": f"Bearer {token}",
        "appkey":        app_key,
        "appsecret":     app_secret,
        "tr_id":         tr_id,
        "content-type":  "application/json; charset=utf-8",
    }

    def _f(val: object) -> float:
        try:
            return float(str(val).replace(",", "")) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    all_rows: list[dict] = []
    ctx_fk = ""
    ctx_nk = ""
    page = 0

    logger.info(f"체결 내역 조회 중 ({from_date} ~ {to_date})...")
    while True:
        page += 1
        params = {
            "CANO":           cano,
            "ACNT_PRDT_CD":   acnt_cd,
            "PDNO":           "",
            "ORD_STRT_DT":    from_date,
            "ORD_END_DT":     to_date,
            "SLL_BUY_DVSN":   "00",
            "CCLD_NCCS_DVSN": "01",
            "OVRS_EXCG_CD":   "",
            "SORT_SQN":       "DS",
            "CTX_AREA_FK200": ctx_fk,
            "CTX_AREA_NK200": ctx_nk,
        }
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = _check_response(resp, f"get_overseas_trades page={page}")

        for item in data.get("output", []) or []:
            qty = _f(item.get("ft_ccld_qty", 0))
            if qty == 0:
                continue
            side_code = item.get("sll_buy_dvsn_cd", "")
            side = "BUY" if side_code == "02" else "SELL" if side_code == "01" else side_code
            all_rows.append({
                "date":             item.get("ord_dt", ""),
                "ticker":           item.get("pdno", ""),
                "side":             side,
                "qty":              qty,
                "price_usd":        _f(item.get("ft_ccld_unpr3")),
                "amount_usd":       _f(item.get("ft_ccld_amt3")),
                "realized_pnl_usd": _f(item.get("rlzt_pfls")),
                "exchange":         item.get("ovrs_excg_cd", ""),
            })

        ctx_nk = data.get("ctx_area_nk200", "").strip()
        ctx_fk = data.get("ctx_area_fk200", "").strip()
        if not ctx_nk:
            break

    if not all_rows:
        logger.info("체결 내역 없음")
        return pd.DataFrame(columns=["date", "ticker", "side", "qty", "price_usd",
                                     "amount_usd", "realized_pnl_usd", "exchange"])

    trades_df = pd.DataFrame(all_rows)
    trades_df["date"] = pd.to_datetime(trades_df["date"], format="%Y%m%d", errors="coerce")
    for col in ["qty", "price_usd", "amount_usd", "realized_pnl_usd"]:
        trades_df[col] = trades_df[col].astype(float)

    logger.info(f"체결 내역 조회 완료: {len(trades_df)}건 (총 {page}페이지)")
    return trades_df


# ─── 4. USD 예수금 조회 ──────────────────────────────────────────────────────

def get_cash_balance(
    token: str,
    app_key: str,
    app_secret: str,
    account_no: str,
) -> dict:
    """
    해외 주식 예수금 조회 (TR: CTRP6504R).

    Args:
        token:      액세스 토큰
        app_key:    KIS 앱 키
        app_secret: KIS 앱 시크릿
        account_no: 계좌번호

    Returns:
        {"total_eval_usd": float, "total_asset_krw": float, "cash_usd": float}
    """
    cano, acnt_cd = _parse_account(account_no)

    url = f"{KIS_BASE_URL}/uapi/overseas-stock/v1/trading/inquire-present-balance"
    headers = {
        "Authorization": f"Bearer {token}",
        "appkey":        app_key,
        "appsecret":     app_secret,
        "tr_id":         "CTRP6504R",
        "content-type":  "application/json; charset=utf-8",
    }
    params = {
        "CANO":            cano,
        "ACNT_PRDT_CD":    acnt_cd,
        "WCRC_FRCR_DVSN_CD": "01",
        "NATN_CD":         "000",
        "TR_MKET_CD":      "00",
        "INQR_DVSN_CD":    "00",
    }

    logger.info("USD 예수금 조회 중...")
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    data = _check_response(resp, "get_cash_balance")

    def _f(val: object) -> float:
        try:
            return float(str(val).replace(",", "")) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    output = data.get("output2", {}) or {}
    if isinstance(output, list):
        output = output[0] if output else {}

    result = {
        "total_eval_usd":  _f(output.get("frcr_evlu_tota")),
        "total_asset_krw": _f(output.get("tot_asst_evlu_amt")),
        "cash_usd":        _f(output.get("frcr_use_psbl_amt")),
    }
    logger.info(f"예수금 조회 완료: 가용 USD {result['cash_usd']:,.2f}")
    return result


# ─── 5. 잔고 스냅샷 저장 ─────────────────────────────────────────────────────

def save_snapshot(
    balance_df: pd.DataFrame,
    cash_dict: dict,
    today: datetime.date,
) -> None:
    """
    일별 잔고 스냅샷을 누적 저장 (같은 날 재실행 시 중복 없이 덮어쓰기).

    Args:
        balance_df: get_overseas_balance()가 반환한 보유 종목 DataFrame
        cash_dict:  get_cash_balance()가 반환한 예수금 dict
        today:      저장 기준일 (datetime.date)
    """
    path = OUTPUT_DIR / "portfolio_snapshot.parquet"

    snap = balance_df.copy()
    snap["date"]           = today
    snap["total_eval_usd"] = cash_dict.get("total_eval_usd", 0.0)
    snap["cash_usd"]       = cash_dict.get("cash_usd", 0.0)

    if path.exists():
        existing = pd.read_parquet(path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        # 오늘 날짜 행 제거 (덮어쓰기 방지)
        existing = existing[existing["date"] != today]
        combined = pd.concat([existing, snap], ignore_index=True)
    else:
        combined = snap

    combined.to_parquet(path, index=False)
    logger.info(f"스냅샷 저장 완료: {path} ({len(snap)}개 종목, {today})")


# ─── 6. 체결 내역 저장 ───────────────────────────────────────────────────────

def save_trades(trades_df: pd.DataFrame) -> None:
    """
    체결 내역을 누적 저장 (date + ticker + side + qty + price_usd 복합키 기준 중복 제거).

    Args:
        trades_df: get_overseas_trades()가 반환한 체결 DataFrame
    """
    if trades_df.empty:
        logger.info("저장할 체결 내역 없음")
        return

    path = OUTPUT_DIR / "trade_history.parquet"

    dedup_keys = ["date", "ticker", "side", "qty", "price_usd"]
    for col in ["qty", "price_usd", "amount_usd", "realized_pnl_usd"]:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].astype(float)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, trades_df], ignore_index=True)
    else:
        combined = trades_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    after  = len(combined)
    if before != after:
        logger.info(f"중복 {before - after}건 제거")

    combined = combined.sort_values("date", ascending=False)
    combined.to_parquet(path, index=False)
    logger.info(f"체결 내역 저장 완료: {path} (총 {len(combined)}건)")


# ─── 7. 메인 파이프라인 ──────────────────────────────────────────────────────

def run_collection(mode: str = "real") -> None:
    """
    KIS API 데이터 수집 전체 파이프라인.

    Args:
        mode: 'real' (실전) 또는 'paper' (모의)
    """
    if mode not in ("real", "paper"):
        raise ValueError(f"mode는 'real' 또는 'paper'여야 합니다: '{mode}'")

    logger.info(f"=== KIS 데이터 수집 시작 (mode={mode}) ===")

    _load_env()
    app_key    = _require_env("KIS_APP_KEY")
    app_secret = _require_env("KIS_APP_SECRET")
    account_no = _require_env("KIS_ACCOUNT_NO")

    today      = datetime.date.today()
    from_date  = (today - datetime.timedelta(days=30)).strftime("%Y%m%d")
    to_date    = today.strftime("%Y%m%d")

    # 1. 토큰 발급
    token = get_access_token(app_key, app_secret)

    # 2. 잔고 조회
    balance_df, _summary = get_overseas_balance(token, app_key, app_secret, account_no, mode)

    # 3. 체결 내역 조회
    trades_df = get_overseas_trades(token, app_key, app_secret, account_no, from_date, to_date, mode)

    # 4. 예수금 조회
    cash_dict = get_cash_balance(token, app_key, app_secret, account_no)

    # 5. 저장
    save_snapshot(balance_df, cash_dict, today)
    save_trades(trades_df)

    logger.info("=== KIS 데이터 수집 완료 ===")


# ─── __main__ ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KIS 해외 주식 데이터 수집기")
    parser.add_argument(
        "--mode",
        choices=["real", "paper"],
        default="real",
        help="실전(real) 또는 모의(paper) 계좌 (기본값: real)",
    )
    args = parser.parse_args()

    try:
        run_collection(args.mode)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"예상치 못한 오류: {exc}")
        sys.exit(1)
