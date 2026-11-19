"""
시장 브리핑 생성기

Google News RSS 피드에서 당일 미국 주식 시장 뉴스를 수집하고,
Google Gemini API를 사용하여 한국어로 3~4줄 시장 브리핑을 생성한다.

결과는 output/market_briefing.json에 저장되어 app.py의 Daily Summary 탭에서
즉시 렌더링된다.

단독 실행:
    python market_briefing.py
    python market_briefing.py --force  # 캐시 무시 강제 갱신

파이프라인에서의 위치:
    run_pipeline.py의 RS 엔진 계산 직후, 대시보드 실행 직전에 호출된다.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("output")
BRIEFING_PATH = OUTPUT_DIR / "market_briefing.json"

# ─── 뉴스 수집 설정 ──────────────────────────────────────────────────────────

# Google News RSS 검색 쿼리 목록 — 각각에서 헤드라인을 수집한 뒤 중복을 제거한다.
_RSS_QUERIES: list[str] = [
    "US stock market when:1d",
    "Wall Street stock when:1d",
    "S&P 500 Nasdaq when:1d",
]

# 우선 수집할 신뢰도 높은 매체 (Google News RSS의 <source> 텍스트 기준).
# 이 목록에 없는 매체의 기사도 수집하되, 정렬 시 이 목록의 매체가 상단에 온다.
_PREFERRED_SOURCES: list[str] = [
    "Reuters", "Bloomberg", "CNBC", "WSJ", "The Wall Street Journal",
    "Barron's", "MarketWatch", "Financial Times", "Investing.com",
    "AP News", "Yahoo Finance",
]

_MAX_HEADLINES = 15        # 최대 헤드라인 수
_RSS_TIMEOUT_SEC = 8       # RSS 요청 타임아웃
_RSS_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RS-Dashboard/1.0"

# ─── LLM 요약 설정 ───────────────────────────────────────────────────────────

# 우선 시도할 Gemini 모델 목록 (일시적 503 부하 또는 모델 변경에 대응)
_GEMINI_MODELS: list[str] = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    # 최후 폴백 — 품질은 낮을 수 있으나 과부하가 모델별로 다를 때 요약 공백을 줄인다.
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]

# Gemini 호출 안정화 — 과부하(503/429) 시 무한 대기·즉시 실패를 모두 피한다.
_GEMINI_TIMEOUT_MS = 45_000                  # 호출 1회당 타임아웃(ms) — 과부하 시 504는 이 시간 뒤에 온다
_GEMINI_BACKOFF_SEC: tuple[int, ...] = (5, 15)  # lite 모델의 재시도 전 대기(지수 백오프) → 최대 3회
_GEMINI_TOTAL_BUDGET_SEC = 240               # 한 번의 요약 생성에 쓰는 총 시간 상한
_GEMINI_RETRY_BUDGET_SEC = 180               # 뉴스 요약 2차(지연) 재시도의 총 시간 상한
SUMMARY_MAX_AGE_HOURS = 24                   # 이전 요약을 유지·표시하는 최대 경과 시간
_TRANSIENT_MARKERS = ("503", "429", "500", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                      "DEADLINE", "TIMEOUT", "TIMED OUT", "OVERLOADED")

# 일반 flash 모델은 과부하일 때 재시도해도 같은 오류가 반복되는 경우가 많아(실측: 'hi' 한 단어도
# 503/504) 1회만 시도하고 바로 다음 모델로 넘어간다. lite 계열만 백오프 재시도한다.
def _max_attempts(model_name: str) -> int:
    return len(_GEMINI_BACKOFF_SEC) + 1 if "lite" in model_name else 1


# 브리핑 캐시 유효 시간 — 이 시간 이내에 재실행하면 기존 결과를 재사용한다.
BRIEFING_CACHE_HOURS = 4


# ─── 뉴스 수집 ────────────────────────────────────────────────────────────────

def _fetch_google_news_rss(query: str) -> list[dict[str, str]]:
    """단일 Google News RSS 쿼리에서 헤드라인 목록을 수집한다.

    Returns:
        [{"title": ..., "link": ..., "source": ..., "pub_date": ...}, ...]
    """
    encoded_q = urllib.request.quote(query, safe="")
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _RSS_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_RSS_TIMEOUT_SEC) as resp:
            tree = ET.fromstring(resp.read())
    except Exception:
        traceback.print_exc()
        return []

    results: list[dict[str, str]] = []
    for item in tree.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        source_el = item.find("source")
        results.append({
            "title": title_el.text.strip() if title_el is not None and title_el.text else "",
            "link": link_el.text.strip() if link_el is not None and link_el.text else "",
            "source": source_el.text.strip() if source_el is not None and source_el.text else "Unknown",
            "pub_date": pub_el.text.strip() if pub_el is not None and pub_el.text else "",
        })
    return results


def collect_headlines() -> list[dict[str, str]]:
    """여러 RSS 쿼리에서 헤드라인을 수집하고 중복 제거·정렬한다."""
    all_items: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for query in _RSS_QUERIES:
        items = _fetch_google_news_rss(query)
        for item in items:
            # 제목 기반 중복 제거 (Google News는 같은 기사가 여러 쿼리에 걸린다)
            norm_title = item["title"].lower().strip()
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                all_items.append(item)

    # 우선 매체 → 나머지 순서로 정렬, 같은 그룹 내에서는 원래 순서(최신순) 유지
    preferred_set = {s.lower() for s in _PREFERRED_SOURCES}

    def _sort_key(item: dict[str, str]) -> tuple[int, int]:
        source_lower = item["source"].lower()
        is_preferred = 0 if any(p in source_lower for p in preferred_set) else 1
        return (is_preferred, 0)

    all_items.sort(key=_sort_key)
    return all_items[:_MAX_HEADLINES]


# ─── 시장 컨텍스트 로드 ───────────────────────────────────────────────────────

def _load_market_context() -> str:
    """output/ 디렉터리의 기존 분석 결과에서 시장 컨텍스트를 추출한다.

    LLM 프롬프트에 정량 데이터를 주입하여 뉴스와 데이터를 일치시킨다.
    파일이 없으면 빈 문자열을 반환한다.
    """
    context_parts: list[str] = []

    # 섹터 요약
    sector_path = OUTPUT_DIR / "sector_summary.parquet"
    if sector_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(sector_path)
            if "ticker" in df.columns and "rs_score" in df.columns:
                top3 = df.nlargest(3, "rs_score")[["ticker", "rs_score"]]
                bot3 = df.nsmallest(3, "rs_score")[["ticker", "rs_score"]]
                context_parts.append(
                    f"Leading sectors (RS Score): {top3.to_dict('records')}"
                )
                context_parts.append(
                    f"Lagging sectors (RS Score): {bot3.to_dict('records')}"
                )
        except Exception:
            pass

    # Breadth
    breadth_path = OUTPUT_DIR / "breadth_metrics.parquet"
    if breadth_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(breadth_path)
            if not df.empty:
                latest = df.sort_values("date").iloc[-1]
                nnh = latest.get("net_new_high", 0)
                bs = latest.get("breadth_score", 0)
                context_parts.append(
                    f"Latest breadth: Net New High={nnh}, Breadth Score={bs}/5"
                )
        except Exception:
            pass

    return "\n".join(context_parts)


# ─── LLM 요약 ────────────────────────────────────────────────────────────────

def _build_prompt(headlines: list[dict[str, str]], market_context: str) -> str:
    """Gemini에 보낼 시장 브리핑 프롬프트를 생성한다."""
    headline_text = "\n".join(
        f"- [{h['source']}] {h['title']}"
        for h in headlines
    )

    prompt = f"""당신은 미국 주식 시장 전문 애널리스트입니다.
아래의 오늘 뉴스 헤드라인과 시장 정량 데이터를 바탕으로,
한국어로 간결한 시장 브리핑(3~4개 불릿 포인트)을 작성하세요.

## 규칙
1. 각 불릿 포인트는 1~2문장으로 핵심만 간결하게 작성합니다.
2. 첫 번째 불릿은 시장 전체 방향성과 핵심 드라이버를 다룹니다.
3. 두 번째 불릿은 주도/약세 섹터나 종목을 다룹니다.
4. 세 번째 불릿은 거시경제 이벤트(금리, 고용, 지정학 등)를 다룹니다.
5. (선택) 네 번째 불릿은 투자자 유의사항이나 향후 주요 이벤트를 다룹니다.
6. 불릿 포인트 앞에 적절한 이모지를 1개 붙여주세요.
7. 불릿 포인트만 출력하세요. 서두/맺음말/제목은 불필요합니다.

## 오늘의 뉴스 헤드라인
{headline_text}
"""
    if market_context:
        prompt += f"""
## 시장 정량 데이터
{market_context}
"""
    return prompt


def _is_transient_error(e: Exception) -> bool:
    """재시도할 가치가 있는 일시적 오류(과부하·레이트리밋·타임아웃)인지 판별한다."""
    code = getattr(e, "code", None)
    if isinstance(code, int) and code in (429, 500, 503, 504):
        return True
    msg = f"{type(e).__name__} {e}".upper()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _call_gemini(prompt: str, label: str, parse=None, budget_sec: float | None = None):
    """Gemini를 호출해 결과를 반환한다. 전부 실패하면 None.

    - 호출마다 타임아웃(_GEMINI_TIMEOUT_MS)을 걸어 서버 과부하 시 장시간 매달리지 않는다.
    - 일반 flash는 1회만 시도하고 실패 시 바로 다음 모델로, lite 계열은 일시적 오류에 한해
      지수 백오프로 재시도한 뒤 다음 모델로 넘어간다.
    - 영구 오류(인증·잘못된 모델명 등)는 재시도 없이 다음 모델로 넘어간다.
    - budget_sec로 총 시간 상한을 호출별로 조정할 수 있다(기본 _GEMINI_TOTAL_BUDGET_SEC).
    - parse가 주어지면 응답 텍스트를 변환하며, ValueError는 해당 모델 포기로 처리한다.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Info] GEMINI_API_KEY가 설정되지 않아 LLM 요약을 건너뜁니다.")
        return None

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS),
        )
    except Exception as e:
        print(f"  [Warning] {label} Gemini 클라이언트 초기화 실패: {e}")
        return None

    budget = _GEMINI_TOTAL_BUDGET_SEC if budget_sec is None else budget_sec
    deadline = time.time() + budget
    for model_name in _GEMINI_MODELS:
        max_attempts = _max_attempts(model_name)
        for attempt in range(max_attempts):
            if time.time() >= deadline:
                print(f"  [Warning] {label} 총 시간 상한({budget:.0f}s) 초과 — 중단합니다.")
                return None
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("빈 응답")
                result = parse(text) if parse else text
                print(f"  {label} 생성 성공: {model_name}")
                return result
            except ValueError as e:  # 빈 응답·JSON 파싱 실패 (json.JSONDecodeError 포함)
                print(f"  [Warning] {label} {model_name} 응답 처리 실패: {e}")
                break
            except Exception as e:
                if _is_transient_error(e) and attempt < max_attempts - 1:
                    wait = _GEMINI_BACKOFF_SEC[attempt]
                    print(f"  [Warning] {label} {model_name} 일시 오류({attempt + 1}회차), {wait}s 후 재시도: {str(e)[:120]}")
                    time.sleep(wait)
                    continue
                print(f"  [Warning] {label} {model_name} 실패: {str(e)[:200]}")
                break

    print(f"  [Warning] {label}: 모든 Gemini 모델 호출이 실패했습니다.")
    return None


def _parse_json_response(raw: str) -> dict[str, Any]:
    """코드블록으로 감싸진 JSON 응답을 파싱한다."""
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


def _summarize_with_gemini(
    headlines: list[dict[str, str]],
    market_context: str,
    budget_sec: float | None = None,
) -> str | None:
    """Gemini API로 시장 브리핑을 생성한다. 실패 시 None 반환."""
    return _call_gemini(_build_prompt(headlines, market_context), "AI 브리핑", budget_sec=budget_sec)


def _summarize_trader_tweets_with_gemini(
    tweets: list[dict[str, Any]],
    market_context: str,
) -> dict[str, Any] | None:
    """Gemini API로 트레이더들의 트윗을 분석하여 시장 심리 및 포지셔닝 브리핑을 생성한다."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not tweets:
        return None

    tweet_lines = []
    for t in tweets[:25]:
        handle = t.get("handle", "")
        name = t.get("name", handle)
        text = t.get("text", "").replace("\n", " ")
        tweet_lines.append(f"- [@{handle}] ({name}): {text}")

    tweet_text = "\n".join(tweet_lines)

    prompt = f"""당신은 윌리엄 오닐 스타일의 주도주 및 모멘텀 트레이딩 전문가입니다.
아래는 Mark Minervini, Dan Zanger, Kristjan Qullamaggie 등 세계적인 모멘텀 트레이더들이 최근 X(트위터)에 게시한 글들입니다.
이 코멘트들을 심층 분석하여 실전 트레이더 관점의 시장 심리 및 포지셔닝 브리핑을 작성하세요.

## 트레이더 게시글 목록
{tweet_text}

## 시장 정량 데이터
{market_context}

## 출력 요구사항
반드시 다음 JSON 포맷으로만 응답하세요 (마크다운 코드블록이나 불필요한 서두 없이 순수 JSON만 출력):
{{
  "sentiment_stance": "강세 적극 매수 / 주의 및 선별 매수 / 방어 및 현금 확보 중 가장 적합한 1개 문구",
  "sentiment_color": "success, warning, danger 중 1개 (강세=success, 주의=warning, 약세=danger)",
  "cash_guidance": "트레이더들이 공통적으로 시사하는 현금 비중 및 리스크 관리 조언 (1~2문장)",
  "mentioned_tickers": ["언급된 주요 주식 티커들 최대 5개 (예: NVDA, TSLA)"],
  "summary_bullets": [
    "트레이더들의 시장 상태(브레이크아웃 성공률, 지수 흐름)에 대한 핵심 평가 (1~2문장)",
    "주도주/섹터 및 개별 종목 움직임에 대한 트레이더들의 시각 (1~2문장)",
    "현재 시점에서 트레이더들이 취하고 있는 포지션 및 대응 전략 (1~2문장)"
  ]
}}
"""

    return _call_gemini(prompt, "트레이더 브리핑", parse=_parse_json_response)


def filter_recent_tweets(tweets: list[dict[str, Any]], max_hours: int = 24) -> list[dict[str, Any]]:
    """트윗의 pub_date를 기준으로 호출 시점에서 max_hours(기본 24시간) 이내에 작성된 글만 필터링한다."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_hours)
    recent = []

    for t in tweets:
        pub_str = t.get("pub_date", "")
        if not pub_str:
            continue
        try:
            dt = datetime.fromisoformat(pub_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                recent.append(t)
        except Exception:
            # ISO 변환 실패 시 비정규 텍스트(예: 수개월 전 날짜 등)는 제외
            continue

    return recent


# ─── 결과 저장/로드 ───────────────────────────────────────────────────────────

def _save_briefing(data: dict[str, Any]) -> None:
    """브리핑 결과를 JSON으로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BRIEFING_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  브리핑 저장 완료: {BRIEFING_PATH}")


def load_briefing() -> dict[str, Any] | None:
    """저장된 브리핑 JSON을 로드한다. 없으면 None."""
    if not BRIEFING_PATH.exists():
        return None
    try:
        with open(BRIEFING_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_cache_fresh(force: bool = False) -> bool:
    """캐시된 브리핑이 아직 유효한지 판단한다."""
    if force:
        return False
    data = load_briefing()
    if data is None:
        return False
    generated_at = data.get("generated_at", "")
    if not generated_at:
        return False
    # 요약이 없거나 이전 요약을 대신 보여주는 중이면 신선한 캐시로 보지 않는다 — 재생성을 다시 시도한다.
    # (API 키가 없으면 어차피 생성할 수 없으므로 매번 재시도하지 않도록 이 조건을 건너뛴다.)
    if os.environ.get("GEMINI_API_KEY", "").strip() and (
        not data.get("summary") or data.get("summary_stale")
    ):
        return False
    try:
        gen_time = datetime.fromisoformat(generated_at)
        now = datetime.now(timezone.utc)
        return (now - gen_time) < timedelta(hours=BRIEFING_CACHE_HOURS)
    except (ValueError, TypeError):
        return False


def _hours_since(iso_str: str | None) -> float | None:
    """ISO 시각 문자열로부터 경과 시간(시)을 반환한다. 파싱 불가면 None."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except (ValueError, TypeError):
        return None


# ─── 메인 실행 ────────────────────────────────────────────────────────────────

def generate_briefing(force: bool = False) -> dict[str, Any]:
    """뉴스 수집 → LLM 요약 → JSON 저장 전체 플로우를 실행한다.

    Args:
        force: True이면 캐시 유효 기간과 무관하게 강제 재생성.

    Returns:
        저장된 브리핑 딕셔너리.
    """
    # 캐시 확인
    if _is_cache_fresh(force):
        print(f"  기존 브리핑이 유효합니다 (캐시 {BRIEFING_CACHE_HOURS}시간 이내). 건너뜁니다.")
        return load_briefing()  # type: ignore[return-value]

    # X(트위터) 트레이더 트윗 수집 (세션이 저장되어 있는 경우 자동 수집)
    try:
        from x_collector import is_session_valid, collect_trader_tweets
        if is_session_valid():
            collect_trader_tweets(max_tweets_per_trader=5)
    except Exception as e:
        print(f"  [Notice] X 트레이더 트윗 수집 건너뜀: {e}")

    print("  뉴스 헤드라인 수집 중...")
    t0 = time.time()
    headlines = collect_headlines()
    print(f"  수집 완료: {len(headlines)}건 ({time.time() - t0:.1f}s)")

    if not headlines:
        result: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": None,
            "headlines": [],
            "error": "뉴스 헤드라인을 수집하지 못했습니다.",
        }
        _save_briefing(result)
        return result

    # 시장 컨텍스트 로드
    market_context = _load_market_context()

    # LLM 요약 생성
    print("  AI 브리핑 생성 중...")
    t1 = time.time()
    summary = _summarize_with_gemini(headlines, market_context)
    if summary:
        print(f"  AI 브리핑 생성 완료 ({time.time() - t1:.1f}s)")
    else:
        print("  AI 요약을 사용할 수 없습니다. 헤드라인만 저장합니다.")

    # 트레이더 트윗 요약 (X 수집 데이터가 있는 경우)
    trader_briefing = None
    tweets_file = OUTPUT_DIR / "x_trader_tweets.json"
    if tweets_file.exists():
        try:
            with open(tweets_file, encoding="utf-8") as f:
                tweets_data = json.load(f)
            tweets_list = tweets_data.get("tweets", [])
            # [24시간 필터] 호출 시점 기준 24시간 이내 포스팅된 트윗만 선별
            recent_tweets = filter_recent_tweets(tweets_list, max_hours=24)
            if recent_tweets:
                print(f"  AI 트레이더 심리 브리핑 생성 중 (최근 24시간 내 {len(recent_tweets)}건)...")
                t2 = time.time()
                trader_summary = _summarize_trader_tweets_with_gemini(recent_tweets, market_context)
                if trader_summary:
                    trader_briefing = trader_summary
                    trader_briefing["updated_at"] = tweets_data.get("updated_at")
                    trader_briefing["total_traders"] = len({t.get("handle") for t in recent_tweets})
                    trader_briefing["tweets"] = recent_tweets
                    print(f"  AI 트레이더 심리 브리핑 완료 ({time.time() - t2:.1f}s)")
            elif tweets_list:
                print("  [Info] 최근 24시간 이내에 포스팅된 트레이더 게시글이 없습니다.")
                trader_briefing = {
                    "sentiment_stance": "관망 및 모니터링",
                    "sentiment_color": "warning",
                    "cash_guidance": "최근 24시간 이내 트레이더들의 신규 포스팅이 없습니다. 장 시작 전 무리한 진입을 자제하고 보수적인 현금 관리를 유지하세요.",
                    "mentioned_tickers": [],
                    "summary_bullets": [
                        "최근 24시간 이내 주요 트레이더들의 신규 게시글이 등록되지 않았습니다.",
                        "주말 및 휴일 또는 관망 국면으로 인해 새로운 방향성 코멘트가 부재한 상태입니다.",
                        "시장 변동성 확인 전까지는 기존 원칙에 따른 보수적 포지션 유지가 권장됩니다."
                    ],
                    "updated_at": tweets_data.get("updated_at"),
                    "total_traders": 0,
                    "tweets": [],
                }
        except Exception as e:
            print(f"  [Warning] 트레이더 트윗 요약 처리 중 오류: {e}")

    # 뉴스 요약 2차(지연) 재시도 — 과부하는 수 분 뒤 풀리는 경우가 많다.
    if not summary:
        print(f"  AI 브리핑 지연 재시도 중 (최대 {_GEMINI_RETRY_BUDGET_SEC}s)...")
        t3 = time.time()
        summary = _summarize_with_gemini(headlines, market_context, budget_sec=_GEMINI_RETRY_BUDGET_SEC)
        if summary:
            print(f"  AI 브리핑 지연 재시도 성공 ({time.time() - t3:.1f}s)")

    # 기존 브리핑의 trader_briefing 보존 (새로 생성되지 않았을 때)
    old_briefing = load_briefing()
    if trader_briefing is None and old_briefing and old_briefing.get("trader_briefing"):
        trader_briefing = old_briefing.get("trader_briefing")

    # 뉴스 요약이 끝내 실패하면 이전 요약을 유지한다 (SUMMARY_MAX_AGE_HOURS 이내일 때만).
    now_iso = datetime.now(timezone.utc).isoformat()
    summary_generated_at: str | None = now_iso if summary else None
    summary_stale = False
    if not summary and old_briefing and old_briefing.get("summary"):
        old_at = old_briefing.get("summary_generated_at") or old_briefing.get("generated_at")
        age_h = _hours_since(old_at)
        if age_h is not None and age_h <= SUMMARY_MAX_AGE_HOURS:
            summary = old_briefing["summary"]
            summary_generated_at = old_at
            summary_stale = True
            print(f"  [Info] 새 AI 요약 실패 — {age_h:.1f}시간 전 이전 요약을 유지합니다.")
        else:
            print(f"  [Info] 이전 요약이 {SUMMARY_MAX_AGE_HOURS}시간을 초과해 폐기합니다.")

    result = {
        "generated_at": now_iso,
        "summary": summary,
        "summary_generated_at": summary_generated_at,
        "summary_stale": summary_stale,
        "headlines": [
            {
                "title": h["title"],
                "link": h["link"],
                "source": h["source"],
                "pub_date": h["pub_date"],
            }
            for h in headlines
        ],
        "trader_briefing": trader_briefing,
        "error": None,
    }
    _save_briefing(result)
    return result


def run_briefing(force: bool = False) -> None:
    """파이프라인 호출용 진입점."""
    from dotenv import load_dotenv
    load_dotenv()
    generate_briefing(force=force)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    # Windows 콘솔 인코딩 대응 (이모지 및 특수문자 출력 허용)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="시장 브리핑 생성기")
    parser.add_argument("--force", action="store_true", help="캐시 무시 강제 갱신")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    result = generate_briefing(force=args.force)
    print("\n=== 브리핑 결과 ===")
    if result.get("summary"):
        print(result["summary"])
    else:
        print("(AI 요약 없음 - 헤드라인만 수집됨)")
    print(f"\n헤드라인 {len(result.get('headlines', []))}건")
