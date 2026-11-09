"""
X (구 트위터) 트레이더 게시글 수집기

설정된 트레이더 목록(traders.json)의 최근 트윗을 Playwright 헤드리스 브라우저로 수집한다.
수집된 트윗은 output/x_trader_tweets.json에 저장되어 market_briefing.py에서
AI 트레이더 심리 요약에 사용된다.

사용법:
    python x_collector.py --login   # 1회용 로그인 창을 열어 세션 저장 (.cache/x_session.json)
    python x_collector.py           # 저장된 세션으로 트윗 수집 (헤드리스)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Windows 콘솔 UTF-8 설정
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CACHE_DIR = Path(".cache")
OUTPUT_DIR = Path("output")
SESSION_PATH = CACHE_DIR / "x_session.json"
TRADERS_CONFIG_PATH = Path("traders.json")
OUTPUT_TWEETS_PATH = OUTPUT_DIR / "x_trader_tweets.json"


def load_traders() -> list[dict[str, str]]:
    """traders.json에서 트레이더 목록을 로드한다."""
    if not TRADERS_CONFIG_PATH.exists():
        return []
    try:
        with open(TRADERS_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [Warning] traders.json 로드 실패: {e}")
        return []


def save_session_from_cookie(auth_token: str, ct0: str = "") -> bool:
    """사용자가 제공한 auth_token (및 선택적 ct0) 쿠키로 x_session.json을 생성한다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    auth_token = auth_token.strip().strip('"').strip("'")
    ct0 = ct0.strip().strip('"').strip("'")

    cookies = [
        {
            "name": "auth_token",
            "value": auth_token,
            "domain": ".x.com",
            "path": "/",
            "expires": -1,
            "httpOnly": True,
            "secure": True,
            "sameSite": "None",
        },
        {
            "name": "auth_token",
            "value": auth_token,
            "domain": ".twitter.com",
            "path": "/",
            "expires": -1,
            "httpOnly": True,
            "secure": True,
            "sameSite": "None",
        }
    ]
    if ct0:
        cookies.extend([
            {
                "name": "ct0",
                "value": ct0,
                "domain": ".x.com",
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            },
            {
                "name": "ct0",
                "value": ct0,
                "domain": ".twitter.com",
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }
        ])

    data = {
        "cookies": cookies,
        "origins": []
    }
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  [성공] 쿠키 세션이 저장되었습니다: {SESSION_PATH}")
    return True


def login_interactive() -> bool:
    """실제 설치된 Chrome 브라우저를 띄워 봇 감지를 우회하고 로그인 세션을 저장한다."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [Error] playwright가 설치되어 있지 않습니다. pip install playwright를 실행하세요.")
        return False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("  [X(트위터) 브라우저 로그인]")
    print("  Google Chrome 브라우저 창이 열리면 X에 로그인해주세요.")
    print("=" * 65 + "\n")

    with sync_playwright() as p:
        # 실제 설치된 Chrome 사용 시도 (없으면 기본 chromium)
        launch_kwargs: dict[str, Any] = {
            "headless": False,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--start-maximized",
            ]
        }
        try:
            browser = p.chromium.launch(channel="chrome", **launch_kwargs)
            print("  실제 Chrome 브라우저 엔진으로 실행합니다 (봇 감지 우회).")
        except Exception:
            browser = p.chromium.launch(**launch_kwargs)
            print("  Playwright Chromium 엔진으로 실행합니다.")

        context = browser.new_context(
            no_viewport=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        )
        page = context.new_page()

        # webdriver 플래그 은폐 (X의 봇 감지 무한 로딩 방지)
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
        """)

        try:
            # wait_until을 commit으로 설정하여 리액트 SPA 번들 로딩 지연으로 인한 멈춤 방지
            print("  X 로그인 페이지로 이동 중...")
            page.goto("https://x.com/login", wait_until="commit", timeout=30000)

            print("\n  >> 브라우저 창에서 로그인을 완료한 후, 콘솔에서 Enter를 누르세요...")
            input("  >> 로그인을 완료했으면 Enter를 누르세요: ")

            # 세션 스토리지 저장
            context.storage_state(path=str(SESSION_PATH))
            print(f"\n  [성공] 로그인 세션이 저장되었습니다: {SESSION_PATH}")
            print("  이제부터는 브라우저 창 없이 백그라운드로 자동 수집됩니다.\n")
            return True
        except Exception as e:
            print(f"  [Error] 로그인 세션 저장 실패: {e}")
            return False
        finally:
            try:
                browser.close()
            except Exception:
                pass


def is_session_valid() -> bool:
    """세션 파일이 존재하는지 확인한다."""
    return SESSION_PATH.exists() and SESSION_PATH.stat().st_size > 10


def is_tweets_cache_fresh(max_hours: int = 4) -> bool:
    """수집된 트윗 캐시가 max_hours 이내에 수집된 유효한 데이터인지 확인한다."""
    if not OUTPUT_TWEETS_PATH.exists():
        return False
    try:
        from datetime import datetime, timezone, timedelta
        with open(OUTPUT_TWEETS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("tweets") or len(data["tweets"]) == 0:
            return False
        updated_at = data.get("updated_at", "")
        if not updated_at:
            return False
        gen_dt = datetime.fromisoformat(updated_at)
        return (datetime.now(timezone.utc) - gen_dt) < timedelta(hours=max_hours)
    except Exception:
        return False


def _parse_twitter_date(raw_date: str) -> str:
    """트위터 날짜 문자열(예: 'Mon Nov 09 10:15:32 +0000 2026')을 ISO 8601 포맷으로 변환한다."""
    if not raw_date:
        return ""
    try:
        dt = datetime.strptime(raw_date, "%a %b %d %H:%M:%S %z %Y")
        return dt.isoformat()
    except Exception:
        return raw_date


def _extract_tweet_from_result(result_obj: dict[str, Any], handle: str) -> dict[str, Any] | None:
    """GraphQL tweet_results.result 객체에서 순수 트윗 데이터를 추출한다."""
    if not isinstance(result_obj, dict):
        return None

    # TweetWithVisibilityResults 래퍼 처리
    if result_obj.get("__typename") == "TweetWithVisibilityResults":
        result_obj = result_obj.get("tweet", {})

    legacy = result_obj.get("legacy")
    if not isinstance(legacy, dict):
        return None

    # 1. 리트윗(Retweet) 필터링: 타인의 글 재게시는 제외
    if legacy.get("retweeted_status_result") or legacy.get("retweeted_status_id_str"):
        return None

    # 2. 타인에게 보낸 답글(Reply) 필터링: 원문 브로드캐스트 글만 수집
    in_reply_to_user = legacy.get("in_reply_to_screen_name")
    if legacy.get("in_reply_to_status_id_str"):
        # 본인 스레드 첫머리가 아닌 타인과의 대화형 답글은 제외
        if in_reply_to_user and in_reply_to_user.lower() != handle.lower():
            return None

    full_text = legacy.get("full_text", "").strip()
    if not full_text:
        return None

    tweet_id = legacy.get("id_str") or result_obj.get("rest_id", "")
    pub_date_raw = legacy.get("created_at", "")
    pub_date = _parse_twitter_date(pub_date_raw)
    link = f"https://x.com/{handle}/status/{tweet_id}" if tweet_id else ""

    # 미디어(차트 이미지) 링크 추출
    media_urls = []
    extended = legacy.get("extended_entities", {})
    if isinstance(extended, dict) and "media" in extended:
        for m in extended["media"]:
            if isinstance(m, dict) and m.get("media_url_https"):
                media_urls.append(m["media_url_https"])

    return {
        "text": full_text,
        "pub_date": pub_date,
        "link": link,
        "is_pinned": False,
        "media_urls": media_urls,
    }


def parse_graphql_timeline(data: dict[str, Any], handle: str, max_count: int = 2) -> list[dict[str, Any]]:
    """X 프론트엔드의 UserTweets/UserTimeline GraphQL 응답 JSON을 파싱하여 정제된 트윗 목록을 반환한다."""
    tweets: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return tweets

    try:
        user_data = data.get("data", {}).get("user", {}).get("result", {})
        timeline = user_data.get("timeline_v2", {}).get("timeline", {}) or user_data.get("timeline", {})
        instructions = timeline.get("instructions", [])
    except Exception:
        return tweets

    for instr in instructions:
        if not isinstance(instr, dict):
            continue
        instr_type = instr.get("type", "")

        # 1. 고정 트윗(Pinned Entry)은 시황 왜곡 방지를 위해 명시적으로 제외
        if instr_type == "TimelinePinEntry":
            continue

        if instr_type == "TimelineAddEntries":
            entries = instr.get("entries", [])
            for entry in entries:
                if len(tweets) >= max_count:
                    break
                if not isinstance(entry, dict):
                    continue

                entry_id = entry.get("entryId", "")
                # 고정 트윗 및 광고 트윗 제외
                if entry_id.startswith("tweet-pinned-") or entry_id.startswith("promoted-tweet-"):
                    continue

                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                tweet_results = item_content.get("tweet_results", {})
                result_obj = tweet_results.get("result")

                if not result_obj:
                    # 스레드/모듈 형태인 경우 (TimelineTimelineModule)
                    items = content.get("items", [])
                    for mod_item in items:
                        if len(tweets) >= max_count:
                            break
                        mod_res = (
                            mod_item.get("item", {})
                            .get("itemContent", {})
                            .get("tweet_results", {})
                            .get("result")
                        )
                        tw = _extract_tweet_from_result(mod_res, handle)
                        if tw:
                            tweets.append(tw)
                    continue

                tw = _extract_tweet_from_result(result_obj, handle)
                if tw:
                    tweets.append(tw)

    return tweets


def _snowflake_to_iso(tweet_id_str: str) -> str:
    """트윗 고유 ID(Snowflake)로부터 밀리초 단위의 정확한 UTC ISO 8601 시각을 역산한다."""
    if not tweet_id_str or not tweet_id_str.isdigit():
        return ""
    try:
        tid = int(tweet_id_str)
        ms = (tid >> 22) + 1288834974657
        dt = datetime.fromtimestamp(ms / 1000.0, timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""


def _parse_tweet_article(article: Any, handle: str) -> dict[str, Any] | None:
    """트위터의 다양한 DOM 구조(신규 Tailwind 레이아웃 및 구형 data-testid)를 모두 지원하여 트윗을 파싱한다."""
    # 1. 상태 링크(Status URL) 및 트윗 ID 추출
    status_link = ""
    tweet_id = ""
    for a in article.query_selector_all('a[href*="/status/"]'):
        href = a.get_attribute("href") or ""
        if "/status/" in href and not any(href.endswith(s) for s in ["/photo/1", "/photo/2", "/photo/3", "/photo/4", "/analytics"]):
            status_link = f"https://x.com{href}" if href.startswith("/") else href
            parts = href.split("/status/")
            if len(parts) > 1:
                cand_id = parts[1].split("/")[0].split("?")[0]
                if cand_id.isdigit():
                    tweet_id = cand_id
            break

    # Snowflake ID 기반 정밀 ISO 8601 일시 계산
    pub_date = _snowflake_to_iso(tweet_id)

    # 2. 미디어(차트 이미지) 추출
    media_urls: list[str] = []
    for img in article.query_selector_all("img"):
        src = img.get_attribute("src") or ""
        if "pbs.twimg.com/media" in src and src not in media_urls:
            media_urls.append(src)

    # 3. 구형 UI 방식 텍스트 시도
    text_el = article.query_selector('div[data-testid="tweetText"]')
    time_el = article.query_selector("time")
    if not pub_date and time_el:
        pub_date = time_el.get_attribute("datetime") or ""

    article_text = article.inner_text()
    lines = [line.strip() for line in article_text.split("\n") if line.strip()]
    if not lines:
        return None

    # [피드백 반영 1] 고정 트윗 판별 (한국어 "메인에 올림", 영어 "Pinned" 등)
    is_pinned = any(
        kw in l for kw in ["메인에 올림", "고정된", "고정 트윗", "고정", "Pinned", "Pinned Tweet"]
        for l in lines[:3]
    )

    # [피드백 반영 2] 리트윗(Retweet) 필터링: 타인 글 재게시는 제외
    is_retweet = any(
        kw in l for kw in ["재게시", "reposted", "Reposted", "님이 재게시했습니다"]
        for l in lines[:2]
    )
    if is_retweet:
        return None

    # [피드백 반영 3] 단순 답글(Reply) 필터링: 본문 브로드캐스트 글만 수집
    is_reply = any(
        kw in l for kw in ["님에게 보내는 답글", "Replying to"]
        for l in lines[:4]
    )
    if is_reply:
        return None

    if text_el and text_el.inner_text().strip():
        return {
            "text": text_el.inner_text().strip(),
            "pub_date": pub_date,
            "link": status_link,
            "is_pinned": is_pinned,
            "media_urls": media_urls,
        }

    # 4. 신규 UI (Tailwind 및 일반 DOM) 방식 본문 파싱
    if len(lines) < 3:
        return None

    content_start_idx = 0
    for idx, line in enumerate(lines[:5]):
        if line.startswith("@") or line.lower() == f"@{handle.lower()}":
            content_start_idx = idx + 2
            break

    if content_start_idx == 0:
        content_start_idx = min(3, len(lines))

    # 하단 메트릭 숫자/버튼 제거
    content_end_idx = len(lines)
    while content_end_idx > content_start_idx:
        last_line = lines[content_end_idx - 1]
        cleaned = last_line.replace(",", "").replace(".", "").replace("K", "").replace("M", "").strip()
        if cleaned.isdigit() or last_line in ["답글", "재게시", "마음에 들어요", "조회수", "Reply", "Repost", "Like"]:
            content_end_idx -= 1
        else:
            break

    body_lines = lines[content_start_idx:content_end_idx]
    body_text = "\n".join(body_lines).strip()
    if not body_text:
        return None

    if not pub_date:
        pub_date = lines[content_start_idx - 1] if content_start_idx > 0 and len(lines) >= content_start_idx else ""

    return {
        "text": body_text,
        "pub_date": pub_date,
        "link": status_link,
        "is_pinned": is_pinned,
        "media_urls": media_urls,
    }


def collect_trader_tweets(
    max_tweets_per_trader: int = 2,
    max_age_hours: int | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """저장된 세션을 사용하여 트레이더들의 최근 트윗을 수집한다.
    
    Args:
        max_tweets_per_trader: 트레이더당 수집할 최대 트윗 수.
        max_age_hours: 지정 시 현재 기준 해당 시간(예: 24시간) 이내에 작성된 글만 수집.
        force: 캐시 무시 여부.
    """
    if not force and is_tweets_cache_fresh(max_hours=4):
        print("  최근 4시간 이내 수집된 트레이더 트윗 캐시가 유효합니다. 건너뜁니다.")
        data = load_collected_tweets()
        return data.get("tweets", []) if data else []

    if not is_session_valid():
        print("  [Info] X 세션 파일이 없습니다. `python x_collector.py --login`을 먼저 실행하세요.")
        return []

    traders = load_traders()
    if not traders:
        print("  [Info] traders.json에 트레이더 목록이 비어 있습니다.")
        return []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [Error] playwright 모듈이 필요합니다.")
        return []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, Any]] = []

    print(f"  트레이더 {len(traders)}명의 최근 트윗 수집 시작...")

    with sync_playwright() as p:
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": ["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        }
        try:
            browser = p.chromium.launch(channel="chrome", **launch_kwargs)
        except Exception:
            browser = p.chromium.launch(**launch_kwargs)

        context = browser.new_context(
            storage_state=str(SESSION_PATH),
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        )
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
        """)

        for idx, trader in enumerate(traders, 1):
            handle = trader["handle"]
            name = trader.get("name", handle)
            url = f"https://x.com/{handle}"
            captured_graphql_tweets: list[dict[str, Any]] = []

            # 1. GraphQL 응답 가로채기(Response Interception) 핸들러 등록
            def handle_response(response: Any) -> None:
                resp_url = response.url
                if "/graphql/" in resp_url and any(ep in resp_url for ep in ["UserTweets", "UserTimeline", "UserTweetsAndReplies", "Profile"]):
                    try:
                        if response.status == 200:
                            json_data = response.json()
                            parsed = parse_graphql_timeline(json_data, handle, max_count=max_tweets_per_trader * 2)
                            if parsed:
                                captured_graphql_tweets.extend(parsed)
                    except Exception:
                        pass

            page.on("response", handle_response)

            try:
                page.goto(url, wait_until="commit", timeout=20000)

                # GraphQL 응답 수신 대기 (최대 2.5초)
                start_wait = time.time()
                while time.time() - start_wait < 2.5:
                    if len(captured_graphql_tweets) >= max_tweets_per_trader:
                        break
                    page.wait_for_timeout(250)

                # GraphQL 가로채기 성공 여부 확인
                trader_tweets: list[dict[str, Any]] = []
                seen_links = set()

                if captured_graphql_tweets:
                    for tw in captured_graphql_tweets:
                        if len(trader_tweets) >= max_tweets_per_trader:
                            break
                        if tw["link"] and tw["link"] in seen_links:
                            continue
                        seen_links.add(tw["link"])
                        trader_tweets.append(tw)

                # 2. GraphQL 응답이 없거나 부족할 경우 스마트 DOM 파싱 수행
                if len(trader_tweets) < max_tweets_per_trader:
                    try:
                        page.wait_for_selector('article', timeout=3000)
                    except Exception:
                        page.wait_for_timeout(1000)

                    articles = page.query_selector_all('article[data-testid="tweet"]')
                    if not articles:
                        articles = page.query_selector_all('article')

                    for article in articles:
                        if len(trader_tweets) >= max_tweets_per_trader:
                            break

                        tw_data = _parse_tweet_article(article, handle)
                        if not tw_data or not tw_data["text"]:
                            continue

                        # [피드백 반영 1] 고정 트윗(Pinned)은 시황 왜곡 방지를 위해 제외
                        if tw_data.get("is_pinned"):
                            continue

                        # [피드백 반영 2] 최대 경과 시간(max_age_hours) 초과 트윗 제외
                        if max_age_hours is not None and tw_data.get("pub_date"):
                            try:
                                from datetime import timedelta
                                p_dt = datetime.fromisoformat(tw_data["pub_date"])
                                if p_dt.tzinfo is None:
                                    p_dt = p_dt.replace(tzinfo=timezone.utc)
                                if (datetime.now(timezone.utc) - p_dt) > timedelta(hours=max_age_hours):
                                    continue
                            except Exception:
                                pass

                        if tw_data["link"] and tw_data["link"] in seen_links:
                            continue
                        seen_links.add(tw_data["link"])
                        trader_tweets.append(tw_data)

                # 수집된 트윗 통합 목록에 저장
                for tw in trader_tweets:
                    collected.append({
                        "handle": handle,
                        "name": name,
                        "text": tw["text"],
                        "pub_date": tw["pub_date"],
                        "link": tw["link"],
                        "is_pinned": tw.get("is_pinned", False),
                        "media_urls": tw.get("media_urls", []),
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    })

                source_type = "GraphQL" if captured_graphql_tweets else "DOM 폴백"
                print(f"    [{idx}/{len(traders)}] @{handle} ({name}): 트윗 {len(trader_tweets)}건 수집 ({source_type})")

            except Exception as e:
                print(f"    [{idx}/{len(traders)}] @{handle} 수집 중 오류: {e}")

            finally:
                # 다음 트레이더 순회를 위해 현재 리스너 제거
                try:
                    page.remove_listener("response", handle_response)
                except Exception:
                    pass

            # 레이트리밋 방어: 1.5 ~ 2.5초 랜덤 딜레이
            sleep_sec = random.uniform(1.5, 2.5)
            page.wait_for_timeout(int(sleep_sec * 1000))

        browser.close()

    # 결과 저장
    result_data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_tweets": len(collected),
        "tweets": collected,
    }
    with open(OUTPUT_TWEETS_PATH, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"  [완료] 총 {len(collected)}건의 트레이더 트윗 저장: {OUTPUT_TWEETS_PATH}")
    return collected


def load_collected_tweets() -> dict[str, Any] | None:
    """수집된 트윗 JSON을 로드한다."""
    if not OUTPUT_TWEETS_PATH.exists():
        return None
    try:
        with open(OUTPUT_TWEETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X 트레이더 트윗 수집기")
    parser.add_argument("--login", action="store_true", help="브라우저 창을 열어 로그인 세션 저장")
    parser.add_argument("--cookie", type=str, default="", help="브라우저 개발자 도구의 auth_token 쿠키 값 직접 입력")
    parser.add_argument("--ct0", type=str, default="", help="브라우저 개발자 도구의 ct0 쿠키 값 (선택)")
    parser.add_argument("--count", type=int, default=2, help="트레이더당 수집할 최대 트윗 수")
    parser.add_argument("--hours", type=int, default=None, help="현재 시점 기준 수집할 최대 경과 시간 (예: 24)")
    parser.add_argument("--force", action="store_true", help="캐시 무시 강제 수집")
    args = parser.parse_args()

    if args.cookie:
        save_session_from_cookie(args.cookie, args.ct0)
    elif args.login:
        login_interactive()
    else:
        collect_trader_tweets(max_tweets_per_trader=args.count, max_age_hours=args.hours, force=args.force)

