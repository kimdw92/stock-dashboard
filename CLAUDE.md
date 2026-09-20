# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

미국 주식 RS(Relative Strength) 분석 대시보드. 야후 파이낸스에서 OHLCV를 받아 IBD 방식 RS 지표를 계산하고, Streamlit 대시보드로 섹터/산업/테마/스크리너/브레드스/포트폴리오를 본다. 코드 주석과 UI는 한국어다.

## 실행

```powershell
.venv\Scripts\python.exe run_pipeline.py                  # 전체 (다운로드 → RS 계산 → 계좌 수집 → 대시보드)
.venv\Scripts\python.exe run_pipeline.py --skip-download  # 캐시된 시세로 RS만 재계산
.venv\Scripts\python.exe run_pipeline.py --skip-account   # .env 토스 자격증명 없을 때 필수
.venv\Scripts\python.exe run_pipeline.py --no-app         # 데이터 갱신만
.venv\Scripts\python.exe -m streamlit run app.py          # 대시보드만 (데이터 이미 있을 때)
```

각 단계는 단독 실행도 된다: `python data_downloader.py`, `python rs_engine.py`, `python toss_collector.py [--full|--dry-run]`.

테스트 프레임워크·린터·빌드 설정이 없다. 검증은 `python -m py_compile app.py rs_engine.py` 후 실제 파이프라인을 돌려 확인하는 방식이다. app.py만 고쳤다면 `--skip-download --skip-engine --skip-account`로 대시보드만 띄우는 게 가장 빠르다.

## 데이터 흐름

```
NASDAQ FTP + yfinance          토스증권 Open API
        │                              │
data_downloader.py             toss_collector.py
  (data_loader.py 사용)                │
        ↓                              ↓
  .intermediate/*.parquet      output/portfolio_snapshot.parquet
   close, high, low, volume,          output/trade_history.parquet
   etf_*, spy_close, universe,
   sector_info, market_caps
        ↓
   rs_engine.py
        ↓
   output/*.parquet
    rs_full, stock_screener, sector_summary, theme_summary,
    breadth_metrics, sector_breadth, technical_indicators
        ↓
     app.py (Streamlit)
```

디렉터리 역할이 고정돼 있다. `.cache/`는 원본 마스터 캐시(증분 append), `.intermediate/`는 다운로더 산출물, `output/`은 엔진·수집기 산출물이자 대시보드 입력. 셋 다 gitignore된다. app.py는 `output/` 뿐 아니라 `.intermediate/`도 직접 읽는다(ETF 종가/메타데이터, close.parquet은 여기에만 있음).

## 각 모듈

**data_loader.py** (2,000줄) — 다운로드 계층의 전부. 이 파일이 이 저장소에서 가장 까다롭고, 상수 위의 긴 주석들이 실측 근거를 담고 있으니 임계값을 건드리기 전에 반드시 읽을 것.
- **마스터 캐시 증분 구조**: `.cache/ohlcv_master_cache.parquet`은 `DatetimeIndex × MultiIndex(티커, OHLCV필드)`. 매 실행 캐시 이후 날짜만 요청해 append한다. 사이드카 JSON 3종이 캐시 옆에 붙는다 — `.stale.json`(응답 없는 티커 카운트), `.splitfix.json`(보정 이력), `.provisional`(장중 미확정 봉).
- **레이트리밋 관리**: `yf.download`는 티커당 HTTP 요청 1건이라 배치 크기 = 요청 수다. 야후 IP 한도(~3,000건 근처)를 피하려고 `DOWNLOAD_CHUNK_SIZE`/`DOWNLOAD_THREADS`/청크 간 대기/`REQUESTS_BEFORE_PAUSE`가 전부 낮게 고정돼 있다. 올리면 429가 난다.
- **미조정 스플릿 감지**: 증분 append 구조의 구조적 결함 — 스플릿 이전 저장분은 옛 가격 스케일로 남아 RS가 이음매를 수익률로 오독한다. 리버스(`_detect_split_artifacts`)와 정방향(`_detect_forward_splits`)을 거래량/달러거래량 보존 여부로 실제 급등·폭락과 구분한다.
- **유동성 필터**: 일 거래대금 중앙값 `UNIVERSE_MIN_DOLLAR_VOLUME` 미달은 '휴면' 처리해 반환 프레임에서 빼되, 매 실행 일부만 갱신해 복귀 판정 여지를 남긴다.
- **캐시는 raw로 저장한다** (ffill 전). 과거에 ffill된 채 저장돼 굳은 값이 있으면 `repair_ffill_cache.py`로 되돌린다(일회성 복구 스크립트, `--apply` 전 dry-run이 기본).

**data_downloader.py** — 7단계 다운로드 오케스트레이션. 큐레이션 ETF 목록(`CURATED_ETFS`)이 여기 하드코딩돼 있고, 이게 테마 분류의 원천이다(ETF의 `Theme` 필드 → `theme_summary`, ETF 보유종목 → 종목의 `parent_etf`/`theme`). ETF를 추가하려면 이 리스트에 넣는다. `_save_intermediate`는 빈 결과가 기존 파일을 덮어쓰지 않게 막는다.

**rs_engine.py** — 순수 계산 계층(I/O는 parquet 읽고 쓰기뿐). 핵심 개념 구분:
- `rs_score`: IBD 방식 가중 합(63/126/189/252일, 2+1+1+1). 데이터 없는 구간은 SPY 수익률로 대체해 신규 상장도 동일 가중 구조로 비교 가능하게 한다.
- `rs_line`: 단순 가격비(종목/SPY). 아래 파생 지표가 전부 이 위에 얹힌다.
- `rs_line_phase`: 21EMA 기준 강세/약세 (이탈 3일까지는 유예).
- `rs_line_ma_stack` / `rs_line_stack_days`: RS Line의 20/50 SMA 정배열·역배열·혼조와 연속 정배열 일수. phase가 최근 며칠을 보는 것과 달리 수주~수개월 추세용.
- 계산은 가능한 한 wide 프레임(date × ticker)에서 벡터로 하고 마지막에 long으로 melt한다. long format `groupby.transform`은 유니버스 규모에서 훨씬 느리다.
- `build_stock_screener`는 **필터링을 하지 않는다** — 최신일 스냅샷 전체를 저장하고 필터는 app.py가 건다.
- Breadth는 NYSE/NASDAQ/AMEX 종목만 대상(`universe.parquet`의 `Exchange` 필터).

**app.py** (3,800줄) — 단일 파일 Streamlit 앱. `main()`이 8개 탭(`_show_*_tab` 함수)을 부른다. 규칙:
- 모든 parquet 로딩은 `@st.cache_data` 함수를 거친다. `st.tabs`는 숨은 탭 본문도 매 리렌더 실행하므로, 여러 탭이 공유하는 프레임은 `main()`에서 한 번 로드해 인자로 넘긴다.
- 스크리너 필터 상태는 `screener_config.json`에 저장/복원된다. 필터 위젯을 추가하면 `_SCREENER_CFG_KEYS`에도 키를 넣어야 유지된다.
- 색상 상수(`COLOR_*`)와 `dashboard.css`가 표·차트 스타일의 단일 출처다.
- Watch List 종합 점수는 `_calc_priority_score()`가 5개 항목 × 20점으로 계산하며, 배점은 각 `_calc_priority_score_*()` 안에 있다.
- `TREND_PERIOD_MONTHS` 같은 공유 상수는 탭마다 복제하지 말 것(실제로 어긋나 버그가 났던 이력).

**toss_collector.py** — 토스증권 Open API로 미국 주식 잔고·주문내역·예수금 수집. `.env`의 `TOSS_CLIENT_ID`/`TOSS_CLIENT_SECRET`/`TOSS_ACCOUNT_SEQ`를 쓰고, 토큰은 `.cache/toss_token.json`에 캐시한다. API 그룹별 TPS 제한(ACCOUNT는 1 TPS)을 `RATE_LIMIT_TPS`로 강제한다. 등록되지 않은 IP에서 호출하면 403 edge-blocked가 난다. `output/trade_history.parquet`이 없으면 자동으로 전체 기간을 조회한다.

## 주의사항

- 캐시 계층이 여러 겹이라 "데이터가 이상하다"의 원인 후보가 넓다. 순서대로 의심할 것: Streamlit `@st.cache_data` → `output/` 산출물이 오래됨 → `.intermediate/` → `.cache/` 마스터 캐시의 스플릿 이음매·ffill 흔적.
- 전체 다운로드는 유니버스 규모 때문에 매우 오래 걸리고 레이트리밋에 걸린다. RS 로직만 고칠 때는 항상 `--skip-download`를 쓴다.
- `output/rs_full.parquet`은 200MB대(약 220만 행)다. app.py에서 이걸 피벗하거나 전체 순회하는 코드를 새로 넣기 전에, 더 싼 소스(`close.parquet`, `stock_screener.parquet`)로 될 일인지 먼저 볼 것.
- `.env`는 gitignore돼 있다. 자격증명이 없는 환경에서는 `--skip-account` 없이 실행하면 파이프라인 전체가 중단된다.
