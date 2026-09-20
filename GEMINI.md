# 프로젝트 개요

이 프로젝트는 윌리엄 오닐(William O'Neil)의 CAN SLIM 및 IBD(Investor's Business Daily) 상대 강도(Relative Strength, RS) 방법론을 기반으로 미국 주식 시장의 주도주(Leader Stocks), 주도 섹터 및 테마를 발굴·분석하는 풀스택 금융 분석 및 스크리닝 플랫폼입니다.

NASDAQ FTP 서버와 Yahoo Finance로부터 대규모 OHLCV 시세 데이터를 수집하고 증분 캐싱하며, IBD 가중 RS 점수 및 기술적 지표를 벡터 연산으로 고속 산출한 뒤, Streamlit 기반의 대화형 대시보드와 토스증권 실계좌 포트폴리오 연동 기능을 제공합니다.

전체 워크플로우는 `run_pipeline.py` 오케스트레이터를 통해 데이터 수집, 지표 연산, 계좌 수집, 웹 대시보드 실행까지 원클릭으로 구동됩니다.

---

## 핵심 기술 스택

* **언어:** Python 3.10+
* **데이터 분석 & 과학 계산:** pandas, numpy, scipy (선형회귀 기울기 등)
* **데이터 수집:** yfinance, requests, curl_cffi (안정적인 브라우저 세션 모방)
* **데이터 저장소 (컬럼형 스토리지):** pyarrow / fastparquet (모든 캐시 및 산출물 .parquet 저장)
* **웹 대시보드:** Streamlit
* **데이터 시각화:** Plotly (인터랙티브 캔들스틱, RS Line, 산점도/버블 차트)
* **스타일링:** CSS (`dashboard.css` 커스텀 테마 주입)
* **계좌 연동:** 토스증권 Open API (OAuth 2.0 기반 토큰 발급 및 잔고·주문내역 조회)

---

## 아키텍처 & 데이터 흐름

```
[데이터 소스]
  - NASDAQ FTP (유니버스 종목 기본 정보)
  - Yahoo Finance (주식/ETF 16개월 OHLCV 시세)
  - SSGA / iShares / VanEck 등 (ETF 보유종목 현황)
  - 토스증권 Open API (실계좌 잔고, 체결내역, 예수금)
          │
          ▼
[1단계: 수집 및 증분 캐싱]
  data_downloader.py (내부적으로 data_loader.py 활용)
    ├─ .cache/ohlcv_master_cache.parquet (MultiIndex 마스터 증분 캐시)
    ├─ .cache/*.json (.stale, .splitfix, .provisional 사이드카)
    └─ .intermediate/*.parquet 저장:
         - close, high, low, volume, spy_close, market_caps
         - etf_close, etf_high, etf_low, etf_info, etf_top_holdings
         - universe, sector_info
          │
          ▼
[2단계: RS 및 기술적 지표 계산]
  rs_engine.py
    └─ output/*.parquet 저장:
         - rs_full.parquet (전 종목 일자별 RS Score, RS Line, Phase, MA Stack 등)
         - stock_screener.parquet (최신일 스크리너 원본 스냅샷)
         - sector_summary.parquet & theme_summary.parquet
         - breadth_metrics.parquet & sector_breadth.parquet
         - technical_indicators.parquet (50 SMA, 14 ATR, ATR Multiple)
          │
          ▼
[3단계: 선택적 계좌 수집]
  toss_collector.py (.env 자격증명 필요)
    └─ output/portfolio_snapshot.parquet & output/trade_history.parquet
          │
          ▼
[4단계: 대시보드 시각화 & 인터랙션]
  app.py (Streamlit 대시보드)
    - output/ 디렉터리의 분석 결과 시각화
    - .intermediate/ 디렉터리의 원본 종가/ETF 메타데이터 직접 참조
    - screener_config.json에 사용자 필터 설정 영속화
```

### 디렉터리 역할 규약

* **`.cache/`**: 원본 마스터 시세 캐시. 실행 시마다 캐시된 최신 날짜 이후 데이터만 증분(append)하여 다운로드 부담을 최소화합니다. (Git 제외)
* **`.intermediate/`**: `data_downloader.py`의 다운로드 정제 산출물. `rs_engine.py`의 입력 소스이자 `app.py`의 52주/YTD 성과 계산용 참조 소스입니다. (Git 제외)
* **`output/`**: `rs_engine.py` 및 `toss_collector.py`의 최종 계산 산출물. `app.py`가 직접 로드하여 사용자 화면을 구성합니다. (Git 제외)

---

## 주요 모듈 및 상세 설명

### 1. `run_pipeline.py` (통합 파이프라인 진입점)
전체 파이프라인을 4단계로 순차 제어합니다.
* `[1/4] 데이터 다운로드` (`data_downloader.py`)
* `[2/4] RS 엔진 계산` (`rs_engine.py`)
* `[3/4] 계좌 수집` (`toss_collector.py`)
* `[4/4] 대시보드 실행` (`app.py`)

각 단계별 스킵 플래그를 제공하여 유연하게 실행할 수 있습니다.

### 2. `data_loader.py` (다운로드 및 캐시 엔진)
프로젝트에서 가장 복잡하고 정교한 시세 다운로드 계층입니다 (약 2,000줄).
* **마스터 캐시 증분 구조**: `.cache/ohlcv_master_cache.parquet`에 `DatetimeIndex × MultiIndex(티커, OHLCV필드)` 형식으로 저장하며, 날짜가 누락된 부분만 선별 요청합니다.
* **사이드카 메타데이터**:
  - `.stale.json`: 응답이 없거나 상장폐지된 티커 카운트 추적
  - `.splitfix.json`: 액면분할/병합 보정 이력 관리
  - `.provisional`: 장중 미확정 봉 캐싱 방지
* **야후 레이트리밋(429) 방어**: 청크 단위 분할(`DOWNLOAD_CHUNK_SIZE`), 스레드 제한(`DOWNLOAD_THREADS`), 요청 간 딜레이 및 휴식(`REQUESTS_BEFORE_PAUSE`)을 엄격하게 관리하여 IP 밴을 방지합니다.
* **미조정 스플릿 자동 감지**: 증분 캐시의 단점인 분할 이전 시세 왜곡을 방지하기 위해 거래량/달러거래량 보존 여부를 추적하여 리버스/정방향 스플릿을 감지하고 보정합니다.
* **Raw 캐시 원칙**: 캐시에는 `ffill()` 이전 순수 원본 데이터를 저장하여 결측치와 실제 횡보를 구분합니다.
* **유동성 필터**: 일일 거래대금 중앙값(`UNIVERSE_MIN_DOLLAR_VOLUME`) 미달 종목을 휴면 처리하여 메모리와 네트워크 낭비를 줄입니다.

### 3. `data_downloader.py` (데이터 다운로드 오케스트레이션)
7단계 다운로드 프로세스를 순차 실행합니다.
* **`CURATED_ETFS` 목록**: 11대 섹터 SPDR ETF, 시장 벤치마크(SPY, QQQ, IWM 등), 주요 산업 및 세부 테마(반도체, AI, 로보틱스, 포토닉스, 클라우드, 사이버보안, 원자력 등) 큐레이션 ETF가 하드코딩되어 있으며, 테마 분석의 기준이 됩니다.
* **ETF 보유종목 캐싱**: `ETF_HOLDINGS_MAX_AGE_DAYS`(기본 30일) 이내의 캐시가 있으면 재수집을 건너뛰어 불필요한 네트워크 요청을 줄입니다.
* **안전 저장 장치**: `_save_intermediate`는 다운로드 실패 등으로 인한 빈 DataFrame이 기존 캐시 파일을 덮어쓰지 못하도록 방어합니다.

### 4. `rs_engine.py` (상대 강도 & 기술적 지표 연산 엔진)
외부 네트워크 I/O 없이 로컬 parquet 파일만을 사용하여 고속 벡터 연산을 수행합니다.
* **IBD 가중 RS Score**:
  - 최근 4개 분기(63일, 126일, 189일, 252일) 수익률을 `2 : 1 : 1 : 1` 가중치로 합산.
  - 신규 상장 등으로 데이터가 부족한 기간은 동일 기간 **SPY 수익률로 대체**하여 모든 종목을 공정하게 비교.
* **RS Line & 파생 지표**:
  - `rs_line`: 종목 종가 / SPY 종가 비율
  - `rs_line_phase`: RS Line의 21 EMA 기준 상회/하회 여부 (3일 유예 버퍼 적용)
  - `rs_line_ma_stack` & `rs_line_stack_days`: RS Line의 20 SMA / 50 SMA 정배열·역배열 상태 및 연속 정배열 일수
  - `rs_slope`: 최근 20거래일 RS Line의 선형회귀 기울기
* **기술적 지표 (`technical_indicators.parquet`)**:
  - 50 SMA, 14일 ATR(True Range 기반), ATR 비율(%), 50일선 이격도(dist_pct_50sma), ATR Multiple(이격 배수 - Extension 과열 여부 판단).
* **Market Breadth**:
  - NYSE, NASDAQ, AMEX 상장 종목 대상 Net New High(52주 신고가 - 신저가), Breadth Score(0~5점), 섹터별 브레드스 산출.
* **벡터 연산 성능 원칙**: Wide format(date × ticker) 행렬 연산 후 필요한 시점에만 Long format으로 변환(melt)하여 대규모 유니버스를 수 초 내에 연산합니다.
* **무필터 스냅샷**: `stock_screener.parquet`은 사전문턱 없이 최신일 전체 종목 스냅샷을 저장하며, 실제 필터링은 대시보드 UI에서 수행합니다.

### 5. `app.py` (Streamlit 대화형 대시보드)
약 3,900줄 규모의 단일 파일 대시보드 앱으로, 8개 전용 탭으로 구성되어 있습니다.
* **8개 분석 탭**:
  1. `📅 Daily Summary`: 주요 지수 성과, 시장 성격(Market Character) 진단, Net New High & Breadth Score, SPY/QQQ 파워 트렌드(Price > 21 EMA > 50 SMA), 세부 시장/섹터/테마 성과표.
  2. `🩺 Market Breadth`: 시장 폭 지표 추이, 52주 신고가/신저가 모니터링, 섹터별 브레드스.
  3. `📊 섹터 분석`: 11개 섹터 RS 랭킹, RS Line 모멘텀 추이, 섹터 히트맵, 시장 RS 순위표(대형 vs 소형, 동일가중 비교).
  4. `🏭 산업 분석`: GICS 산업별 RS 강도 및 산업 내 선도 주식 목록.
  5. `🎯 테마 분석`: 큐레이션된 테마 ETF 기준 상대강도 순위 및 상위 10대 편입종목.
  6. `🔍 종목 스크리너`: RS 백분위(1/3/12개월), 정배열 일수, ATR Multiple, 시가총액, 거래대금, 산업 순위 등 복합 조건 스크리닝 및 CSV 내보내기. (`screener_config.json`과 양방향 연동)
  7. `💼 포트폴리오`: 토스증권 계좌 잔고, 실현손익, 보유종목의 RS 점수 및 기술적 상태 모니터링.
  8. `👀 Watch List`: 관심종목 우선순위 종합 점수(5개 항목 × 20점 = 100점 만점: RS Score, RS 3M, RS Phase, RS MA Stack, Extension) 산출 및 RS 백분위 vs 산업 강도 버블 차트 제공.
* **성능 및 렌더링 최적화**:
  - `@st.cache_data`를 통한 메모리 캐싱.
  - 탭 간 공유 프레임은 `main()`에서 단 1회 로드하여 하위 함수로 전달.
  - 대용량 파일(`rs_full.parquet`, 약 200MB/220만 행)의 반복 피벗 및 전체 순회를 지양하고 요약 파일 우선 활용.

### 6. `toss_collector.py` (토스증권 포트폴리오 수집기)
토스증권 Open API를 통해 보유 잔고, 주문 체결 내역, 예수금을 수집합니다.
* `.env` 파일의 API Key(`TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`, `TOSS_ACCOUNT_SEQ`) 사용.
* 토큰은 `.cache/toss_token.json`에 저장하여 유효기간 동안 재사용.
* 계좌 API 그룹의 1 TPS 레이트리밋 엄격 준수 (`RATE_LIMIT_TPS`).
* 첫 실행 시 전 기간 자동 조회, 이후 실행부터는 직전 체결일 기준 증분 조회.

### 7. `repair_ffill_cache.py` (캐시 복구 유틸리티)
과거 버전에서 잘못 적용되었던 마스터 캐시의 `ffill()` 복사본(Close/High/Low/Volume이 모두 전일과 동일한 봉)을 감지하여 NaN으로 되돌려주는 복구 도구입니다.

---

## 실행 방법

### 1. 가상환경 및 종속성 설치

```powershell
# Windows PowerShell 기준
python -m venv .venv
.venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. 토스증권 자격증명 설정 (선택 사항)

토스증권 포트폴리오를 연동하려면 프로젝트 루트에 `.env` 파일을 작성합니다:
```ini
TOSS_CLIENT_ID=your_client_id_here
TOSS_CLIENT_SECRET=your_client_secret_here
TOSS_ACCOUNT_SEQ=your_account_seq_here
```
> [!NOTE]
> `.env` 설정이 없거나 등록되지 않은 IP 환경에서는 반드시 `--skip-account` 플래그를 붙여 실행해야 파이프라인이 중단되지 않습니다.

### 3. 파이프라인 실행 명령어 모음

```powershell
# [기본] 전체 파이프라인 실행 (다운로드 → RS 계산 → 계좌 수집 → 대시보드)
python run_pipeline.py

# [자주 사용] 시세 다운로드를 건너뛰고 RS 재계산 및 대시보드 실행 (가장 추천)
python run_pipeline.py --skip-download

# [토스 미사용 시] 계좌 수집 단계 제외하고 실행
python run_pipeline.py --skip-account

# [빠른 UI 확인] 데이터/엔진 연산 없이 대시보드만 바로 실행
python run_pipeline.py --skip-download --skip-engine --skip-account
# 또는 직접 실행:
streamlit run app.py

# [백그라운드 데이터 갱신] 대시보드 실행 없이 데이터만 갱신
python run_pipeline.py --no-app
```

### 4. 개별 모듈 단독 실행

```powershell
python data_downloader.py                    # 시세 데이터 수집만 실행
python rs_engine.py                          # 지표 및 RS 엔진만 재계산
python toss_collector.py                     # 계좌 정보만 수집 (증분)
python toss_collector.py --full              # 계좌 체결내역 전체 기간 재수집
python repair_ffill_cache.py                 # 캐시 오염 여부 미리보기 (Dry-run)
python repair_ffill_cache.py --apply         # 캐시 오염 복구 실제 적용
```

---

## 개발 규칙 & 엔지니어링 가이드라인

### 1. 코드 검증 및 빌드
* 별도의 pytest 테스트 슈트나 린터 설정이 없으므로, 코드 수정 후 반드시 문법 검증을 선행합니다:
  ```powershell
  python -m py_compile app.py rs_engine.py data_loader.py data_downloader.py
  ```
* UI 로직 수정 시에는 `python run_pipeline.py --skip-download --skip-engine --skip-account`로 빠르게 대시보드를 띄워 확인합니다.

### 2. 레이트리밋 및 네트워크 주의사항
* Yahoo Finance는 IP당 요청량(~3,000건 근처) 한계가 있습니다. `data_loader.py`에 정의된 `DOWNLOAD_CHUNK_SIZE`, `DOWNLOAD_THREADS`, 대기 시간 상수들을 임의로 상향 조정하지 마십시오. (429 Too Many Requests 발생 위험)
* RS 수식이나 UI를 수정할 때는 절대 전체 다운로드를 다시 돌리지 말고 `--skip-download`를 사용하십시오.

### 3. 대용량 데이터 처리 원칙
* `output/rs_full.parquet`은 약 220만 행(200MB 이상) 규모의 시계열 데이터입니다.
* `app.py`에서 새로운 분석이나 차트를 추가할 때 `rs_full`을 전체 순회하거나 피벗하는 연산을 추가하기 전에, 더 가벼운 데이터 소스(`close.parquet`, `stock_screener.parquet`)로 해결할 수 있는지 먼저 검토하십시오.

### 4. 스크리너 설정 영속화
* 종목 스크리너의 기본값과 UI 필터 설정은 `screener_config.json`에 저장됩니다.
* 스크리너에 신규 필터 위젯을 추가할 때는 `app.py` 상단의 `_SCREENER_CFG_KEYS` 리스트에도 해당 세션 키를 등록해야 설정이 브라우저 재시작 시에도 유지됩니다.

### 5. 트러블슈팅: "데이터가 이상하게 표시될 때" 점검 순서
1. **Streamlit 캐시**: 대시보드 우측 상단 `Clear cache` 클릭 후 새로고침
2. **최종 산출물 확인**: `output/`의 parquet 파일 생성 일시가 최신인지 확인
3. **중간 데이터 확인**: `.intermediate/`의 종가 및 유니버스 데이터 검증
4. **마스터 캐시 결측/스플릿 확인**: `.cache/` 마스터 캐시에 비정상 급등락이나 굳어버린 ffill이 있는지 확인하고 필요 시 `repair_ffill_cache.py` 실행
