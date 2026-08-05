"""
RS Dashboard 통합 파이프라인

실행:
  python run_pipeline.py                  # 전체 실행 (다운로드 → RS 계산 → 계좌 수집 → 대시보드)
  python run_pipeline.py --skip-download  # 다운로드 건너뛰기
  python run_pipeline.py --skip-engine    # RS 계산 건너뛰기
  python run_pipeline.py --skip-account   # 계좌 수집 건너뛰기 (.env 미설정 시 필요)
  python run_pipeline.py --no-app         # 대시보드 실행 없이 데이터 갱신만

계좌 수집 단계는 .env의 토스증권 자격증명을 사용한다. 자격증명이 없거나 인증에
실패하면 다른 단계와 동일하게 파이프라인이 중단되므로, 아직 설정 전이라면
--skip-account를 붙여 실행할 것.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd


def _step_download() -> None:
    from data_downloader import run_download
    run_download()


def _step_engine() -> None:
    from rs_engine import run_pipeline

    _NYSE_NASDAQ_AMEX = {"NASDAQ", "N", "A", "P"}
    _universe_path = ".intermediate/universe.parquet"
    _breadth_tickers: list[str] = []
    if os.path.exists(_universe_path):
        try:
            _uni = pd.read_parquet(_universe_path)
            if "Exchange" in _uni.columns:
                _uni = _uni[_uni["Exchange"].isin(_NYSE_NASDAQ_AMEX)]
            _breadth_tickers = _uni.index.tolist()
            print(f"  Market Breadth 티커: {len(_breadth_tickers):,}개 (NYSE/NASDAQ/AMEX)")
        except Exception as e:
            print(f"  [Warning] Universe 로드 실패, Breadth 계산 건너뜀: {e}")

    config = {
        "close_path":       ".intermediate/close.parquet",
        "etf_close_path":   ".intermediate/etf_close.parquet",
        "etf_info_path":    ".intermediate/etf_info.parquet",
        "etf_holdings_path":".intermediate/etf_top_holdings.parquet",
        "spy_close_path":   ".intermediate/spy_close.parquet",
        "volume_path":      ".intermediate/volume.parquet",
        "market_cap_path":  ".intermediate/market_caps.parquet",
        "sector_info_path": ".intermediate/sector_info.parquet",
        "high_path":        ".intermediate/high.parquet",
        "low_path":         ".intermediate/low.parquet",
        "output_dir":       "output",
        "benchmark":        "SPY",
        "breadth_tickers":  _breadth_tickers,
    }

    if not os.path.exists(config["close_path"]):
        print(f"  [Error] 입력 파일 없음: {config['close_path']}")
        print("  data_downloader.py를 먼저 실행하세요.")
        sys.exit(1)

    run_pipeline(config)


def _step_account() -> None:
    # 인자 없이 호출한다. 첫 실행은 output/trade_history.parquet이 없어 자동으로
    # 전체 기간을 조회하고(_incremental_from_date()가 None), 이후로는 증분만 받는다.
    from toss_collector import run_collection
    run_collection()


def _step_app() -> None:
    print("  Streamlit 대시보드를 시작합니다. 종료하려면 Ctrl+C를 누르세요.\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=False,
        )
    except KeyboardInterrupt:
        pass  # 사용자가 Ctrl+C로 정상 종료 — 에러가 아님


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RS Dashboard 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python run_pipeline.py                  전체 실행\n"
            "  python run_pipeline.py --skip-download  RS 계산 + 계좌 수집 + 대시보드\n"
            "  python run_pipeline.py --skip-account   계좌 수집 제외 (.env 미설정 시)\n"
            "  python run_pipeline.py --no-app         데이터 갱신만 (대시보드 미실행)\n"
        ),
    )
    parser.add_argument("--skip-download", action="store_true", help="데이터 다운로드 건너뜀")
    parser.add_argument("--skip-engine",   action="store_true", help="RS 엔진 계산 건너뜀")
    parser.add_argument("--skip-account",  action="store_true",
                        help="계좌 수집 건너뜀 (.env 자격증명이 없으면 필요)")
    parser.add_argument("--no-app",        action="store_true", help="대시보드 실행 안 함")
    args = parser.parse_args()

    steps = [
        ("[1/4] 데이터 다운로드  (data_downloader.py)", _step_download, args.skip_download),
        ("[2/4] RS 엔진 계산     (rs_engine.py)",       _step_engine,   args.skip_engine),
        ("[3/4] 계좌 수집        (toss_collector.py)",  _step_account,  args.skip_account),
        ("[4/4] 대시보드 실행    (app.py)",              _step_app,      args.no_app),
    ]

    pipeline_start = time.time()

    for label, fn, skip in steps:
        sep = "=" * 60
        if skip:
            print(f"\n{sep}\n[SKIP] {label}\n{sep}")
            continue

        print(f"\n{sep}\n[START] {label}\n{sep}")
        t0 = time.time()
        try:
            fn()
        except SystemExit:
            raise
        except KeyboardInterrupt:
            print("\n[중단] 사용자가 파이프라인을 중단했습니다.")
            sys.exit(0)
        except Exception as exc:
            print(f"\n[FAILED] {label}\n{exc}", file=sys.stderr)
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"\n[DONE]  {label}  ({elapsed:.1f}s)")

    total = time.time() - pipeline_start
    print(f"\n{'='*60}\n전체 파이프라인 완료  ({total:.1f}s)\n{'='*60}")


if __name__ == "__main__":
    main()
