"""마스터 캐시에 굳어버린 ffill 흔적을 되돌리는 일회성 복구 스크립트.

배경
----
예전 download_ohlcv는 ffill()을 마스터 캐시에 *저장하기 전에* 걸었다. 그 결과
다운로드에 실패한 봉이 전일 값 복사본으로 캐시에 남아, "받지 못한 봉"과
"가격이 그대로인 봉"을 구분할 수 없게 됐다. 이 상태에서는
  - 레이트리밋으로 빠진 며칠이 횡보 가격으로 굳어 RS/모멘텀 계산을 오염시키고,
  - 어떤 티커가 밀렸는지 알 수 없어 다음 실행이 이어받지 못한다.
data_loader는 이제 캐시를 raw(ffill 전)로 저장하지만, 이미 굳은 과거 값은
그대로 남아 있다. 이 스크립트가 그 흔적을 NaN으로 되돌려 재요청 대상으로 만든다.

판정 기준
--------
Close/High/Low/Volume 네 값이 *모두* 전일과 정확히 일치하면 ffill 흔적으로 본다.
실제 봉이 전일과 거래량까지 소수점 단위로 같을 확률은 사실상 0이다
(예: 관측된 AACB는 27090주가 4거래일 연속 동일 — 명백한 복사본).

말단 연속 flat 길이로 두 집단을 구분한다:
  - 짧은 연속(기본 8거래일 이하) = 다운로드 실패 피해자 → 되돌리면 재수신 가능
  - 긴 연속(그 이상)             = 상장폐지/거래정지로 야후에 새 봉이 아예 없음
                                   → 되돌려봐야 응답이 없으므로 건드리지 않는다
후자를 NaN으로 만들면 우선순위 큐 앞자리만 영구히 점거하므로 제외하는 편이 낫다.

사용법
-----
    python repair_ffill_cache.py                 # 미리보기(dry-run, 기본값)
    python repair_ffill_cache.py --apply         # 실제 적용 (.bak 백업 자동 생성)
    python repair_ffill_cache.py --max-run 5 --apply
    python repair_ffill_cache.py --cache etf_ohlcv_master_cache.parquet --apply

적용 후 data_downloader.py를 다시 실행하면 되돌린 구간이 우선 재요청된다.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / ".cache"
FIELDS = ("Close", "High", "Low", "Volume")
DEFAULT_MAX_RUN = 8       # 이 길이 이하의 말단 연속 flat만 복구 대상으로 본다
DEFAULT_MIN_VALID = 30    # 복구 후 유효봉이 이보다 적어지면 되돌리지 않는다


def _split_fields(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(티커, 필드) MultiIndex 캐시를 필드별 DataFrame으로 분리."""
    return {f: master.xs(f, level=1, axis=1) for f in FIELDS}


def _rebuild(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """필드별 DataFrame을 원래 캐시 레이아웃으로 재조립 (download_ohlcv와 동일)."""
    parts = []
    for f in FIELDS:
        d = frames[f].copy()
        d.columns = pd.MultiIndex.from_product([d.columns, [f]])
        parts.append(d)
    return pd.concat(parts, axis=1).sort_index(axis=1)


def _flat_mask(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """네 필드가 모두 전일과 동일한 셀 = ffill 복사본으로 판정."""
    mask = None
    for f in FIELDS:
        df = frames[f]
        eq = df == df.shift(1)  # NaN == NaN 은 False → 이미 결측인 셀은 걸리지 않음
        mask = eq if mask is None else (mask & eq)
    assert mask is not None
    return mask


def _trailing_run_lengths(mask: pd.DataFrame) -> pd.Series:
    """티커별로 마지막 행에서 뒤로 이어지는 연속 flat 길이."""
    arr = mask.values
    n, t = arr.shape
    run = np.zeros(t, dtype=int)
    for i in range(n - 1, -1, -1):
        still = run == (n - 1 - i)   # 아직 끊기지 않고 이어져 온 티커만 연장
        run[still] += arr[i][still]
    return pd.Series(run, index=mask.columns)


def repair(
    cache_path: Path,
    max_run: int = DEFAULT_MAX_RUN,
    min_valid: int = DEFAULT_MIN_VALID,
    apply: bool = False,
) -> None:
    if not cache_path.exists():
        print(f"[Skip] 캐시 없음: {cache_path}")
        return

    master = pd.read_parquet(cache_path)
    master.index = pd.to_datetime(master.index)
    frames = _split_fields(master)
    close = frames["Close"]
    print(f"\n{'=' * 68}\n{cache_path.name}  ({close.shape[0]}행 × {close.shape[1]}티커)\n{'=' * 68}")

    mask = _flat_mask(frames)
    runs = _trailing_run_lengths(mask)

    # 복구 대상 = 말단 연속 flat이 1..max_run 인 티커. 그 티커의 말단 구간만 되돌린다.
    targets = runs[(runs > 0) & (runs <= max_run)]
    dead = runs[runs > max_run]
    print(f"  말단 flat 있는 티커      : {int((runs > 0).sum()):,}")
    print(f"  복구 대상 (1~{max_run}거래일)  : {len(targets):,}")
    print(f"  제외: 장기 정지 (>{max_run}일) : {len(dead):,}")

    if targets.empty:
        print("  -> 되돌릴 구간이 없습니다.")
        return

    # 되돌릴 셀 마스크: 대상 티커의 말단 연속 구간만 True
    repair_mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    n = len(close.index)
    for ticker, run_len in targets.items():
        repair_mask.iloc[n - int(run_len):, repair_mask.columns.get_loc(ticker)] = True

    # 안전장치: 되돌린 뒤 유효봉이 min_valid 미만이 되는 티커는 제외한다.
    # (그런 티커는 다음 실행에서 need_full로 분류돼 16개월치를 통째로 다시 받게 되고,
    #  그 요청 폭증이 애초에 문제였던 레이트리밋을 다시 부른다.)
    remaining = (close.notna() & ~repair_mask).sum()
    unsafe = remaining[remaining < min_valid].index
    if len(unsafe) > 0:
        repair_mask[unsafe] = False
        print(f"  제외: 복구 시 유효봉 {min_valid}개 미만 : {len(unsafe):,}")

    cells = int(repair_mask.values.sum())
    tickers_n = int(repair_mask.any().sum())
    print(f"  -> 되돌릴 (티커, 날짜): {cells:,}건 / 티커 {tickers_n:,}개 "
          f"— 필드 {len(FIELDS)}종에 동시 적용되므로 총 {cells * len(FIELDS):,}개 값")

    by_date = repair_mask.sum(axis=1)
    by_date = by_date[by_date > 0]
    print("\n  날짜별 되돌릴 티커 수:")
    for d, cnt in by_date.tail(12).items():
        print(f"     {pd.Timestamp(d).date()}  {int(cnt):,}")

    if not apply:
        print("\n  [Dry-run] 실제로 반영하려면 --apply 를 붙여 다시 실행하세요.")
        return

    backup = cache_path.with_suffix(cache_path.suffix + ".bak")
    shutil.copy2(cache_path, backup)
    print(f"\n  [Backup] {backup.name}")

    for f in FIELDS:
        frames[f] = frames[f].mask(repair_mask)

    repaired = _rebuild(frames)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    repaired.to_parquet(tmp)
    tmp.replace(cache_path)
    print(f"  [Apply] {cache_path.name} 갱신 완료")

    # 미수신 카운터 초기화: 되돌린 티커는 아직 "물어본 적 없는" 상태로 되돌려
    # 다음 실행에서 후순위로 밀리지 않고 정상적으로 우선 재요청되게 한다.
    stale = cache_path.with_name(cache_path.stem + ".stale.json")
    if stale.exists():
        stale.unlink()
        print(f"  [Reset] {stale.name} 삭제 (미수신 카운터 초기화)")

    after = _trailing_run_lengths(_flat_mask(frames))
    print(f"  [Verify] 복구 후 말단 flat 티커: {int((after > 0).sum()):,} "
          f"(장기 정지 {len(dead):,}개는 의도적으로 보존)")
    print("\n  다음 단계: python data_downloader.py 를 실행하면 되돌린 구간이 우선 재요청됩니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="마스터 캐시의 ffill 흔적을 NaN으로 되돌린다.")
    ap.add_argument("--cache", action="append", default=None,
                    help="대상 캐시 파일명 (기본: 주식/ETF 마스터 캐시 모두)")
    ap.add_argument("--max-run", type=int, default=DEFAULT_MAX_RUN,
                    help=f"복구 대상으로 볼 말단 연속 flat 최대 거래일 수 (기본 {DEFAULT_MAX_RUN})")
    ap.add_argument("--min-valid", type=int, default=DEFAULT_MIN_VALID,
                    help=f"복구 후 남아야 할 최소 유효봉 수 (기본 {DEFAULT_MIN_VALID})")
    ap.add_argument("--apply", action="store_true",
                    help="실제로 캐시에 반영 (미지정 시 미리보기만)")
    args = ap.parse_args()

    names = args.cache or ["ohlcv_master_cache.parquet", "etf_ohlcv_master_cache.parquet"]
    for name in names:
        repair(CACHE_DIR / name, max_run=args.max_run,
               min_valid=args.min_valid, apply=args.apply)


if __name__ == "__main__":
    main()
