"""
RS Dashboard 실행 런처 (Interactive Launcher)
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    # 윈도우 콘솔 UTF-8 인코딩 대응
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 작업 디렉터리를 프로젝트 루트로 고정
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workspace_dir)

    python_exe = sys.executable
    streamlit_exe = os.path.join(os.path.dirname(python_exe), "streamlit.exe")
    if not os.path.exists(streamlit_exe):
        streamlit_cmd = [python_exe, "-m", "streamlit"]
    else:
        streamlit_cmd = [streamlit_exe]

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 68)
        print("                 [*] RS Dashboard 실행 런처")
        print("=" * 68)
        print()
        print("  [1] 빠른 대시보드 실행 (데이터 재계산 없이 바로 열기) - [추천]")
        print("  [2] RS 지표 재계산 후 대시보드 실행 (--skip-download)")
        print("  [3] 전체 파이프라인 실행 (다운로드 + RS 계산 + 대시보드)")
        print("  [4] 백그라운드 데이터 갱신만 수행 (대시보드 미실행)")
        print("  [5] 토스 계좌 연동 포함 전체 실행 (API 키 설정 필요)")
        print("  [0] 종료")
        print()
        print("=" * 68)

        try:
            choice = input("실행할 번호를 선택하세요 [0-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n런처를 종료합니다.")
            break

        if choice == "0":
            break
        elif choice == "1":
            print("\n" + "-" * 50)
            print("Streamlit 대시보드를 바로 시작합니다...")
            print("-" * 50)
            try:
                subprocess.run(streamlit_cmd + ["run", "app.py"])
            except KeyboardInterrupt:
                pass
            break
        elif choice == "2":
            print("\n" + "-" * 50)
            print("다운로드를 생략하고 RS 지표 재계산 후 대시보드를 실행합니다...")
            print("-" * 50)
            try:
                subprocess.run([python_exe, "run_pipeline.py", "--skip-download", "--skip-account"])
            except KeyboardInterrupt:
                pass
            break
        elif choice == "3":
            print("\n" + "-" * 50)
            print("전체 파이프라인 (다운로드 + RS 계산 + 대시보드)을 실행합니다...")
            print("-" * 50)
            try:
                subprocess.run([python_exe, "run_pipeline.py", "--skip-account"])
            except KeyboardInterrupt:
                pass
            break
        elif choice == "4":
            print("\n" + "-" * 50)
            print("데이터 다운로드 및 RS 재계산만 수행합니다...")
            print("-" * 50)
            try:
                subprocess.run([python_exe, "run_pipeline.py", "--skip-account", "--no-app"])
            except KeyboardInterrupt:
                pass
            print("\n작업이 완료되었습니다.")
            input("종료하려면 Enter를 누르세요...")
            break
        elif choice == "5":
            print("\n" + "-" * 50)
            print("토스증권 계좌 수집을 포함한 전체 파이프라인을 실행합니다...")
            print("-" * 50)
            try:
                subprocess.run([python_exe, "run_pipeline.py"])
            except KeyboardInterrupt:
                pass
            break
        else:
            print("\n[알림] 올바른 번호(0~5)를 입력해주세요.")
            input("계속하려면 Enter를 누르세요...")


if __name__ == "__main__":
    main()
