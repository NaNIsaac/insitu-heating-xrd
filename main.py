"""진입점.

python main.py                 ← 인자 없이 실행(VS Code ▶ 버튼 포함): experiments.toml 로 실험 비교 (compare)
python main.py <명령> [옵션]    ← overview, compare, overlay, image ... (자세히: python main.py -h)
"""

import sys

from insitu_xrd.cli import main

# 인자 없이 실행했을 때 할 일 (바꾸고 싶으면 여기만 수정)
DEFAULT_ARGS = ["compare"]  # 예: ["compare", "다른설정.toml"], ["overview", "IGOx1", "--mpl"]

if __name__ == "__main__":
    args = sys.argv[1:]
    # 명령 없이 옵션만 주면 (예: python main.py --no-show) 기본 명령에 붙임. -h 는 전체 도움말
    if not args or (args[0].startswith("-") and args[0] not in ("-h", "--help")):
        args = DEFAULT_ARGS + args
    main(args)
