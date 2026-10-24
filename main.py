"""Entry point.

python main.py                  no arguments (also the VS Code run button): compare the experiments in experiments.toml
python main.py <command> [options]  overview, compare, overlay, image ... (see python main.py -h)
"""

import sys

from insitu_xrd.cli import main

DEFAULT_ARGS = ["compare"]  # e.g. ["compare", "other.toml"], ["overview", "IGOx1", "--mpl"]

if __name__ == "__main__":
    args = sys.argv[1:]
    # options without a command (python main.py --no-show) go to the default command; -h stays global help
    if not args or (args[0].startswith("-") and args[0] not in ("-h", "--help")):
        args = DEFAULT_ARGS + args
    main(args)
