"""Helper script that pins ``train.train`` to the Qwen3-Next architecture."""

import sys

from train.train import main


def main_qwen3next() -> None:
    if "--architecture" not in sys.argv:
        sys.argv.extend(["--architecture", "qwen3next"])
    main()


if __name__ == "__main__":
    main_qwen3next()
