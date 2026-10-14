"""PoC phase 1: bf16, no quantization; the short form of Vladimir Ivanov's train_6m.py (fx2-cmix-transformer, GPL-3)

    uv run torchrun --nproc_per_node 1 -m tft.train_poc
"""

from dotenv import load_dotenv

from tft.runner import run_phase


def main() -> None:
    run_phase("bf16")


if __name__ == "__main__":
    load_dotenv()
    main()
