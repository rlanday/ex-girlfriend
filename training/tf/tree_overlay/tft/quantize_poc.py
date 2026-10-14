"""PoC phase 2: quantization-aware training, the short form of Vladimir Ivanov's quantize.py (fx2-cmix-transformer, GPL-3)

    uv run torchrun --nproc_per_node 1 -m tft.quantize_poc
"""

from dotenv import load_dotenv

from tft.runner import run_phase


def main() -> None:
    run_phase("qat")


if __name__ == "__main__":
    load_dotenv()
    main()
