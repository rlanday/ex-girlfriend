"""PoC phase 3: fp32 cooldown, the short form of Vladimir Ivanov's cooldown.py (fx2-cmix-transformer, GPL-3)

The export reference is the checkpoint this phase writes: pysrc/export_weights.py
quantizes from fp32 parameters, and his flex_attention/fp32 config is what the
C++ port was written against.

    uv run torchrun --nproc_per_node 1 -m tft.cooldown_poc
"""

from dotenv import load_dotenv

from tft.runner import run_phase


def main() -> None:
    run_phase("fp32")


if __name__ == "__main__":
    load_dotenv()
    main()
