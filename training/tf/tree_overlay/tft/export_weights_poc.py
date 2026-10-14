"""208-capable weight export, the parametric twin of pysrc/export_weights.py

Derived from Vladimir Ivanov's pysrc/export_weights.py (fx2-cmix-transformer,
GPL-3).  All the arithmetic (the bit-exact replication of QuantizeFunction, the
bf16 scale round trip, the rope tables, the container layout) is his and is
IMPORTED, not copied.  What this file changes:

  1. the config comes from tft/config.py, so the vocabulary is 208 and the
     geometry is the PoC one instead of the hard coded 6m,
  2. his three magic tensor counts (111 / 119 / 88) become formulas in the
     layer count (see expected_counts()); the formulas reproduce his three
     numbers exactly for his 12 layer geometry, which is the regression test,
  3. his `assert config_ints.tolist() == [205, 192, ...]` becomes a comparison
     against the config that was actually built,
  4. the vocabulary bitmap travels with the weights (sidecar by default).

It also fixes the blocker that makes his own export unrunnable: his
make_reference_config() omits the required field `query_key_norm_gain` and
raises TypeError before exporting anything.

    uv run python -m tft.export_weights_poc            # needs CUDA
    TFT_GEOMETRY=poc4m uv run python -m tft.export_weights_poc
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch import linspace

from pysrc.export_weights import (
    DTYPE_BF16_BITS,
    DTYPE_FLOAT32,
    DTYPE_INT8,
    DTYPE_INT32,
    _ACTIVATION_SCALE_SUFFIXES,
    _UNQUANTIZED_FP32_SUFFIXES,
    bf16_bits,
    quantize_weight_rows,
    read_tensor_file,
    write_tensor_file,
)
from pysrc.model import Transformer, make_rope_args

from tft import config as tftcfg
from tft.param_count import Geometry


def expected_counts(g: Geometry, paket: dict | None = None) -> dict:
    """how many tensors of each kind his export loop must produce

    Derived from which modules carry a weight quantizer in his quantized
    config: the kimi convolutions and the beta projection are NOT quantized
    (kimi_linear_convolution_quantization / _beta_projection_quantization are
    None), so they land in the raw fp32 bucket.

    Regression check (his 12 layer geometry, 9 kimi + 3 attention):
        quantized  = 3 + 2*12 + 4*3 + 8*9 = 111   (his assert)
        raw fp32   = 7*9 + 2*12 + 1      =  88    (his assert)
        activation = (111 - 1) + 3*3     = 119    (his assert)

    A model with the context package (zctx/zpmodel.py, ZPaketEmbedding) adds
    up to four quantized weights (`last_table`, `last_projection`,
    `match_table`, `match_projection`) and one activation scale per
    PROJECTION (the two tables are CastedEmbeddings and quantize no
    activation).  `last_offsets` is registered with persistent=False and is
    therefore not in the state_dict.  Nothing lands in the raw fp32 bucket.
    With `paket=None` the numbers are exactly his, which is the regression.
    """
    n_layers = g.n_layers
    n_kimi = sum(1 for k in g.kimi_linear if k)
    n_attn = n_layers - n_kimi
    n_quantized = 3 + 2 * n_layers + 4 * n_attn + 8 * n_kimi
    n_raw_fp32 = 7 * n_kimi + 2 * n_layers + 1
    n_activation_scales = (n_quantized - 1) + 3 * n_attn
    if paket is not None:
        extra = int(bool(paket["use_last"])) + int(bool(paket["use_match"]))
        n_quantized += 2 * extra          # table + projection per channel
        n_activation_scales += extra      # only the projections
    return {
        "n_quantized": n_quantized,
        "n_activation_scales": n_activation_scales,
        "n_raw_fp32": n_raw_fp32,
    }


def paket_from_env() -> dict | None:
    """the context-package switches of the training run, read as tft/runner.py
    reads them.  Returns None for a plain model.

        ZCTX_ARM=pk PK_LAST=1 PK_MATCH=1 PK_LAST_X=2 PK_SLOT_DIM=16
    """
    if os.environ.get("ZCTX_ARM", "none") not in ("pk", "pkd"):
        return None
    # A "pk" model with both channels off still carries the V+8 embedding rows
    # (install_paket widens the table unconditionally), so the arm stays a
    # context-package arm and the container form must say so.
    use_last = os.environ.get("PK_LAST", "1") != "0"
    use_match = os.environ.get("PK_MATCH", "1") != "0"
    from zctx import zpaket

    return {
        "use_last": use_last,
        "use_match": use_match,
        "x": int(os.environ.get("PK_LAST_X", str(zpaket.DEFAULT_X))),
        "d_slot": int(os.environ.get("PK_SLOT_DIM", "16")),
        "n_sentinel": int(zpaket.N_SENTINEL),
    }


def _self_test() -> None:
    """the formulas must reproduce his three shipped asserts"""
    from tft.param_count import GEOMETRIES

    got = expected_counts(GEOMETRIES["6m"])
    assert got == {
        "n_quantized": 111,
        "n_activation_scales": 119,
        "n_raw_fp32": 88,
    }, got
    # The context package adds four quantized weights and two activation
    # scales (the two tables are CastedEmbeddings and quantize no activation),
    # nothing raw fp32; the plain numbers above must stay untouched (the
    # byte-identity regression of the poc4m container).
    full = {"use_last": True, "use_match": True, "x": 2, "d_slot": 16,
            "n_sentinel": 8}
    assert expected_counts(GEOMETRIES["6m"], full) == {
        "n_quantized": 115,
        "n_activation_scales": 121,
        "n_raw_fp32": 88,
    }
    assert expected_counts(GEOMETRIES["6m"], dict(full, use_match=False)) == {
        "n_quantized": 113,
        "n_activation_scales": 120,
        "n_raw_fp32": 88,
    }


_self_test()


def load_model(device: torch.device, geometry_name: str | None = None) -> Transformer:
    g = tftcfg.geometry(geometry_name)
    cfg = tftcfg.make_config("fp32", geometry_name=g.name, variable_sequence_length=False)
    model = Transformer(cfg).to(device)
    # A context-package checkpoint carries the ZPaketEmbedding tensors, and
    # `install_paket` has to run BEFORE load_state_dict, as zctx/zpmodel.py
    # says and as the trainer does it (tft/runner.py).  Without it
    # `load_state_dict(strict=True)` fails on the extra keys and on the V+8
    # embedding shape.
    paket = paket_from_env()
    if paket is not None:
        from zctx.zpmodel import install_paket

        added = install_paket(
            model, cfg,
            x=paket["x"] if paket["use_last"] else 0,
            n_sentinel=paket["n_sentinel"],
            d_slot=paket["d_slot"],
            use_last=paket["use_last"], use_match=paket["use_match"],
        )
        print("context package installed for export: X=%d last=%s match=%s "
              "sentinel=%d, +%d parameters"
              % (paket["x"], paket["use_last"], paket["use_match"],
                 paket["n_sentinel"], added), flush=True)
    path = tftcfg.model_path("fp32", g.name)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print("loaded " + path, flush=True)
    return model


@torch.inference_mode()
def export_weights(
    model: Transformer, output_filename: str, geometry_name: str | None = None,
    paket: dict | None = None,
) -> dict:
    g = tftcfg.geometry(geometry_name)
    cfg = model.cfg
    state_dict = model.state_dict()
    modules = dict(model.named_modules())
    if paket is None:
        paket = paket_from_env()

    entries: list = []
    quantized_weight_keys = {
        key
        for key in state_dict
        if key.endswith(".weight")
        and key[: -len(".weight")] + ".quantize_weight.scale" in state_dict
    }

    n_quantized = n_activation_scales = n_raw_fp32 = 0
    negative_scale_weights: list = []

    for key, value in state_dict.items():
        value = value.detach()
        if key in quantized_weight_keys:
            prefix = key[: -len(".weight")]
            scale_fp32 = state_dict[prefix + ".quantize_weight.scale"].detach()
            ints, scale_used = quantize_weight_rows(value, scale_fp32)

            quantize_module = modules[prefix].quantize_weight
            ints_reference = quantize_module(
                value, initialize_scales_steps=None, return_ints=True
            )
            dequantized_reference = quantize_module(
                value, initialize_scales_steps=None, return_ints=False
            )
            assert torch.equal(ints_reference, ints.to(torch.int64)), (
                "int mismatch for " + key
            )
            assert torch.equal(
                dequantized_reference,
                ints.to(torch.float32) * scale_used.unsqueeze(-1),
            ), "dequantized mismatch for " + key
            assert int(ints.abs().max().item()) <= 7

            n_negative = int((scale_used < 0).sum().item())
            if n_negative:
                negative_scale_weights.append((key, n_negative))

            entries.append((key + ".q", DTYPE_INT8, ints.cpu().numpy()))
            entries.append((key + ".scale", DTYPE_BF16_BITS, bf16_bits(scale_fp32)))
            n_quantized += 1
        elif key.endswith(".quantize_weight.scale"):
            assert (
                key[: -len(".quantize_weight.scale")] + ".weight"
                in quantized_weight_keys
            )
        elif key.endswith(_ACTIVATION_SCALE_SUFFIXES):
            scale_used = value.to(torch.bfloat16).to(torch.float32)
            assert bool((scale_used > 0).all()), "activation scale not positive: " + key
            entries.append((key, DTYPE_BF16_BITS, bf16_bits(value)))
            n_activation_scales += 1
        else:
            assert key.endswith(_UNQUANTIZED_FP32_SUFFIXES), key
            assert value.dtype == torch.float32
            entries.append((key, DTYPE_FLOAT32, value.cpu().numpy()))
            n_raw_fp32 += 1

    # rope tables, computed exactly as pysrc.model.make_rope_args
    assert cfg.rope_base is not None and not cfg.half_truncate_rope
    device = next(model.parameters()).device
    rope_length = tftcfg.MAX_ARTICLE_TOKENS
    rope_args = make_rope_args(sequence_length=rope_length, cfg=cfg, device=device)
    assert rope_args is not None
    assert rope_args.sines.shape == (rope_length, cfg.d_head // 2)

    inv_freq = 1.0 / (
        cfg.rope_base
        ** linspace(0, 1, steps=cfg.d_head // 2, dtype=torch.float32, device=device)
    )
    t = torch.arange(rope_length, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    assert torch.equal(freqs.sin(), rope_args.sines)
    assert torch.equal(freqs.cos(), rope_args.cosines)

    entries.append(("rope.inv_freq", DTYPE_FLOAT32, inv_freq.cpu().numpy()))
    entries.append(("rope.sin", DTYPE_FLOAT32, rope_args.sines.cpu().numpy()))
    entries.append(("rope.cos", DTYPE_FLOAT32, rope_args.cosines.cpu().numpy()))

    config_ints = np.array(
        [
            cfg.vocabulary_size,
            cfg.d_model,
            cfg.n_layers,
            cfg.d_head,
            cfg.n_query_heads,
            cfg.d_mlp,
            cfg.window_size,
            cfg.kimi_linear_d_head,
            cfg.kimi_linear_n_heads,
            cfg.kimi_linear_convolution_size,
            cfg.rope_base,
        ],
        dtype=np.int32,
    )
    # his hard coded [205, 192, 12, 64, 3, 768, 1024, 64, 3, 4, 10000] becomes
    # a comparison against the geometry that was actually built
    assert config_ints.tolist() == [
        g.vocabulary_size,
        g.d_model,
        g.n_layers,
        g.d_head,
        g.n_query_heads,
        g.d_mlp,
        g.window_size,
        g.kimi_linear_d_head,
        g.kimi_linear_n_heads,
        g.kimi_linear_convolution_size,
        g.rope_base,
    ], config_ints.tolist()
    entries.append(("config.ints", DTYPE_INT32, config_ints))
    entries.append(
        (
            "config.kimi",
            DTYPE_INT32,
            np.array([int(k) for k in cfg.kimi_linear], dtype=np.int32),
        )
    )

    expected = expected_counts(g, paket)
    got = {
        "n_quantized": n_quantized,
        "n_activation_scales": n_activation_scales,
        "n_raw_fp32": n_raw_fp32,
    }
    assert got == expected, "tensor counts " + str(got) + " != expected " + str(expected)
    assert len(entries) == 2 * n_quantized + n_activation_scales + n_raw_fp32 + 5

    # The coder's loader decides the FORM from the
    # tensor list (src/tf1/opt/arena_build.cpp), so the export states the same
    # thing explicitly instead of letting it be a side effect of the geometry.
    by_name = {name: array for name, _dtype, array in entries}
    emb_rows = by_name["embedding.weight.q"].shape[0]
    paket_names = [
        "embedding.last_table.weight.q",
        "embedding.last_projection.weight.q",
        "embedding.match_table.weight.q",
        "embedding.match_projection.weight.q",
    ]
    present = [n for n in paket_names if n in by_name]
    if paket is None:
        assert not present, (
            "a plain export must not carry context package tensors, found " + str(present)
        )
        assert emb_rows == cfg.vocabulary_size, emb_rows
    else:
        want_rows = cfg.vocabulary_size + paket["n_sentinel"]
        assert emb_rows == want_rows, (
            "embedding has " + str(emb_rows) + " rows, expected " + str(want_rows)
        )
        d_slot, X = paket["d_slot"], paket["x"]
        if paket["use_last"]:
            from zctx import zpaket as _zpl
            _last_rows = _zpl.last_rows(cfg.vocabulary_size)
            assert by_name["embedding.last_table.weight.q"].shape == (
                X * _last_rows, d_slot), (
                by_name["embedding.last_table.weight.q"].shape,
                "expected %d rows per slot (X0ROW=%s)"
                % (_last_rows, _zpl.X0ROW))
            assert by_name["embedding.last_projection.weight.q"].shape == (
                cfg.d_model, X * d_slot)
        if paket["use_match"]:
            from zctx import zpaket as _zp

            assert by_name["embedding.match_table.weight.q"].shape == (
                _zp.N_MATCH_ROWS + _zp.N_LEN_CLASSES, d_slot)
            assert by_name["embedding.match_projection.weight.q"].shape == (
                cfg.d_model, 2 * d_slot)
            assert _zp.N_MATCH_ROWS == cfg.vocabulary_size + 1, (
                "the match table must carry V token rows plus the marker; the "
                "coder derives the size of its match table the same way"
            )

    write_tensor_file(output_filename, entries)

    read_back = read_tensor_file(output_filename)
    assert len(read_back) == len(entries)
    for name, dtype_code, array in entries:
        read_dtype_code, read_array = read_back[name]
        assert read_dtype_code == dtype_code, name
        assert read_array.shape == array.shape, name
        assert np.array_equal(read_array, array), name

    result = dict(
        got,
        geometry=g.name,
        vocabulary_size=cfg.vocabulary_size,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        window_size=cfg.window_size,
        rope_sequence_length=rope_length,
        tensors=len(entries),
        bytes=os.path.getsize(output_filename),
        negative_row_scales=sum(n for _, n in negative_scale_weights),
        # The CHANNEL FLAG the coder's loader checks against
        embedding_rows=int(emb_rows),
        paket=(None if paket is None else dict(
            paket, tensors=sorted(
                n for n in by_name
                if n.startswith("embedding.last_") or n.startswith("embedding.match_")
            ),
        )),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result



# --- the manifest fields -------------------------------------------------

def _p11_geom():
    from tft import config as _c
    return _c.geometry(os.environ.get("TFT_GEOMETRY"))


def _p11_meta(feld, fallback):
    """build state from `dump-meta.json` of the package.

    The environment is the wrong source: in dump mode `PK_REDIRMERGE` controls
    nothing (the feature is in the dump), and the manifest has to state what the
    package carries.
    """
    d = os.environ.get("PK_DUMP_DIR")
    if d:
        import json as _j
        try:
            with open(os.path.join(d, "dump-meta.json"), encoding="utf-8") as f:
                return _j.load(f).get(feld, fallback)
        except OSError:
            pass
    return fallback


def _p11_paket() -> str:
    return os.environ.get("PK_PAKET_NAME") or os.path.basename(
        (os.environ.get("PK_DUMP_DIR") or os.environ.get("PK_DIR") or "").rstrip("/"))


def _p11_arme() -> list:
    """the arm list of the package, the training form recorded in the manifest.

    It must state what the training really fed: an arm added to the package
    without an entry here reopens the gap this field exists to close.
    """
    dumpdir = os.environ.get("PK_DUMP_DIR")
    if dumpdir:
        import json as _j
        with open(os.path.join(dumpdir, "dump-meta.json"), encoding="utf-8") as f:
            return list(_j.load(f).get("arme", []))
    an = lambda n, d="1": os.environ.get(n, d) != "0"
    arme = []
    for name, var, d in (("paket", "PK_PAKET", "1"), ("match", "PK_MATCH", "1"),
                         ("z3", "PK_Z3", "1"), ("num", "PK_NUM", "0"),
                         ("z1t", "PK_Z1T", "1"), ("syn", "PK_SYN", "1"),
                         ("alias", "PK_ALIAS", "0"), ("a2", "PK_A2", "1"),
                         ("a2n", "PK_A2N", "1"), ("sent", "PK_SENT", "1"),
                         ("sent_feed", "PK_SENT_FEED", "1"),
                         ("titelmask", "PK_TITLEMASK", "1"),
                         ("x0row", "PK_LAST_X0ROW", "0")):
        if an(var, d):
            arme.append(name)
    return arme


def _p11_prior0_sha():
    """sha256 of the prior0 table of the package"""
    import hashlib
    weg = os.environ.get("PK_PRIOR0")
    if not weg:
        d = os.environ.get("PK_DUMP_DIR") or os.environ.get("PK_DIR")
        weg = os.path.join(d, "z3-prior0.float16") if d else None
    if not weg or not os.path.exists(weg):
        return None
    h = hashlib.sha256()
    with open(weg, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_sidecar(output_filename: str, result: dict) -> None:
    """geometry + vocabulary bitmap next to the container, for the coder side

    The bitmap is what turns a vocabulary INDEX back into a byte value; the
    weights are meaningless without it.  Kept as a sidecar rather than a tensor
    inside the container so that his C++ reader sees exactly the tensor set it
    expects.
    """
    bitmap_path = tftcfg.data_paths()["vocab_bitmap_filename"]
    payload = dict(result)
    # These fields tie the weights to their training form (package, arms,
    # prior0, window, redirect merge, X0ROW); the measurement builds of the
    # development tree check them when they load a container.
    payload["paket"] = payload.get("paket") or _p11_paket()
    payload["arme"] = _p11_arme()
    payload["prior0_sha"] = _p11_prior0_sha()
    payload["fenster"] = int(getattr(_p11_geom(), "window_size", 0))
    payload["redirmerge"] = _p11_meta("redirmerge",
                                      os.environ.get("PK_REDIRMERGE", "0") != "0")
    payload["x0row"] = _p11_meta("x0row",
                                 os.environ.get("PK_LAST_X0ROW", "0") != "0")
    if os.path.isfile(bitmap_path):
        raw = np.fromfile(bitmap_path, dtype=np.uint8)
        assert len(raw) == 32, "vocab bitmap must be 32 bytes"
        popcount = int(np.unpackbits(raw, bitorder="little").sum())
        assert popcount == result["vocabulary_size"], (
            "bitmap popcount "
            + str(popcount)
            + " != vocabulary "
            + str(result["vocabulary_size"])
        )
        payload["vocab_bitmap_hex"] = raw.tobytes().hex()
        payload["vocab_byte_values"] = np.flatnonzero(
            np.unpackbits(raw, bitorder="little")
        ).tolist()
        raw.tofile(output_filename + ".vocab-bitmap.u8")
    else:
        payload["vocab_bitmap_hex"] = None
        print("WARNING: no " + bitmap_path + "; container ships without a bitmap")
    with open(output_filename + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote " + output_filename + ".json", flush=True)


def compress_container(bin_path: str) -> str:
    """the .c2 the coder's loader prefers, written by his own encoder, which
    re-decodes and verifies bit-exactness before it returns
    (pysrc/weights_compress)"""
    from pysrc import weights_compress

    out = os.path.splitext(bin_path)[0] + ".c2"
    argv = sys.argv
    sys.argv = ["weights_compress", "compress", bin_path, out]
    try:
        weights_compress.main()
    finally:
        sys.argv = argv
    print("wrote " + out + " (" + str(os.path.getsize(out)) + " bytes)", flush=True)
    return out


def main() -> None:
    assert torch.cuda.is_available(), "export quantizes on CUDA, exactly as his does"
    g = tftcfg.geometry()
    out_dir = os.environ.get("TFT_EXPORT_DIR", "cpp_infer/data")
    os.makedirs(out_dir, exist_ok=True)
    output_filename = out_dir + "/weights-" + g.name + ".bin"
    model = load_model(torch.device("cuda"), g.name)
    paket = paket_from_env()
    result = export_weights(model, output_filename, g.name, paket)
    if os.environ.get("TFT_EXPORT_C2", "1") != "0":
        result["container_c2"] = os.path.basename(compress_container(output_filename))
    write_sidecar(output_filename, result)


if __name__ == "__main__":
    main()
