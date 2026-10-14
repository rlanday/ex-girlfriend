#!/usr/bin/env python3
"""zone map v2: where the transformer helps, where it hurts, and where it may
be switched off at all

It needs no new run: the probability rows of both sides are available from
the dump run (dump identity checked: 0 of 2,000 sampled rows differ).

Compared with v1 (`zonemap.py`, six zones) this version does three things:

  1. FINER CLASSES along the block structure of the coder stream: the XML
     header is split into comment and rest, lists are separated from tables,
     templates and headings get rows of their own, digit runs are split by
     length (the length carries the signal: runs of three and four digits,
     years, are the only ones where the transformer gains), and words
     (dictionary code words >= 0x80) are separated from punctuation.
  2. For EVERY zone the CAUSAL RECOGNISABILITY of its boundaries, because it
     decides whether a gate can be built at all: a decoder knows only the
     past.  Three levels:
        oeffner   a markup opener switches the zone on, a closer switches it
                  off; fully causal from the opener on (links, templates,
                  table rows, title, XML fields)
        praefix   recognisable only from the k-th character of the zone on;
                  the first k-1 run ungated (digit runs)
        nein      determinable only in hindsight (run length, repetition of a
                  line not yet read); no gate possible, at most a model input
  3. A DERIVED recommendation instead of a hand-written opinion: the decision
     rule is in `empfehlung()` ("recommendation") and can be followed on the
     numbers of each row.

The script is a tool for repeated use: it reads a dump and a probability
file, writes a TSV and a readable table, and can be rerun after every
training.  Optionally the ZONE SHARE is also counted on a second corpus (the
training package); that answers how representative the excerpt is (on the
first run: digits 11.5 % on the excerpt against 3.2 % on the package).

    python -m zctx.zonemap2 <dumpdir> <tf_probs.float16> [pkgdir] [outdir]
"""

from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Classes.  Order = priority: the FIRST matching rule wins.
# Each row: (name, causality level, remark)
# ---------------------------------------------------------------------------
ZONES = [
    ("TITEL",       "oeffner", "article separator .. end of title"),
    ("XML_COMMENT", "oeffner", "comment field in the header"),
    ("XML_REST",    "oeffner", "rest of the XML header (id, revision, timestamp)"),
    ("REPETITIV",   "praefix", "line shares a >=8 token prefix with the previous line "
                               "(deletion log, bot lists, table body)"),
    ("UEBERSCHR",   "oeffner", "line starts with ="),
    ("VORLAGE",     "oeffner", "{{ .. }}"),
    ("LINK",        "oeffner", "[[ .. ]]"),
    ("TABELLE",     "oeffner", "line starts with | or !, or {| .. |}"),
    ("LISTE",       "oeffner", "line starts with * # ; :"),
    ("ZIFF_1_2",    "praefix", "digit run of length 1-2"),
    ("ZIFF_3_4",    "praefix", "digit run of length 3-4 (years)"),
    ("ZIFF_5_6",    "praefix", "digit run of length 5-6"),
    ("ZIFF_7PLUS",  "praefix", "digit run of length >= 7 (coordinates, numbers)"),
    ("WORT",        "oeffner", "dictionary/WRT code word >= 0x80 in running text"),
    ("TEXT_REST",   "oeffner", "rest of the running text (punctuation, white space)"),
]
NAMES = [z[0] for z in ZONES]
IDX = {n: i for i, n in enumerate(NAMES)}
KAUSAL = {n: k for n, k, _ in ZONES}
BEMERKUNG = {n: b for n, _, b in ZONES}

LINE_TABLE = tuple(b"|!")
LINE_LIST = tuple(b"*#;:")
DIGITS = tuple(range(0x30, 0x3A))


def load_bitmap(directory: str) -> np.ndarray:
    bm = np.fromfile(os.path.join(directory, "vocab-bitmap.u8"), dtype=np.uint8)
    return np.array([b * 8 + k for b in range(32) for k in range(8) if bm[b] >> k & 1],
                    dtype=np.uint8)


def find_all(tokens: np.ndarray, pattern: list) -> np.ndarray:
    n, m = len(tokens), len(pattern)
    if n < m:
        return np.zeros(0, dtype=np.int64)
    hit = tokens[: n - m + 1] == pattern[0]
    for k in range(1, m):
        hit &= tokens[k : n - m + 1 + k] == pattern[k]
    return np.flatnonzero(hit)


def classify(tokens: np.ndarray, boundaries: np.ndarray, bitmap: np.ndarray,
             text_marker: list, comment_marker: list) -> np.ndarray:
    """(n,) uint8 zone index; a pure function of the token sequence"""
    from zctx.zonemap import TITLE_TERMINATOR

    n = len(tokens)
    b = bitmap[tokens]
    zone = np.full(n, IDX["TEXT_REST"], dtype=np.uint8)

    # --- words: dictionary/WRT code words ---------------------------------
    zone[b >= 0x80] = IDX["WORT"]

    # --- digit runs, by length --------------------------------------------
    isdig = np.isin(b, np.array(DIGITS, dtype=np.uint8))
    d = np.diff(np.concatenate([[0], isdig.view(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lengths = ends - starts
    for lo, hi, name in ((1, 2, "ZIFF_1_2"), (3, 4, "ZIFF_3_4"),
                         (5, 6, "ZIFF_5_6"), (7, 10**9, "ZIFF_7PLUS")):
        sel = (lengths >= lo) & (lengths <= hi)
        for a, e in zip(starts[sel], ends[sel]):
            zone[a:e] = IDX[name]

    # --- per-line marks (table, list, heading) ----------------------------
    idx = np.arange(n, dtype=np.int64)
    last_nl = np.maximum.accumulate(np.where(b == 0x0A, idx, -1))
    nonspace = b != 0x20
    cs = np.cumsum(nonspace, dtype=np.int64)
    base = np.where(last_nl >= 0, cs[np.maximum(last_nl, 0)], 0)
    first_ns = nonspace & ((cs - base) == 1)
    for marks, name in ((LINE_TABLE, "TABELLE"), (LINE_LIST, "LISTE"),
                        (tuple(b"="), "UEBERSCHR")):
        hit = first_ns & np.isin(b, np.array(marks, dtype=np.uint8))
        last_hit = np.maximum.accumulate(np.where(hit, idx, -1))
        zone[last_hit > last_nl] = IDX[name]

    # --- REPETITIVE lines (deletion logs, bot lists) ----------------------
    # A deletion log has no mark in the coder stream; what marks such pages
    # is repetition: every line starts like the previous one.  That is
    # causally checkable (the previous line has been read) and catches
    # deletion logs, bot lists and table bodies without guessing them.
    nl_pos = np.flatnonzero(b == 0x0A)
    line_start = np.concatenate([[0], nl_pos + 1])
    line_end = np.concatenate([nl_pos, [n]])
    PREFIX_MIN = 8
    for i in range(1, len(line_start)):
        s0, e0 = int(line_start[i - 1]), int(line_end[i - 1])
        s1, e1 = int(line_start[i]), int(line_end[i])
        m = min(e0 - s0, e1 - s1, 64)
        if m < PREFIX_MIN:
            continue
        eq = tokens[s0 : s0 + m] == tokens[s1 : s1 + m]
        k = int(np.argmin(eq)) if not eq.all() else m
        if k >= PREFIX_MIN:
            zone[s1:e1] = IDX["REPETITIV"]

    # --- bracketed regions (template, link) -------------------------------
    for op1, op2, cl1, cl2, name in ((0x7B, 0x7B, 0x7D, 0x7D, "VORLAGE"),
                                     (0x5B, 0x5B, 0x5D, 0x5D, "LINK")):
        o = np.zeros(n, dtype=np.int32); c = np.zeros(n, dtype=np.int32)
        if n > 1:
            o[1:] = ((b[:-1] == op1) & (b[1:] == op2)).astype(np.int32)
            c[1:] = ((b[:-1] == cl1) & (b[1:] == cl2)).astype(np.int32)
        # counter with a floor of 0 (see zgate.exclusion_mask): a plain prefix
        # sum breaks at the first unbalanced closer.
        s = np.cumsum(o) - np.cumsum(c)
        zone[(s - np.minimum.accumulate(np.minimum(s, 0))) > 0] = IDX[name]

    # --- header per article: title, comment, rest of the XML --------------
    tm = list(text_marker) if text_marker else []
    cm = list(comment_marker) if comment_marker else []
    for i in range(len(boundaries) - 1):
        a, e = int(boundaries[i]), int(boundaries[i + 1])
        t = find_all(tokens[a : min(a + 256, e)], TITLE_TERMINATOR)
        title_end = a + int(t[0]) + len(TITLE_TERMINATOR) if len(t) else a
        zone[a:title_end] = IDX["TITEL"]
        head_end = title_end
        if tm:
            m = find_all(tokens[title_end : min(title_end + 2000, e)], tm)
            if len(m):
                head_end = title_end + int(m[0]) + len(tm)
        if head_end > title_end:
            zone[title_end:head_end] = IDX["XML_REST"]
            if cm:
                c = find_all(tokens[title_end:head_end], cm)
                if len(c):
                    zone[title_end + int(c[0]) : head_end] = IDX["XML_COMMENT"]
    return zone


def derive_comment_marker(tokens: np.ndarray, boundaries: np.ndarray,
                          bitmap: np.ndarray, width: int = 6) -> list:
    """the token sequence that opens the comment field in the header

    Derived like the text marker in v1: the sequence that occurs exactly once
    in the second half of the header in the most articles.  If none is found,
    the class XML_COMMENT is dropped (and the map says so instead of
    inventing a marker).
    """
    from zctx.zonemap import TITLE_TERMINATOR

    counts: dict = {}
    tried = 0
    for i in range(min(300, len(boundaries) - 1)):
        a, e = int(boundaries[i]), int(boundaries[i + 1])
        t = find_all(tokens[a : min(a + 256, e)], TITLE_TERMINATOR)
        if not len(t):
            continue
        s = a + int(t[0]) + len(TITLE_TERMINATOR)
        window = tokens[s + 200 : s + 900]
        tried += 1
        seen = set()
        for k in range(0, max(0, len(window) - width)):
            key = tuple(int(v) for v in window[k : k + width])
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    best, hits = max(counts.items(), key=lambda kv: kv[1])
    return list(best) if hits >= 0.6 * max(1, tried) else []


# Exchange rate of the program: what 1 % of time is worth in rate (then
# 15-38 KB per 1 %).  The threshold for "superfluous" is the CONSERVATIVE
# end: dropping a zone that carries more than 15 KB per 1 % of time sells
# too cheaply.
KURS_BYTE_JE_PROZENT = float(os.environ.get("ZKARTE_KURS", "15000"))
# length of the coder stream of enwik9 (positions), for the extrapolation
N_ENWIK9 = float(os.environ.get("ZKARTE_N", "587000000"))


def empfehlung(delta_pro_token: float, anteil_zeit: float, kausal: str,
               bytes_je_prozent: float) -> str:
    """MARGINAL GAIN PER TIME COST, three categories

    The radical form of gating removes whole regions from the stream, in
    TRAINING as well.  The question is then not only "does the transformer
    hurt here?" but "does it gain enough here to earn its run time?".
    Three categories (the fourth text, lower-case "in", marks a zone too small
    to matter for time):

      OUT (hurts)          the transformer is worse than the prior.  No
                           trade-off needed; that is rate AND time.
      OUT (superfluous)    it gains, but less than the program's exchange
                           rate gives for the same share of time; the time
                           is worth more elsewhere, and so is the training
                           budget.
      IN (gains)           it earns its time.

    `bytes_je_prozent` is the gain of the zone, extrapolated to enwik9,
    divided by its share of time in percent: the rate at which time items
    are priced.
    """
    if delta_pro_token > 0.0:
        return "OUT (hurts)"
    if anteil_zeit < 0.5:
        return "in (too small for time)"
    if abs(bytes_je_prozent) < KURS_BYTE_JE_PROZENT:
        return "OUT (superfluous)"
    return "IN (gains)"


def artikelklassen(tokens: np.ndarray, boundaries: np.ndarray, zone: np.ndarray,
                   width: int = 8) -> tuple:
    """(class_per_article, NAMES): the COARSE level, whole articles

    The radical form removes not zones but SEGMENTS: whole census place
    articles, deletion logs, XML regions.  At this level the context gap does
    not matter (the coder resets at the article boundary anyway), and the
    saving is run time AND training budget at once.

    Three classes, all derived from the stream instead of guessed:
      SERIE    ("series") a high share of the 8-token windows of the article
               has already occurred in EARLIER articles; the signature of
               the census place articles (series articles from the same
               template).  The test uses only the past, so it can be
               evaluated causally once the article has been read.
      LOGARTIG ("log-like") more than half of the article lies in the zone
               REPETITIV (deletion logs, bot lists).
      NORMAL   the rest.
    """
    names = ("NORMAL", "SERIE", "LOGARTIG")
    n_art = len(boundaries) - 1
    kl = np.zeros(n_art, dtype=np.uint8)
    seen: set = set()
    rep = IDX["REPETITIV"]
    for a in range(n_art):
        s0, e0 = int(boundaries[a]), int(boundaries[a + 1])
        if e0 - s0 < width + 2:
            continue
        # sample of the windows: every 4th is enough for a share
        idxs = range(s0, e0 - width, 4)
        hit = 0
        total = 0
        neu_keys = []
        for i in idxs:
            key = tokens[i : i + width].tobytes()
            total += 1
            if key in seen:
                hit += 1
            else:
                neu_keys.append(key)
        seen.update(neu_keys)
        if total and hit / total >= 0.75:
            kl[a] = 1
        elif float((zone[s0:e0] == rep).mean()) > 0.5:
            kl[a] = 2
    return kl, names


def nats(probs_path: str, tokens: np.ndarray, v: int, cache: str = "",
         chunk: int = 200_000) -> np.ndarray:
    if cache and os.path.exists(cache):
        return np.load(cache)
    from zctx.zonemap import nats as _n

    out = _n(probs_path, tokens, v)
    if cache:
        np.save(cache, out)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    dumpdir, tf_probs = sys.argv[1], sys.argv[2]
    pkgdir = sys.argv[3] if len(sys.argv) > 3 else ""
    outdir = sys.argv[4] if len(sys.argv) > 4 else "."
    os.makedirs(outdir, exist_ok=True)
    cdir = os.environ.get("ZGATE_CACHE", outdir)

    from zctx.zonemap import derive_text_marker

    tokens = np.fromfile(os.path.join(dumpdir, "ppmd-bytes.uint8"),
                         dtype=np.uint8).astype(np.int64)
    bnd = np.fromfile(os.path.join(dumpdir, "article-boundaries.int32"), dtype=np.int32)
    bitmap = load_bitmap(dumpdir)
    v = len(bitmap)

    tmark, thits, ttried = derive_text_marker(tokens, bnd)
    cmark = derive_comment_marker(tokens, bnd, bitmap)
    zone = classify(tokens, bnd, bitmap, tmark, cmark)

    tf = nats(tf_probs, tokens, v, os.path.join(cdir, "_tf_nats.npy"))
    pp = nats(os.path.join(dumpdir, "ppmd-probs.float16"), tokens, v,
              os.path.join(cdir, "_pp_nats.npy"))
    n = min(len(tf), len(pp))
    tf, pp, z = tf[:n], pp[:n], zone[1:][:n]
    delta = tf - pp

    # zone shares on the second corpus (training package); shares only,
    # there are no transformer rows there.
    pkg_share = {}
    if pkgdir:
        pt = np.fromfile(os.path.join(pkgdir, "ppmd-bytes.uint8"),
                         dtype=np.uint8).astype(np.int64)
        pb = np.fromfile(os.path.join(pkgdir, "article-boundaries.int32"), dtype=np.int32)
        pbm = load_bitmap(pkgdir)
        ptm, _, _ = derive_text_marker(pt, pb)
        pcm = derive_comment_marker(pt, pb, pbm)
        pz = classify(pt, pb, pbm, ptm, pcm)
        for i, name in enumerate(NAMES):
            pkg_share[name] = 100.0 * float((pz == i).mean())

    rows = []
    for i, name in enumerate(NAMES):
        sel = z == i
        k = int(sel.sum())
        if not k:
            continue
        rows.append({
            "zone": name, "token": k, "anteil": 100.0 * k / n,
            "anteil_pkg": pkg_share.get(name, float("nan")),
            "ppmd": float(pp[sel].mean()), "tf": float(tf[sel].mean()),
            "delta_tok": float(delta[sel].mean()), "delta_sum": float(delta[sel].sum()),
            "kausal": KAUSAL[name],
        })
    for r in rows:
        # gain of the zone on enwik9 (delta per token x share in the training stream)
        anteil_zeit = r["anteil_pkg"] if r["anteil_pkg"] == r["anteil_pkg"] else r["anteil"]
        r["anteil_zeit"] = anteil_zeit
        r["bytes_enwik9"] = (r["delta_tok"] * anteil_zeit / 100.0 * N_ENWIK9
                             / np.log(2) / 8)
        r["byte_je_proz"] = (r["bytes_enwik9"] / anteil_zeit) if anteil_zeit > 0 else 0.0
        r["empfehlung"] = empfehlung(r["delta_tok"], anteil_zeit, r["kausal"],
                                     r["byte_je_proz"])

    tsv = os.path.join(outdir, "karte_v2.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("zone\ttoken\tanteil_bett_pct\tanteil_paket_pct\tppmd_nats\ttf_nats\t"
                "delta_nats_pro_token\tdelta_nats_summe\tdelta_bytes\tkausal\t"
                "empfehlung\tbemerkung\n")
        for r in rows:
            f.write("%s\t%d\t%.4f\t%.4f\t%.6f\t%.6f\t%+.6f\t%+.1f\t%+.1f\t%s\t%s\t%s\n"
                    % (r["zone"], r["token"], r["anteil"], r["anteil_pkg"], r["ppmd"],
                       r["tf"], r["delta_tok"], r["delta_sum"],
                       r["delta_sum"] / np.log(2) / 8, r["kausal"], r["empfehlung"],
                       BEMERKUNG[r["zone"]]))

    print("ZONE MAP v2   excerpt %s   %d scored positions   V=%d"
          % (dumpdir, n, v))
    print("text marker in %d of %d articles, comment marker %s"
          % (thits, ttried, "derived" if cmark else "NOT found (class dropped)"))
    print("TF total %.6f   PPMD total %.6f   contribution %+.6f nats/token"
          % (tf.mean(), pp.mean(), delta.mean()))
    print()
    head = ("%-12s %7s %7s %9s %10s %13s %12s  %-8s %s"
            % ("zone", "excpt%", "trng%", "d/token", "dB excpt", "dB enwik9",
               "B/1% time", "causal", "recommendation"))
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda r: r["bytes_enwik9"]):
        print("%-12s %6.2f%% %6.2f%% %+9.5f %+10.1f %+13.1f %+12.1f  %-8s %s"
              % (r["zone"], r["anteil"], r["anteil_zeit"], r["delta_tok"],
                 r["delta_sum"] / np.log(2) / 8, r["bytes_enwik9"],
                 r["byte_je_proz"], r["kausal"], r["empfehlung"]))
    raus = [r for r in rows if r["empfehlung"].startswith("OUT")]
    print()
    print("RADICAL FORM: %s -> %.2f %% of the stream out, rate balance %+.1f bytes "
          "on enwik9 (exchange-rate threshold %.0f B per 1 %% time)"
          % (", ".join(r["zone"] for r in raus) or "none",
             sum(r["anteil_zeit"] for r in raus),
             -sum(r["bytes_enwik9"] for r in raus), KURS_BYTE_JE_PROZENT))
    # ---- second level: whole articles -----------------------------------
    kl, klnamen = artikelklassen(tokens, bnd, zone)
    art_of_pos = np.zeros(len(tokens), dtype=np.uint8)
    for a in range(len(bnd) - 1):
        art_of_pos[int(bnd[a]) : int(bnd[a + 1])] = kl[a]
    ap = art_of_pos[1:][:n]
    print()
    print("ARTICLE CLASSES (the level at which whole segments could be dropped)")
    h2 = ("%-10s %8s %10s %8s %10s %11s %13s  %s"
          % ("class", "articles", "positions", "share", "d/token", "dB excpt",
             "dB enwik9", "recommendation"))
    print(h2); print("-" * len(h2))
    art_rows = []
    for ki, kname in enumerate(klnamen):
        sel = ap == ki
        k = int(sel.sum())
        if not k:
            continue
        anteil = 100.0 * k / n
        dtok = float(delta[sel].mean())
        be = dtok * anteil / 100.0 * N_ENWIK9 / np.log(2) / 8
        bjp = be / anteil if anteil > 0 else 0.0
        emp = empfehlung(dtok, anteil, "oeffner", bjp)
        art_rows.append((kname, int((kl == ki).sum()), k, anteil, dtok,
                         float(delta[sel].sum()) / np.log(2) / 8, be, bjp, emp))
        print("%-10s %8d %10d %7.2f%% %+10.5f %+11.1f %+13.1f  %s"
              % (kname, int((kl == ki).sum()), k, anteil, dtok,
                 float(delta[sel].sum()) / np.log(2) / 8, be, emp))
    with open(os.path.join(outdir, "karte_v2_artikel.tsv"), "w", encoding="utf-8") as f:
        f.write("klasse\tartikel\tpositionen\tanteil_pct\tdelta_pro_token\t"
                "delta_bytes_bett\tdelta_bytes_enwik9\tbytes_je_prozent_zeit\t"
                "empfehlung\n")
        for r in art_rows:
            f.write("%s\t%d\t%d\t%.4f\t%+.6f\t%+.1f\t%+.1f\t%+.1f\t%s\n" % r)

    print()
    print("total damage   %+10.1f bytes   total gain   %+10.1f bytes"
          % (sum(r["delta_sum"] for r in rows if r["delta_sum"] > 0) / np.log(2) / 8,
             sum(r["delta_sum"] for r in rows if r["delta_sum"] < 0) / np.log(2) / 8))
    print("table written: %s" % tsv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
