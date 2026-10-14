# Measurements behind the release

These are retained experiment results, not new benchmark runs. Smaller payloads
are better; these prefix sizes omit executable fees. “160 MB” denotes the
prepared-input fixture (about 100 MB coder stream), not raw enwik9 size.
“12 MB” is the metadata-heavy native fixture. Its complete metadata block
makes it useful for metadata tests but unsuitable for proportional extrapolation
of whole-file savings. Evidence is copied into `verification/experiments/` with
its historical relative paths and is indexed by `verification/evidence.json`.

## Neural controls

| Native neif arm | Payload bytes, 160 MB fixture | Difference from B | Verification |
|---|---:|---:|---|
| A1: neutral expanded mixer geometry, stock-equivalent | 18,304,368 | +9,047 | Stock-equivalent bytes |
| B: one-block FWRH + FW1 | 18,295,321 | 0 | Fresh 160 MB decode/cmp passed |
| F4: four blocks, otherwise B | 18,289,887 | −5,434 | 160 MB encode only; 12 MB decode/cmp passed |

For A1/B, wall times were 33,524.13 / 37,838.54 seconds; user CPU times
33,429.09 / 37,703.42; peak RSS 9,864.5 / 9,865.6 MiB. B's observed wall
increase was 12.87%. F4 recorded 37,286.93 seconds and 9,868.0 MiB on its
separate run. Different contention prevents treating F4's lower wall time as
a speedup or isolating its added compute from these runs.

Earlier zmix experiments motivated the port. FW-DUAL saved 2,459 payload bytes
at 20 MB and 11,648 at 100 MB. At 20 MB, per-feature RMS preconditioning saved
another 133 bytes; adding the selected schedule gave a 156-byte combined
improvement over fixed-step FW-DUAL. Both variants passed fresh decodes.
Direct final-mixer wiring (FW1) saved 463 payload bytes with a 16-byte prefix
fee versus its HS3 control. Those comparisons use a different host and do not
identify the same component effects on neif. No isolated neif FW1-only
measurement is claimed.

## Metadata and table controls

| Native arm | Fixture | Payload bytes | Delta from specified control | Decode evidence |
|---|---|---:|---:|---|
| A0 / COSTDUMP metadata control | 12 MB | 1,653,282 | reference | Reference bytes match A0 smoke |
| E1 | 12 MB | 1,632,500 | −20,782 vs A0 | Encode only in audited standalone evidence |
| E13 = E1 + E3b | 12 MB | 1,630,460 | −22,822 vs A0; −2,040 vs E1 | Passed |
| B | 12 MB | 1,653,021 | neural control | Passed |
| B + E13 | 12 MB | 1,629,832 | −23,189 vs B | Passed |
| B + W3 | 12 MB | 1,652,903 | −118 vs B | Passed |
| FINAL | 12 MB | 1,629,691 | −141 vs B + E13, adding F4 and W3 together | Passed |
| B + W3 | 160 MB | 18,289,825 | −5,496 vs B | Encode only at this size |

The metadata control, E1 and E13 logs use the same input hash prefix
`8634dffef9c3ffc6`, CPU4 and COSTDUMP instrumentation. Their wall seconds are
1,921.07 / 1,832.41 / 1,832.65, and peak RSS is 6,046.3 / 6,044.4 / 6,044.9
MiB. E1's residual representation also shortens that prepared stream by
159,000 bytes. E3b's marginal +0.24 seconds is not a statistically meaningful
timing conclusion. Its added state is a ring of nine revision-id/timestamp pairs.

W3's 160 MB encode recorded 35,826.03 seconds and 10,019.3 MiB, versus B's
37,838.54 seconds and 9,865.6 MiB. They used different CPUs/days, so this is
not a controlled speedup. At 12 MB, the table arm reaches 9,735.1 MiB versus
B's 6,026.1 MiB: the added tables are much larger than the 153.7 MiB difference
in the later high-water marks suggests. Neif's PPM paging shrinks its resident
pool as other RSS grows; table capacity can displace PPM frames. Increased
system time is consistent with that trade-off, but these runs do not isolate
per-component paging costs.

The E1 code derives its forecast only from decoded page IDs and revision IDs.
The 17,000,000 threshold and initial forecast are corpus-informed constants
in source. E3b is a slope against revision ID across eight prior intervals;
it falls back to neif's estimator until enough usable history exists. It
changes a forecast/context, not the timestamp residual grammar. Both are
project adaptations of inherited metadata machinery, not borrowed new models.

## Whole-corpus accounting

| Full enwik9 encode | Payload bytes | SFX archive bytes | Compressor bytes | Wall seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Reproduced stock neif | 87,896,203 | 91,913,343 | 4,000,689 | 188,498.85 | 9,984.4 |
| B | 87,854,686 | 91,867,264 | 3,993,453 | 193,814.01 | 9,983.4 |
| FINAL | 87,764,097 | 91,778,237 | 3,995,122 | 220,071.89 | 10,204.5 |

B's full decode was interrupted; its full-size encode is not a verified
submission candidate. FINAL's replacement fresh full decode **passed** in
241,636.66 seconds with 10,194 MiB peak RSS. These shared-host runs are not
an isolated timing ablation. The full B→FINAL observed wall increase is
26,257.88 seconds (13.55%), and its RSS increase is 221.1 MiB.

FINAL reduces the stock payload by 132,106 bytes and archive overhead by
3,000 bytes. FINAL→B comparison: 90,589 payload bytes saved, with 1,562 more
overhead bytes, leaving an 89,027-byte archive gain. These are joint effects
of modeling, capacity, metadata and binary construction. No full-corpus
factorial separates their contributions; the prefix gains must not be added
or extrapolated as if independent. The separate compressor is counted only
for objectives that charge it, not again in the LTCB archive total.
