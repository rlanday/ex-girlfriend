# Stage (d): GPU training

This directory holds the files to train the transformer on package D and
the record of the training chain of the shipped weights (`poc8m_w2048`, 1,000
bf16 epochs, 32 qat, 6 fp32), with the environment of every run, the phase
scripts, the host setup and the lifting of subnormal rows between bf16 and
qat. Geometry, batching, learning rates and the sections of the chain are in
`../../README.md`, section 5; the trainer tree is built as in its section 1.

## Files

| file | role |
|---|---|
| `chain/CHAIN.tsv` | one row per run: env file, script, warm start and its sha256, package (A to D as in `../../README.md`), window, epochs per phase, continued schedule, cards, gradient accumulation, end state and its sha256, bf16 epochs seen |
| `chain/01_anchor_w2048.env` to `chain/10_fp32.env` | the environment of each run with the values it ran with; a header names start point, package, window, epochs and end state |
| `kette_arm.sh` | one chain arm ("Kette" = chain): bf16, qat, fp32, export; marker `/root/ARM_<tag>_FERTIG` |
| `wiederanlauf.sh` | resume ("Wiederanlauf") of one phase from a checkpoint, continuing its learning-rate schedule |
| `qat_arm.sh` | qat, fp32 and export on an existing bf16 state `models/<geometry>-pkd.tch` |
| `fp32_arm.sh` | fp32 with `TFT_EPOCHS_FP32` epochs on the qat end `models/<geometry>-pkd-q4.tch`, then the export |
| `setup_host.sh` | GPU host setup: venv, window geometries, warm-start check, package and priors with content checks, loader check |
| `geo_zusatz.py` | adds a window variant of an existing geometry to `tft/param_count.py` (row 06 needs `poc8m_w2560`) |
| `epoche_opt.sh` | prints the epoch stored in the `.opt` file of a checkpoint |
| `probe_row_max.py`, `fix_subnormal_rows.py` | the smallest row maxima of a checkpoint, and the lifting of rows below 1e-30 (row 08r) |
| `prioren_server.py` | HTTP server with range requests for the priors, `ppmd-bytes.uint8` and the package tar ("Prioren" = priors) |

The scripts keep their variable names, with `BAUM` = trainer tree, `ENVD` = env
file, `KARTEN` = cards (`CUDA_VISIBLE_DEVICES`) and `PORT` = torchrun master
port; the headers explain the rest. They assume the layout of the GPU hosts
used:

```
/root/fx6m_8m/     trainer tree (the composed tree of ../../README.md section 1)
/root/pkg_v7/      training package and priors (the name was kept for every package)
/root/ckpt/        warm-start checkpoints, under the names the env files use
/root/chain/       the env files of chain/
/root/*.sh, /root/geo_zusatz.py
```

Hardware: 4 GPUs with 32 GB each (RTX 5090 class), driver 580 or later
(CUDA 13); about 6 minutes per bf16 epoch on four such cards. Two cards give
the same step count with twice the gradient accumulation.

## Hosts

**Data host.** One server per file (priors, `ppmd-bytes.uint8`, package tar):

```bash
python3 prioren_server.py --datei <file> --port <port> --minuten 10080
```

It prints the URL with its access token. Before a GPU host is set up, every URL
must answer a ranged request with status 206 and the full length in
`Content-Range` (priors: n_tokens x 416 bytes, ppmd-bytes: n_tokens bytes):

```bash
curl -s -r 0-1023 -o /dev/null -D - "$URL" | grep -i -E '^HTTP|^content-range'
```

**GPU host.** First check the driver, the card count and that the system
python sees the cards (a host with a dead CUDA context lists its cards in
`nvidia-smi` but not in torch):

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Then copy the tree to `/root/fx6m_8m/` (without `.venv` and `models/`), the
scripts and `geo_zusatz.py` to `/root/` and `chain/` to `/root/chain/`, and
write the parameter files listed in the header of `setup_host.sh` (mode, URLs,
expected sha256 values; for a warm start also its path and sha256 prefix).
This chain uses two modes of `setup_host.sh`. `acht` checks no warm start and
starts `kette_arm.sh` with the env file in `ANCHOR_ENV` itself, on cards 0 and
1 (a cold start). `zweig` checks the warm start, which lies with its `.opt` at
the path in `TFT_WARMSTART`, and starts nothing.

## Training on package D

The recipe of `../../README.md` section 5 ("Training on package D") with the
scripts of this directory, on a host prepared as above:

```bash
B=/root/fx6m_8m; C=/root/chain
# the env of row 08 without its warm start, 1,000 bf16 epochs
grep -v -E '^TFT_WARMSTART' $C/08_w2048.env | sed 's/^TFT_EPOCHS_BF16=.*/TFT_EPOCHS_BF16=1000/' > $C/d_cold.env
# setup in mode acht; it ends with /root/BOOT_V7_OK and starts kette_arm.sh in
# tmux "acht" on cards 0,1 (log /root/lauf_8m.log); stop the arm at "### qat"
echo acht > /root/.modus_v7
ANCHOR_ENV=$C/d_cold.env nohup bash /root/setup_host.sh > /root/boot_nohup.log 2>&1 &
# lift subnormal rows of the bf16 end
cd $B && .venv/bin/python /root/probe_row_max.py models/poc8m_w2048-pkd.tch
.venv/bin/python /root/fix_subnormal_rows.py models/poc8m_w2048-pkd.tch /root/fixed.tch
cp /root/fixed.tch models/poc8m_w2048-pkd.tch
# qat 32, fp32 6, export, on four cards
BAUM=$B ENVD=$C/10_fp32.env KARTEN=0,1,2,3 PORT=29931 bash /root/qat_arm.sh
```

## The chain of the shipped weights

Each row of `CHAIN.tsv` starts from the end state of the row before; check its
sha256 against the table, then copy it with its `.opt` to the name the next env
file expects. Run each arm in `tmux` with its output in a log file. Rows 01 to
07 ran on packages A to C, which this package does not rebuild; row 01 ran
cold through `setup_host.sh` in mode `acht`, the others with `kette_arm.sh` or, for
the rows with a `b`, `wiederanlauf.sh`, as `CHAIN.tsv` lists them (row 06 needs
the geometry `poc8m_w2560`, `geo_zusatz.py`). Rows 08 to 10 ran on package D,
on a host set up in mode `zweig` with the end of row 07 as the warm start:

```bash
echo zweig > /root/.modus_v7
nohup bash /root/setup_host.sh > /root/boot_nohup.log 2>&1 &     # ends with /root/BOOT_V7_OK
B=/root/fx6m_8m; C=/root/chain
BAUM=$B ENVD=$C/08_w2048.env        KARTEN=0,1,2,3 PORT=29751 bash /root/kette_arm.sh   # epochs 1 to 123
BAUM=$B ENVD=$C/08b_w2048_resume.env KARTEN=0,1,2,3 PORT=29751 bash /root/kette_arm.sh  # epochs 124 to 200
# 08r  lift subnormal rows of the bf16 end, before qat
cd $B && .venv/bin/python /root/probe_row_max.py models/poc8m_w2048-pkd.tch
.venv/bin/python /root/fix_subnormal_rows.py models/poc8m_w2048-pkd.tch /root/fixed.tch
cp /root/fixed.tch models/poc8m_w2048-pkd.tch
# 09, 10  qat 32 on the lifted state, fp32 6, export
BAUM=$B ENVD=$C/09_qat.env KARTEN=0,1,2,3 PORT=29931 bash /root/qat_arm.sh
BAUM=$B ENVD=$C/10_fp32.env bash /root/fp32_arm.sh
```

`qat_arm.sh` with file 10 runs the same two phases in one go (32 qat, 6 fp32);
with file 09 only the qat phase counts, the fp32 phase follows from file 10.
A bf16 arm carries only its bf16 end on, so stop it once its log shows
`### qat`, after confirming with `epoche_opt.sh models/<geometry>-pkd.tch.opt`
that the bf16 end holds all epochs, and copy `models/<geometry>-pkd.tch` and
its `.opt`. The export of the last run, `weights-poc8m_w2048.bin`, goes to
`../4_export/`.

## Rules

1. **Batching.** `TFT_MICRO_BATCH_TOKENS=131072` per rank and
   `TFT_OPTIMIZER_BATCH_TOKENS=1048576` (2^20) in every run; the trainer derives
   the gradient accumulation, 2 on 4 cards and 4 on 2 cards, so steps per epoch
   do not depend on the card count. In the dump loader the micro-batch is also
   the longest document that is not split. Package D has at most 131,037 rows
   per document, and the log must show `articles split: 0` and
   `dump loader: position limit 131072, longest document 131037 rows, margin 35`.
2. **Warm start or resume.** A warm start begins a new section with a fresh
   schedule, and `TFT_WARMSTART` and `TFT_WARMSTART_OPT` load weights and
   optimizer state (`KINJ_OPTSTATE=1`) into the bf16 phase. A resume continues a
   section, with `wiederanlauf.sh` setting `TFT_SCHED_TOTAL_EPOCHS` and
   `TFT_SCHED_DONE_EPOCHS` and running the remaining epochs; the log shows `continued schedule: block of
   200 epochs (warm-up 133), here epochs 123..200`. Read the done epochs from
   the `.opt` with `epoche_opt.sh`, never estimate them. qat and fp32 always
   load from `models/` (bf16 end, qat end); `kette_arm.sh` and `qat_arm.sh`
   unset the warm-start and schedule variables before them.
3. **Splits change the data order.** The loader draws the epoch permutations
   for the epochs of each run from a generator with seed 0, so every run, a
   resumed one included, starts at the first permutation; the `b` rows of
   `CHAIN.tsv` are such splits; the table records the runs of the shipped
   weights.
4. **Priors are checked by content.** `setup_host.sh` compares the sha256 of the
   first MiB and of the whole file with the expected values and requires 64
   random rows to sum to 1 within 0.01. A length check alone passes on a file
   with holes; a reused host with priors of another package fetches again when
   the first MiB differs.
5. **No qat on a state with subnormal rows.** Before qat, `probe_row_max.py`
   on the bf16 end; a row maximum below 1.18e-38 (the smallest normal float32)
   makes qat produce NaN. Lift such rows with `fix_subnormal_rows.py`, probe
   again, and watch the first qat steps for NaN.
6. **One arm per tree, a fresh log directory per run.** Checkpoints are written
   to `models/<geometry>-<arm>.tch` relative to the tree, so two arms at the same
   time need two trees. The logger refuses to write into an existing jsonl file
   (named after geometry, phase and run tag), so before a resume or a repeated
   phase in the same `TFT_LOG_DIR`, move the old file aside.
7. **Secure before releasing a host.** Copy every checkpoint with its `.opt`
   (bf16, qat and fp32 ends) and the logs off the GPU host and compare sha256
   before the host is released.

Renting hosts and mirroring checkpoints are not part of this package. Before a
start or a resume check the warm-start sha256, the env values, status 206 of
every URL, driver, cards and CUDA context, and the `continued schedule` line in
the log.
