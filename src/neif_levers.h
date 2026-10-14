#ifndef NEIF_LEVERS_H_
#define NEIF_LEVERS_H_

// Metadata-coding levers of work/h3_neif_meta_20261007 (2026-10-07).  Both are
// compile-time switches; every combination builds (stock A0 = 0/0).
//   NEIF_E1   part A: a pre-migration revid (< 17,000,000) is coded as
//             '+' [-]decimal(rv - F(page id)), F interpolating between the
//             nearest already-decoded B1 page ids (b1_interp.h).
//   NEIF_E3B  part B: the timestamp forecast of a B2 record (rank >= 87,813) is
//             the 8-step long-baseline slope over the last 9 B2 records
//             (ts_progress.h); B1 keeps the median-of-32-slopes estimator.
#ifndef NEIF_E1
#define NEIF_E1 1
#endif
#ifndef NEIF_E3B
#define NEIF_E3B 1
#endif

#endif
