
#include <stdint.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace r1_reorder {
namespace {

struct LineRef { size_t start; size_t len; };

struct TailBlock {
  size_t original_index;
  size_t first, count;
  size_t key_start, key_len;
};

void* FastGrow(void* p, size_t* cap, size_t need, size_t esz) {
  size_t c = *cap;
  if (c == 0) c = 16;
  while (c < need) c += c / 2 + 1;
  void* q = realloc(p, c * esz);
  if (q == 0) {
    std::fprintf(stderr, "\nr1 out of memory\n");
    abort();
  }
  *cap = c;
  return q;
}

struct ByteBuf { unsigned char* d; size_t n, cap; };
struct LineVec { LineRef* d; size_t n, cap; };
struct BlockVec { TailBlock* d; size_t n, cap; };
struct IdxVec { size_t* d; size_t n, cap; };
struct U64Vec { uint64_t* d; size_t n, cap; };

void BufFree(ByteBuf* b) { free(b->d); b->d = 0; b->n = b->cap = 0; }
void LineFree(LineVec* v) { free(v->d); v->d = 0; v->n = v->cap = 0; }
void BlockFree(BlockVec* v) { free(v->d); v->d = 0; v->n = v->cap = 0; }
void IdxFree(IdxVec* v) { free(v->d); v->d = 0; v->n = v->cap = 0; }
void U64Free(U64Vec* v) { free(v->d); v->d = 0; v->n = v->cap = 0; }

void BufReserve(ByteBuf* b, size_t need) {
  if (need > b->cap) b->d = (unsigned char*)FastGrow(b->d, &b->cap, need, 1);
}
void BufPush(ByteBuf* b, unsigned char v) {
  if (b->n == b->cap) b->d = (unsigned char*)FastGrow(b->d, &b->cap, b->n + 1, 1);
  b->d[b->n++] = v;
}
void BufAppend(ByteBuf* b, const unsigned char* src, size_t len) {
  if (b->n + len > b->cap) b->d = (unsigned char*)FastGrow(b->d, &b->cap, b->n + len, 1);
  if (len) memcpy(b->d + b->n, src, len);
  b->n += len;
}
void LinePush(LineVec* v, size_t start, size_t len) {
  if (v->n == v->cap) v->d = (LineRef*)FastGrow(v->d, &v->cap, v->n + 1, sizeof(LineRef));
  v->d[v->n].start = start; v->d[v->n].len = len; ++v->n;
}
void IdxPush(IdxVec* v, size_t value) {
  if (v->n == v->cap) v->d = (size_t*)FastGrow(v->d, &v->cap, v->n + 1, sizeof(size_t));
  v->d[v->n++] = value;
}
void IdxResize(IdxVec* v, size_t n) {
  if (n > v->cap) v->d = (size_t*)FastGrow(v->d, &v->cap, n, sizeof(size_t));
  v->n = n;
}
void U64Resize(U64Vec* v, size_t n) {
  if (n > v->cap) v->d = (uint64_t*)FastGrow(v->d, &v->cap, n, sizeof(uint64_t));
  v->n = n;
}
void BlockPush(BlockVec* v, const TailBlock* b) {
  if (v->n == v->cap) v->d = (TailBlock*)FastGrow(v->d, &v->cap, v->n + 1, sizeof(TailBlock));
  v->d[v->n++] = *b;
}

// --- comparison functions (strict total orders) ---------------------------
int Fast2CmpD86a(const void* pa, const void* pb, void* ctx) {
  const size_t a = *(const size_t*)pa;
  const size_t b = *(const size_t*)pb;
  const uint64_t* d86a = (const uint64_t*)ctx;
  if (d86a[a] != d86a[b]) return d86a[a] < d86a[b] ? -1 : 1;
  return a < b ? -1 : (a > b ? 1 : 0);
}

struct BlockCmpCtx { const unsigned char* data; const TailBlock* blocks; };

int Fast2CmpBlock(const void* pa, const void* pb, void* ctx) {
  const size_t a = *(const size_t*)pa;
  const size_t b = *(const size_t*)pb;
  const BlockCmpCtx* c = (const BlockCmpCtx*)ctx;
  const TailBlock* A = &c->blocks[a];
  const TailBlock* B = &c->blocks[b];
  const size_t m = A->key_len < B->key_len ? A->key_len : B->key_len;
  if (m) {
    const int r = memcmp(c->data + A->key_start, c->data + B->key_start, m);
    if (r) return r < 0 ? -1 : 1;
  }
  if (A->key_len != B->key_len) return A->key_len < B->key_len ? -1 : 1;
  return A->original_index < B->original_index ? -1 :
      (A->original_index > B->original_index ? 1 : 0);
}

const unsigned char kSideMagicD86[] = {'R', '1', 'O', 'R', 'D', '3', '\n'};
const unsigned char kFooterMagic[] = {'R', '1', 'O', 'R', 'D', 'F', 'T', 'R'};
const unsigned char kD99Line[] = {0xDF, 0x99, 'N'};
const unsigned char kD86Prefix[] = {0xDF, 0x86, 'N'};

size_t g_tail_start = 0, g_tail_len = 0;
size_t g_regime1_start = 0, g_regime2_start = 0;

const size_t kMaxD99Gap = 1024;

struct SideMeta {
  size_t tail_len;
  size_t r0_count, r1_count, r2_count;
  size_t prelude_count, suffix_count;
  IdxVec sorted_to_original;
};

bool SizeTFromU64(uint64_t value, size_t* out) {
  if (value > (uint64_t)(size_t)-1) return false;
  *out = (size_t)value;
  return true;
}

bool ReadFile(const char* path, ByteBuf* data) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return false; }
  const off_t size = ftello(f);
  if (size < 0) { fclose(f); return false; }
  rewind(f);
  BufReserve(data, (size_t)size);
  data->n = (size_t)size;
  size_t got = 0;
  if (data->n) got = fread(data->d, 1, data->n, f);
  fclose(f);
  return got == data->n;
}

bool WriteFile(const char* path, const unsigned char* d, size_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  if (n && fwrite(d, 1, n, f) != n) { fclose(f); return false; }
  return fclose(f) == 0;
}

bool HasBytesAt(const unsigned char* d, size_t n, size_t pos,
    const unsigned char* bytes, size_t len) {
  return pos <= n && n - pos >= len && memcmp(d + pos, bytes, len) == 0;
}

size_t BodyEnd(const unsigned char* d, size_t n, LineRef line) {
  if (line.start > n || line.len > n - line.start) return line.start;
  size_t end = line.start + line.len;
  if (end > line.start && d[end - 1] == '\n') --end;
  if (end > line.start && d[end - 1] == '\r') --end;
  return end;
}

void SplitLines(size_t start, size_t len, const unsigned char* d, size_t n,
    LineVec* out) {
  out->n = 0;
  if (start > n || len > n - start) return;
  const size_t end = start + len;
  size_t line_start = start;
  for (size_t i = start; i < end; ++i) {
    if (d[i] == '\n') {
      LinePush(out, line_start, i + 1 - line_start);
      line_start = i + 1;
    }
  }
  if (line_start < end) LinePush(out, line_start, end - line_start);
}

bool IsExactD99Line(const unsigned char* d, size_t n, LineRef line) {
  const size_t end = BodyEnd(d, n, line);
  return end - line.start == sizeof(kD99Line) &&
      memcmp(d + line.start, kD99Line, sizeof(kD99Line)) == 0;
}

bool IsAsciiNumberLine(const unsigned char* d, size_t n, LineRef line) {
  const size_t end = BodyEnd(d, n, line);
  if (line.start == end) return false;
  for (size_t i = line.start; i < end; ++i) {
    const unsigned char c = d[i];
    if ((c < '0' || c > '9') && c != '-') return false;
  }
  return true;
}

bool ParseUnsignedExact(const unsigned char* d, size_t pos, size_t end,
    uint64_t* value) {
  if (pos >= end || d[pos] < '0' || d[pos] > '9') return false;
  uint64_t result = 0;
  while (pos < end && d[pos] >= '0' && d[pos] <= '9') {
    const uint64_t digit = (uint64_t)(d[pos++] - '0');
    if (result > ((uint64_t)-1 - digit) / 10) return false;
    result = result * 10 + digit;
  }
  if (pos != end) return false;
  *value = result;
  return true;
}

bool ParseFirstD86a(const unsigned char* d, size_t n, const LineRef* lines,
    const TailBlock* block, uint64_t* value) {
  if (block->count < 2) return false;
  const LineRef line = lines[block->first + 1];
  const size_t end = BodyEnd(d, n, line);
  if (end < line.start + sizeof(kD86Prefix) ||
      memcmp(d + line.start, kD86Prefix, sizeof(kD86Prefix)) != 0) {
    return false;
  }
  return ParseUnsignedExact(d, line.start + sizeof(kD86Prefix), end, value);
}

void AppendLines(ByteBuf* out, const unsigned char* d, const LineRef* lines,
    size_t first, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    BufAppend(out, d + lines[first + i].start, lines[first + i].len);
  }
}

void AppendVarint(ByteBuf* out, uint64_t value) {
  while (value >= 0x80) {
    BufPush(out, (unsigned char)((value & 0x7F) | 0x80));
    value >>= 7;
  }
  BufPush(out, (unsigned char)value);
}

bool ReadVarint(const unsigned char* d, size_t n, size_t* pos,
    uint64_t* value) {
  uint64_t result = 0;
  int shift = 0;
  while (*pos < n && shift <= 63) {
    const unsigned char byte = d[(*pos)++];
    if (shift == 63 && (byte & 0x7F) > 1) return false;
    result |= (uint64_t)(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      *value = result;
      return true;
    }
    shift += 7;
  }
  return false;
}

void AppendU64LE(ByteBuf* out, uint64_t value) {
  for (int i = 0; i < 8; ++i)
    BufPush(out, (unsigned char)((value >> (8 * i)) & 0xFF));
}

bool ReadU64LE(const unsigned char* d, size_t n, size_t pos, uint64_t* value) {
  if (pos > n || n - pos < 8) return false;
  uint64_t result = 0;
  for (int i = 0; i < 8; ++i) result |= (uint64_t)d[pos + i] << (8 * i);
  *value = result;
  return true;
}

// --- Fenwick (vector<int> -> int*) ----------------------------------------
struct Fenwick { int* tree; size_t size; };

void FenwickAdd(Fenwick* f, size_t index, int delta) {
  for (++index; index < f->size; index += index & (0 - index)) {
    f->tree[index] += delta;
  }
}

void FenwickInit(Fenwick* f, size_t n) {
  f->size = n + 1;
  f->tree = (int*)calloc(f->size ? f->size : 1, sizeof(int));
  if (!f->tree) { std::fprintf(stderr, "\nr1 out of memory\n"); abort(); }
  for (size_t i = 0; i < n; ++i) FenwickAdd(f, i, 1);
}

void FenwickFree(Fenwick* f) { free(f->tree); f->tree = 0; f->size = 0; }

size_t FenwickSumLessThan(const Fenwick* f, size_t index) {
  size_t result = 0;
  while (index > 0) {
    result += (size_t)f->tree[index];
    index -= index & (0 - index);
  }
  return result;
}

size_t FenwickFindByOrder(const Fenwick* f, size_t rank) {
  size_t index = 0;
  size_t bit = 1;
  while ((bit << 1) < f->size) bit <<= 1;
  while (bit != 0) {
    const size_t next = index + bit;
    if (next < f->size && (size_t)f->tree[next] <= rank) {
      index = next;
      rank -= (size_t)f->tree[next];
    }
    bit >>= 1;
  }
  return index;
}

bool AppendLehmerPermutation(ByteBuf* out, const size_t* permutation,
    size_t count) {
  Fenwick fenwick;
  FenwickInit(&fenwick, count);
  unsigned char* seen = (unsigned char*)calloc(count ? count : 1, 1);
  if (!seen) { std::fprintf(stderr, "\nr1 out of memory\n"); abort(); }
  bool ok = true;
  for (size_t i = 0; i < count; ++i) {
    const size_t value = permutation[i];
    if (value >= count || seen[value]) { ok = false; break; }
    seen[value] = 1;
    AppendVarint(out, FenwickSumLessThan(&fenwick, value));
    FenwickAdd(&fenwick, value, -1);
  }
  free(seen);
  FenwickFree(&fenwick);
  return ok;
}

bool ReadLehmerPermutation(const unsigned char* d, size_t n, size_t* pos,
    size_t count, IdxVec* permutation) {
  Fenwick fenwick;
  FenwickInit(&fenwick, count);
  IdxResize(permutation, count);
  bool ok = true;
  for (size_t i = 0; i < count; ++i) {
    uint64_t rank64 = 0;
    if (!ReadVarint(d, n, pos, &rank64) || rank64 >= count - i) { ok = false; break; }
    const size_t value = FenwickFindByOrder(&fenwick, (size_t)rank64);
    permutation->d[i] = value;
    FenwickAdd(&fenwick, value, -1);
  }
  FenwickFree(&fenwick);
  return ok;
}

bool PartitionTailLines(const LineRef* lines, size_t count, size_t* r0_count,
    size_t* r1_count, size_t* r2_count) {
  *r0_count = *r1_count = *r2_count = 0;
  size_t pos = 0;
  for (size_t i = 0; i < count; ++i) {
    if (pos < g_regime1_start) ++*r0_count;
    else if (pos < g_regime2_start) ++*r1_count;
    else ++*r2_count;
    if (lines[i].len > g_tail_len - pos) return false;
    pos += lines[i].len;
  }
  return pos == g_tail_len;
}

size_t GuessLastHugeBlockEnd(const unsigned char* d, size_t n,
    const LineRef* lines, size_t start, size_t limit) {
  size_t end = start + 7 < limit ? start + 7 : limit;
  if (end < limit && IsAsciiNumberLine(d, n, lines[end])) {
    ++end;
    if (end < limit && !IsExactD99Line(d, n, lines[end])) {
      const size_t body_end = BodyEnd(d, n, lines[end]);
      if (body_end > lines[end].start &&
          body_end - lines[end].start <= 16 &&
          d[lines[end].start] >= 0x80) {
        ++end;
      }
    }
  }
  return end;
}

bool ParseTailBlocks(const unsigned char* d, size_t n, const LineRef* r1_lines,
    size_t r1_count, size_t* prelude_count, BlockVec* blocks,
    size_t* suffix_first, size_t* suffix_count) {
  IdxVec starts; starts.d = 0; starts.n = starts.cap = 0;
  for (size_t i = 0; i < r1_count; ++i) {
    if (IsExactD99Line(d, n, r1_lines[i])) IdxPush(&starts, i);
  }
  if (starts.n == 0) { IdxFree(&starts); return false; }
  *prelude_count = starts.d[0];
  blocks->n = 0;
  if (starts.n > blocks->cap) {
    blocks->d = (TailBlock*)FastGrow(blocks->d, &blocks->cap, starts.n,
        sizeof(TailBlock));
  }
  size_t cursor = starts.d[0];
  for (size_t block_index = 0; block_index < starts.n; ++block_index) {
    const size_t start = starts.d[block_index];
    if (start != cursor) { IdxFree(&starts); return false; }
    const size_t next_start =
        block_index + 1 < starts.n ? starts.d[block_index + 1] : r1_count;
    const size_t end = (next_start - start <= 16) ? next_start :
        GuessLastHugeBlockEnd(d, n, r1_lines, start, next_start);
    TailBlock block;
    block.original_index = block_index;
    block.first = start;
    block.count = end - start;
    if (block.count > 2) {
      block.key_start = r1_lines[start + 2].start;
      const LineRef last = r1_lines[end - 1];
      block.key_len = last.start + last.len - block.key_start;
    } else {
      block.key_start = 0;
      block.key_len = 0;
    }
    BlockPush(blocks, &block);
    cursor = end;
  }
  *suffix_first = cursor;
  *suffix_count = r1_count - cursor;
  IdxFree(&starts);
  return true;
}

bool ComputeD86aOrder(const unsigned char* d, size_t n, const LineRef* lines,
    const TailBlock* blocks, size_t block_count, IdxVec* order) {
  U64Vec d86a; d86a.d = 0; d86a.n = d86a.cap = 0;
  U64Resize(&d86a, block_count);
  for (size_t i = 0; i < block_count; ++i) {
    if (!ParseFirstD86a(d, n, lines, &blocks[i], &d86a.d[i])) {
      U64Free(&d86a);
      return false;
    }
  }
  IdxResize(order, block_count);
  for (size_t i = 0; i < block_count; ++i) order->d[i] = i;
  if (block_count) {
    qsort_r(order->d, block_count, sizeof(size_t), Fast2CmpD86a,
        (void*)d86a.d);
  }
  U64Free(&d86a);
  return true;
}

bool MakeSide(const unsigned char* d, size_t n, const LineRef* lines,
    const TailBlock* blocks, size_t block_count, const size_t* sorted_indices,
    size_t sorted_count, size_t r0_count, size_t r1_count, size_t r2_count,
    size_t prelude_count, size_t suffix_count, ByteBuf* side) {
  U64Vec d86a; d86a.d = 0; d86a.n = d86a.cap = 0;
  U64Resize(&d86a, sorted_count);
  for (size_t sorted_pos = 0; sorted_pos < sorted_count; ++sorted_pos) {
    const size_t index = sorted_indices[sorted_pos];
    if (index >= block_count ||
        !ParseFirstD86a(d, n, lines, &blocks[index], &d86a.d[sorted_pos])) {
      U64Free(&d86a);
      return false;
    }
  }
  IdxVec d86_order; d86_order.d = 0; d86_order.n = d86_order.cap = 0;
  IdxResize(&d86_order, sorted_count);
  for (size_t i = 0; i < sorted_count; ++i) d86_order.d[i] = i;
  if (sorted_count) {
    qsort_r(d86_order.d, sorted_count, sizeof(size_t), Fast2CmpD86a,
        (void*)d86a.d);
  }
  U64Free(&d86a);

  IdxVec original_by_d86a; original_by_d86a.d = 0;
  original_by_d86a.n = original_by_d86a.cap = 0;
  IdxResize(&original_by_d86a, sorted_count);
  for (size_t i = 0; i < sorted_count; ++i) {
    original_by_d86a.d[i] = sorted_indices[d86_order.d[i]];
  }
  IdxFree(&d86_order);

  side->n = 0;
  BufReserve(side, 32 + original_by_d86a.n * 3);
  BufAppend(side, kSideMagicD86, sizeof(kSideMagicD86));
  AppendVarint(side, g_tail_len);
  AppendVarint(side, r0_count);
  AppendVarint(side, r1_count);
  AppendVarint(side, r2_count);
  AppendVarint(side, prelude_count);
  AppendVarint(side, suffix_count);
  AppendVarint(side, original_by_d86a.n);
  const bool ok = AppendLehmerPermutation(side, original_by_d86a.d,
      original_by_d86a.n);
  IdxFree(&original_by_d86a);
  return ok;
}

bool ParseSide(const unsigned char* side, size_t side_len, SideMeta* meta) {
  if (!HasBytesAt(side, side_len, 0, kSideMagicD86, sizeof(kSideMagicD86)))
    return false;
  size_t pos = sizeof(kSideMagicD86);
  uint64_t tail_len = 0, r0 = 0, r1 = 0, r2 = 0, prelude = 0, suffix = 0;
  uint64_t count = 0;
  if (!ReadVarint(side, side_len, &pos, &tail_len) || tail_len == 0 ||
      !ReadVarint(side, side_len, &pos, &r0) ||
      !ReadVarint(side, side_len, &pos, &r1) ||
      !ReadVarint(side, side_len, &pos, &r2) ||
      !ReadVarint(side, side_len, &pos, &prelude) ||
      !ReadVarint(side, side_len, &pos, &suffix) ||
      !ReadVarint(side, side_len, &pos, &count)) {
    return false;
  }
  if (!SizeTFromU64(tail_len, &meta->tail_len) ||
      !SizeTFromU64(r0, &meta->r0_count) ||
      !SizeTFromU64(r1, &meta->r1_count) ||
      !SizeTFromU64(r2, &meta->r2_count) ||
      !SizeTFromU64(prelude, &meta->prelude_count) ||
      !SizeTFromU64(suffix, &meta->suffix_count)) {
    return false;
  }
  size_t count_size = 0;
  if (!SizeTFromU64(count, &count_size)) return false;
  if (!ReadLehmerPermutation(side, side_len, &pos, count_size,
          &meta->sorted_to_original)) {
    return false;
  }
  return pos == side_len;
}

bool DeriveGeometry(const unsigned char* data, size_t n, LineVec* lines_out) {
  IdxVec d99; d99.d = 0; d99.n = d99.cap = 0;
  for (size_t i = 0; i + 5 <= n; ++i) {
    if (data[i] != '\n') continue;
    if (data[i + 1] == kD99Line[0] && data[i + 2] == kD99Line[1] &&
        data[i + 3] == kD99Line[2] && data[i + 4] == '\n') {
      IdxPush(&d99, i + 1);
    }
  }
  if (d99.n < 2) { IdxFree(&d99); return false; }
  size_t best_b = 0, best_e = 1, b = 0;
  for (size_t i = 1; i <= d99.n; ++i) {
    if (i == d99.n || d99.d[i] - d99.d[i - 1] > kMaxD99Gap) {
      if (i - b >= best_e - best_b) { best_b = b; best_e = i; }
      b = i;
    }
  }
  if (best_e - best_b < 2) { IdxFree(&d99); return false; }
  const size_t tail_start = d99.d[best_b];
  const size_t tail_len = n - tail_start;
  SplitLines(tail_start, tail_len, data, n, lines_out);
  const size_t last_off = d99.d[best_e - 1];
  size_t last_idx = lines_out->n;
  for (size_t i = 0; i < lines_out->n; ++i) {
    if (lines_out->d[i].start == last_off) { last_idx = i; break; }
  }
  if (last_idx == lines_out->n) { IdxFree(&d99); return false; }
  const size_t end = GuessLastHugeBlockEnd(data, n, lines_out->d, last_idx,
      lines_out->n);
  g_tail_start = tail_start;
  g_tail_len = tail_len;
  g_regime1_start = 0;
  g_regime2_start = (end < lines_out->n ? lines_out->d[end].start : n) - tail_start;
  const size_t blocks = best_e - best_b;
  IdxFree(&d99);
  (void)0;
  return true;
}

}  // namespace

bool ReorderEncodedTailFile(const std::string& path,
    const std::string& side_path) {
  ByteBuf input; input.d = 0; input.n = input.cap = 0;
  if (!ReadFile(path.c_str(), &input)) { BufFree(&input); return false; }
  LineVec lines; lines.d = 0; lines.n = lines.cap = 0;
  if (!DeriveGeometry(input.d, input.n, &lines)) {
    std::fprintf(stderr,
        "\nr1 encoded tail reorder refused: no D99 block run (stream=%zu)\n",
        input.n);
    LineFree(&lines); BufFree(&input);
    return false;
  }

  SplitLines(g_tail_start, g_tail_len, input.d, input.n, &lines);
  size_t r0_count = 0, r1_count = 0, r2_count = 0;
  if (!PartitionTailLines(lines.d, lines.n, &r0_count, &r1_count, &r2_count)) {
    LineFree(&lines); BufFree(&input);
    return false;
  }
  const size_t r1_first = r0_count;

  size_t prelude_count = 0, suffix_first = 0, suffix_count = 0;
  BlockVec blocks; blocks.d = 0; blocks.n = blocks.cap = 0;
  if (!ParseTailBlocks(input.d, input.n, lines.d + r1_first, r1_count,
          &prelude_count, &blocks, &suffix_first, &suffix_count)) {
    BlockFree(&blocks); LineFree(&lines); BufFree(&input);
    return false;
  }

  IdxVec sorted_indices; sorted_indices.d = 0;
  sorted_indices.n = sorted_indices.cap = 0;
  IdxResize(&sorted_indices, blocks.n);
  for (size_t i = 0; i < blocks.n; ++i) sorted_indices.d[i] = i;
  if (blocks.n) {
    BlockCmpCtx ctx;
    ctx.data = input.d;
    ctx.blocks = blocks.d;
    qsort_r(sorted_indices.d, blocks.n, sizeof(size_t), Fast2CmpBlock,
        (void*)&ctx);
  }

  ByteBuf side; side.d = 0; side.n = side.cap = 0;
  if (!MakeSide(input.d, input.n, lines.d + r1_first, blocks.d, blocks.n,
          sorted_indices.d, sorted_indices.n, r0_count, r1_count, r2_count,
          prelude_count, suffix_count, &side) ||
      !WriteFile(side_path.c_str(), side.d, side.n)) {
    BufFree(&side); IdxFree(&sorted_indices); BlockFree(&blocks);
    LineFree(&lines); BufFree(&input);
    return false;
  }

  ByteBuf output; output.d = 0; output.n = output.cap = 0;
  BufReserve(&output, input.n + side.n + sizeof(kFooterMagic) + 8);
  BufAppend(&output, input.d, g_tail_start);
  AppendLines(&output, input.d, lines.d, 0, r0_count);
  AppendLines(&output, input.d, lines.d + r1_first, 0, prelude_count);
  for (size_t i = 0; i < sorted_indices.n; ++i) {
    const TailBlock* blk = &blocks.d[sorted_indices.d[i]];
    AppendLines(&output, input.d, lines.d + r1_first, blk->first, blk->count);
  }
  AppendLines(&output, input.d, lines.d + r1_first, suffix_first, suffix_count);
  AppendLines(&output, input.d, lines.d, r1_first + r1_count, r2_count);
  bool ok = output.n == input.n;
  if (ok) {
    BufAppend(&output, side.d, side.n);
    BufAppend(&output, kFooterMagic, sizeof(kFooterMagic));
    AppendU64LE(&output, side.n);
    ok = WriteFile(path.c_str(), output.d, output.n);
  }
  if (ok) {
    (void)0;
  }
  BufFree(&output); BufFree(&side); IdxFree(&sorted_indices);
  BlockFree(&blocks); LineFree(&lines); BufFree(&input);
  return ok;
}

bool ExtractSideFromFile(const std::string& path,
    const std::string& side_path) {
  ByteBuf data; data.d = 0; data.n = data.cap = 0;
  if (!ReadFile(path.c_str(), &data) || data.n < sizeof(kFooterMagic) + 8) {
    BufFree(&data);
    return false;
  }
  const size_t footer_pos = data.n - sizeof(kFooterMagic) - 8;
  if (!HasBytesAt(data.d, data.n, footer_pos, kFooterMagic,
          sizeof(kFooterMagic))) {
    BufFree(&data);
    return false;
  }
  uint64_t side_len64 = 0;
  if (!ReadU64LE(data.d, data.n, footer_pos + sizeof(kFooterMagic),
          &side_len64) || side_len64 > footer_pos) {
    BufFree(&data);
    return false;
  }
  const size_t side_len = (size_t)side_len64;
  const size_t side_pos = footer_pos - side_len;
  const bool ok = WriteFile(side_path.c_str(), data.d + side_pos, side_len) &&
      WriteFile(path.c_str(), data.d, side_pos);
  BufFree(&data);
  return ok;
}

bool RestoreEncodedTailFile(const std::string& path,
    const std::string& side_path) {
  ByteBuf input; input.d = 0; input.n = input.cap = 0;
  ByteBuf side; side.d = 0; side.n = side.cap = 0;
  LineVec lines; lines.d = 0; lines.n = lines.cap = 0;
  BlockVec sorted_blocks; sorted_blocks.d = 0;
  sorted_blocks.n = sorted_blocks.cap = 0;
  IdxVec d86_order; d86_order.d = 0; d86_order.n = d86_order.cap = 0;
  SideMeta meta;
  meta.tail_len = meta.r0_count = meta.r1_count = meta.r2_count = 0;
  meta.prelude_count = meta.suffix_count = 0;
  meta.sorted_to_original.d = 0;
  meta.sorted_to_original.n = meta.sorted_to_original.cap = 0;
  ByteBuf output; output.d = 0; output.n = output.cap = 0;
  const TailBlock** original_blocks = 0;
  bool ok = false;

  if (!ReadFile(path.c_str(), &input) || !ReadFile(side_path.c_str(), &side))
    goto done;
  if (!ParseSide(side.d, side.n, &meta)) goto done;
  if (meta.tail_len == 0 || meta.tail_len > input.n) {
    std::fprintf(stderr,
        "\nr1 encoded tail restore refused: stream=%zu tail_len=%zu\n",
        input.n, meta.tail_len);
    goto done;
  }
  g_tail_len = meta.tail_len;
  g_tail_start = input.n - g_tail_len;
  SplitLines(g_tail_start, g_tail_len, input.d, input.n, &lines);
  if (lines.n != meta.r0_count + meta.r1_count + meta.r2_count) goto done;
  if (meta.prelude_count + meta.suffix_count > meta.r1_count) goto done;

  {
    const size_t r1_first = meta.r0_count;
    const size_t middle_first = r1_first + meta.prelude_count;
    const size_t middle_count =
        meta.r1_count - meta.prelude_count - meta.suffix_count;
    size_t middle_prelude = 0, middle_suffix_first = 0, middle_suffix = 0;
    if (!ParseTailBlocks(input.d, input.n, lines.d + middle_first,
            middle_count, &middle_prelude, &sorted_blocks,
            &middle_suffix_first, &middle_suffix) ||
        middle_prelude != 0 || middle_suffix != 0 ||
        sorted_blocks.n != meta.sorted_to_original.n) {
      goto done;
    }

    original_blocks =
        (const TailBlock**)calloc(sorted_blocks.n ? sorted_blocks.n : 1,
            sizeof(const TailBlock*));
    if (!original_blocks) { std::fprintf(stderr, "\nr1 out of memory\n"); abort(); }
    if (!ComputeD86aOrder(input.d, input.n, lines.d + middle_first,
            sorted_blocks.d, sorted_blocks.n, &d86_order)) {
      goto done;
    }
    for (size_t d86_pos = 0; d86_pos < d86_order.n; ++d86_pos) {
      const size_t sorted_pos = d86_order.d[d86_pos];
      const size_t original_index = meta.sorted_to_original.d[d86_pos];
      if (original_index >= sorted_blocks.n ||
          original_blocks[original_index]) {
        goto done;
      }
      original_blocks[original_index] = &sorted_blocks.d[sorted_pos];
    }

    BufReserve(&output, input.n);
    BufAppend(&output, input.d, g_tail_start);
    AppendLines(&output, input.d, lines.d, 0, meta.r0_count);
    AppendLines(&output, input.d, lines.d + r1_first, 0, meta.prelude_count);
    for (size_t i = 0; i < sorted_blocks.n; ++i) {
      const TailBlock* blk = original_blocks[i];
      if (!blk) goto done;
      AppendLines(&output, input.d, lines.d + middle_first, blk->first,
          blk->count);
    }
    AppendLines(&output, input.d, lines.d + r1_first,
        meta.r1_count - meta.suffix_count, meta.suffix_count);
    AppendLines(&output, input.d, lines.d, r1_first + meta.r1_count,
        meta.r2_count);
    if (output.n != input.n || !WriteFile(path.c_str(), output.d, output.n))
      goto done;

    (void)0;
    ok = true;
  }

done:
  free(original_blocks);
  BufFree(&output);
  IdxFree(&d86_order);
  IdxFree(&meta.sorted_to_original);
  BlockFree(&sorted_blocks);
  LineFree(&lines);
  BufFree(&side);
  BufFree(&input);
  return ok;
}

}  // namespace r1_reorder
