#ifndef CONTEXT_MAP_COMPACT_H_
#define CONTEXT_MAP_COMPACT_H_

namespace cm_compact {

enum { KEEP_TIEFE = 4 };

// Index-independent check value: 11 bits from the top of a full avalanche
// over the 32-bit context.  Bits 0..7 = byte check value (SIMD compare),
// bits 8..10 = tag of the T forms.  Related to the checksum of the zmix
// build option `-Dcmc2-hash64` (origin zmix 1.0 / James Byrne,
// GPL-3.0-or-later, src/models/fxcm26/cm3.zig; idea and parameter of the
// first mixing stage taken); the second stage is the multiply-shift finalizer
// of MurmurHash3 (Austin Appleby, public domain); the cut to 8+3 bits is ours.
static inline U32 chk_hash(const U32 cx, const int i) {
    U32 h = (cx ^ (cx >> 15)) * 2654435761u + (U32)i;
    h ^= h >> 13;
    h *= 0x85ebca6bu;
    h ^= h >> 16;
    return h >> 21;   // 11 Bit
}
// Variant with a 12-bit check value (8 in the check byte, 4 tag bits).
static inline U32 chk_hash12(const U32 cx, const int i) {
    U32 h = (cx ^ (cx >> 15)) * 2654435761u + (U32)i;
    h ^= h >> 13;
    h *= 0x85ebca6bu;
    h ^= h >> 16;
    return h >> 20;   // 12 Bit
}

enum { CMK_VERBATIM12_MINBITS = 20 };
#define CMK_C0H12(c) ((unsigned)(((unsigned)(c) * 0x529u) & 0xFFFu))

enum { CMK_VERBATIM_MINBITS = 21 };

static inline U32 chk_v(const U32 cx, const int i, const int hashen) {
    return hashen ? chk_hash(cx, i) : (cx >> 21);   // 11 bits on both paths
}

template <const int A>
static inline unsigned FindChk8(const U8 *c, const U8 ch) {
    static_assert(A >= 1 && A <= 16, "FindChk8: A outside 1..16");
    const __m128i target = _mm_set1_epi8((char)ch);
    const __m128i v = _mm_loadu_si128((const __m128i*)c);
    unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(v, target));
    if (A < 16) m &= (1u << A) - 1u;
    return m;
}

template <const int SLOTS, const int TAGBITS>
union EKA {
    struct {
        U16 last;
        U8  chk[SLOTS];        // 2..
        U8  bh[SLOTS][7];      //          bit histories, 7 B per slot as before
        U16 tag3[3];           //          3 bit je Slot, 5 Slots je U16
    };
    U8 pad[128];
    enum { A = SLOTS };

    inline int tagget(const int s) const { return (tag3[s / 5] >> (3 * (s % 5))) & 7; }
    inline void tagset(const int s, const int v) {
        const int sh = 3 * (s % 5);
        U16 &w = tag3[s / 5];
        w = (U16)((w & (U16)(~(7u << sh))) | (U16)((U32)v << sh));
    }
    inline bool tagok(const int s, const int tg) const {
        return TAGBITS == 0 || tagget(s) == tg;
    }

    inline U8* get(U32 ch, int keep) {

        const U8  cb = (U8)(ch & 0xff);
        const int tg = TAGBITS ? (int)((ch >> 8) & 7) : 0;
        const int recent0 = last & 15;
        const int recent1 = (last >> 4) & 15;
        const int recent2 = (last >> 8) & 15;
        const int recent3 = (last >> 12) & 15;
        if (recent0 < A && chk[recent0] == cb && tagok(recent0, tg)) {

            return &bh[recent0][0];
        }
        unsigned m = FindChk8<SLOTS>(chk, cb);
        while (m) {
            const int hit = (int)__builtin_ctz(m);
            if (tagok(hit, tg)) {

                last = (U16)((last << 4) | hit);
                return &bh[hit][0];
            }
            m &= m - 1;
        }
        int b = 0xffff, bi = 0;
        for (int i = 0; i < A; ++i) {
            if (SLOT_FREE3(i)) {
                const int pri = bh[i][0];
                if (pri < b) b = pri, bi = i;
            }
        }

        last = (U16)((last << 4) | bi | keep);
        chk[bi] = cb;
        if (TAGBITS) tagset(bi, tg);
        return (U8*)memset(&bh[bi][0], 0, 7);
    }
};

union EKB {
    struct {
        U8 chk[16];      // 0..15
        U8 bh[16][7];    // 16..127
    };
    U8 pad[128];
    enum { A = 16 };
    inline U8* get(U32 ch, int keep) {

        (void)keep;
        const U8 cb = (U8)(ch & 0xff);
        const unsigned m = FindChk8<16>(chk, cb);
        if (m) {
            const int hit = (int)__builtin_ctz(m);

            return &bh[hit][0];
        }
        int b = 0xffff, bi = 0;
        for (int i = 0; i < 16; ++i) {
            const int pri = bh[i][0];
            if (pri < b) b = pri, bi = i;
        }

        chk[bi] = cb;
        return (U8*)memset(&bh[bi][0], 0, 7);
    }
};

union alignas(16) EKD {
    struct {
        U8 chk[16];      // bits 0..6 check value, bit 7 protection
        U8 bh[16][7];
    };
    U8 pad[128];
    enum { A = 16, KEEP = KEEP_TIEFE };
    inline U8* get(U32 ch, int keep) {

        (void)keep;
        const U8 cb = (U8)(ch & 0x7f);
        const __m128i mask7f = _mm_set1_epi8(0x7f);
        __m128i v = _mm_load_si128((const __m128i*)chk);
        const unsigned hitm = (unsigned)_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_and_si128(v, mask7f), _mm_set1_epi8((char)cb)));
        int sl;
        if (hitm) {
            sl = (int)__builtin_ctz(hitm);

        } else {
            const unsigned prot = (unsigned)_mm_movemask_epi8(v);
            int b = 0x10000, bi = -1;
            for (int i = 0; i < 16; ++i) {
                if ((prot >> i) & 1u) continue;
                const int pri = bh[i][0];
                if (pri < b) b = pri, bi = i;
            }
            if (bi < 0) {
                bi = 0;
                for (int i = 1; i < 16; ++i) if (bh[i][0] < bh[bi][0]) bi = i;
            }

            chk[bi] = cb;
            memset(&bh[bi][0], 0, 7);
            sl = bi;
            v = _mm_load_si128((const __m128i*)chk);
        }
        unsigned prot = (unsigned)_mm_movemask_epi8(v) | (1u << sl);
        if (__builtin_popcount(prot) > KEEP) {
            _mm_store_si128((__m128i*)chk, _mm_and_si128(v, mask7f));
        }
        chk[sl] = (U8)(chk[sl] | 0x80);
        return &bh[sl][0];
    }
};

union EKC {
    struct {
        U8 chk[4];      // 0..3
        U8 bh[4][7];    // 4..31
    };
    U8 pad[32];
    enum { A = 4 };
    inline U8* get(U32 ch, int keep) {

        (void)keep;
        const U8 cb = (U8)(ch & 0xff);
        int b = 0xffff, bi = 0;
        for (int i = 0; i < 4; ++i) {
            if (chk[i] == cb) {

                return &bh[i][0];
            }
            const int pri = bh[i][0];
            if (pri < b) b = pri, bi = i;
        }

        chk[bi] = cb;
        return (U8*)memset(&bh[bi][0], 0, 7);
    }
};

union EKE {
    struct {
        U8 chk[15];                 // 0..14
        U8 bh[15][7];               // 15..119
        unsigned long long aux;     // 120..127
    };
    U8 pad[128];
    enum { A = 15 };
    inline int tg4(const int s) const { return (int)((aux >> (4 + 4 * s)) & 15); }
    inline U8* get(U32 ch, int keep) {

        (void)keep;
        const U8  cb = (U8)(ch & 0xff);
        const int tg = (int)((ch >> 8) & 15);
        const int recent0 = (int)(aux & 15);
        if (recent0 < 15 && chk[recent0] == cb && tg4(recent0) == tg) {

            return &bh[recent0][0];
        }
        unsigned m = FindChk8<15>(chk, cb);
        while (m) {
            const int hit = (int)__builtin_ctz(m);
            if (tg4(hit) == tg) {

                aux = (aux & ~15ull) | (unsigned long long)hit;
                return &bh[hit][0];
            }
            m &= m - 1;
        }
        int b = 0xffff, bi = 0;
        for (int i = 0; i < 15; ++i) {
            if (i != recent0) {
                const int pri = bh[i][0];
                if (pri < b) b = pri, bi = i;
            }
        }

        chk[bi] = cb;
        const int sh = 4 + 4 * bi;
        aux = (aux & ~((15ull << sh) | 15ull)) | ((unsigned long long)tg << sh) | (unsigned long long)bi;
        return (U8*)memset(&bh[bi][0], 0, 7);
    }
};

template <int K0, int K25>
union EKM {
    enum { N = K0 + K25, A = N, QB = (N > 16 ? 5 : 4), INV = (1 << QB) - 1,
           AUXB = 128 - N - 5 * K0 - 7 * K25 };
    struct {
        U8 chk[N];
        U8 p0[K0 ? K0 : 1][5];
        U8 p25[K25][7];
        U8 aux[AUXB];
    };
    struct { U8 _o[N]; U8 bh[1][7]; };
    U8 pad[128];
    static_assert(3 * N + 4 * QB <= 8 * AUXB, "EKM: header does not fit");
    static_assert(N <= 31, "EKM: too many slots");
    typedef unsigned __int128 W; enum { WB = AUXB < 16 ? AUXB : 16 };   // head bits <= 113; form 15 has AUXB 24 > sizeof(W)
    inline W ld() const { W v = 0; memcpy(&v, aux, WB); return v; }
    inline void st(W v) { memcpy(aux, &v, WB); }
    inline unsigned bits(int off, int n) const { return (unsigned)(ld() >> off) & ((1u << n) - 1u); }
    inline void setbits(int off, int n, unsigned v) {
        const W m = (W)((1u << n) - 1u) << off;
        st((ld() & ~m) | ((W)v << off));
    }
    inline int tag(int s) const { return (int)bits(3 * s, 3); }
    inline int q(int k) const { return (int)bits(3 * N + QB * k, QB); }
    inline void push(int s, int poison) {
        const unsigned qm = (1u << (4 * QB)) - 1u;
        const unsigned alt = bits(3 * N, 4 * QB);
        const unsigned neu = poison ? ((qm & ~(unsigned)INV) | (unsigned)s)
                                    : (((alt << QB) | (unsigned)s) & qm);
        setbits(3 * N, 4 * QB, neu);
    }
    inline U8* slotp(int s) { return s < K0 ? &p0[s][0] : &p25[s - K0][0]; }
    template <int ART>
    __attribute__((noinline)) U8* get(U32 ch, int keep) {

        const int lo = ART ? K0 : 0, hi = ART ? N : K0;
        const U8 cb = (U8)(ch & 0xff);
        const int tg = (int)((ch >> 8) & 7);
        const int r0 = q(0);
        if (r0 >= lo && r0 < hi && chk[r0] == cb && tag(r0) == tg) {

            return slotp(r0);
        }
        const __m256i target = _mm256_set1_epi8((char)cb);
        const __m256i v = _mm256_loadu_si256((const __m256i*)chk);
        unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, target));
        m &= ((1u << hi) - 1u) & ~((1u << lo) - 1u);
        while (m) {
            const int hit = (int)__builtin_ctz(m);
            if (tag(hit) == tg) {

                push(hit, 0);
                return slotp(hit);
            }
            m &= m - 1;
        }
        const int r1 = q(1), r2 = q(2), r3 = q(3);
        int b = 0xffff, bi = lo;
        for (int s = lo; s < hi; ++s) {
            if (s == r0 || s == r1 || s == r2 || s == r3) continue;
            const int pri = slotp(s)[0];
            if (pri < b) b = pri, bi = s;
        }
        if (b == 0xffff) {
            for (int s = lo; s < hi; ++s) { const int pri = slotp(s)[0]; if (pri < b) b = pri, bi = s; }
        }

        chk[bi] = cb;
        setbits(3 * bi, 3, (unsigned)tg);
        push(bi, keep);
        return (U8*)memset(slotp(bi), 0, ART ? 7 : 5);
    }
};

template <int K0, int K25>
union EKMC {
    enum { N = K0 + K25, A = N, QB = 4, INV = 15, AUXB = 128 - 2 * N - 5 * K0 - 7 * K25 };
    struct {
        U8 chk[N];
        U8 ctr[N];
        U8 p0[K0][5];
        U8 p25[K25][7];
        U8 aux[AUXB];
    };
    struct { U8 _o[2 * N]; U8 bh[1][7]; };
    U8 pad[128];
    static_assert(3 * N + 4 * QB <= 8 * AUXB, "EKMC: header does not fit");
    static_assert(N <= 15, "EKMC: the queue holds 4-bit indices");
    typedef unsigned __int128 W;
    inline W ld() const { W v = 0; memcpy(&v, aux, AUXB); return v; }
    inline void st(W v) { memcpy(aux, &v, AUXB); }
    inline unsigned bits(int off, int n) const { return (unsigned)(ld() >> off) & ((1u << n) - 1u); }
    inline void setbits(int off, int n, unsigned v) {
        const W m = (W)((1u << n) - 1u) << off;
        st((ld() & ~m) | ((W)v << off));
    }
    inline int tag(int s) const { return (int)bits(3 * s, 3); }
    inline int q(int k) const { return (int)bits(3 * N + QB * k, QB); }
    inline void push(int s, int poison) {
        const unsigned qm = (1u << (4 * QB)) - 1u;
        const unsigned alt = bits(3 * N, 4 * QB);
        const unsigned neu = poison ? ((qm & ~(unsigned)INV) | (unsigned)s)
                                    : (((alt << QB) | (unsigned)s) & qm);
        setbits(3 * N, 4 * QB, neu);
    }
    inline void zaehle(int s) {
        if (ctr[s] == 255) { for (int k = 0; k < N; ++k) ctr[k] = (U8)(ctr[k] >> 1); }
        ctr[s] = (U8)(ctr[s] + 1);
    }
    inline U8* slotp(int s) { return s < K0 ? &p0[s][0] : &p25[s - K0][0]; }
    template <int ART>
    __attribute__((noinline)) U8* get(U32 ch, int keep) {

        const int lo = ART ? K0 : 0, hi = ART ? N : K0;
        const U8 cb = (U8)(ch & 0xff);
        const int tg = (int)((ch >> 8) & 7);
        const int r0 = q(0);
        if (r0 >= lo && r0 < hi && chk[r0] == cb && tag(r0) == tg) {

            zaehle(r0);
            return slotp(r0);
        }
        unsigned m = FindChk8<N>(chk, cb);
        m &= ((1u << hi) - 1u) & ~((1u << lo) - 1u);
        while (m) {
            const int hit = (int)__builtin_ctz(m);
            if (tag(hit) == tg) {

                zaehle(hit);
                push(hit, 0);
                return slotp(hit);
            }
            m &= m - 1;
        }
        const int r1 = q(1), r2 = q(2), r3 = q(3);
        int b = 0x7fffffff, bi = lo;
        for (int pass = 0; pass < 2 && b == 0x7fffffff; ++pass) {
            for (int s = lo; s < hi; ++s) {
                if (!pass && (s == r0 || s == r1 || s == r2 || s == r3)) continue;
                const int key = (int)ctr[s] * 256 + slotp(s)[0];
                if (key < b) b = key, bi = s;
            }
        }

        chk[bi] = cb;
        ctr[bi] = 1;
        setbits(3 * bi, 3, (unsigned)tg);
        push(bi, keep);
        return (U8*)memset(slotp(bi), 0, ART ? 7 : 5);
    }
};

template <int K0, int K25>
union EKMC6 {
    enum { N = K0 + K25, A = N, QB = 4, INV = 15, CMAX = 63,
           AUXB = 128 - N - 5 * K0 - 7 * K25 };
    struct {
        U8 chk[N];
        U8 p0[K0][5];
        U8 p25[K25][7];
        U8 aux[AUXB];      // 0..7: tags (3 bits per slot) and queue; 8..19: counters (6 bits per slot)
    };
    struct { U8 _o[N]; U8 bh[1][7]; };
    U8 pad[128];
    static_assert(N == 16, "EKMC6: layout for 16 slots");
    static_assert(AUXB >= 20, "EKMC6: header does not fit");
    typedef unsigned long long Q;
    typedef unsigned __int128 W;
    inline Q hd() const { Q v; memcpy(&v, aux, 8); return v; }
    inline void sethd(Q v) { memcpy(aux, &v, 8); }
    inline W cw() const { W v = 0; memcpy(&v, aux + 8, 12); return v; }
    inline void setcw(W v) { memcpy(aux + 8, &v, 12); }
    inline int tag(int s) const { return (int)((hd() >> (3 * s)) & 7); }
    inline int q(int k) const { return (int)((hd() >> (48 + 4 * k)) & 15); }
    inline void push(int s, int poison) {
        const Q h0 = hd();
        const Q alt = (h0 >> 48) & 0xffffull;
        const Q neu = poison ? (0xfff0ull | (Q)s) : (((alt << 4) | (Q)s) & 0xffffull);
        sethd((h0 & 0x0000ffffffffffffull) | (neu << 48));
    }
    inline void settag(int s, int tg) {
        const Q h0 = hd();
        sethd((h0 & ~(7ull << (3 * s))) | ((Q)tg << (3 * s)));
    }
    inline int ctr(W c, int s) const { return (int)((c >> (6 * s)) & 63); }
    inline void zaehle(int s) {
        W c = cw();
        if (ctr(c, s) == CMAX) {
            W r = 0;
            for (int k = 0; k < N; ++k) r |= (W)(ctr(c, k) >> 1) << (6 * k);
            c = r;
        }
        c += (W)1 << (6 * s);
        setcw(c);
    }
    inline void setctr(int s, int v) {
        W c = cw();
        c = (c & ~((W)63 << (6 * s))) | ((W)v << (6 * s));
        setcw(c);
    }
    inline U8* slotp(int s) { return s < K0 ? &p0[s][0] : &p25[s - K0][0]; }
    template <int ART>
    __attribute__((noinline)) U8* get(U32 ch, int keep) {

        const int lo = ART ? K0 : 0, hi = ART ? N : K0;
        const U8 cb = (U8)(ch & 0xff);
        const int tg = (int)((ch >> 8) & 7);
        const int r0 = q(0);
        if (r0 >= lo && r0 < hi && chk[r0] == cb && tag(r0) == tg) {

            zaehle(r0);
            return slotp(r0);
        }
        unsigned m = FindChk8<N>(chk, cb);
        m &= ((1u << hi) - 1u) & ~((1u << lo) - 1u);
        while (m) {
            const int hit = (int)__builtin_ctz(m);
            if (tag(hit) == tg) {

                zaehle(hit);
                push(hit, 0);
                return slotp(hit);
            }
            m &= m - 1;
        }
        const int r1 = q(1), r2 = q(2), r3 = q(3);
        const W c = cw();
        int b = 0x7fffffff, bi = lo;
        for (int pass = 0; pass < 2 && b == 0x7fffffff; ++pass) {
            for (int s = lo; s < hi; ++s) {
                if (!pass && (s == r0 || s == r1 || s == r2 || s == r3)) continue;
                const int key = ctr(c, s) * 256 + slotp(s)[0];
                if (key < b) b = key, bi = s;
            }
        }

        chk[bi] = cb;
        setctr(bi, 1);
        settag(bi, tg);
        push(bi, keep);
        return (U8*)memset(slotp(bi), 0, ART ? 7 : 5);
    }
};

}

#endif
