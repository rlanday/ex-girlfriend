#!/usr/bin/env python3
# cut_dpath.py - build guard for the encoder block.
#
# The linker script declares ~150 functions encoder-only; the packer cuts
# them out of .decomp_bin and fills the gap with zeros.  If one of them is in
# fact on the decompression path, the decompressor jumps into nothing, and
# only a full decompression would show it (SIGSEGV).
#
# A name comparison against a decoder-only build does not work: there the
# compiler inlines the static functions, their names disappear although the
# code is there, and the comparison would report nothing.  So the shipped
# build itself is examined.  Steps:
#   1. .encoder_only range from the linker map.
#   2. main and selfextract_decomp from the linker map.
#   3. entry of the argc==1 branch = block start of the call site of
#      selfextract_decomp (walk backwards to just past the last terminator).
#   3a. the prelude of main: every instruction of main from which that entry
#      is reachable inside main (backward reachability in the CFG of main).
#      archive9 runs it first on every start (progress clock, layout tag),
#      so its callees are roots as well.  The one exception is the argument
#      vector of comp9 (data, see below): the prelude reads it on purpose.
#      The rest of main (the branches behind argc==1) is free.
#   4. reachability from this entry and the prelude: CFG within main, then
#      the complete call graph over all reached functions.
#      Direct call/jmp targets AND address-taking are considered
#      (lea/mov $imm/rip-relative), so that function pointers do not slip through.
#   4a. Which mov/lea/push immediates are addresses (entry branch, 2026-11-09).
#      The image is linked -no-pie, so "mov $0x2950da,%edi" can be a function
#      pointer; but the same form carries plain numbers (a memset length
#      0x23fbfc), and every number between the image base and the end of
#      .bss falls into SOME input section.  Rule: an immediate that lies
#      strictly inside a text input section (file.o:(.text...)) is an address
#      only if a symbol of the map sits exactly there; code is entered at
#      symbols, never in the middle of a function (the tree has no computed
#      goto, the one construct that takes an inner code address).  Such inner
#      immediates are numbers: they are not followed, and they are listed in
#      the output so that a reader can check them.  Immediates on a section
#      start, on a symbol, in data, or in no input section at all are
#      followed as before.  (Before this rule a memset length that happened
#      to fall inside CmxRecodeMain made its callees 16 false duds:
#      builds/resm_b22dd10.)
#   5. Every reached function that lies in .encoder_only is a DUD.
#   6. Roots from .preinit_array/.init_array/.fini_array (they run in every
#      process: archive9 without arguments, every helper process, verify).
#      Destructors registered with __cxa_atexit are address-takings there and
#      thus already edges of the call graph.
#   7. Data closure: if reached code takes the address of a data object
#      (vtable, jump or function table), its 8-byte code pointers become
#      nodes, so virtual methods and table calls are seen.
# Identical code folding does not disturb the guard (it works on addresses),
# but makes the result build dependent: a folded copy can happen to lie in
# .text.  Therefore no COMDAT/STL instances belong in the linker script.
#
# LIMIT.  The method sees only what is in the linked image.  A call via an
# address computed at run time (dlsym or similar) would be invisible; there
# is none in the tree.  Rule 4a would miss the address of a label inside a
# function taken by an immediate (GNU computed goto, &&label); there is none
# in the tree either (grep 'goto \*' src), and such a target stays inside its
# own function anyway.  A plain number that happens to equal a function entry
# or to fall into a data object is still followed (false duds possible, never
# a missed one).
#
# USAGE
#   cut_dpath.py <image> <linker-map> [<linker-script>]
#   Return 0 = clean, 1 = dud found (with name list),
#             2 = tool/input error (also a build abort!)
#
# The linker map comes from -Wl,-Map=<file>; the binary is bit-identical with
# and without the map.

import bisect
import collections
import struct
import os
import re
import subprocess
import sys


def die(msg, rc=2):
    sys.stderr.write("cut_dpath: %s\n" % msg)
    sys.exit(rc)


# ---------------------------------------------------------------- linker map
def read_map(path):
    """read the lld map.  Returns
       out_sections : {name: (vma, size)}
       in_sections  : [(vma, size, "file.o:(.text.SYM)")]
       symbols      : [(vma, raw_name)]  - not demangled, as in the .o
    """
    out_sec, in_sec, syms = {}, [], []
    rx = re.compile(r'^\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+(\d+)\s(\s*)(.*)$')
    for line in open(path, errors='replace'):
        m = rx.match(line.rstrip('\n'))
        if not m:
            continue
        vma = int(m.group(1), 16)
        size = int(m.group(3), 16)
        indent = len(m.group(5))
        rest = m.group(6)
        if not rest:
            continue
        # lld indents in steps: 0 = output section, 8 = input section,
        # 16 = symbol.  Do NOT distinguish by the parenthesis - demangled
        # symbol names like "selfextract_decomp()" also have one.
        if indent == 0:
            out_sec[rest.strip()] = (vma, size)
        elif indent <= 8:
            in_sec.append((vma, size, rest.strip()))
        else:
            syms.append((vma, rest.strip()))
    return out_sec, in_sec, syms


# ---------------------------------------------------------------- disassembly
def disasm(binary):
    """objdump -d -> {func_start: name}, {address: (mnemo, operands)}"""
    try:
        p = subprocess.run(["objdump", "-d", "--no-show-raw-insn", binary],
                           capture_output=True, text=True)
    except FileNotFoundError:
        die("objdump not found - the build guard cannot run")
    if p.returncode != 0:
        die("objdump rc=%d: %s" % (p.returncode, p.stderr.strip()[:200]))
    ins = {}
    rx_i = re.compile(r'^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$')
    for line in p.stdout.splitlines():
        m = rx_i.match(line)
        if m:
            ins[int(m.group(1), 16)] = (m.group(2), m.group(3))
    if not ins:
        die("objdump returned no instructions (image without section headers?)")
    return ins


TERMINATORS = ('jmp', 'ret', 'retq', 'hlt', 'ud2')


def main():
    if len(sys.argv) < 3:
        die("Usage: cut_dpath.py <image> <linker-map> [<linker-script>]")
    binary, mappath = sys.argv[1], sys.argv[2]
    ldspath = sys.argv[3] if len(sys.argv) > 3 else None
    for f in (binary, mappath):
        if not os.path.isfile(f):
            die("missing: %s" % f)

    out_sec, in_sec, syms = read_map(mappath)
    if '.encoder_only' not in out_sec:
        die("the linker map has no section .encoder_only - wrong map "
            "or the linker script had no effect")
    ENC_LO, ENC_SZ = out_sec['.encoder_only']
    ENC_HI = ENC_LO + ENC_SZ

    # Address -> input section name (that is the unit that the
    # linker script addresses).
    in_sec.sort()
    sec_addr = [x[0] for x in in_sec]

    def sec_of(a):
        i = bisect.bisect_right(sec_addr, a) - 1
        if i < 0:
            return None
        vma, size, name = in_sec[i]
        if size and a >= vma + size:
            return None
        return name

    # Symbol table from the map (raw + demangled are both in it; we
    # need the raw names for main/selfextract_decomp).
    sym_by_name = {}
    for vma, n in syms:
        sym_by_name.setdefault(n, vma)

    def find_sym(*cands):
        for c in cands:
            if c in sym_by_name:
                return sym_by_name[c]
        return None

    main_a = find_sym('main')
    sxd_a = find_sym('_Z18selfextract_decompv', 'selfextract_decomp()')
    if main_a is None:
        die("main is not in the linker map")
    if sxd_a is None:
        die("selfextract_decomp is not in the linker map - without this "
            "anchor the argc==1 branch cannot be determined")

    ins = disasm(binary)
    order = sorted(ins)

    def nxt(a):
        i = bisect.bisect_right(order, a)
        return order[i] if i < len(order) else None

    def branch_target(ops):
        m = re.match(r'^(?:0x)?([0-9a-f]+)\s*', ops)
        return int(m.group(1), 16) if m else None

    # Rule 4a (see the header): an immediate strictly inside a text input
    # section is an address only if a symbol of the map sits exactly there.
    sym_addr = set(vma for vma, n in syms)
    imm_inner = {}                    # (site, value) -> text input section

    def imm_is_addr(v, at):
        i = bisect.bisect_right(sec_addr, v) - 1
        if i < 0:
            return True               # below every input section: as before
        vma, size, name = in_sec[i]
        if ':(.text' not in name:
            return True               # data or linker-made section: as before
        if v == vma or v >= vma + size:
            return True               # a section start, or in no section: as before
        if v in sym_addr:
            return True               # an entry inside a multi-symbol section
        imm_inner[(at, v)] = (vma, name)
        return False

    def imm_targets(mn, ops, at=None):
        """address-taking: mov $0x..., lea ...(%rip) # 0x... <sym>"""
        out = []
        # Only instructions that can materialise a pointer take an address;
        # a compare or mask with a large constant does not (a cmp immediate
        # in the XML model once fell into the encoder block by layout).
        if mn.startswith(('mov', 'lea', 'push')):
            for m in re.finditer(r'\$0x([0-9a-f]{5,})', ops):
                v = int(m.group(1), 16)
                if imm_is_addr(v, at):
                    out.append(v)
        m = re.search(r'#\s*(?:0x)?([0-9a-f]{5,})\s', ops + ' ')
        if m and mn in ('lea', 'mov', 'movq', 'movl'):
            out.append(int(m.group(1), 16))
        return out

    # --- entry of the argc==1 branch -------------------------------------
    call_sxd = [a for a in order
                if ins[a][0] == 'call' and branch_target(ins[a][1]) == sxd_a]
    if len(call_sxd) != 1:
        die("selfextract_decomp is called %d times (expected: exactly 1) - "
            "the anchor for the argc==1 branch is not unique"
            % len(call_sxd))
    entry = call_sxd[0]
    # Backwards to just past the last terminator = block start.
    i = order.index(entry)
    while i > 0:
        pmn = ins[order[i - 1]][0]
        if pmn.startswith(TERMINATORS) or (pmn[0] == 'j' and pmn != 'jmp'):
            break
        i -= 1
    dentry = order[i]

    # --- 0. prelude of main (step 3a) --------------------------------------
    # CFG of main within its input section; jumps leaving it are tail calls.
    mi = bisect.bisect_right(sec_addr, main_a) - 1
    if mi < 0:
        die("main is not in any input section of the map")
    MAIN_LO, MAIN_HI = main_a, in_sec[mi][0] + in_sec[mi][1]
    succ, pre_t, pre_ind = {}, [], []
    for a in order[bisect.bisect_left(order, MAIN_LO):bisect.bisect_left(order, MAIN_HI)]:
        mn, ops = ins[a]
        s = []
        if mn.startswith(('ret', 'hlt', 'ud2')):
            pass
        elif mn == 'jmp':
            t = branch_target(ops)
            if t is not None:
                s.append(t)
        elif mn[0] == 'j':
            t = branch_target(ops)
            if t is not None:
                s.append(t)
            s.append(nxt(a))
        else:
            s.append(nxt(a))
        succ[a] = [b for b in s if b is not None and MAIN_LO <= b < MAIN_HI]
    if dentry not in succ:
        die("the argc==1 entry 0x%x is not in main" % dentry)
    pred = collections.defaultdict(list)
    for a, l in succ.items():
        for b in l:
            pred[b].append(a)
    fw, st = set(), [dentry]                 # forward from the entry
    while st:
        a = st.pop()
        if a not in fw:
            fw.add(a)
            st.extend(succ.get(a, ()))
    prel, st = set(), [dentry]               # backward to the entry
    while st:
        a = st.pop()
        if a not in prel:
            prel.add(a)
            st.extend(pred.get(a, ()))
    prel -= fw
    for a in prel:
        mn, ops = ins[a]
        if mn == 'call' or mn[0] == 'j':
            t = branch_target(ops)
            if t is not None and not (MAIN_LO <= t < MAIN_HI):
                pre_t.append(t)
            elif t is None and '*' in ops:
                pre_ind.append(a)
        pre_t += imm_targets(mn, ops, a)

    # --- 1. CFG within main from dentry ----------------------------------
    seen_ins, stack, callees, indirect = set(), [dentry], set(), []
    while stack:
        a = stack.pop()
        if a in seen_ins or a not in ins:
            continue
        seen_ins.add(a)
        mn, ops = ins[a]
        for t in imm_targets(mn, ops, a):
            callees.add(t)
        if mn.startswith(('ret', 'hlt', 'ud2')):
            continue
        if mn == 'call':
            t = branch_target(ops)
            if t is not None:
                callees.add(t)
            elif '*' in ops:
                indirect.append(a)
            stack.append(nxt(a))
            continue
        if mn == 'jmp':
            t = branch_target(ops)
            if t is not None:
                stack.append(t)
            elif '*' in ops:
                indirect.append(a)
            continue
        if mn[0] == 'j':
            t = branch_target(ops)
            if t is not None:
                stack.append(t)
            stack.append(nxt(a))
            continue
        stack.append(nxt(a))

    # --- 2. call graph over all reached functions -------------------
    # Function boundaries from the input sections of the map.
    def func_start(a):
        i = bisect.bisect_right(sec_addr, a) - 1
        return in_sec[i][0] if i >= 0 else None

    def body(start):
        i = bisect.bisect_left(order, start)
        vma, size, _ = in_sec[bisect.bisect_right(sec_addr, start) - 1]
        end = vma + size if size else start + 1
        out = []
        while i < len(order) and order[i] < end:
            out.append(order[i])
            i += 1
        return out

    # --- 2a. data hull and initialisation roots (H1) ----------------------
    # Data objects (input sections outside the code ranges) -> the
    # function starts their 8-byte words point to.
    try:
        rs = subprocess.run(["readelf", "-SW", binary], capture_output=True, text=True)
    except FileNotFoundError:
        die("readelf not found - the initialisation/data hull cannot run")
    if rs.returncode != 0:
        die("readelf rc=%d" % rs.returncode)
    elf_secs = []
    for line in rs.stdout.splitlines():
        m = re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+\S+\s+(\S*)', line)
        if m:
            nm, typ, addr, off, size, flg = m.groups()
            elf_secs.append((nm, typ, int(addr, 16), int(off, 16), int(size, 16), flg))
    raw = open(binary, 'rb').read()

    # The fixed argument vector of comp9 (started without arguments) must lie
    # in the encoder block.  archive9 is started without arguments as well and
    # knows it is not the compressor only because that block is zeros there.
    # Placed anywhere else, archive9 would start as a compressor.  Found by its
    # text, not by a section name, so a renamed section cannot slip past; every
    # build carries the vector, so not finding it is an error as well.
    oa_text = b'-e\x00enwik9\x00archive9\x00'
    oa_n, k = 0, raw.find(oa_text)
    vec_rng = []                      # input sections holding the vector (data only)
    while k >= 0:
        va = None
        for nm, typ, a, o, z, flg in elf_secs:
            if typ != 'NOBITS' and 'A' in flg and o <= k < o + z:
                va = a + (k - o)
                break
        if va is not None:            # only what is loaded counts
            if not (ENC_LO <= va < ENC_HI):
                die("argument vector at vaddr 0x%x is not in .encoder_only "
                    "(0x%x..0x%x) - archive9 would compress"
                    % (va, ENC_LO, ENC_HI), rc=1)
            oa_n += 1
            vi = bisect.bisect_right(sec_addr, va) - 1
            if vi >= 0 and va < in_sec[vi][0] + in_sec[vi][1] and '(.rodata' in in_sec[vi][2]:
                vec_rng.append((in_sec[vi][0], in_sec[vi][0] + in_sec[vi][1]))
        k = raw.find(oa_text, k + 1)
    if oa_n == 0:
        die("argument vector of the compressor not found in the image", rc=1)
    print("cut_dpath: argument vector %d x in .encoder_only" % oa_n)

    # Prelude targets; the vector is the one thing in .encoder_only it may read.
    pre_vec = [t for t in pre_t if any(lo <= t < hi for lo, hi in vec_rng)]
    pre_t = [t for t in pre_t if not any(lo <= t < hi for lo, hi in vec_rng)]
    print("cut_dpath: prelude of main %d instructions (CFG from the argc==1 entry %d, "
          "main %d), %d targets, %d of them on the argument vector"
          % (len(prel), len(fw), len(succ), len(pre_t) + len(pre_vec), len(pre_vec)))
    if pre_ind:
        print("cut_dpath: NOTE %d indirect jumps in the prelude of main" % len(pre_ind))

    exec_rng = [(a, a + z) for nm, typ, a, o, z, flg in elf_secs if 'X' in flg and 'A' in flg]
    def is_code(v):
        return any(lo <= v < hi for lo, hi in exec_rng)
    data_ptrs = collections.defaultdict(set)
    init_roots = set()
    for nm, typ, a, o, z, flg in elf_secs:
        if 'A' not in flg or 'X' in flg or typ not in ('PROGBITS', 'INIT_ARRAY', 'FINI_ARRAY', 'PREINIT_ARRAY'):
            continue
        for k in range(0, z - 7, 8):
            v = struct.unpack_from('<Q', raw, o + k)[0]
            if not is_code(v):
                continue
            fs = func_start(v)
            if fs is None or fs != v:
                continue
            if nm in ('.init_array', '.preinit_array', '.fini_array'):
                init_roots.add(v)
            else:
                d = func_start(a + k)
                if d is not None:
                    data_ptrs[d].add(v)
    if not init_roots:
        die(".init_array/.fini_array without code pointers - image without section headers "
            "or wrong image; the initialisation roots are missing")

    reached, stack = set(), ([func_start(t) for t in callees] + [func_start(t) for t in pre_t]
                             + sorted(init_roots))
    data_seen = set()
    while stack:
        f = stack.pop()
        if f is None or f in reached:
            continue
        reached.add(f)
        for a in body(f):
            mn, ops = ins[a]
            tl = []
            if mn == 'call' or mn[0] == 'j':
                t = branch_target(ops)
                if t is not None:
                    tl.append(t)
            tl += imm_targets(mn, ops, a)
            for t in tl:
                s = func_start(t)
                if s is None:
                    continue
                if s in data_ptrs:                 # Datenobjekt: Codezeiger folgen
                    if s not in data_seen:
                        data_seen.add(s)
                        stack.extend(data_ptrs[s])
                    continue
                if s not in reached:
                    stack.append(s)

    # --- 3. duds ---------------------------------------------------
    blind = sorted(f for f in reached if ENC_LO <= f < ENC_HI)
    names = {}
    for f in blind:
        names[f] = sec_of(f) or ("0x%x" % f)

    print("cut_dpath: .encoder_only 0x%x..0x%x (%d B), %d functions reachable "
          "from the argc==1 entry 0x%x + prelude of main + %d init/fini roots "
          "(data hull: %d data objects with code pointers)"
          % (ENC_LO, ENC_HI, ENC_SZ, len(reached), dentry, len(init_roots), len(data_seen)))
    if indirect:
        print("cut_dpath: NOTE %d indirect jumps on the -d path; the "
              "address-taking check covers them, as long as the target address is in the "
              "image." % len(indirect))

    # Rule 4a: the immediates taken as numbers, for the reader of the log.
    # "walked" counts only the code the guard looked at (main from the
    # argc==1 entry, its prelude, and every reached function).
    if imm_inner:
        inner_enc = sum(1 for (at, v) in imm_inner if ENC_LO <= v < ENC_HI)
        print("cut_dpath: rule 4a: %d mov/lea/push immediates in the walked code lie inside a "
              "function body where no symbol sits (%d of them inside .encoder_only): taken as "
              "numbers, not followed:" % (len(imm_inner), inner_enc))
        for (at, v) in sorted(imm_inner, key=lambda k: (k[1], k[0] or 0)):
            vma, name = imm_inner[(at, v)]
            print("    at 0x%-8x $0x%x = %s+0x%x%s%s"
                  % (at or 0, v, name, v - vma,
                     "" if v in ins else " (not an instruction boundary)",
                     "  [.encoder_only]" if ENC_LO <= v < ENC_HI else ""))
    else:
        print("cut_dpath: rule 4a: no mov/lea/push immediate in the walked code lies inside a function body.")

    if not blind:
        print("cut_dpath: clean - no .encoder_only function on the -d path.")
        return 0

    print("")
    print("cut_dpath: *** %d DUDS *** - these functions are in the "
          "encoder block, but are needed by the DECOMPRESSOR:" % len(blind))
    for f in blind:
        print("    0x%-8x  %s" % (f, names[f]))
    print("")
    if ldspath and os.path.isfile(ldspath):
        want = set()
        for f in blind:
            n = names[f]
            m = re.search(r':\((\.[^)]+)\)$', n)
            if m:
                want.add(m.group(1))
        # A linker script line names up to three prefixes per symbol
        # (.text./.text.unlikely./.text.hot.) in ONE *( ... ); the symbol
        # name is compared, so that multi-pattern lines are found too.
        def sym_of(tok):
            m2 = re.match(r'\.text(?:\.unlikely|\.hot)?\.(.+)$', tok)
            return m2.group(1) if m2 else tok
        want_sym = {sym_of(w) for w in want}
        lines, hit = [], set()
        for l in open(ldspath):
            m2 = re.match(r'\s*\*\((.+)\)\s*$', l)
            if not m2:
                continue
            syms_l = {sym_of(t) for t in m2.group(1).split()}
            if syms_l & want_sym:
                lines.append(l)
                hit |= syms_l & want_sym
        print("cut_dpath: to DELETE from %s:" % ldspath)
        for l in lines:
            print("   " + l.rstrip())
        missing = want_sym - hit
        if missing:
            print("cut_dpath: NOT in the linker script (come from a "
                  "collection rule?): %s" % ", ".join(sorted(missing)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
