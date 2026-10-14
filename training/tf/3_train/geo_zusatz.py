#!/usr/bin/env python3
"""Add a missing window geometry to tft/param_count.py ("Zusatz" = addition).

The new entry is a copy of an existing variant of the same base (or of the
base itself) in which only window_size changes; no tensor shape depends on
it.  Example: poc8m_w2560 is derived from the first poc8m_w<N> entry.

Usage:   python3 geo_zusatz.py <geometry name> [path to param_count.py]
Returns  0 = present or added, 2 = cannot be derived, 3 = write error.
The original file is kept as <path>.vor_<name> ("vor" = before).
"""
# German names: hauptteil = main part, quelle = source text, treffer = match,
# basis = base name, fenster = window, vorlage = template entry,
# tiefe = bracket depth, neu = new block, hinweis = note, kandidat = candidate.
import ast
import os
import re
import shutil
import sys


def hauptteil(name, pfad):
    quelle = open(pfad).read()
    if 'name="%s"' % name in quelle:
        print("geometry %s is already present" % name)
        return 0

    treffer = re.fullmatch(r"(.+)_w(\d+)", name)
    if not treffer:
        print("CANNOT DERIVE: %s has no _w<window> suffix" % name)
        return 2
    basis, fenster = treffer.group(1), int(treffer.group(2))

    vorlagen = re.findall(r'name="(%s_w\d+)"' % re.escape(basis), quelle)
    vorlage = vorlagen[0] if vorlagen else basis
    if 'name="%s"' % vorlage not in quelle:
        print("CANNOT DERIVE: neither a window variant nor the base %s" % basis)
        return 2

    # Cut out the _add(...) block of the template by counting brackets.
    i = quelle.index('name="%s"' % vorlage)
    start = quelle.rindex("_add(", 0, i)
    tiefe, j = 0, start
    while j < len(quelle):
        if quelle[j] == "(":
            tiefe += 1
        elif quelle[j] == ")":
            tiefe -= 1
            if tiefe == 0:
                break
        j += 1
    block = quelle[start:j + 1]

    neu = block.replace('name="%s"' % vorlage, 'name="%s"' % name)
    neu, anzahl = re.subn(r"window_size=\d+", "window_size=%d" % fenster, neu)
    if anzahl != 1:
        print("CANNOT DERIVE: %d window_size lines in the block" % anzahl)
        return 2

    hinweis = "# derived at setup from %s (window_size only)" % vorlage
    kandidat = quelle.rstrip("\n") + "\n\n\n" + hinweis + "\n" + neu + "\n"
    try:
        ast.parse(kandidat)          # check first, then write
    except SyntaxError as fehler:
        print("ABORT: the result would not be valid Python (%s)" % fehler)
        return 3
    shutil.copy(pfad, pfad + ".vor_" + name)
    open(pfad, "w").write(kandidat)
    print("geometry %s derived from %s (window=%d)" % (name, vorlage, fenster))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: geo_zusatz.py <name> [param_count.py]")
        sys.exit(2)
    ziel = sys.argv[2] if len(sys.argv) > 2 else "tft/param_count.py"
    if not os.path.isfile(ziel):
        print("param_count.py not found: %s" % ziel)
        sys.exit(3)
    sys.exit(hauptteil(sys.argv[1], ziel))
