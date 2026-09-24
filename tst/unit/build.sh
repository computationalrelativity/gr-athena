#!/bin/bash
# Build the standalone EOS unit drivers against src/z4c/primitive/.
set -e
cd "$(dirname "$0")"
SRC=../../src
OUT=${OUT:-.build}
mkdir -p "$OUT"
for f in eos_compose eos_eir eos_transition ps_error reset_floor_transition; do
  g++ -std=c++17 -O2 -g -I"$SRC" -c "$SRC/z4c/primitive/$f.cpp" -o "$OUT/$f.o"
done
for t in test_eos_entropy_inversion; do
  g++ -std=c++17 -O2 -g -I"$SRC" "$t.cpp" "$OUT"/*.o -lhdf5 -lhdf5_hl -o "$t"
done
