# Unit drivers for the primitive EOS policies

Standalone programs that link the sources under `src/z4c/primitive/`
directly, with `Globals` stubbed, so an EOS change can be exercised
without building or running the full code.

They need the SFHo CompOSE table and the EIR electron table. `build.sh`
looks for them where the standalone primitive-solver repository keeps its
test data; point `EOS_TEST_DATA` elsewhere if yours lives somewhere else.

```
cd tst/unit && ./build.sh && ./test_eos_entropy_inversion
```
