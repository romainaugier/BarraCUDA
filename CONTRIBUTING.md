# Contributing to BarraCUDA

You're more than welcome to submit a PR. I'm happy to look at it and give it a fair shake.

If you're reading this and going "Ah heck I don't think I can do that, I'll get it wrong", that's perfectly fine. Submit the PR anyway and I am always happy to guide and assist. I am always learning myself.

## The Style

BarraCUDA is written in a defensive C99 style. I spent too much time staring at NASA code in my Halmat project and it stuck. The rules exist because they eliminate entire categories of bugs by construction rather than by testing.

**No dynamic allocation in hot paths.** Pre-allocated, fixed-size buffers. If a pool overflows, return a sentinel, never corrupt a counter. If you can't have unbounded allocation you can't have memory leaks.

**No recursion.** Every function call is a known-depth call. Stack usage is predictable. If you can't recurse you can't blow the stack.

**All loops must be bounded.** If you're iterating to a fixpoint, there's a guard counter. No infinite loops, no "this should always converge." If every loop is bounded you can't hang.

**Bounds-check array accesses** from external or untrusted indices. Trust internal bookkeeping, verify everything else.

**Stack-allocated where possible.** Deterministic behaviour, deterministic cleanup.

**Strict error checking.** Check return values. Handle the failure path.

**No floats where integers will do.** If you're comparing ratios, cross-multiply. Floating point is for GPU shader maths, not compiler internals.

### Naming

Function and variable names are short, 4-7 characters. Think of it like reading a motorway sign at 100km/h, you want "SH1 NORTH" not "STATE_HIGHWAY_ONE_NORTHBOUND_DIRECTION". When you're reading a thousand lines of instruction selector at 2am you want `ra_gc` not `regalloc_graphcolor`. Look at the newer code for the pattern: `isel_emit`, `mk_hash`, `enc_vop3`, `xt_meta`, `dce_copy`.

Some older code still uses longer names from when I wanted things readable for reviewers on Reddit. That's changed now. If you're touching a file and spot a verbose name, feel free to shorten it.

### Comments

Comments explain the *why*, not the *what*. Any C programmer can read the code and understand what it does. The comments are there to explain intent, context, and the reasoning behind non-obvious decisions.

Section headers look like this:
```c
/* ---- Section Name ---- */
```

Humour is welcome and encouraged. You're welcome to add your own personality and wit to anything you write. Arrogance is not welcome. Self-deprecating dry wit is the house style but it's not mandatory.

When refactoring or moving code, please keep existing comments with it. They took thought to write.

## Where to Help

Check `Issues` for current priorities. In general the most impactful areas are:

**Language features** that real CUDA code needs, things the parser or sema doesn't handle yet. If you've got a .cu file that doesn't compile, that's a useful bug report even if you don't have a fix.

**Backend work.** New architecture targets are always interesting. The compiler is designed for this, BIR is backend-agnostic and each target is self-contained. If you're a deep tech startup and need CUDA support for your hardware, reach out.

**Test cases.** Real CUDA kernels that break things are genuinely valuable. The weirder the better.

**Optimisation passes.** DCE, constant folding, and instruction scheduling already exist. If you want to add something like loop unrolling or better spill heuristics, open an issue first so we can chat about the approach.

If you're not sure whether something is worth a PR, open an issue. I also love to, as we say here in New Zealand, spin a few yarns. A quick conversation up front saves everyone time.

## GPU Targets

BarraCUDA currently supports:

- **CDNA 2** (gfx90a, MI250)
- **CDNA 3** (gfx942, MI300X)
- **RDNA 2** (gfx1030 family)
- **RDNA 3** (gfx1100 family, gfx1150 family)
- **RDNA 4** (gfx1200 family)

The frontend lowers to BIR (BarraCUDA IR) in SSA form. Each backend is a self-contained target that consumes BIR. Adding a new GPU architecture means writing a new instruction selector and emitter, the rest of the pipeline is shared.

## Building & Testing

```bash
# Build
make

# Run the test suite
make test

# Run the emulator test suite (RDNA3, requires tinygrad mockgpu in WSL)
python tests/emu/run_emu.py
```

Verify your changes don't introduce encoding regressions:
```bash
llvm-objdump -d --mcpu=gfx1100 output.hsaco
# Zero decode failures = good
```

## License

BarraCUDA is Apache 2.0. By submitting a PR, you agree your contribution is licensed under the same terms.

---

You've read this long, here's your prize, the island across the harbour from my house!

![Rangitoto](docs/harbour.png)
