# Particle Species Support
There are two ways particle species could be implemented in Athena++, each with its own advantages and disadvantages.
1. Adapt Athena's existing infrastructure for passive scalars.
2. Add additional hydro variables.

# Passive Scalars in Athena++
Athena includes support for passive scalars that can be advected around the grid like any other fluid quantity. They are enabled by running `configure.py` with the `--nscalars=N` flag. To evolve these scalars, they must also be enabled in the `MeshBlock::ProblemGenerator` function. See [https://github.com/PrincetonUniversity/athena/wiki/Passive-Scalars](the Athena++ wiki) for more details.

Scalars are stored separately from the other hydro variables; the primitive and conserved variables are stored on the `MeshBlock` as part of the `MeshBlock::phydro` object, but the scalars are stored in the `s`, `s1`, and `s2` variables (each representing a different register for the RK solver) for the conserved variables (i.e., `D*Y[i]`) and the `r` for the primitive variables (i.e., `Y[i]`) on the `MeshBlock` as part of the `MeshBlock::pscalars` object.

# Needed Changes
## Changes Needed Everywhere
1. In `src/eos/eostaudyn_ps_hydro_gr.cpp`, there are several spots marked with a `FIXME` that indicate where particle species support is missing. This includes `PrimitiveToConservedSingle`, `ConservedToPrimitive`, `SoundSpeedsGR`, and `ApplyPrimitiveFloors`. In particular, `SoundSpeedsGR` will need to be modified to accept a variable containing the primitive particle abundances.
2. The code makes several calls to `GetTemperatureFromP` and `GetEnthalpy`, which are `PrimitiveSolver` functions that depend on particle abundances, that are currently filled with empty dummy variables. Many, though probably not all, of these are marked with `FIXME` labels. The files that need to be updated are `src/coordinates/gr_dynamical.cpp`, `src/hydro/rsolvers/hydro/llftaudyn_rel_no_transform.cpp`, `src/hydro/rsolvers/hydro/hlletaudyn_rel_no_transform.cpp`, and `src/z4c/z4c.cpp`. Most of these should also be inside `USETM` preprocessor blocks.
3. Because the `SoundSpeedsGR` function arguments will change, the declaration in `src/eos/eos.hpp` will need to be modified, too. This redefinition should probably be wrapped in a `USETM` preprocessor block to avoid breaking code that doesn't use the `PrimitiveSolver` framework.
4. GR initial data/problem generators need to be generalized to deal with the presence of particle abundances.

## Changes needed only for passive scalars
1. Though you can access the passive scalars via the `MeshBlock` in all of the functions above, it's not always clear which RK stage or register of these variables is needed. Therefore, most, if not all, of these functions need to be modified to accept an array of the primitive and/or conserved scalar variables. This will also require changes in the task list where these functions are called.
2. When floors are applied to the conserved or primitive variables, these changes need to be copied back into the scalar arrays. However, the passive scalars have their own `ConservedToPrimitive` and `PrimitiveToConserved` routines, so the version of these variables calculated via `PrimitiveSolver` should *not* be copied back into the scalar arrays.

# Changes needed only for adding additional hydro variables
1. The `NHYDRO` variable defined in `athena.hpp` needs to be modified to include every additional particle species at compile-time. I recommend using the `MAX_SPECIES` variable for this. It should be configurable via `configure.py`.
2. The Riemann solvers, `llftaudyn_rel_no_transform.cpp` and `hlletaudyn_rel_no_transform.cpp`, need to be updated to include flux calculations for the additional particle species.
