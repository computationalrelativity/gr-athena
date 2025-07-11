#ifndef M1_CALC_EQUILIBRIUM_HPP
#define M1_CALC_EQUILIBRIUM_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_closure.hpp"
#include "m1_calc_update.hpp"

// ============================================================================
namespace M1::Equilibrium {
// ============================================================================

void SetEquilibrium(
  M1 & pm1,
  M1::vars_Lab & U_C,
  M1::vars_Source & U_S,
  const int k,
  const int j,
  const int i);

// Enforce equilibrium only on species using energy averages.
//
// - Takes internal sc_avg_nrg (should be populated during opacity calc.)
// - Takes current J; Optionally fiducial frame is reconstructed at the point
//   based on current (E, F_d)
// - Expression: sc_avg_nrg = J / n
/*
void SetEquilibrium_n_nG(
  M1 & pm1,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::SourceMetaVector & S,
  const int k,
  const int j,
  const int i,
  const bool construct_fiducial,
  const bool construct_src_nG,
  const bool construct_src_E_F_d,
  const bool use_diff_src
);
*/

/*
// Set also (E, F_d)
void SetEquilibrium_E_F_d_n_nG(
  M1 & pm1,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::SourceMetaVector & S,
  const int k,
  const int j,
  const int i,
  const bool construct_src_nG,
  const bool construct_src_E_F_d,
  const bool use_diff_src
);
*/

// Enforce equilibrium only on species using energy averages.
// - Suppose we have previous data U:=(N=nG, E, F_d).
// - Optionally perform initial (E*,F_d*) evolution with or without sources
// - Optionally constructs current J* based on current (E*, F_d*)
// - Set n* <- n using sc_avg_nrg = J* / n (computed during opac. calc.)
// - Set N* <- n* \times Gamma[U*]
//
// Optionally:
// - Construct new sources from state-vector difference S:=U*-U
// - Take final result as explicit evolution with this S

void SetEquilibrium_n_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL_C,
  const int k,
  const int j,
  const int i
);


// Set also (E, F_d) based on equilibrium considerations
void SetEquilibrium_E_F_d_n_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL_C,
  const int k,
  const int j,
  const int i
);

// Map reference eq (J,n) to state-vector; map also to Euler frame
void MapReferenceEquilibrium(
  M1 & pm1,
  M1::vars_Eql & eq,
  Update::StateMetaVector & C,
  const int k,
  const int j,
  const int i
);

// ============================================================================
} // namespace M1::Equilibrium
// ============================================================================


#endif // M1_CALC_EQUILIBRIUM_HPP

