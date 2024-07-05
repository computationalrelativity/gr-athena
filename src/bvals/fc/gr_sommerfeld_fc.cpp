// C, C++ headers
#include <iostream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "bvals_fc.hpp"

void FaceCenteredBoundaryVariable::GRSommerfeldInnerX1(
    Real time, Real dt, int il, int jl, int ju, int kl, int ku, int ngh)
{
  FaceCenteredBoundaryVariable::OutflowInnerX1(
    time, dt, il, jl, ju, kl, ku, ngh
  );
  return;
}

void FaceCenteredBoundaryVariable::GRSommerfeldOuterX1(
    Real time, Real dt, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  FaceCenteredBoundaryVariable::OutflowOuterX1(
    time, dt, iu, jl, ju, kl, ku, ngh
  );
  return;
}

void FaceCenteredBoundaryVariable::GRSommerfeldInnerX2(
    Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh)
{
  FaceCenteredBoundaryVariable::OutflowInnerX2(
    time, dt, il, iu, jl, kl, ku, ngh
  );
  return;
}

void FaceCenteredBoundaryVariable::GRSommerfeldOuterX2(
    Real time, Real dt, int il, int iu, int ju, int kl, int ku, int ngh)
{
  FaceCenteredBoundaryVariable::OutflowOuterX2(
    time, dt, il, iu, ju, kl, ku, ngh
  );
  return;
}

void FaceCenteredBoundaryVariable::GRSommerfeldInnerX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ngh)
{
  FaceCenteredBoundaryVariable::OutflowInnerX3(
    time, dt, il, iu, jl, ju, kl, ngh
  );
  return;
}

void FaceCenteredBoundaryVariable::GRSommerfeldOuterX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int ku, int ngh)
{
  FaceCenteredBoundaryVariable::OutflowOuterX3(
    time, dt, il, iu, jl, ju, ku, ngh
  );
  return;
}

//
// :D
//