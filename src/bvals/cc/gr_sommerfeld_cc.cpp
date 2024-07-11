// C, C++ headers
#include <iostream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "bvals_cc.hpp"

void CellCenteredBoundaryVariable::GRSommerfeldInnerX1(
    Real time, Real dt, int il, int jl, int ju, int kl, int ku, int ngh)
{
  CellCenteredBoundaryVariable::OutflowInnerX1(
    time, dt, il, jl, ju, kl, ku, ngh
  );
  return;
}

void CellCenteredBoundaryVariable::GRSommerfeldOuterX1(
    Real time, Real dt, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  CellCenteredBoundaryVariable::OutflowOuterX1(
    time, dt, iu, jl, ju, kl, ku, ngh
  );
  return;
}

void CellCenteredBoundaryVariable::GRSommerfeldInnerX2(
    Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh)
{
  CellCenteredBoundaryVariable::OutflowInnerX2(
    time, dt, il, iu, jl, kl, ku, ngh
  );
  return;
}

void CellCenteredBoundaryVariable::GRSommerfeldOuterX2(
    Real time, Real dt, int il, int iu, int ju, int kl, int ku, int ngh)
{
  CellCenteredBoundaryVariable::OutflowOuterX2(
    time, dt, il, iu, ju, kl, ku, ngh
  );
  return;
}

void CellCenteredBoundaryVariable::GRSommerfeldInnerX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ngh)
{
  CellCenteredBoundaryVariable::OutflowInnerX3(
    time, dt, il, iu, jl, ju, kl, ngh
  );
  return;
}

void CellCenteredBoundaryVariable::GRSommerfeldOuterX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int ku, int ngh)
{
  CellCenteredBoundaryVariable::OutflowOuterX3(
    time, dt, il, iu, jl, ju, ku, ngh
  );
  return;
}

//
// :D
//