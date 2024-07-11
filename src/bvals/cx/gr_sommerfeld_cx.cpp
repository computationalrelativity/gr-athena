// C, C++ headers
#include <iostream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "bvals_cx.hpp"

void CellCenteredXBoundaryVariable::GRSommerfeldInnerX1(
    Real time, Real dt, int il, int jl, int ju, int kl, int ku, int ngh)
{
  CellCenteredXBoundaryVariable::ExtrapolateOutflowInnerX1(
    time, dt, il, jl, ju, kl, ku, ngh
  );
  return;
}

void CellCenteredXBoundaryVariable::GRSommerfeldOuterX1(
    Real time, Real dt, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  CellCenteredXBoundaryVariable::ExtrapolateOutflowOuterX1(
    time, dt, iu, jl, ju, kl, ku, ngh
  );
  return;
}

void CellCenteredXBoundaryVariable::GRSommerfeldInnerX2(
    Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh)
{
  CellCenteredXBoundaryVariable::ExtrapolateOutflowInnerX2(
    time, dt, il, iu, jl, kl, ku, ngh
  );
  return;
}

void CellCenteredXBoundaryVariable::GRSommerfeldOuterX2(
    Real time, Real dt, int il, int iu, int ju, int kl, int ku, int ngh)
{
  CellCenteredXBoundaryVariable::ExtrapolateOutflowOuterX2(
    time, dt, il, iu, ju, kl, ku, ngh
  );
  return;
}

void CellCenteredXBoundaryVariable::GRSommerfeldInnerX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ngh)
{
  CellCenteredXBoundaryVariable::ExtrapolateOutflowInnerX3(
    time, dt, il, iu, jl, ju, kl, ngh
  );
  return;
}

void CellCenteredXBoundaryVariable::GRSommerfeldOuterX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int ku, int ngh)
{
  CellCenteredXBoundaryVariable::ExtrapolateOutflowOuterX3(
    time, dt, il, iu, jl, ju, ku, ngh
  );
  return;
}

//
// :D
//