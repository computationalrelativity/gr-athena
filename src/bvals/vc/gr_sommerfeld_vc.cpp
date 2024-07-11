// C, C++ headers
#include <iostream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "bvals_vc.hpp"

void VertexCenteredBoundaryVariable::GRSommerfeldInnerX1(
    Real time, Real dt, int il, int jl, int ju, int kl, int ku, int ngh)
{
  VertexCenteredBoundaryVariable::ExtrapolateOutflowInnerX1(
    time, dt, il, jl, ju, kl, ku, ngh
  );
  return;
}

void VertexCenteredBoundaryVariable::GRSommerfeldOuterX1(
    Real time, Real dt, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  VertexCenteredBoundaryVariable::ExtrapolateOutflowOuterX1(
    time, dt, iu, jl, ju, kl, ku, ngh
  );
  return;
}

void VertexCenteredBoundaryVariable::GRSommerfeldInnerX2(
    Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh)
{
  VertexCenteredBoundaryVariable::ExtrapolateOutflowInnerX2(
    time, dt, il, iu, jl, kl, ku, ngh
  );
  return;
}

void VertexCenteredBoundaryVariable::GRSommerfeldOuterX2(
    Real time, Real dt, int il, int iu, int ju, int kl, int ku, int ngh)
{
  VertexCenteredBoundaryVariable::ExtrapolateOutflowOuterX2(
    time, dt, il, iu, ju, kl, ku, ngh
  );
  return;
}

void VertexCenteredBoundaryVariable::GRSommerfeldInnerX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ngh)
{
  VertexCenteredBoundaryVariable::ExtrapolateOutflowInnerX3(
    time, dt, il, iu, jl, ju, kl, ngh
  );
  return;
}

void VertexCenteredBoundaryVariable::GRSommerfeldOuterX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int ku, int ngh)
{
  VertexCenteredBoundaryVariable::ExtrapolateOutflowOuterX3(
    time, dt, il, iu, jl, ju, ku, ngh
  );
  return;
}

//
// :D
//