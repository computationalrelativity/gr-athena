#include <cassert> // assert
#include <cmath> // abs, exp, sin, fmod
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../wave/wave.hpp"
#include "../trackers/extrema_tracker.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

namespace {

int RefinementCondition(MeshBlock *pmb);

Real wv_ph(const Real x1, const Real x2, const Real x3)
{
  // (r * sin(r)) * exp(-r)
  // return x1 * std::sin(x1) * std::exp(-x1);

  const Real a = 10;
  return (x1-a) * std::cos(x1-a) * std::exp(-POW2(x1-a));
  // return POW2(x1-a) * std::exp(-POW2(x1-a));
}


Real wv_pi(const Real x1, const Real x2, const Real x3)
{
  return 0;
}

void BCSInnerParityX1(MeshBlock *pmb,
                      Coordinates *pco,
                      Real time, Real dt,
                      int il, int iu,
                      int jl, int ju,
                      int kl, int ku, int ngh)
{
  // impose parity conditions (both fields even)
  AthenaArray<Real> Ph, Pi;

  Ph.InitWithShallowSlice(pmb->pwave->u, 0, 1);
  Pi.InitWithShallowSlice(pmb->pwave->u, 1, 1);


  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    for (int i=1; i<=ngh; ++i)
    {
      Ph(k,j,il-i) = Ph(k,j,il);
      Pi(k,j,il-i) = Pi(k,j,il);
    }
  }
}

} // namespace

//----------------------------------------------------------------------------------------

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BCSInnerParityX1);

  if(adaptive==true)
  {
    EnrollUserRefinementCondition(RefinementCondition);
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  int il = pwave->mbi.il, iu = pwave->mbi.iu;
  int kl = pwave->mbi.kl, ku = pwave->mbi.ku;
  int jl = pwave->mbi.jl, ju = pwave->mbi.ju;

  Real c = pwave->c;

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    Real x1 = pwave->mbi.x1(i);
    Real x2 = pwave->mbi.x2(j);
    Real x3 = pwave->mbi.x3(k);

    pwave->u(0,k,j,i) = wv_ph(x1,x2,x3);
    pwave->u(1,k,j,i) = wv_pi(x1,x2,x3);

    pwave->exact(k,j,i) = 0.0;
    pwave->error(k,j,i) = 0.0;
  }
}

void MeshBlock::WaveUserWorkInLoop()
{
  Real max_err = 0;
  Real fun_max = 0;

  Real c = pwave->c;

  const Real t_end = pmy_mesh->time + pmy_mesh->dt;

  for (int k=0; k<pwave->mbi.nn3; ++k)
  for (int j=0; j<pwave->mbi.nn2; ++j)
  for (int i=0; i<pwave->mbi.nn1; ++i)
  {
    const Real x1 = pwave->mbi.x1(i);
    const Real x2 = pwave->mbi.x2(j);
    const Real x3 = pwave->mbi.x3(k);

    const Real ct = c * t_end;

    // pwave->exact(k,j,i) = 0.0;
    // pwave->error(k,j,i) = pwave->u(0,k,j,i) - pwave->exact(k,j,i);

    // tracker reference field (full block- mock ref. field as communicated)
    pwave->ref_tra(k,j,i) = pwave->u(0,k,j,i);
  }
}

namespace {

int RefinementCondition(MeshBlock *pmb)
{
  Mesh * pmesh = pmb->pmy_mesh;
  ExtremaTracker * ptracker_extrema = pmesh->ptracker_extrema;

  int root_level = ptracker_extrema->root_level;
  int mb_physical_level = pmb->loc.level - root_level;


  // Iterate over refinement levels offered by trackers.
  //
  // By default if a point is not in any sphere, completely de-refine.
  int req_level = 0;

  for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained = false;
    int cur_req_level = ptracker_extrema->ref_level(n-1);

    {
      if (ptracker_extrema->ref_type(n-1) == 0)
      {
        is_contained = pmb->PointContained(
          ptracker_extrema->c_x1(n-1),
          ptracker_extrema->c_x2(n-1),
          ptracker_extrema->c_x3(n-1)
        );
      }
      else if (ptracker_extrema->ref_type(n-1) == 1)
      {
        is_contained = pmb->SphereIntersects(
          ptracker_extrema->c_x1(n-1),
          ptracker_extrema->c_x2(n-1),
          ptracker_extrema->c_x3(n-1),
          ptracker_extrema->ref_zone_radius(n-1)
        );
      }
    }

    if (is_contained)
    {
      req_level = std::max(cur_req_level, req_level);
    }

  }

  if (req_level > mb_physical_level)
  {
    return 1;  // currently too coarse, refine
  }
  else if (req_level == mb_physical_level)
  {
    return 0;  // level satisfied, do nothing
  }

  // otherwise de-refine
  return -1;

}

}