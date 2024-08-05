//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_wave_rhs.cpp
//  \brief Calculate wave equation RHS

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/finite_differencing.hpp"
#include "wave.hpp"

// test
#include "../utils/interp_barycentric.hpp"

//-----------------------------------------------------------------------------

//! \fn void Wave::WaveRHS
//  \brief Calculate RHS for the wave equation using finite-differencing
void Wave::WaveRHS(AthenaArray<Real> & u)
{
  MeshBlock *pmb = pmy_block;

  AthenaArray<Real> wu, wpi;
  // internal dimension inferred
  wu.InitWithShallowSlice(u, 0, 1);
  wpi.InitWithShallowSlice(u, 1, 1);

  Real c_2 = SQR(c);

  static const bool is_spherical_polar = std::strcmp(COORDINATE_SYSTEM,
                                                     "spherical_polar") == 0;

  const bool contains_origin = pmb->PointContained(0,0,0);
  bool regularization_needed = false;
  int regularization_ix = -1;

  for(int i = mbi.il; i <= mbi.iu; ++i)
  {
    if (mbi.x1(i) == 0)
    {
      regularization_needed = true;
      regularization_ix = i;
      break;
    }
  }

  for(int k = mbi.kl; k <= mbi.ku; ++k)
  for(int j = mbi.jl; j <= mbi.ju; ++j)
  {
    #pragma omp simd
    for(int i = mbi.il; i <= mbi.iu; ++i)
    {
      rhs(0,k,j,i) = wpi(k,j,i);
      rhs(1,k,j,i) = 0.0;
    }

    if (is_spherical_polar)
    {
      // Laplacian in spherical coordinates:
      const Real sin_th = std::sin(mbi.x2(j));
      // const Real csc_th = 1.0 / sin_th;

      const Real cot_th = sin_th / std::cos(mbi.x2(j));

      switch (mbi.ndim)
      {
        case 3:
        {
          // phi
          #pragma omp simd
          for(int i = mbi.il; i <= mbi.iu; ++i)
          {
            const Real oo_r = 1.0 / mbi.x1(i);
            rhs(1,k,j,i) += POW2(oo_r) * fd->Dxx(2, wu(k,j,i));
          }
        }
        case 2:
        {
          // theta
          #pragma omp simd
          for(int i = mbi.il; i <= mbi.iu; ++i)
          {
            const Real oo_r = 1.0 / mbi.x1(i);

            rhs(1,k,j,i) += POW2(oo_r) * (
              fd->Dxx(1, wu(k,j,i)) +
              fd->Dx( 1, wu(k,j,i)) * cot_th
            );
          }

        }
        case 1:
        {
          // radial part
          #pragma omp simd
          for(int i = mbi.il; i <= mbi.iu; ++i)
          {
            const Real oo_r = 1.0 / mbi.x1(i);
            rhs(1,k,j,i) += (fd->Dxx(0, wu(k,j,i)) +
                              2.0 * oo_r * fd->Dx(0, wu(k,j,i)));
          }

          break;
        }
        default:
        {
          assert(false);
        }
      }


      #pragma omp simd
      for(int i = mbi.il; i <= mbi.iu; ++i)
      {
        rhs(1,k,j,i) = c_2 * rhs(1,k,j,i);
      }

      if (regularization_needed)
      {
        const int RIX = regularization_ix;
        rhs(0,k,j,RIX) = 0;
        rhs(1,k,j,RIX) = 0;
      }
    }
    else
    {
      for(int a = 0; a < 3; ++a)
      {
        #pragma omp simd
        for(int i = mbi.il; i <= mbi.iu; ++i)
        {
          rhs(1,k,j,i) += c_2 * fd->Dxx(a, wu(k,j,i));
        }
      }
    }
  }

  /*
  // cx rat. der. test
  AthenaArray<Real> D1_wu;
  D1_wu.NewAthenaArray(pmb->ncells1);

  numprox::interpolation::Floater_Hormann::D1(
    &(pmb->pcoord->x1v(0)),
    &(wu(0,0,0,0)),
    &(D1_wu(0,0,0)),
    pmb->ncells1-1,
    16,
    0
  );

  numprox::interpolation::Floater_Hormann::D2(
    &(pmb->pcoord->x1v(0)),
    &(wu(0,0,0,0)),
    &(D1_wu(0,0,0)),
    &(rhs(1,0,0,0)),
    pmb->ncells1-1-2*0,
    16,
    0
  );
  */

  /*
  // vc rat. der. test
  AthenaArray<Real> D1_wu;
  D1_wu.NewAthenaArray(pmb->nverts1);

  numprox::interpolation::Floater_Hormann::D1(
    &(pmb->pcoord->x1f(0)),
    &(wu(0,0,0,0)),
    &(D1_wu(0,0,0)),
    pmb->nverts1-1,
    6,
    0
  );

  numprox::interpolation::Floater_Hormann::D2(
    &(pmb->pcoord->x1f(0)),
    &(wu(0,0,0,0)),
    &(D1_wu(0,0,0)),
    &(rhs(1,0,0,0)),
    pmb->nverts1-1-2*0,
    6,
    0
  );
  */

}