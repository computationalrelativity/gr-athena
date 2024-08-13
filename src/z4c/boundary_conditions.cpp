// C++ standard headers
#include <algorithm> // max
#include <cmath> // exp, pow, sqrt

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

//-----------------------------------------------------------------------------
// Compute the boundary RHS given the state vector and matter state
//
// This function operates only on a thin layer of points at the physical
// boundary of the domain.
void Z4c::Z4cBoundaryRHS(AA &u, AA &u_mat, AA &u_rhs)
{
  MeshBlock *pmb = pmy_block;
  BoundaryValues *pbval = pmy_block->pbval;

  if(pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.il,
                   mbi.jl, mbi.ju,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.iu, mbi.iu,
                   mbi.jl, mbi.ju,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.jl, mbi.jl,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.ju, mbi.ju,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::inner_x3] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.jl, mbi.ju,
                   mbi.kl, mbi.kl);
  }
  if(pbval->block_bcs[BoundaryFace::outer_x3] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.jl, mbi.ju,
                   mbi.ku, mbi.ku);
  }
}

// ----------------------------------------------------------------------------
// Apply Sommerfeld BCs to the given set of points
void Z4c::Z4cSommerfeld_(
  AA & u,
  AA & u_rhs,
  const int  is, const int ie,
  const int  js, const int je,
  const int  ks, const int ke)
{

  Z4c_vars z4c, rhs;
  SetZ4cAliases(u, z4c);
  SetZ4cAliases(u_rhs, rhs);

  for(int k = ks; k <= ke; ++k)
  for(int j = js; j <= je; ++j) {
    // ------------------------------------------------------------------------
    // 1st derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        dKhat_d(a,i) = fd->Ds(a, z4c.Khat(k,j,i));
        dTheta_d(a,i) = fd->Ds(a, z4c.Theta(k,j,i));
      }
    }
    // Vectors
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        dGam_du(b,a,i) = fd->Ds(b, z4c.Gam_u(a,k,j,i));
      }
    }
    // Tensors
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        dA_ddd(c,a,b,i) = fd->Ds(c, z4c.A_dd(a,b,k,j,i));
      }
    }

    // ------------------------------------------------------------------------
    // Compute pseudo-radial vector
    //
    #pragma omp simd
    for (int i = is; i <= ie; ++i)
    {
      r(i) = std::sqrt(SQR(mbi.x1(i)) + SQR(mbi.x2(j)) + SQR(mbi.x3(k)));
      s_u(0, i) = mbi.x1(i) / r(i);
      s_u(1, i) = mbi.x2(j) / r(i);
      s_u(2, i) = mbi.x3(k) / r(i);
    }

    // ------------------------------------------------------------------------
    // Boundary RHS for scalars
    //
    #pragma omp simd
    for (int i = is; i <= ie; ++i)
    {
      rhs.Theta(k, j, i) = -z4c.Theta(k, j, i) / r(i);
      rhs.Khat(k, j, i) = -std::sqrt(2.) * z4c.Khat(k, j, i) / r(i);
    }

    for (int a = 0; a < NDIM; ++a)
    {
      #pragma omp simd
      for (int i = is; i <= ie; ++i)
      {
        rhs.Theta(k, j, i) -= s_u(a, i) * dTheta_d(a, i);
        rhs.Khat(k, j, i) -= std::sqrt(2.) * s_u(a, i) * dKhat_d(a, i);
      }
    }

    // ------------------------------------------------------------------------
    // Boundary RHS for the Gamma's
    //
    for(int a = 0; a < NDIM; ++a) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        rhs.Gam_u(a,k,j,i) = - z4c.Gam_u(a,k,j,i)/r(i);
      }
      for(int b = 0; b < NDIM; ++b) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
          rhs.Gam_u(a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
        }
      }
    }

    // ------------------------------------------------------------------------
    // Boundary RHS for the A_ab
    //
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        rhs.A_dd(a,b,k,j,i) = - z4c.A_dd(a,b,k,j,i)/r(i);
      }
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
          rhs.A_dd(a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
        }
      }
    }
  }
}

//
// :D
//