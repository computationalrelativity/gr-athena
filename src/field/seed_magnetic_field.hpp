#ifndef FIELD_SEED_MAGNETIC_FIELD_HPP_
#define FIELD_SEED_MAGNETIC_FIELD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file seed_magnetic_field.hpp
//  \brief CT-consistent magnetic field initialization from an edge-centred
//         vector potential via the discrete Stokes theorem.
//
//  The face-centred magnetic field is computed as
//
//    B^i_face = (1/A_face) \oint_{\partial face} A \cdot dl
//
//  which in Cartesian coordinates reduces to simple finite differences of the
//  vector potential evaluated at cell edges.  Because the discrete curl of any
//  discrete vector potential satisfies the discrete divergence identity
//  exactly, the resulting face-centred field has div(B) = 0 to machine
//  precision.
//
//  Usage:
//    The caller provides a callback that, given a coordinate position (x,y,z)
//    and interpolated primitive values (pressure, density), returns the three
//    components of the vector potential A = (Ax, Ay, Az).  Both pressure and
//    density are interpolated to the edge locations from the four nearest
//    cell-centred values using bilinear averaging.
//
//  Note:
//    This utility assumes Cartesian coordinates with uniform spacing within
//    each MeshBlock (which is always the case for gr-athena++ Cartesian).  The
//    coordinate face positions x1f, x2f, x3f and cell-centre positions x1v,
//    x2v, x3v are used directly.

#include <algorithm>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn SeedFaceBFromEdgePotential
//  \brief Initialise face-centred B from an edge-centred vector potential.
//
//  VectorPotentialFn signature:
//    void fn(Real x, Real y, Real z, Real p, Real rho,
//            Real &Ax, Real &Ay, Real &Az);
//
//  After setting the face fields, bcc is recomputed via
//  Field::CalculateCellCenteredField so that it is consistent with the new
//  face data.

template <typename VectorPotentialFn>
inline void SeedFaceBFromEdgePotential(MeshBlock* pmb,
                                       VectorPotentialFn A_func)
{
  Coordinates* pcoord = pmb->pcoord;
  Field* pfield       = pmb->pfield;
  Hydro* phydro       = pmb->phydro;

  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;

  // Mesh dimensionality flags
  const bool three_d = pmb->pmy_mesh->f3;  // k-dimension active
  const bool two_d   = pmb->pmy_mesh->f2;  // j-dimension active

  // --- Zero all face and cell-centred B arrays ---
  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();

  // ---------------------------------------------------------------------------
  // Allocate temporary edge-centred vector potential arrays.
  //
  // Edge locations (Cartesian):
  //   A_x on x1-edges: (x1v(i), x2f(j), x3f(k))
  //   A_y on x2-edges: (x1f(i), x2v(j), x3f(k))
  //   A_z on x3-edges: (x1f(i), x2f(j), x3v(k))
  //
  // We allocate over the full interior + one layer of boundary faces so that
  // the Stokes differences can reach all interior faces.
  // ---------------------------------------------------------------------------
  const int nk = (three_d) ? (ke - ks + 2) : 1;  // ke+1 - ks + 1 for 3D
  const int nj = (two_d) ? (je - js + 2) : 1;    // je+1 - js + 1 for 2D
  const int ni = ie - is + 2;                    // ie+1 - is + 1

  // Allocate all three at the maximum envelope: (nk, nj, ni)
  AthenaArray<Real> Ax, Ay, Az;
  Ax.NewAthenaArray(nk, nj, ni);
  Ay.NewAthenaArray(nk, nj, ni);
  Az.NewAthenaArray(nk, nj, ni);

  // ---------------------------------------------------------------------------
  // Bilinear interpolation of CC primitives to edge locations.
  //
  // Each edge type is shared by 4 neighbouring cell centres in the two
  // transverse directions.  For Cartesian with uniform spacing within a block
  // the bilinear weights are all 1/4.
  //
  //   x1-edge at (x1v(i), x2f(j), x3f(k)): average over (j-1,j) x (k-1,k)
  //   x2-edge at (x1f(i), x2v(j), x3f(k)): average over (i-1,i) x (k-1,k)
  //   x3-edge at (x1f(i), x2f(j), x3v(k)): average over (i-1,i) x (j-1,j)
  //
  // In 2D (no k-dim): k-1 averages collapse.  In 1D: j-1 also collapses.
  // ---------------------------------------------------------------------------

  const AthenaArray<Real>& prim = phydro->w;

  // --- Fill edge potentials ---

  // A_x at x1-edges: position (x1v(i), x2f(j), x3f(k))
  // k in [ks, ke+f3],  j in [js, je+f2],  i in [is, ie]
  for (int k = ks; k <= ke + (three_d ? 1 : 0); ++k)
  {
    for (int j = js; j <= je + (two_d ? 1 : 0); ++j)
    {
      for (int i = is; i <= ie; ++i)
      {
        Real p_e, rho_e;
        if (three_d && two_d)
        {
          p_e   = 0.25 * (prim(IPR, k - 1, j - 1, i) + prim(IPR, k - 1, j, i) +
                          prim(IPR, k, j - 1, i) + prim(IPR, k, j, i));
          rho_e = 0.25 * (prim(IDN, k - 1, j - 1, i) + prim(IDN, k - 1, j, i) +
                          prim(IDN, k, j - 1, i) + prim(IDN, k, j, i));
        }
        else if (two_d)
        {
          p_e   = 0.5 * (prim(IPR, k, j - 1, i) + prim(IPR, k, j, i));
          rho_e = 0.5 * (prim(IDN, k, j - 1, i) + prim(IDN, k, j, i));
        }
        else
        {
          p_e   = prim(IPR, k, j, i);
          rho_e = prim(IDN, k, j, i);
        }
        Real ax, ay, az;
        A_func(pcoord->x1v(i),
               pcoord->x2f(j),
               pcoord->x3f(k),
               p_e,
               rho_e,
               ax,
               ay,
               az);
        Ax(k - ks, j - js, i - is) = ax;
      }
    }
  }

  // A_y at x2-edges: position (x1f(i), x2v(j), x3f(k))
  // k in [ks, ke+f3],  j in [js, je],  i in [is, ie+1]
  for (int k = ks; k <= ke + (three_d ? 1 : 0); ++k)
  {
    for (int j = js; j <= je; ++j)
    {
      for (int i = is; i <= ie + 1; ++i)
      {
        Real p_e, rho_e;
        if (three_d)
        {
          p_e   = 0.25 * (prim(IPR, k - 1, j, i - 1) + prim(IPR, k - 1, j, i) +
                          prim(IPR, k, j, i - 1) + prim(IPR, k, j, i));
          rho_e = 0.25 * (prim(IDN, k - 1, j, i - 1) + prim(IDN, k - 1, j, i) +
                          prim(IDN, k, j, i - 1) + prim(IDN, k, j, i));
        }
        else
        {
          p_e   = 0.5 * (prim(IPR, k, j, i - 1) + prim(IPR, k, j, i));
          rho_e = 0.5 * (prim(IDN, k, j, i - 1) + prim(IDN, k, j, i));
        }
        Real ax, ay, az;
        A_func(pcoord->x1f(i),
               pcoord->x2v(j),
               pcoord->x3f(k),
               p_e,
               rho_e,
               ax,
               ay,
               az);
        Ay(k - ks, j - js, i - is) = ay;
      }
    }
  }

  // A_z at x3-edges: position (x1f(i), x2f(j), x3v(k))
  // k in [ks, ke],  j in [js, je+f2],  i in [is, ie+1]
  for (int k = ks; k <= ke; ++k)
  {
    for (int j = js; j <= je + (two_d ? 1 : 0); ++j)
    {
      for (int i = is; i <= ie + 1; ++i)
      {
        Real p_e, rho_e;
        if (two_d)
        {
          p_e   = 0.25 * (prim(IPR, k, j - 1, i - 1) + prim(IPR, k, j - 1, i) +
                          prim(IPR, k, j, i - 1) + prim(IPR, k, j, i));
          rho_e = 0.25 * (prim(IDN, k, j - 1, i - 1) + prim(IDN, k, j - 1, i) +
                          prim(IDN, k, j, i - 1) + prim(IDN, k, j, i));
        }
        else
        {
          p_e   = 0.5 * (prim(IPR, k, j, i - 1) + prim(IPR, k, j, i));
          rho_e = 0.5 * (prim(IDN, k, j, i - 1) + prim(IDN, k, j, i));
        }
        Real ax, ay, az;
        A_func(pcoord->x1f(i),
               pcoord->x2f(j),
               pcoord->x3v(k),
               p_e,
               rho_e,
               ax,
               ay,
               az);
        Az(k - ks, j - js, i - is) = az;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Compute face-centred B via discrete Stokes theorem (Cartesian).
  //
  //   b.x1f(k,j,i) = dAz/dy - dAy/dz
  //   b.x2f(k,j,i) = dAx/dz - dAz/dx
  //   b.x3f(k,j,i) = dAy/dx - dAx/dy
  // ---------------------------------------------------------------------------

  // b.x1f: i in [is, ie+1], j in [js, je], k in [ks, ke]
  for (int k = ks; k <= ke; ++k)
  {
    for (int j = js; j <= je; ++j)
    {
      for (int i = is; i <= ie + 1; ++i)
      {
        Real val = 0.0;
        if (two_d)
        {
          const Real dy = pcoord->dx2f(j);
          val +=
            (Az(k - ks, j + 1 - js, i - is) - Az(k - ks, j - js, i - is)) / dy;
        }
        if (three_d)
        {
          const Real dz = pcoord->dx3f(k);
          val -=
            (Ay(k + 1 - ks, j - js, i - is) - Ay(k - ks, j - js, i - is)) / dz;
        }
        pfield->b.x1f(k, j, i) = val;
      }
    }
  }

  // b.x2f: i in [is, ie], j in [js, je+f2], k in [ks, ke]
  for (int k = ks; k <= ke; ++k)
  {
    for (int j = js; j <= je + (two_d ? 1 : 0); ++j)
    {
      for (int i = is; i <= ie; ++i)
      {
        Real val = 0.0;
        if (three_d)
        {
          const Real dz = pcoord->dx3f(k);
          val +=
            (Ax(k + 1 - ks, j - js, i - is) - Ax(k - ks, j - js, i - is)) / dz;
        }
        {
          const Real dx = pcoord->dx1f(i);
          val -=
            (Az(k - ks, j - js, i + 1 - is) - Az(k - ks, j - js, i - is)) / dx;
        }
        pfield->b.x2f(k, j, i) = val;
      }
    }
  }

  // b.x3f: i in [is, ie], j in [js, je], k in [ks, ke+f3]
  for (int k = ks; k <= ke + (three_d ? 1 : 0); ++k)
  {
    for (int j = js; j <= je; ++j)
    {
      for (int i = is; i <= ie; ++i)
      {
        Real val = 0.0;
        {
          const Real dx = pcoord->dx1f(i);
          val +=
            (Ay(k - ks, j - js, i + 1 - is) - Ay(k - ks, j - js, i - is)) / dx;
        }
        if (two_d)
        {
          const Real dy = pcoord->dx2f(j);
          val -=
            (Ax(k - ks, j + 1 - js, i - is) - Ax(k - ks, j - js, i - is)) / dy;
        }
        pfield->b.x3f(k, j, i) = val;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Derive cell-centred B from the face-centred field.
  // ---------------------------------------------------------------------------
  const int il = 0, iu = pmb->ncells1 - 1;
  const int jl = 0, ju = pmb->ncells2 - 1;
  const int kl = 0, ku = pmb->ncells3 - 1;
  pfield->CalculateCellCenteredField(
    pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl, ku);
}

#endif  // FIELD_SEED_MAGNETIC_FIELD_HPP_
