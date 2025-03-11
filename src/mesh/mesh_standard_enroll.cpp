//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

// C headers
// pre-C11: needed before including inttypes.h, else won't define int64_t for C++ code
// #define __STDC_FORMAT_MACROS

// C++ headers
#include <algorithm>
#include <cinttypes>  // format macro "PRId64" for fixed-width integer type std::int64_t
#include <cmath>      // std::abs(), std::pow()
#include <cstdint>    // std::int64_t fixed-wdith integer type alias
#include <cstdlib>
#include <cstring>    // std::memcpy()
#include <iomanip>    // std::setprecision()
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "mesh.hpp"

#include "../z4c/z4c.hpp"
#include "../m1/m1.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Enroll standard quantities used in multiple pgens --------------------------
#if FLUID_ENABLED
namespace {

Real max_rho(MeshBlock *pmb, int iout)
{
  Real max_rho_ = 0.0;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AA &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    max_rho_ = std::max(std::abs(w(IDN,k,j,i)), max_rho_);
  }

  return max_rho_;
}

Real max_T(MeshBlock *pmb, int iout)
{
  Real max_T = -std::numeric_limits<Real>::infinity();
  AA temperature;
  temperature.InitWithShallowSlice(pmb->phydro->derived_ms, IX_T, 1);

  CC_ILOOP3(k, j, i)
  {
    max_T = std::max(max_T, temperature(k,j,i));
  }
  return max_T;
}

Real num_c2p_fail(MeshBlock *pmb, int iout)
{
  Real sum_ = 0;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  // Reset the status
  AA c2p_status;
  c2p_status.InitWithShallowSlice(pmb->phydro->derived_ms, IX_C2P, 1);

  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    if (c2p_status(k,j,i) > 0)
      sum_++;
  }

  return sum_;
}

}
#endif // FLUID_ENABLED

void Mesh::EnrollUserStandardHydro()
{
#if FLUID_ENABLED
  EnrollUserHistoryOutput(max_rho, "max_rho",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(max_T, "max_T",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(num_c2p_fail, "num_c2p_fail",
                          UserHistoryOperation::max);
#endif // FLUID_ENABLED
}

// ----------------------------------------------------------------------------

#if MAGNETIC_FIELDS_ENABLED
namespace {

Real DivBface(MeshBlock *pmb, int iout)
{
  Field *pf = pmb->pfield;

  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += ((pf->b.x1f(k,j,i+1) - pf->b.x1f(k,j,i))/ dx +
                 (pf->b.x2f(k,j+1,i) - pf->b.x2f(k,j,i))/ dy +
                 (pf->b.x3f(k+1,j,i) - pf->b.x3f(k,j,i))/ dz) * vol;
      }
    }
  }
  return divB;
}

}
#endif // MAGNETIC_FIELDS_ENABLED

void Mesh::EnrollUserStandardField()
{
#if MAGNETIC_FIELDS_ENABLED
  EnrollUserHistoryOutput(DivBface, "div_B",
                          UserHistoryOperation::max);

#endif // MAGNETIC_FIELDS_ENABLED
}

// ----------------------------------------------------------------------------
#if Z4C_ENABLED
namespace {

Real min_alpha(MeshBlock *pmb, int iout)
{
  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_alpha = std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_alpha = std::min(alpha(k,j,i), m_alpha);
  }

  return m_alpha;
}

Real max_abs_con_H(MeshBlock *pmb, int iout)
{
  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca con_H(pmb->pz4c->storage.con, Z4c::I_CON_H);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_abs_con_H = -std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_abs_con_H = std::max(std::abs(con_H(k,j,i)), m_abs_con_H);
  }

  return m_abs_con_H;
}

}
#endif // Z4C_ENABLED

void Mesh::EnrollUserStandardZ4c()
{
#if Z4C_ENABLED
  EnrollUserHistoryOutput(min_alpha, "min_alpha",
                          UserHistoryOperation::min);
  EnrollUserHistoryOutput(max_abs_con_H, "max_abs_con.H",
                          UserHistoryOperation::max);
#endif // Z4C_ENABLED
}

// ----------------------------------------------------------------------------
#if M1_ENABLED
namespace {

Real max_sc_nG_00(MeshBlock *pmb, int iout)
{
  Real max_sc_nG_00 = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    // const Real oo_sc_sqrt_det_g = OO(pmb->pm1->geom.sc_sqrt_det_g(k,j,i));
    max_sc_nG_00 = std::max(max_sc_nG_00,
                            // oo_sc_sqrt_det_g *
                            pmb->pm1->lab.sc_nG(0,0)(k,j,i));
  }
  return max_sc_nG_00;
}

Real max_sc_E_00(MeshBlock *pmb, int iout)
{
  Real max_sc_E_00 = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    // const Real oo_sc_sqrt_det_g = OO(pmb->pm1->geom.sc_sqrt_det_g(k,j,i));
    max_sc_E_00 = std::max(max_sc_E_00,
                          //  oo_sc_sqrt_det_g *
                            pmb->pm1->lab.sc_E(0,0)(k,j,i));
  }
  return max_sc_E_00;
}

Real min_sc_nG_00(MeshBlock *pmb, int iout)
{
  Real min_sc_nG_00 = +std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    // const Real oo_sc_sqrt_det_g = OO(pmb->pm1->geom.sc_sqrt_det_g(k,j,i));
    min_sc_nG_00 = std::min(min_sc_nG_00,
                            pmb->pm1->lab.sc_nG(0,0)(k,j,i));
  }
  return min_sc_nG_00;
}

Real min_sc_E_00(MeshBlock *pmb, int iout)
{
  Real min_sc_E_00 = +std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    // const Real oo_sc_sqrt_det_g = OO(pmb->pm1->geom.sc_sqrt_det_g(k,j,i));
    min_sc_E_00 = std::min(min_sc_E_00,
                          //  oo_sc_sqrt_det_g *
                            pmb->pm1->lab.sc_E(0,0)(k,j,i));
  }
  return min_sc_E_00;
}

Real min_sc_n_00(MeshBlock *pmb, int iout)
{
  Real min_sc_n_00 = +std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    // const Real oo_sc_sqrt_det_g = OO(pmb->pm1->geom.sc_sqrt_det_g(k,j,i));
    min_sc_n_00 = std::min(min_sc_n_00,
                            pmb->pm1->rad.sc_n(0,0)(k,j,i));
  }
  return min_sc_n_00;
}

Real min_sc_J_00(MeshBlock *pmb, int iout)
{
  Real min_sc_J_00 = +std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    // const Real oo_sc_sqrt_det_g = OO(pmb->pm1->geom.sc_sqrt_det_g(k,j,i));
    min_sc_J_00 = std::min(min_sc_J_00,
                          //  oo_sc_sqrt_det_g *
                            pmb->pm1->rad.sc_J(0,0)(k,j,i));
  }
  return min_sc_J_00;
}
}
#endif

void Mesh::EnrollUserStandardM1()
{
#if M1_ENABLED
  EnrollUserHistoryOutput(max_sc_nG_00,
                          "max_sc_nG_00", UserHistoryOperation::max);
  EnrollUserHistoryOutput(max_sc_E_00,
                          "max_sc_E_00", UserHistoryOperation::max);

  EnrollUserHistoryOutput(min_sc_nG_00,
                          "min_sc_nG_00", UserHistoryOperation::min);
  EnrollUserHistoryOutput(min_sc_E_00,
                          "min_sc_E_00", UserHistoryOperation::min);

  EnrollUserHistoryOutput(min_sc_n_00,
                          "min_sc_n_00", UserHistoryOperation::min);
  EnrollUserHistoryOutput(min_sc_J_00,
                          "min_sc_J_00", UserHistoryOperation::min);

#endif // M1_ENABLED
}