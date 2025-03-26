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

#include <functional>

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

int windowed_npts(MeshBlock *pmb,
                  const int IWN, AA & window,
                  const Real win_min,
                  const Real win_max,
                  int iout)
{
  int n_pts = 0;

  CC_ILOOP3(k,j,i)
  {
    if ((win_min <= window(IWN,k,j,i)) &&
         win_max >= window(IWN,k,j,i))
    {
      n_pts += 1;
    }
  }
  return n_pts;
}

Real windowed_int(MeshBlock *pmb,
                  const int IQ, AA & quantity,
                  const int IWN, AA & window,
                  const Real win_min,
                  const Real win_max,
                  int iout)
{
  Real sum_Q = 0;

  CC_ILOOP3(k,j,i)
  {
    if ((win_min <= window(IWN,k,j,i)) &&
         win_max >= window(IWN,k,j,i))
    {

      const Real dx1 = pmb->pcoord->dx1v(i);
      const Real dx2 = pmb->pcoord->dx2v(i);
      const Real dx3 = pmb->pcoord->dx3v(i);
      const Real w = dx1*dx2*dx3;
      sum_Q += quantity(IQ,k,j,i)*w;
    }
  }
  return sum_Q;
}

Real weighted_avg(MeshBlock *pmb,
                  const int IQ, AA & quantity,
                  const int IWG, AA & weight,
                  int iout)
{
  Real sum_Q = 0;

  CC_ILOOP3(k,j,i)
  {
    const Real dx1 = pmb->pcoord->dx1v(i);
    const Real dx2 = pmb->pcoord->dx2v(i);
    const Real dx3 = pmb->pcoord->dx3v(i);
    const Real w = dx1*dx2*dx3 * weight(IWG,k,j,i);
    sum_Q += quantity(IQ,k,j,i) * w;
  }
  return sum_Q;
}

Real windowed_weighted_avg(MeshBlock *pmb,
                  const int IQ, AA & quantity,
                  const int IWG, AA & weight,
                  const int IWN, AA & window,
                  const Real win_min,
                  const Real win_max,
                  int iout)
{
  Real sum_Q = 0;

  CC_ILOOP3(k,j,i)
  {
    if ((win_min <= window(IWN,k,j,i)) &&
         win_max >= window(IWN,k,j,i))
    {
      const Real dx1 = pmb->pcoord->dx1v(i);
      const Real dx2 = pmb->pcoord->dx2v(i);
      const Real dx3 = pmb->pcoord->dx3v(i);
      const Real w = dx1*dx2*dx3 * weight(IWG,k,j,i);
      sum_Q += quantity(IQ,k,j,i) * w;
    }
  }
  return sum_Q;
}
}
#endif // FLUID_ENABLED

void Mesh::EnrollUserStandardHydro(ParameterInput * pin)
{
#if FLUID_ENABLED
  EnrollUserHistoryOutput(max_rho, "max_rho",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(max_T, "max_T",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(num_c2p_fail, "num_c2p_fail",
                          UserHistoryOperation::max);

  // Enroll all average [windowed] quantities ---------------------------------
  InputBlock *pib = pin->pfirst_block;
  while (pib != nullptr)
  {
    if (pib->block_name.compare(0, 12, "hst_windowed")  == 0)
    {
      std::string hstwn = pib->block_name.substr(12); // cnt starts at 0
      const int par_ix = atoi(hstwn.c_str());

      const std::string str_n_pts = "hw_n_pts_" + hstwn;

      const Real rho_min = pin->GetOrAddReal(
        pib->block_name,
        "rho_min", 0.0);
      const Real rho_max = pin->GetOrAddReal(
        pib->block_name,
        "rho_max", 1e99);

       // get number of points that satisfy the density window
      auto fcn_hw_n_pts = [rho_min,rho_max](MeshBlock *pmb, int iout)
       {
         return windowed_npts(pmb,
                              IDN, pmb->phydro->w,
                              rho_min, rho_max, iout);
       };

      // sum quantities that live in the density window -----------------------
      auto fcn_hw_sum = [rho_min,rho_max](MeshBlock *pmb, int iout)
      {
        return windowed_int(pmb,
                            IDN, pmb->phydro->u,
                            IDN, pmb->phydro->w,
                            rho_min, rho_max, iout);
      };

      auto fcn_hw_sum_T = [rho_min,rho_max](MeshBlock *pmb, int iout)
      {
        return windowed_weighted_avg(pmb,
                                     IX_T, pmb->phydro->derived_ms,
                                     IDN, pmb->phydro->u,
                                     IDN, pmb->phydro->w,
                                     rho_min, rho_max, iout);
      };

      auto fcn_hw_sum_Y = [rho_min,rho_max](MeshBlock *pmb, int iout)
      {
        const int IX_Y = 0;
        return windowed_weighted_avg(pmb,
                                     IX_Y, pmb->pscalars->r,
                                     IDN, pmb->phydro->u,
                                     IDN, pmb->phydro->w,
                                     rho_min, rho_max, iout);
      };

      EnrollUserHistoryOutput(
        fcn_hw_n_pts, str_n_pts.c_str(),
        UserHistoryOperation::sum);

      EnrollUserHistoryOutput(
        fcn_hw_sum, ("hw_M_" + hstwn).c_str(),
        UserHistoryOperation::sum);

      EnrollUserHistoryOutput(
        fcn_hw_sum_T, ("hw_T_" + hstwn).c_str(),
        UserHistoryOperation::sum);

#if NSCALARS > 0
      EnrollUserHistoryOutput(
        fcn_hw_sum_Y, ("hw_Y_" + hstwn).c_str(),
        UserHistoryOperation::sum);
#endif
    }

    pib = pib->pnext;
  }

  auto fcn_mass_geodesic = [&](MeshBlock *pmb, int iout)
  {
    return windowed_int(pmb,
                        IDN, pmb->phydro->u,
                        IX_U_D_0, pmb->phydro->derived_ms,
                        -1e99, -1, iout);
  };

  EnrollUserHistoryOutput(
    fcn_mass_geodesic, "m_ej_geod",
    UserHistoryOperation::sum);

  auto fcn_mass_bernulli = [&](MeshBlock *pmb, int iout)
  {
    return windowed_int(pmb,
                        IDN, pmb->phydro->u,
                        IX_HU0, pmb->phydro->derived_int,
                        -1e99, -1, iout);
  };

  EnrollUserHistoryOutput(
    fcn_mass_bernulli, "m_ej_bern",
    UserHistoryOperation::sum);

  // --------------------------------------------------------------------------
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

Real max_B2(MeshBlock *pmb, int iout)
{
  Field *pf = pmb->pfield;

  Real max_B2_ = 0.0;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  CC_ILOOP3(k, j, i)
  {
    max_B2_ = std::max(
      std::abs(pf->derived_ms(IX_B2,k,j,i)),
      max_B2_);
  }

  return max_B2_;
}

}
#endif // MAGNETIC_FIELDS_ENABLED

void Mesh::EnrollUserStandardField(ParameterInput * pin)
{
#if MAGNETIC_FIELDS_ENABLED
  EnrollUserHistoryOutput(max_B2, "max_B2",
                          UserHistoryOperation::max);

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
  AT_N_sca alpha(pmb->pz4c->storage.adm, Z4c::I_ADM_alpha);

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

void Mesh::EnrollUserStandardZ4c(ParameterInput * pin)
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

Real min_sc_nG(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real min_sc_nG_ = std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    min_sc_nG_ = std::min(min_sc_nG_, pmb->pm1->lab.sc_nG(ix_g,ix_s)(k,j,i));
  }
  return min_sc_nG_;
}

Real max_sc_nG(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real max_sc_nG_ = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    max_sc_nG_ = std::max(max_sc_nG_, pmb->pm1->lab.sc_nG(ix_g,ix_s)(k,j,i));
  }
  return max_sc_nG_;
}

Real min_sc_E(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real min_sc_E_ = std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    min_sc_E_ = std::min(min_sc_E_, pmb->pm1->lab.sc_E(ix_g,ix_s)(k,j,i));
  }
  return min_sc_E_;
}

Real max_sc_E(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real max_sc_E_ = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    max_sc_E_ = std::max(max_sc_E_, pmb->pm1->lab.sc_E(ix_g,ix_s)(k,j,i));
  }
  return max_sc_E_;
}

Real min_sc_J(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real min_sc_J_ = std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    min_sc_J_ = std::min(min_sc_J_, pmb->pm1->rad.sc_J(ix_g,ix_s)(k,j,i));
  }
  return min_sc_J_;
}

Real max_sc_J(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real max_sc_J_ = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    max_sc_J_ = std::max(max_sc_J_, pmb->pm1->rad.sc_J(ix_g,ix_s)(k,j,i));
  }
  return max_sc_J_;
}

Real min_sc_n(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real min_sc_n_ = std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    min_sc_n_ = std::min(min_sc_n_, pmb->pm1->rad.sc_n(ix_g,ix_s)(k,j,i));
  }
  return min_sc_n_;
}

Real max_sc_n(const int ix_g, const int ix_s, MeshBlock *pmb, int iout)
{
  Real max_sc_n_ = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    max_sc_n_ = std::max(max_sc_n_, pmb->pm1->rad.sc_n(ix_g,ix_s)(k,j,i));
  }
  return max_sc_n_;
}

}
#endif

void Mesh::EnrollUserStandardM1(ParameterInput * pin)
{
#if M1_ENABLED
  const int N_GRPS = pin->GetOrAddInteger("M1", "ngroups",  1);
  const int N_SPCS = pin->GetOrAddInteger("M1", "nspecies", 1);

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    auto max_sc_nG_ix_gs = [ig=ix_g, is=ix_s](MeshBlock *pmb, int iout)
    {
      return max_sc_nG(ig, is, pmb, iout);
    };

    auto max_sc_E_ix_gs = [ig=ix_g, is=ix_s](MeshBlock *pmb, int iout)
    {
      return max_sc_E(ig, is, pmb, iout);
    };

    EnrollUserHistoryOutput(
      max_sc_nG_ix_gs,
      ("max_sc_nG_" + std::to_string(ix_g) + std::to_string(ix_s)).c_str(),
      UserHistoryOperation::max
    );

    EnrollUserHistoryOutput(
      max_sc_E_ix_gs,
      ("max_sc_E_" + std::to_string(ix_g) + std::to_string(ix_s)).c_str(),
      UserHistoryOperation::max
    );

    auto min_sc_nG_ix_gs = [ig=ix_g, is=ix_s](MeshBlock *pmb, int iout)
    {
      return min_sc_nG(ig, is, pmb, iout);
    };

    auto min_sc_E_ix_gs = [ig=ix_g, is=ix_s](MeshBlock *pmb, int iout)
    {
      return min_sc_E(ig, is, pmb, iout);
    };

    EnrollUserHistoryOutput(
      min_sc_nG_ix_gs,
      ("min_sc_nG_" + std::to_string(ix_g) + std::to_string(ix_s)).c_str(),
      UserHistoryOperation::min
    );

    EnrollUserHistoryOutput(
      min_sc_E_ix_gs,
      ("min_sc_E_" + std::to_string(ix_g) + std::to_string(ix_s)).c_str(),
      UserHistoryOperation::min
    );

  }

#endif // M1_ENABLED
}