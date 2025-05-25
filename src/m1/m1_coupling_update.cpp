// c++
// ...

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"
#include "../utils/linear_algebra.hpp"
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include <cmath>

// ============================================================================
namespace M1 {
// ============================================================================

// Helper for M1::CoupleSourcesADM
// Warning: This will throw the internal geom vector into different t states
void Update_sqrt_det_g(M1 & pm1)
{
  using namespace LinearAlgebra;

  Z4c * pz4c = pm1.pmy_block->pz4c;

#ifndef Z4C_CX_ENABLED

  AT_N_sym sl_g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);

  // scratch quantities
  AT_N_sym g_dd_(    pm1.mbi.nn1);
  AT_C_sca detgamma_(pm1.mbi.nn1);  // spatial met det

  ILOOP2(k,j)
  {
    pm1.pmy_coord->GetGeometricFieldCC(g_dd_, sl_g_dd, k, j);

    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    ILOOP1(i)
    {
      pm1.geom.sp_g_dd(a,b,k,j,i) = g_dd_(a,b,i);
    }

    Det3Metric(detgamma_,pm1.geom.sp_g_dd,
               k,j,pz4c->mbi.il, pz4c->mbi.iu);

    ILOOP1(i)
    {
      pm1.geom.sc_sqrt_det_g(k,j,i) = std::sqrt(detgamma_(i));
      pm1.geom.sc_oo_sqrt_det_g(k,j,i) = OO(pm1.geom.sc_sqrt_det_g(k,j,i));
    }
  }
#else
  ILOOP3(k,j,i)
  {
    pm1.geom.sc_oo_sqrt_det_g(k,j,i) = OO(pm1.geom.sc_sqrt_det_g(k,j,i));
  }
#endif // Z4C_CX_ENABLED
}

void M1::CoupleSourcesADM(AT_C_sca &A_rho, AT_N_vec &A_S_d, AT_N_sym & A_S_dd)
{
  Z4c * pz4c = pmy_block->pz4c;

  // BD: TODO - best place to put this?
  //
  // When Z4c is coupled this uses updated geom but lagging M1 data

  // Update subset of internal representation of the geometry
  Update_sqrt_det_g(*this);

  // point to scratches -------------------------------------------------------
  AT_N_sym & sp_P_dd_ = scratch.sp_P_dd_;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = lab.sc_E(ix_g,ix_s);
    AT_N_vec & sp_F_d  = lab.sp_F_d(ix_g,ix_s);
    // AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);
    AT_C_sca & sc_chi  = lab_aux.sc_chi(ix_g,ix_s);

    AT_C_sca & sc_oo_sqrt_det_g = geom.sc_oo_sqrt_det_g;

    ILOOP2(k,j)
    {
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        A_rho(k,j,i) += sc_E(k,j,i) * sc_oo_sqrt_det_g(k,j,i);

        if (!std::isfinite(sc_E(k,j,i)))
        {
          pm1->StatePrintPoint(
            "CoupleSourcesADM [sc_E] non-finite",
            ix_g, ix_s,
            k, j, i, true);
        }
        if (!std::isfinite(sc_oo_sqrt_det_g(k,j,i)))
        {
          pm1->StatePrintPoint(
            "CoupleSourcesADM [sc_oo_sqrt_det_g] non-finite",
            ix_g, ix_s,
            k, j, i, true);
        }
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        A_S_d(a,k,j,i) += sp_F_d(a,k,j,i) * sc_oo_sqrt_det_g(k,j,i);
        if (!std::isfinite(sp_F_d(a,k,j,i)))
        {
          pm1->StatePrintPoint(
            "CoupleSourcesADM [sp_F_d] non-finite",
            ix_g, ix_s,
            k, j, i, true);
        }
      }

      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        Assemble::Frames::sp_P_dd_(
          *this, sp_P_dd_, sc_chi, sc_E, sp_F_d,
          k, j, i, i
        );
      }

      for (int a=0; a<N; ++a)
      for (int b=a; b<N; ++b)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        A_S_dd(a,b,k,j,i) += sp_P_dd_(a,b,i) * sc_oo_sqrt_det_g(k,j,i);
        if (!std::isfinite(sp_P_dd_(a,b,i)))
        {
          pm1->StatePrintPoint(
            "CoupleSourcesADM [sp_P_dd_] non-finite",
            ix_g, ix_s,
            k, j, i, true);
        }
      }
    }
  }
};


void M1::CoupleSourcesHydro(AA & cons)
{
  Z4c * pz4c = pmy_block->pz4c;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & S_sc_E   = sources.sc_E(ix_g,ix_s);
    AT_N_vec & S_sp_F_d = sources.sp_F_d(ix_g,ix_s);

    ILOOP2(k,j)
    {
      /*
      // tau source -----------------------------------------------------------
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IEN,k,j,i) -= S_sc_E(k,j,i);
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        // Sign here is correct
        cons(IEN,k,j,i) += geom.sp_beta_u(a,k,j,i) * S_sp_F_d(a,k,j,i);
      }

      // S_j source -----------------------------------------------------------
      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        // cons(IM1+a,k,j,i) -= geom.sc_alpha(k,j,i) * S_sp_F_d(a,k,j,i);
        cons(IM1+a,k,j,i) += geom.sc_alpha(k,j,i) * S_sp_F_d(a,k,j,i);
      }
      */

      // tau source -----------------------------------------------------------
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IEN,k,j,i) -= S_sc_E(k,j,i);

        if (!std::isfinite(S_sc_E(k,j,i)))
        {
          pm1->StatePrintPoint(
            "CoupleSourcesHydro [S_sc_E] non-finite",
            ix_g, ix_s,
            k, j, i, true);
        }
      }

      // S_j source -----------------------------------------------------------
      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IM1+a,k,j,i) -= S_sp_F_d(a,k,j,i);
        if (!std::isfinite(S_sp_F_d(k,j,i)))
        {
          pm1->StatePrintPoint(
            "CoupleSourcesHydro [S_sp_F_d] non-finite",
            ix_g, ix_s,
            k, j, i, true);
        }
      }
    }
  }
}

// This is specific to 3 species (nu, nu_bar, nu_x); note ordering!
void  M1::CoupleSourcesYe(const Real mb, AA &ps)
{
  Z4c * pz4c = pmy_block->pz4c;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  {
    AT_C_sca & S_sc_nG_nue = sources.sc_nG(ix_g,0);
    AT_C_sca & S_sc_nG_nua = sources.sc_nG(ix_g,1);

    // sources of the form:
    // alpha * (
    //  sqrt(gamma) sc_eta_0 - kap_a_0 * n_tilde
    //)

    ILOOP3(k,j,i)
    if (MaskGet(k, j, i))
    {
      ps(0,k,j,i) += mb * (
        S_sc_nG_nua(k,j,i) - S_sc_nG_nue(k,j,i)
      );
      if (!std::isfinite(S_sc_nG_nue(k,j,i)) ||
          !std::isfinite(S_sc_nG_nua(k,j,i)))
      {
        pm1->StatePrintPoint(
          "CoupleSourcesYe [S_sc_nG_nu(e/a)] non-finite",
          ix_g, 0,
          k, j, i, true);
      }
    }
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//