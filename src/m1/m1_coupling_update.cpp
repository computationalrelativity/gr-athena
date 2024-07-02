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

// ============================================================================
namespace M1 {
// ============================================================================

// Helper for M1::CoupleSourcesADM
// Warning: This will throw the internal geom vector into different t states
void Update_sqrt_det_g(M1 & pm1)
{
  using namespace LinearAlgebra;

  Z4c * pz4c = pm1.pmy_block->pz4c;

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
    }
  }
}


void M1::CoupleSourcesADM(AT_C_sca &A_rho, AT_N_vec &A_S_d, AT_N_sym & A_S_dd)
{
  Z4c * pz4c = pmy_block->pz4c;

  // TODO: best place to put this?
  //
  // When Z4c is coupled this uses updated geom but lagging M1 data

  // Update subset of internal representation of the geometry
  Update_sqrt_det_g(*this);

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = lab.sc_E(ix_g,ix_s);
    AT_N_vec & sp_F_d  = lab.sp_F_d(ix_g,ix_s);
    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    ILOOP2(k,j)
    {
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        A_rho(k,j,i) += sc_E(k,j,i) / geom.sc_sqrt_det_g(k,j,i);
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        A_S_d(a,k,j,i) += sp_F_d(a,k,j,i) / geom.sc_sqrt_det_g(k,j,i);
      }

      for (int a=0; a<N; ++a)
      for (int b=a; b<N; ++b)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        A_S_dd(a,b,k,j,i) += sp_P_dd(a,b,k,j,i) / geom.sc_sqrt_det_g(k,j,i);
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
    AT_C_sca & sc_S1   = sources.sc_S1(ix_g,ix_s);
    AT_N_vec & sp_S1_d = sources.sp_S1_d(ix_g,ix_s);

    ILOOP2(k,j)
    {
      // tau source -----------------------------------------------------------
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IEN,k,j,i) -= sc_S1(k,j,i);
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IEN,k,j,i) += geom.sp_beta_u(a,k,j,i) * sp_S1_d(a,k,j,i); // Sign here is correct
      }

      // S_j source -----------------------------------------------------------
      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IM1+a,k,j,i) -= geom.sc_alpha(k,j,i) * sp_S1_d(a,k,j,i);
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
    AT_C_sca & sc_S0_nue = sources.sc_S0(ix_g,0);
    AT_C_sca & sc_S0_nua = sources.sc_S0(ix_g,1);

    ILOOP3(k,j,i)
    if (MaskGet(k, j, i))
    {
      ps(0,k,j,i) += geom.sc_alpha(k,j,i) * mb * (
        sc_S0_nua(k,j,i) - sc_S0_nue(k,j,i)
      );
    }
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//