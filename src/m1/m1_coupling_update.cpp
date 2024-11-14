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

  // BD: TODO - best place to put this?
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

void M1::CoupleSourcesHydro(const Real weight, AA & cons)
{
  Z4c * pz4c = pmy_block->pz4c;

  // scratch quantities
  AT_C_sca & dotFv_  = scratch.sc_A_;
  AT_C_sca & dotPvv_ = scratch.sc_B_;
  AT_N_vec & dotPv_d_  = scratch.sp_vec_A_;

  AT_C_sca & S_   = scratch.sc_C_;
  AT_N_vec & S_d_ = scratch.sp_vec_B_;

  dotPv_d_.ZeroClear();
  dotPvv_.ZeroClear();

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = lab.sc_E(ix_g,ix_s);
    AT_N_vec & sp_F_d  = lab.sp_F_d(ix_g,ix_s);
    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
    AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);
    AT_C_sca & sc_eta   = radmat.sc_eta(  ix_g,ix_s);

    ILOOP2(k,j)
    {

      // contracted quantities
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        dotFv_(i) = Assemble::sc_dot_dense_sp__(sp_F_d, fidu.sp_v_u, k, j, i);
      }

      for (int a=0; a<N; ++a)
      for (int b=0; b<N; ++b)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        dotPv_d_(a,i) += sp_P_dd(a,b,k,j,i) * fidu.sp_v_u(b,k,j,i);
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        dotPvv_(i) += dotPv_d_(a,i) * fidu.sp_v_u(a,k,j,i);
      }

      // assemble sources
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        const Real kap_as = sc_kap_a(k,j,i) + sc_kap_s(k,j,i);

        const Real W  = fidu.sc_W(k,j,i);
        const Real W2 = SQR(W);

        S_(i) = W * geom.sc_alpha(k,j,i) * (
          -kap_as * sc_E(k,j,i) + kap_as * dotFv_(i) +
          (
            geom.sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) +
            sc_kap_s(k,j,i) * W2 * (
              sc_E(k,j,i) - 2 * dotFv_(i) + dotPvv_(i)
            )
          )
        );
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        const Real kap_as = sc_kap_a(k,j,i) + sc_kap_s(k,j,i);

        const Real W  = fidu.sc_W(k,j,i);
        const Real W2 = SQR(W);

        S_d_(a,i) = W * geom.sc_alpha(k,j,i) * (
          -kap_as * dotPv_d_(a,i) - kap_as * sp_F_d(a,k,j,i) +
          (
            geom.sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) +
            sc_kap_s(k,j,i) * W2 * (
              sc_E(k,j,i) - 2 * dotFv_(i) + dotPvv_(i)
            )
          ) * fidu.sp_v_d(a,k,j,i)
        );
      }

      // tau source -----------------------------------------------------------
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IEN,k,j,i) += weight * S_(i);
      }

      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IEN,k,j,i) += -weight * geom.sp_beta_u(a,k,j,i) * S_d_(a,i);
      }

      // S_j source -----------------------------------------------------------
      for (int a=0; a<N; ++a)
      ILOOP1(i)
      if (MaskGet(k, j, i))
      {
        cons(IM1+a,k,j,i) += -weight * geom.sc_alpha(k,j,i) * S_d_(a,i);
      }
    }
  }

}

// This is specific to 3 species (nu, nu_bar, nu_x); note ordering!
void M1::CoupleSourcesYe(const Real weight, const Real mb, AA &ps)
{
  Z4c * pz4c = pmy_block->pz4c;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  {
    // AT_C_sca & sc_E    = lab.sc_E(ix_g,ix_s);
    // AT_N_vec & sp_F_d  = lab.sp_F_d(ix_g,ix_s);
    // AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_kap_a_0_nu  = radmat.sc_kap_a_0(ix_g,0);
    AT_C_sca & sc_kap_a_0_nub = radmat.sc_kap_a_0(ix_g,1);
    // AT_C_sca & sc_kap_a_0_nux = radmat.sc_kap_a_0(ix_g,2);

    AT_C_sca & sc_eta_0_nu  = radmat.sc_eta_0(ix_g,0);
    AT_C_sca & sc_eta_0_nub = radmat.sc_eta_0(ix_g,1);
    // AT_C_sca & sc_eta_0_nux = radmat.sc_eta_0(ix_g,2);

    AT_C_sca & sc_n_nu  = rad.sc_n(ix_g,0);
    AT_C_sca & sc_n_nub = rad.sc_n(ix_g,1);

    ILOOP3(k,j,i)
    if (MaskGet(k, j, i))
    {
      ps(0,k,j,i) += SQR(geom.sc_alpha(k,j,i)) * mb * (
        geom.sc_sqrt_det_g(k,j,i) * sc_eta_0_nub(k,j,i) -
        sc_kap_a_0_nub(k,j,i) * sc_n_nub(k,j,i) -
        geom.sc_sqrt_det_g(k,j,i) * sc_eta_0_nu(k,j,i) +
        sc_kap_a_0_nu(k,j,i) * sc_n_nu(k,j,i)
      );
    }
  }
}


// `master_m1` style ----------------------------------------------------------

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
        cons(IM1+a,k,j,i) -= geom.sc_alpha(k,j,i) * S_sp_F_d(a,k,j,i);
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

    ILOOP3(k,j,i)
    if (MaskGet(k, j, i))
    {
      ps(0,k,j,i) += geom.sc_alpha(k,j,i) * mb * (
        S_sc_nG_nua(k,j,i) - S_sc_nG_nue(k,j,i)
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