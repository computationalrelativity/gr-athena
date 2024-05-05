// c++
// ...

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"
#include "../utils/linear_algebra.hpp"
#include "m1.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Prepare coupled (background) geometry (only during M1 ctor)
void M1::InitializeGeometry(vars_Geom & geom, vars_Scratch & scratch)
{
  // Allocate dense storage ---------------------------------------------------
  geom.sc_alpha.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sp_beta_u.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sp_g_dd.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);
  geom.sp_K_dd.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sp_dalpha_d.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sp_dbeta_du.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sp_dg_ddd.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  // For derived quantities
  geom.sp_beta_d.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sc_sqrt_det_g.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  geom.sp_g_uu.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  // Populate storage ---------------------------------------------------------
  if (!Z4C_ENABLED)
  {
    // Do not have z4c data; populate Minkowski in geodesic gauge.
    geom.sc_alpha.Fill(1.0);

    geom.sp_beta_u.Fill(0.0);

    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    M1_GLOOP3(k,j,i)
    {
      geom.sp_g_dd(a,b,k,j,i) = (a==b);
    }

    geom.sp_K_dd.Fill(0.0);
    geom.sp_dg_ddd.Fill(0.0);
  }

  // Derived quantities (sqrt_det, spatial inv)
  DerivedGeometry(geom, scratch);
  return;
}


// ----------------------------------------------------------------------------
// Update coupled (background) geometry (only _after_ M1 ctor)
void M1::UpdateGeometry(vars_Geom & geom, vars_Scratch & scratch)
{
  // Populate storage ---------------------------------------------------------
  if (Z4C_ENABLED)
  {
    Z4c * pz4c = pmy_block->pz4c;

    // sliced quantities
    AT_C_sca sl_alpha(pz4c->storage.u, Z4c::I_Z4c_alpha);

    AT_N_vec sl_beta_u(pz4c->storage.u, Z4c::I_Z4c_betax);

    AT_N_sym sl_g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
    AT_N_sym sl_K_dd(pz4c->storage.adm, Z4c::I_ADM_Kxx);

    // scratch quantities
    AT_C_sca alpha_(mbi.nn1);

    AT_N_vec beta_u_(mbi.nn1);

    AT_N_sym g_dd_(mbi.nn1);
    AT_N_sym K_dd_(mbi.nn1);

    AT_N_D1sca dalpha_d_(mbi.nn1);

    AT_N_D1vec dbeta_du_(mbi.nn1);

    AT_N_D1sym dg_ddd_(mbi.nn1);

    // M1: on CC --------------------------------------------------------------
    int IL, IU, JL, JU, KL, KU;
    pmy_coord->GetGeometricFieldCCIdxRanges(
      IL, IU,
      JL, JU,
      KL, KU);

    // const int IL = pmy_block->is, IU = pmy_block->ie;
    // const int JL = pmy_block->js, JU = pmy_block->je;
    // const int KL = pmy_block->ks, KU = pmy_block->ke;

    for (int k=KL; k<=KU; ++k)
    for (int j=JL; j<=JU; ++j)
    {
      pmy_coord->GetGeometricFieldCC(alpha_, sl_alpha, k, j);

      pmy_coord->GetGeometricFieldCC(beta_u_, sl_beta_u, k, j);

      pmy_coord->GetGeometricFieldCC(g_dd_, sl_g_dd, k, j);
      pmy_coord->GetGeometricFieldCC(K_dd_, sl_K_dd, k, j);

      for(int a=0; a<NDIM; ++a)
      {
        pmy_coord->GetGeometricFieldDerCC(dalpha_d_, sl_alpha,  a, k, j);
        pmy_coord->GetGeometricFieldDerCC(dbeta_du_, sl_beta_u, a, k, j);
        pmy_coord->GetGeometricFieldDerCC(dg_ddd_,   sl_g_dd,   a, k, j);
      }

      // Copy to dense storage ------------------------------------------------
      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        geom.sc_alpha(k,j,i) = alpha_(i);
      }

      for(int a = 0; a < NDIM; ++a)
      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        geom.sp_beta_u(a,k,j,i) = beta_u_(a,i);
      }

      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        geom.sp_g_dd(a,b,k,j,i) = g_dd_(a,b,i);
        geom.sp_K_dd(a,b,k,j,i) = K_dd_(a,b,i);
      }

      for(int c = 0; c < NDIM; ++c)
      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        geom.sp_dalpha_d(c,k,j,i) = dalpha_d_(c,i);
      }

      for(int c = 0; c < NDIM; ++c)
      for(int a = 0; a < NDIM; ++a)
      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        geom.sp_dbeta_du(c,a,k,j,i) = dbeta_du_(c,a,i);
      }


      for(int c = 0; c < NDIM; ++c)
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        geom.sp_dg_ddd(c,a,b,k,j,i) = dg_ddd_(c,a,b,i);
      }

    }

    // Derived quantities (sqrt_det, spatial inv)
    DerivedGeometry(geom, scratch);
  }
}

// ----------------------------------------------------------------------------
// Derived quantities based on coupled (background) geometry
void M1::DerivedGeometry(vars_Geom & geom, vars_Scratch & scratch)
{
  using namespace LinearAlgebra;

  AT_C_sca detgamma_(   mbi.nn1);  // spatial met det
  AT_C_sca oo_detgamma_(mbi.nn1);  // 1 / spatial met det

  AT_N_sym sp_g_uu_(mbi.nn1);

  // M1: on CC ----------------------------------------------------------------
  int IL, IU, JL, JU, KL, KU;
  pmy_coord->GetGeometricFieldCCIdxRanges(
    IL, IU,
    JL, JU,
    KL, KU);

  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  {
    Det3Metric(detgamma_,geom.sp_g_dd,
               k,j,IL,IU);

    #pragma omp simd
    for (int i=IL; i<=IU; ++i)
    {
      oo_detgamma_(i) = 1.0 / detgamma_(i);
      geom.sc_sqrt_det_g(k,j,i) = std::sqrt(detgamma_(i));
    }

    Inv3Metric(oo_detgamma_, geom.sp_g_dd, sp_g_uu_,
               k,j,IL,IU);

    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    #pragma omp simd
    for (int i=IL; i<=IU; ++i)
    {
      geom.sp_g_uu(a,b,k,j,i) = sp_g_uu_(a,b,i);
    }

    Assemble::sp_beta_d(geom.sp_beta_d, geom.sp_beta_u, geom.sp_g_dd, scratch,
                        k, j, IL, IU);

  }


}

// ----------------------------------------------------------------------------
// Prepare coupled (background) hydro (only during M1 ctor)
void M1::InitializeHydro(vars_Hydro & hydro,
                         vars_Geom & geom,
                         vars_Scratch & scratch)
{
  // Allocate dense storage ---------------------------------------------------
  hydro.sc_W.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);

  if (FLUID_ENABLED)
  {
    Hydro * phydro = pmy_block->phydro;

    // slice primitives
    hydro.sc_w_rho.InitWithShallowSlice(   phydro->w, IDN);
    hydro.sp_w_util_u.InitWithShallowSlice(phydro->w, IVX);
    hydro.sc_w_p.InitWithShallowSlice(     phydro->w, IPR);
  }
  else
  {
    // Fixed hydro background
    hydro.sc_w_rho.NewAthenaTensor(   mbi.nn3, mbi.nn2, mbi.nn1);
    hydro.sp_w_util_u.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);
    hydro.sc_w_p.NewAthenaTensor(     mbi.nn3, mbi.nn2, mbi.nn1);

    // set constant Gamma=2 EoS with K=1 for debug
    const Real K = 1;
    const Real Gamma = 2;

    hydro.sc_w_rho.Fill(1e-10);
    hydro.sp_w_util_u.Fill(0.);

    M1_GLOOP3(k,j,i)
    {
      hydro.sc_w_p(k,j,i) = K * std::pow(hydro.sc_w_rho(k,j,i), Gamma);
    }
  }

  // Lorentz factor etc
  DerivedHydro(hydro, geom, scratch);
  return;
}

// ----------------------------------------------------------------------------
// Update coupled (background) hydro (only _after_ M1 ctor)
void M1::UpdateHydro(vars_Hydro & hydro,
                     vars_Geom & geom,
                     vars_Scratch & scratch)
{
  if (FLUID_ENABLED)
  {
    // Lorentz factor etc
    DerivedHydro(hydro, geom, scratch);
  }
  return;
}

// ----------------------------------------------------------------------------
// Derived quantities based on coupled (background) geometry
void M1::DerivedHydro(vars_Hydro & hydro,
                      vars_Geom & geom,
                      vars_Scratch & scratch)
{
  using namespace LinearAlgebra;

  // Lorentz factor
  M1_GLOOP2(k,j)
  {
    M1_GLOOP1(i)
    {
      const Real norm2_util = InnerProductVecMetric(
        hydro.sp_w_util_u, geom.sp_g_dd,
        k,j,i
      );
      hydro.sc_W(k,j,i) = std::sqrt(1. + norm2_util);
    }
  }

}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//