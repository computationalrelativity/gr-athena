// c++
// ...

// Athena++ headers
#include "m1.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Compute the numerical fluxes using a simple 2nd order flux-limited method
// with high-Peclet limit fix. Cf. Hydro::CalculateFluxes(...)
void M1::CalcFluxes(AthenaArray<Real> & u)
{
  M1_ILOOP2(k,j)
  {
    Assemble::st_beta_u_(geom.st_beta_u_, geom.sp_beta_u,
                         k, j, mbi.il, mbi.iu);

    Assemble::st_g_uu_(geom.st_g_uu_,
                       geom.sp_g_uu, geom.sc_alpha, geom.sp_beta_u, scratch,
                       k, j, mbi.il, mbi.iu);

    Assemble::st_g_dd_(geom.st_g_dd_,
                       geom.sp_g_dd,
                       geom.sc_alpha,
                       geom.sp_beta_d,
                       geom.sp_beta_u,
                       scratch,
                       k, j, mbi.il, mbi.iu);

    Assemble::st_n_u_(geom.st_n_u_, geom.sc_alpha, geom.sp_beta_u,
                      k, j, mbi.il, mbi.iu);

    Assemble::st_n_d_(geom.st_n_d_, geom.sc_alpha,
                      k, j, mbi.il, mbi.iu);

    Assemble::st_P_ud_(geom.st_P_ud_, geom.st_n_u_, geom.st_n_d_,
                       k, j, mbi.il, mbi.iu);


  }

  return;
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//