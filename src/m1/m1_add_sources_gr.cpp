// c++
// ...

// Athena++ headers
#include "m1.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Calculate source with geometry terms and add to r.h.s.
void M1::AddGRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
{
  /*
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

  AT_C_sca F_sc_E_( mbi.nn1);
  AT_C_sca F_sc_Eb_(mbi.nn1);
  */

  return;
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//