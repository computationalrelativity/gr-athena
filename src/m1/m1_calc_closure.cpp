// c++
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1::closures {
// ============================================================================

// P_{i j} in Eq.(15) of [1]
void thin(M1 * pm1)
{
  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    AT_N_sym & sp_P_dd = pm1->rad.sp_P_dd(ix_g,ix_s);
    AT_N_vec & sp_F_d  = pm1->lab.sp_F_d( ix_g,ix_s);
    AT_C_sca & sc_E    = pm1->lab.sc_E(   ix_g,ix_s);

    M1_ILOOP2(k,j)
    {
      Assemble::st_g_uu_(pm1->geom.st_g_uu_,
                         pm1->geom.sp_g_uu,
                         pm1->geom.sc_alpha,
                         pm1->geom.sp_beta_u,
                         pm1->scratch,
                         k, j, pm1->mbi.il, pm1->mbi.iu);

      Assemble::st_F_d_(pm1->lab.st_F_d_,
                        sp_F_d,
                        pm1->geom.sp_beta_u,
                        k, j, pm1->mbi.il, pm1->mbi.iu);

      Assemble::Norm_st_(pm1->lab.sc_norm_st_F_,
                         pm1->lab.st_F_d_,
                         pm1->geom.st_g_uu_,
                         k, j, pm1->mbi.il, pm1->mbi.iu);

      for (int a=0; a<N; ++a)
      for (int b=a; b<N; ++b)
      M1_ILOOP1(i)
      {
        const Real F2 = pm1->lab.sc_norm_st_F_(i);
        const Real fac = (F2 > 0) ? sc_E(k,j,i) / F2
                                  : 0.0;
        sp_P_dd(a,b,k,j,i) = fac * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
      }

    }
  }
}

// ============================================================================
} // namespace M1::closures
// ============================================================================


// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Computes the closure on a mesh block
void M1::CalcClosure(AthenaArray<Real> & u)
{
  // Thin limit
  closures::thin(pm1);

  return;
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//