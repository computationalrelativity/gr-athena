// c++
// ...

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_update.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Calculate geometric source contribution based on u, add to u_inh
void M1::AddSourceGR(AA & u, AA & u_inh)
{
  using namespace Update;

  vars_Lab U_C { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u,     U_C);

  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, U_I);

  // scratches
  AT_N_vec & sp_F_u_  = scratch.sp_vec_A_;

  AT_N_sym & sp_P_dd_ = scratch.sp_P_dd_;
  AT_N_sym & sp_P_uu_ = scratch.sp_P_uu_;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(*this, U_C, ix_g, ix_s);
    StateMetaVector I = ConstructStateMetaVector(*this, U_I, ix_g, ix_s);

    M1_ILOOP3(k,j,i)
    if (MaskGet(k, j, i))
    {

      Assemble::Frames::sp_P_dd_(
        *this, sp_P_dd_, C.sc_chi, C.sc_E, C.sp_F_d,
        k, j, i, i
      );

      Assemble::sp_d_to_u_(this, sp_F_u_,  C.sp_F_d,  k, j, i, i);
      Assemble::sp_dd_to_uu_(this, sp_P_uu_, sp_P_dd_, k, j, i, i);

      for (int a=0; a<N; ++a)
      {
        for (int b=0; b<N; ++b)
        {
          I.sc_E(k,j,i) += geom.sc_alpha(k,j,i) *
                            sp_P_uu_(a,b,i) *
                            geom.sp_K_dd(a,b,k,j,i);
        }

        I.sc_E(k,j,i) -= sp_F_u_(a,i) * geom.sp_dalpha_d(a,k,j,i);
      }

      for (int a=0; a<N; ++a)
      {
        I.sp_F_d(a,k,j,i) -= C.sc_E(k,j,i) * geom.sp_dalpha_d(a,k,j,i);

        for (int b=0; b<N; ++b)
        {
          I.sp_F_d(a,k,j,i) += C.sp_F_d(b,k,j,i) *
                               geom.sp_dbeta_du(a,b,k,j,i);

          for (int c=0; c<N; ++c)
          {
            I.sp_F_d(a,k,j,i) += 0.5 * geom.sc_alpha(k,j,i) *
                                       sp_P_uu_(b,c,i) *
                                       geom.sp_dg_ddd(a,b,c,k,j,i);
          }
        }
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