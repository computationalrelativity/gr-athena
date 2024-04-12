// c++
// ...

// Athena++ headers
#include "m1.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Calculate geometric source contribution based on u, add to u_inh
void M1::AddGRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_inh)
{
  vars_Lab U_n { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U_n);

  vars_Lab I   { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, I);

  AT_N_vec & sp_F_u_  = scratch.sp_vec_;
  AT_N_sym & sp_P_uu_ = scratch.sp_sym_A_;

  AT_N_D1sca & sp_dalpha_d = geom.sp_dalpha_d;
  AT_N_D1vec & sp_dbeta_du = geom.sp_dbeta_du;
  AT_N_D1sym & sp_dg_ddd   = geom.sp_dg_ddd;


  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & Un_sc_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & Un_sp_F_d  = U_n.sp_F_d(ix_g,ix_s);
    AT_N_sym & Un_sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & I_nG   = I.sc_nG( ix_g,ix_s);
    AT_C_sca & I_E    = I.sc_E(  ix_g,ix_s);
    AT_N_vec & I_F_d  = I.sp_F_d(ix_g,ix_s);

    M1_ILOOP2(k,j)
    {
      Assemble::sp_d_to_u_(  this, sp_F_u_,  Un_sp_F_d,  k, j, mbi.il, mbi.iu);
      Assemble::sp_dd_to_uu_(this, sp_P_uu_, Un_sp_P_dd, k, j, mbi.il, mbi.iu);

      for (int a=0; a<N; ++a)
      for (int b=0; b<N; ++b)
      M1_ILOOP1(i)
      {
        I_E(k,j,i) += geom.sc_alpha(k,j,i) *
                      sp_P_uu_(a,b,i) *
                      geom.sp_K_dd(a,b,k,j,i);
      }

      for (int a=0; a<N; ++a)
      M1_ILOOP1(i)
      {
        I_E(k,j,i) -= sp_F_u_(a,i) * sp_dalpha_d(a,k,j,i);
      }

      for (int a=0; a<N; ++a)
      {
        M1_ILOOP1(i)
        {
          I_F_d(a,k,j,i) -= Un_sc_E(k,j,i) * sp_dalpha_d(a,k,j,i);
        }

        for (int b=0; b<N; ++b)
        {
          M1_ILOOP1(i)
          {
            I_F_d(a,k,j,i) += Un_sp_F_d(b,k,j,i) * sp_dbeta_du(a,b,k,j,i);
          }

          for (int c=0; c<N; ++c)
          M1_ILOOP1(i)
          {
            I_F_d(a,k,j,i) += 0.5 * sp_P_uu_(b,c,i) * sp_dg_ddd(a,b,c,k,j,i);
          }
        }
      }
    }
  }

  return;
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//