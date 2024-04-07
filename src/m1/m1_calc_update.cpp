// C++ standard headers
// ...

// Athena++ headers
#include "m1.hpp"
// #include "m1_macro.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// -----------------------------------------------------------------------------
// Function to update the radiation fields
void M1::CalcUpdate(Real const dt,
                    AthenaArray<Real> & u_pre,
                    AthenaArray<Real> & u_cur,
		                AthenaArray<Real> & u_inh)
{

  vars_Lab U_n { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_n);

  vars_Lab U_p { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_p);

  vars_Lab I   { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, I);

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & Un_nG   = U_n.sc_nG( ix_g,ix_s);
    AT_C_sca & Un_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & Un_F_d  = U_n.sp_F_d(ix_g,ix_s);

    AT_C_sca & Up_nG   = U_p.sc_nG( ix_g,ix_s);
    AT_C_sca & Up_E    = U_p.sc_E(  ix_g,ix_s);
    AT_N_vec & Up_F_d  = U_p.sp_F_d(ix_g,ix_s);

    AT_C_sca & I_nG   = I.sc_nG( ix_g,ix_s);
    AT_C_sca & I_E    = I.sc_E(  ix_g,ix_s);
    AT_N_vec & I_F_d  = I.sp_F_d(ix_g,ix_s);

    M1_ILOOP2(k, j)
    {
      M1_ILOOP1(i)
      {
        Un_nG(k,j,i) = Up_nG(k,j,i) + dt * I_nG(k,j,i);
        Un_E(k,j,i)  = Up_E(k,j,i)  + dt * I_E(k,j,i);
      }

      for (int a=0; a<N; ++a)
      {
        M1_ILOOP1(i)
        {
          Un_F_d(a,k,j,i) = Up_F_d(a,k,j,i) + dt * I_F_d(a,k,j,i);
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