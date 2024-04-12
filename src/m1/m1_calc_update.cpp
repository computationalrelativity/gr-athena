// C++ standard headers
// ...

// Athena++ headers
#include "m1.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Calculate matter source contribution based on u, add to u_inh
void M1::AddMatterSources(AthenaArray<Real> & u, AthenaArray<Real> & u_inh)
{
  vars_Lab U { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U);

  vars_Lab I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, I);

  // required geometric quantities
  AT_C_sca & sc_alpha      = geom.sc_alpha;
  AT_C_sca & sc_sqrt_det_g = geom.sc_sqrt_det_g;

  // required matter quantities
  AT_C_sca & sc_W   = fidu.sc_W;
  AT_N_vec & sp_v_d = fidu.sp_v_d;
  AT_N_vec & sp_v_u = fidu.sp_v_u;

  // point to scratches -------------------------------------------------------
  AT_C_sca & sc_G_ = scratch.sc_G_;
  AT_C_sca & dotFv_ = pm1->scratch.sc_A_;

  // indicial ranges ----------------------------------------------------------
  const int il = pm1->mbi.il;
  const int iu = pm1->mbi.iu;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
    AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
    AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

    AT_C_sca & I_nG   = I.sc_nG( ix_g,ix_s);
    AT_C_sca & I_E    = I.sc_E(  ix_g,ix_s);
    AT_N_vec & I_F_d  = I.sp_F_d(ix_g,ix_s);

    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);
    AT_C_sca & sc_n    = lab_aux.sc_n(   ix_g,ix_s);

    AT_C_sca & sc_J   = rad.sc_J(ix_g,ix_s);
    AT_C_sca & sc_H_t = rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d = rad.sp_H_d(ix_g,ix_s);

    // radiation-matter variables
    AT_C_sca & sc_eta   = radmat.sc_eta(  ix_g,ix_s);
    AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
    AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

    AT_C_sca & sc_eta_0   = radmat.sc_eta_0(  ix_g,ix_s);
    AT_C_sca & sc_kap_a_0 = radmat.sc_kap_a_0(ix_g,ix_s);

    M1_ILOOP2(k, j)
    {
      dotFv_.ZeroClear();

      for (int a=0; a<N; ++a)
      M1_GLOOP1(i)
      {
        dotFv_(i) += U_F_d(a,k,j,i) * sp_v_u(a,k,j,i);
      }

      // prepare \tilde{n}
      Assemble::sc_G_(sc_G_, sc_W, U_E, sc_J, dotFv_,
                      opt.fl_E, opt.fl_J, opt.eps_E, k, j, il, iu);

      M1_ILOOP1(i)
      {
        sc_n(k,j,i) = U_nG(k,j,i) / sc_G_(i);
      }

      // prepare J
      M1_ILOOP1(i)
      {
        sc_J(k,j,i) = (U_E(k,j,i) - 2.0 * dotFv_(i));
      }

      for (int a=0; a<N; ++a)
      for (int b=0; b<N; ++b)
      M1_ILOOP1(i)
      {
        sc_J(k,j,i) += sp_v_u(a,k,j,i) * sp_v_u(b,k,j,i) * sp_P_dd(a,b,k,j,i);
      }

      M1_ILOOP1(i)
      {
        sc_J(k,j,i) = SQR(sc_W(k,j,i)) * sc_J(k,j,i);
      }

      // prepare H_t
      M1_ILOOP1(i)
      {
        sc_H_t(k,j,i) = sc_W(k,j,i) * (U_E(k,j,i) - sc_J(k,j,i) - dotFv_(i));
      }

      // prepare H_d
      for (int a=0; a<N; ++a)
      {
        M1_ILOOP1(i)
        {
          sp_H_d(a,k,j,i) = (
            U_F_d(a,k,j,i) - sc_J(k,j,i) * sp_v_d(a,k,j,i)
          );
        }

        for (int b=0; b<N; ++b)
        M1_ILOOP1(i)
        {
          sp_H_d(a,k,j,i) -= (
            sp_v_u(b,k,j,i) * sp_P_dd(b,a,k,j,i)
          );
        }

        M1_ILOOP1(i)
        {
          sp_H_d(a,k,j,i) = sc_W(k,j,i) * sp_H_d(a,k,j,i);
        }
      }

      // populate sources
      M1_ILOOP1(i)
      {
        I_nG(k,j,i) += sc_alpha(k,j,i) * (
          sc_sqrt_det_g(k,j,i) * sc_eta_0(k,j,i) -
          sc_kap_a_0(k,j,i) * sc_n(k,j,i)
        );

        I_E(k,j,i) += sc_alpha(k,j,i) * (
          (sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) -
           sc_kap_a(k,j,i) * sc_J(k,j,i)) * sc_W(k,j,i) -
          (sc_kap_a(k,j,i) + sc_kap_s(k,j,i)) * sc_H_t(k,j,i)
        );
      }

      for (int a=0; a<N; ++a)
      M1_ILOOP1(i)
      {
        I_F_d(a,k,j,i) += sc_alpha(k,j,i) * (
          (sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) -
           sc_kap_a(k,j,i) * sc_J(k,j,i)) * sc_W(k,j,i) * sp_v_d(a,k,j,i) -
          (sc_kap_a(k,j,i) + sc_kap_s(k,j,i)) * sp_H_d(a,k,j,i)
        );
      }
    }

  }

}

// ----------------------------------------------------------------------------
// Function to update the state vector
void M1::CalcUpdate(Real const dt,
                    AthenaArray<Real> & u_pre,
                    AthenaArray<Real> & u_cur,
		                AthenaArray<Real> & u_inh)
{
  AddMatterSources(u_pre, u_inh);

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