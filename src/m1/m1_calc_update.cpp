// C++ standard headers
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_closure.hpp"
#include "m1_macro.hpp"
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

    M1_ILOOP2(k,j)
    {
      dotFv_.ZeroClear();

      for (int a=0; a<N; ++a)
      M1_GLOOP1(i)
      {
        dotFv_(i) += U_F_d(a,k,j,i) * sp_v_u(a,k,j,i);
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

      // prepare \tilde{n}
      Assemble::sc_G_(sc_G_, sc_W, U_E, sc_J, dotFv_,
                      opt.fl_E, opt.fl_J, opt.eps_E, k, j, il, iu);

      M1_ILOOP1(i)
      {
        sc_n(k,j,i) = U_nG(k,j,i) / sc_G_(i);
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
// Explicit update strategy for state vector
void M1::CalcExplicitUpdate(Real const dt,
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

    M1_ILOOP2(k,j)
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

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (Picard iteration)
void M1::CalcImplicitUpdatePicardFrozenP(
  Real const dt,
  AthenaArray<Real> & u_pre,
  AthenaArray<Real> & u_cur,
  AthenaArray<Real> & u_inh)
{
  // (imp.) iterated state vector
  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_I);

  vars_Lab R { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, R);

  // u1 / previous
  vars_Lab U_o { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_o);

  // fiducial quantities
  AT_C_sca & sc_W = fidu.sc_W;
  AT_N_vec & sp_v_u  = fidu.sp_v_u;
  AT_N_vec & sp_v_d  = fidu.sp_v_d;

  // required geometric quantities
  AT_C_sca & sc_alpha      = geom.sc_alpha;
  AT_C_sca & sc_sqrt_det_g = geom.sc_sqrt_det_g;

  std::array<Real, 1> iI_E;
  std::array<Real, N> iI_F_d;

  // point to scratches -------------------------------------------------------
  AT_N_vec & sp_H_d_     = scratch.sp_vec_A_;

  // indicial ranges ----------------------------------------------------------
  const int il = pm1->mbi.il;
  const int iu = pm1->mbi.iu;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & R_nG   = R.sc_nG( ix_g,ix_s);
    AT_C_sca & R_E    = R.sc_E(  ix_g,ix_s);
    AT_N_vec & R_F_d  = R.sp_F_d(ix_g,ix_s);

    AT_C_sca & O_nG   = U_o.sc_nG( ix_g,ix_s);
    AT_C_sca & O_E    = U_o.sc_E(  ix_g,ix_s);
    AT_N_vec & O_F_d  = U_o.sp_F_d(ix_g,ix_s);

    AT_C_sca & I_nG   = U_I.sc_nG( ix_g,ix_s);
    AT_C_sca & I_E    = U_I.sc_E(  ix_g,ix_s);
    AT_N_vec & I_F_d  = U_I.sp_F_d(ix_g,ix_s);

    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_eta_0   = radmat.sc_eta_0(  ix_g,ix_s);
    AT_C_sca & sc_kap_a_0 = radmat.sc_kap_a_0(ix_g,ix_s);

    AT_C_sca & sc_eta   = radmat.sc_eta(  ix_g,ix_s);
    AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
    AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

    M1_ILOOP3(k,j,i)
    {
      // nop if mask has not been set
      if (!pm1->MaskGet(k,j,i))
      {
        continue;
      }

      // Iterate (S_1, S_{1+k})
      const int iter_max_P = opt.max_iter_P;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = opt.max_iter_P_rst;  // maximum number of restarts
      Real w_opt = opt.w_opt_ini;  // underrelaxation factor
      Real e_P_abs_tol = opt.eps_P_abs_tol;
      // maximum error amplification factor between iters.
      Real fac_PA = opt.fac_amp_P;

      // retain values for potential restarts
      iI_E[0] = I_E(k,j,i);
      for (int a=0; a<N; ++a)
      {
        iI_F_d[a] = I_F_d(a,k,j,i);
      }

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      const Real kap_as = sc_kap_a(k,j,i) + sc_kap_s(k,j,i);

      Real dotPvv (0);
      for (int a=0; a<N; ++a)
      for (int b=0; b<N; ++b)
      {
        dotPvv += sp_P_dd(a,b,k,j,i) * sp_v_u(a,k,j,i) * sp_v_u(b,k,j,i);
      }

      // solver loop ----------------------------------------------------------
      const Real W    = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      do
      {
        pit++;

        Real dotFv = Assemble::sc_dot_dense_sp__(I_F_d, sp_v_u, k, j, i);

        Real S_fac (0);
        S_fac = sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i);
        S_fac += sc_kap_s(k,j,i) * W2 * (
          I_E(k,j,i) - 2.0 * dotFv + dotPvv
        );

        Real S1 = sc_alpha(k,j,i) * W * (
          kap_as * (dotFv - I_E(k,j,i)) + S_fac
        );

        // debug ------------------------------------
        // const Real W  = W;
        // const Real W2 = SQR(W);
        // const Real J = W2 * (
        //   I_E(k,j,i) - 2.0 * dotFv + dotPvv
        // );

        // const Real H_n = W * (I_E(k,j,i) - J - dotFv);

        // for (int a=0; a<N; ++a)
        // {
        //   sp_H_d_(a,i) = W * (
        //     I_F_d(a,k,j,i) - J * sp_v_d(a,k,j,i)
        //   );
        //   for (int b=0; b<N; ++b)
        //   {
        //     sp_H_d_(a,i) -= W * sp_v_u(b,k,j,i) * sp_P_dd(a,b,k,j,i);
        //   }
        // }

        // S1 = sc_alpha(k,j,i) * (
        //   (sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) -
        //     sc_kap_a(k,j,i) * J) * W -
        //   (sc_kap_a(k,j,i) + sc_kap_s(k,j,i)) * H_n
        // );
        // -------------------------------------------

        Real WE = O_E(k,j,i) + dt * R_E(k,j,i);
        Real ZE = I_E(k,j,i) - dt * S1 - WE;

        // Ensure update preserves energy non-negativity
        I_E(k,j,i) = std::abs(I_E(k,j,i) - w_opt * ZE);
        I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

        e_abs_cur = std::abs(ZE);

        for (int a=0; a<N; ++a)
        {
          Real dotPv (0);
          for (int b=0; b<N; ++b)
          {
            dotPv += sp_P_dd(a,b,k,j,i) * sp_v_u(b,k,j,i);
          }

          Real S1pk = sc_alpha(k,j,i) * W * (
            kap_as * (dotPv - I_F_d(a,k,j,i)) + S_fac * sp_v_d(a,k,j,i)
          );

          // debug ------------------------------------
          // S1pk = sc_alpha(k,j,i) * (
          //   (sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i) -
          //   sc_kap_a(k,j,i) * J) * W * sp_v_d(a,k,j,i) -
          //   (sc_kap_a(k,j,i) + sc_kap_s(k,j,i)) * sp_H_d_(a,i)
          // );
          // ------------------------------------------

          Real WF_d = O_F_d(a,k,j,i) + dt * R_F_d(a,k,j,i);
          Real ZF_d = I_F_d(a,k,j,i) - dt * S1pk - WF_d;

          I_F_d(a,k,j,i) = I_F_d(a,k,j,i) - w_opt * ZF_d;

          e_abs_cur = std::max(std::abs(ZF_d), e_abs_cur);
        }

        // scale error tol by step
        // e_abs_cur = w_opt * e_abs_cur;

        if (e_abs_cur > fac_PA * e_abs_old)
        {
          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          I_E(k,j,i) = iI_E[0];
          for (int a=0; a<N; ++a)
          {
            I_F_d(a,k,j,i) = iI_F_d[a];
          }

          // restart iteration
          e_abs_old = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::CalcImplicitUpdatePicardFrozenP max restarts exceeded.";
            std::cout << msg.str().c_str() << std::endl;
            std::exit(0);
          }
        }
        else
        {
          e_abs_old = e_abs_cur;
        }

      } while ((pit < iter_max_P) && (e_abs_cur >= e_P_abs_tol));

      // Ensure energy is non-negative
      I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

      // Neutrino current evolution is linearly implicit; assemble directly
      // Note: could be moved outside loop but more convenient to put here
      const Real dotFv = Assemble::sc_dot_dense_sp__(I_F_d, sp_v_u, k, j, i);

      const Real I_J = Assemble::sc_J__(
        W2, dotFv, I_E, sp_v_u, sp_P_dd,
        k, j, i
      );

      const Real WnGam = O_nG(k,j,i) + dt * R_nG(k,j,i);
      const Real I_Gam = Assemble::sc_G_(
        W, I_E(k,j,i), I_J, dotFv,
        opt.fl_E, opt.fl_J, opt.eps_E
      );

      I_nG(k,j,i) = WnGam / (
        1.0 - dt * sc_alpha(k,j,i) * sc_kap_a_0(k,j,i) / I_Gam
      );

      // dump some information ------------------------------------------------
      if (opt.verbose_iter_P)
      {
        if (e_abs_cur >= e_P_abs_tol)
        {
          std::cout << "Tol. not achieved: " << e_abs_cur << std::endl;
        }
      }

    }

  }

}

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (Picard iteration)
//
// Semi-implicit iteration is performed (with closure)
void M1::CalcImplicitUpdatePicardMinerboP(
  Real const dt,
  AthenaArray<Real> & u_pre,
  AthenaArray<Real> & u_cur,
  AthenaArray<Real> & u_inh)
{
  // (imp.) iterated state vector
  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_I);

  vars_Lab R { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, R);

  // u1 / previous
  vars_Lab U_o { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_o);

  // fiducial quantities
  AT_C_sca & sc_W = fidu.sc_W;
  AT_N_vec & sp_v_u = fidu.sp_v_u;
  AT_N_vec & sp_v_d = fidu.sp_v_d;

  // required geometric quantities
  AT_C_sca & sc_alpha      = geom.sc_alpha;
  AT_C_sca & sc_sqrt_det_g = geom.sc_sqrt_det_g;
  AT_N_sym & sp_g_dd       = geom.sp_g_dd;
  AT_N_sym & sp_g_uu       = geom.sp_g_uu;

  // point to scratches -------------------------------------------------------
  AT_N_vec & sp_dH_d_    = scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = scratch.sp_sym_C_;

  std::array<Real, 1> iI_xi;
  std::array<Real, 1> iI_E;
  std::array<Real, N> iI_F_d;

  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // indicial ranges ----------------------------------------------------------
  const int il = mbi.il;
  const int iu = mbi.iu;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & R_nG   = R.sc_nG( ix_g,ix_s);
    AT_C_sca & R_E    = R.sc_E(  ix_g,ix_s);
    AT_N_vec & R_F_d  = R.sp_F_d(ix_g,ix_s);

    AT_C_sca & O_nG   = U_o.sc_nG( ix_g,ix_s);
    AT_C_sca & O_E    = U_o.sc_E(  ix_g,ix_s);
    AT_N_vec & O_F_d  = U_o.sp_F_d(ix_g,ix_s);

    AT_C_sca & I_nG   = U_I.sc_nG( ix_g,ix_s);
    AT_C_sca & I_E    = U_I.sc_E(  ix_g,ix_s);
    AT_N_vec & I_F_d  = U_I.sp_F_d(ix_g,ix_s);

    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);
    AT_C_sca & sc_chi  = lab_aux.sc_chi( ix_g,ix_s);
    AT_C_sca & sc_xi   = lab_aux.sc_xi(  ix_g,ix_s);

    AT_C_sca & sc_J    = rad.sc_J(  ix_g,ix_s);
    AT_C_sca & sc_H_t  = rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d  = rad.sp_H_d(ix_g,ix_s);

    AT_C_sca & sc_eta_0   = radmat.sc_eta_0(  ix_g,ix_s);
    AT_C_sca & sc_kap_a_0 = radmat.sc_kap_a_0(ix_g,ix_s);

    AT_C_sca & sc_eta   = radmat.sc_eta(  ix_g,ix_s);
    AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
    AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

    // initialize struct for root-finding
    Closures::Minerbo::DataRootfinder drf {
      sp_g_uu,
      I_E,
      I_F_d,
      sp_P_dd,
      sc_chi,
      sc_xi,
      sc_J,
      sc_H_t,
      sp_H_d,
      pm1->fidu.sc_W,
      pm1->fidu.sp_v_u,
      pm1->fidu.sp_v_d,
      sp_dH_d_,
      sp_P_tn_dd_,
      sp_P_tk_dd_,
      sp_dP_dd_
    };

    // // Shift explicit update onto I_* - this fixes the first guess
    // M1_ILOOP3(k,j,i)
    // {
    //   // nop if mask has not been set
    //   if (!pm1->MaskGet(k,j,i))
    //   {
    //     continue;
    //   }

    //   I_E(k,j,i) = O_E(k,j,i) + dt * R_E(k,j,i);
    //   I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

    //   for (int a=0; a<N; ++a)
    //   {
    //     I_F_d(a,k,j,i) = O_F_d(a,k,j,i) + dt * R_F_d(a,k,j,i);
    //   }
    // }

    // // Compute closure based on first guess
    // // This also updates internally stored sc_J, sc_H_t, sp_H_d
    // CalcClosure(u_cur);

    M1_ILOOP3(k,j,i)
    {
      // nop if mask has not been set
      if (!pm1->MaskGet(k,j,i))
      {
        continue;
      }

      // Iterate (S_1, S_{1+k})
      const int iter_max_P = opt.max_iter_P;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = opt.max_iter_P_rst;  // maximum number of restarts
      Real w_opt = opt.w_opt_ini;  // underrelaxation factor
      Real e_P_abs_tol = opt.eps_P_abs_tol;
      Real e_C_abs_tol = opt.eps_C;
      // maximum error amplification factor between iters.
      Real fac_PA   = opt.fac_amp_P;
      Real fac_PA_C = opt.fac_amp_C;

      // retain values for potential restarts
      iI_xi[0] = sc_xi(k,j,i);
      iI_E[0]  = I_E(k,j,i);
      for (int a=0; a<N; ++a)
      {
        iI_F_d[a] = I_F_d(a,k,j,i);
      }

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      Real e_abs_old_C = std::numeric_limits<Real>::infinity();
      Real e_abs_cur_C = 0;

      // solver loop ----------------------------------------------------------
      const Real kap_as = sc_kap_a(k,j,i) + sc_kap_s(k,j,i);

      const Real W    = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      drf.i = i;
      drf.j = j;
      drf.k = k;

      do
      {
        pit++;

        // Minerbo assembly ---------------------------------------------------
        const Real xi = sc_xi(k,j,i);
        sc_chi(k,j,i) = Closures::Minerbo::chi(sc_xi(k,j,i));

        Real dotFv = Assemble::sc_dot_dense_sp__(I_F_d, sp_v_u, k, j, i);

        Closures::ClosureThin(this, sp_P_tn_dd_, I_E, I_F_d, k, j, i);
        Closures::ClosureThick(
          this,
          sp_P_tk_dd_,
          dotFv,
          I_E,
          I_F_d,
          k, j, i);

        Closures::Minerbo::sp_P_dd(sp_P_dd,
                                   sc_chi,
                                   sp_P_tn_dd_,
                                   sp_P_tk_dd_,
                                   k, j, i);

        // assemble zero functional
        const Real Z_xi = Closures::Minerbo::R(xi, &drf);

        sc_xi(k,j,i) = std::abs(sc_xi(k,j,i) - w_opt * Z_xi);
        e_abs_cur_C = std::abs(Z_xi);

        // if iteration pushes outside admissible range reset
        if (opt.reset_thin)
        {
          if ((sc_xi(k,j,i) < xi_min) || (sc_xi(k,j,i) > xi_max))
          {
            sc_xi(k,j,i) = xi_max;
            sc_chi(k,j,i) = Closures::Minerbo::chi(sc_xi(k,j,i));

            for (int a=0; a<N; ++a)
            for (int b=a; b<N; ++b)
            {
              sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
            }
          }
          e_abs_cur_C = 0.0;
        }

        // state-vector non-linear subsystem ----------------------------------

        Real S_fac (0);
        S_fac = sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i);
        S_fac += sc_kap_s(k,j,i) * drf.sc_J(k,j,i);

        Real S1 = sc_alpha(k,j,i) * W * (
          kap_as * (dotFv - I_E(k,j,i)) + S_fac
        );

        Real WE = O_E(k,j,i) + dt * R_E(k,j,i);
        Real ZE = I_E(k,j,i) - dt * S1 - WE;
        // Ensure update preserves energy non-negativity
        I_E(k,j,i) = I_E(k,j,i) - w_opt * ZE;
        I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

        e_abs_cur = std::abs(ZE);

        for (int a=0; a<N; ++a)
        {
          Real dotPv (0);
          for (int b=0; b<N; ++b)
          {
            dotPv += sp_P_dd(a,b,k,j,i) * sp_v_u(b,k,j,i);
          }

          Real S1pk = sc_alpha(k,j,i) * W * (
            kap_as * (dotPv - I_F_d(a,k,j,i)) + S_fac * sp_v_d(a,k,j,i)
          );

          Real WF_d = O_F_d(a,k,j,i) + dt * R_F_d(a,k,j,i);
          Real ZF_d = I_F_d(a,k,j,i) - dt * S1pk - WF_d;

          I_F_d(a,k,j,i) = I_F_d(a,k,j,i) - w_opt * ZF_d;

          e_abs_cur = std::max(std::abs(ZF_d), e_abs_cur);
        }

        // scale error tol by step
        // e_abs_cur = w_opt * e_abs_cur;

        if (e_abs_cur > fac_PA * e_abs_old)
        {
          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          sc_xi(k,j,i) = iI_xi[0];
          I_E(k,j,i) = iI_E[0];
          for (int a=0; a<N; ++a)
          {
            I_F_d(a,k,j,i) = iI_F_d[a];
          }

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::CalcImplicitUpdatePicardMinerboP max restarts exceeded.";
            std::cout << msg.str().c_str() << std::endl;
            Closures::InfoDump(this, ix_g, ix_s, k, j, i);
            std::cout << e_abs_cur << "\n";
            std::cout << e_abs_old << "\n";
            std::exit(0);
          }

          // restart iteration
          e_abs_old   = std::numeric_limits<Real>::infinity();
          e_abs_old_C = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

        }
        else
        {
          e_abs_old   = e_abs_cur;
          e_abs_old_C = e_abs_cur_C;
        }

      } while ((pit < iter_max_P) &&
               (e_abs_cur >= e_P_abs_tol));

      // Ensure energy is non-negative
      I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

      // Neutrino current evolution is linearly implicit; assemble directly
      // Note: could be moved outside loop but more convenient to put here
      const Real dotFv = Assemble::sc_dot_dense_sp__(I_F_d, sp_v_u, k, j, i);

      const Real I_J = Assemble::sc_J__(
        W2, dotFv, I_E, drf.sp_v_u, drf.sp_P_dd,
        k, j, i
      );

      const Real WnGam = O_nG(k,j,i) + dt * R_nG(k,j,i);
      const Real I_Gam = Assemble::sc_G_(
        W, I_E(k,j,i), I_J, dotFv,
        opt.fl_E, opt.fl_J, opt.eps_E
      );

      I_nG(k,j,i) = WnGam / (
        1.0 - dt * sc_alpha(k,j,i) * sc_kap_a_0(k,j,i) / I_Gam
      );


      // dump some information ------------------------------------------------

      if (opt.verbose_iter_P)
      {
        if (e_abs_cur >= e_P_abs_tol)
        {
          std::cout << "M1::CalcImplicitUpdatePicardMinerboP:\n";
          std::cout << "Tol. not achieved: " << e_abs_cur << "\n";
          std::cout << "(pit,rit): " << pit << "," << rit << "\n";
          std::cout << "chi: " << sc_chi(k,j,i) << "\n";
        }
      }

    }

  }

}

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (Picard iteration)
//
// Closure is computed based on explicit update
// Semi-implicit iteration is performed (without closure)
// Closure is recomputed
//
// This gives incorrect speed in moving-medium diffusion test
void M1::CalcImplicitUpdatePicardMinerboPC(
  Real const dt,
  AthenaArray<Real> & u_pre,
  AthenaArray<Real> & u_cur,
  AthenaArray<Real> & u_inh)
{
  // (imp.) iterated state vector
  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_I);

  vars_Lab R { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, R);

  // u1 / previous
  vars_Lab U_o { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_o);

  // fiducial quantities
  AT_C_sca & sc_W = fidu.sc_W;
  AT_N_vec & sp_v_u = fidu.sp_v_u;
  AT_N_vec & sp_v_d = fidu.sp_v_d;

  // required geometric quantities
  AT_C_sca & sc_alpha      = geom.sc_alpha;
  AT_C_sca & sc_sqrt_det_g = geom.sc_sqrt_det_g;
  AT_N_sym & sp_g_dd       = geom.sp_g_dd;
  AT_N_sym & sp_g_uu       = geom.sp_g_uu;

  // point to scratches -------------------------------------------------------
  AT_N_vec & sp_dH_d_    = scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = scratch.sp_sym_C_;

  std::array<Real, 1> iI_xi;
  std::array<Real, 1> iI_E;
  std::array<Real, N> iI_F_d;

  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // indicial ranges ----------------------------------------------------------
  const int il = mbi.il;
  const int iu = mbi.iu;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & R_nG   = R.sc_nG( ix_g,ix_s);
    AT_C_sca & R_E    = R.sc_E(  ix_g,ix_s);
    AT_N_vec & R_F_d  = R.sp_F_d(ix_g,ix_s);

    AT_C_sca & O_nG   = U_o.sc_nG( ix_g,ix_s);
    AT_C_sca & O_E    = U_o.sc_E(  ix_g,ix_s);
    AT_N_vec & O_F_d  = U_o.sp_F_d(ix_g,ix_s);

    AT_C_sca & I_nG   = U_I.sc_nG( ix_g,ix_s);
    AT_C_sca & I_E    = U_I.sc_E(  ix_g,ix_s);
    AT_N_vec & I_F_d  = U_I.sp_F_d(ix_g,ix_s);

    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);
    AT_C_sca & sc_chi  = lab_aux.sc_chi( ix_g,ix_s);
    AT_C_sca & sc_xi   = lab_aux.sc_xi(  ix_g,ix_s);

    AT_C_sca & sc_J    = rad.sc_J(  ix_g,ix_s);
    AT_C_sca & sc_H_t  = rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d  = rad.sp_H_d(ix_g,ix_s);

    AT_C_sca & sc_eta_0   = radmat.sc_eta_0(  ix_g,ix_s);
    AT_C_sca & sc_kap_a_0 = radmat.sc_kap_a_0(ix_g,ix_s);

    AT_C_sca & sc_eta   = radmat.sc_eta(  ix_g,ix_s);
    AT_C_sca & sc_kap_a = radmat.sc_kap_a(ix_g,ix_s);
    AT_C_sca & sc_kap_s = radmat.sc_kap_s(ix_g,ix_s);

    // initialize struct for root-finding
    Closures::Minerbo::DataRootfinder drf {
      sp_g_uu,
      I_E,
      I_F_d,
      sp_P_dd,
      sc_chi,
      sc_xi,
      sc_J,
      sc_H_t,
      sp_H_d,
      pm1->fidu.sc_W,
      pm1->fidu.sp_v_u,
      pm1->fidu.sp_v_d,
      sp_dH_d_,
      sp_P_tn_dd_,
      sp_P_tk_dd_,
      sp_dP_dd_
    };

    // Shift explicit update onto I_* - this fixes the first guess
    M1_ILOOP3(k,j,i)
    {
      // nop if mask has not been set
      if (!pm1->MaskGet(k,j,i))
      {
        continue;
      }

      I_E(k,j,i) = O_E(k,j,i) + dt * R_E(k,j,i);
      I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

      for (int a=0; a<N; ++a)
      {
        I_F_d(a,k,j,i) = I_F_d(a,k,j,i) + dt * R_F_d(a,k,j,i);
      }
    }

    // Compute closure based on first guess
    // This also updates internally stored sc_J, sc_H_t, sp_H_d
    CalcClosure(u_cur);

    // Semi-implicit loop -----------------------------------------------------
    M1_ILOOP3(k,j,i)
    {
      // nop if mask has not been set
      if (!pm1->MaskGet(k,j,i))
      {
        continue;
      }

      // Iterate (S_1, S_{1+k})
      const int iter_max_P = opt.max_iter_P;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = opt.max_iter_P_rst;  // maximum number of restarts
      Real w_opt = opt.w_opt_ini;  // underrelaxation factor
      Real e_P_abs_tol = opt.eps_P_abs_tol;
      Real e_C_abs_tol = opt.eps_C;
      // maximum error amplification factor between iters.
      Real fac_PA   = opt.fac_amp_P;
      Real fac_PA_C = opt.fac_amp_C;

      // retain values for potential restarts
      iI_xi[0] = sc_xi(k,j,i);
      iI_E[0]  = I_E(k,j,i);
      for (int a=0; a<N; ++a)
      {
        iI_F_d[a] = I_F_d(a,k,j,i);
      }

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      Real e_abs_old_C = std::numeric_limits<Real>::infinity();
      Real e_abs_cur_C = 0;

      // solver loop ----------------------------------------------------------
      const Real kap_as = sc_kap_a(k,j,i) + sc_kap_s(k,j,i);

      const Real W    = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      drf.i = i;
      drf.j = j;
      drf.k = k;

      do
      {
        pit++;

        // state-vector non-linear subsystem ----------------------------------
        Real dotFv = Assemble::sc_dot_dense_sp__(I_F_d, sp_v_u, k, j, i);

        Real S_fac (0);
        S_fac = sc_sqrt_det_g(k,j,i) * sc_eta(k,j,i);
        S_fac += sc_kap_s(k,j,i) * drf.sc_J(k,j,i);

        Real S1 = sc_alpha(k,j,i) * W * (
          kap_as * (dotFv - I_E(k,j,i)) + S_fac
        );

        Real WE = O_E(k,j,i) + dt * R_E(k,j,i);
        Real ZE = I_E(k,j,i) - dt * S1 - WE;
        // Ensure update preserves energy non-negativity
        I_E(k,j,i) = I_E(k,j,i) - w_opt * ZE;
        I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

        e_abs_cur = std::abs(ZE);

        for (int a=0; a<N; ++a)
        {
          Real dotPv (0);
          for (int b=0; b<N; ++b)
          {
            dotPv += sp_P_dd(a,b,k,j,i) * sp_v_u(b,k,j,i);
          }

          Real S1pk = sc_alpha(k,j,i) * W * (
            kap_as * (dotPv - I_F_d(a,k,j,i)) + S_fac * sp_v_d(a,k,j,i)
          );

          Real WF_d = O_F_d(a,k,j,i) + dt * R_F_d(a,k,j,i);
          Real ZF_d = I_F_d(a,k,j,i) - dt * S1pk - WF_d;

          I_F_d(a,k,j,i) = I_F_d(a,k,j,i) - w_opt * ZF_d;

          e_abs_cur = std::max(std::abs(ZF_d), e_abs_cur);
        }

        // scale error tol by step
        // e_abs_cur = w_opt * e_abs_cur;

        if (e_abs_cur > fac_PA * e_abs_old)
        {
          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          sc_xi(k,j,i) = iI_xi[0];
          I_E(k,j,i) = iI_E[0];
          for (int a=0; a<N; ++a)
          {
            I_F_d(a,k,j,i) = iI_F_d[a];
          }

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::CalcImplicitUpdatePicardMinerboPC max restarts exceeded.";
            std::cout << msg.str().c_str() << std::endl;
            Closures::InfoDump(this, ix_g, ix_s, k, j, i);
            std::cout << e_abs_cur << "\n";
            std::cout << e_abs_old << "\n";
            std::exit(0);
          }

          // restart iteration
          e_abs_old   = std::numeric_limits<Real>::infinity();
          e_abs_old_C = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

        }
        else
        {
          e_abs_old   = e_abs_cur;
          e_abs_old_C = e_abs_cur_C;
        }

      } while ((pit < iter_max_P) &&
               (e_abs_cur >= e_P_abs_tol));


      // Ensure energy is non-negative
      I_E(k,j,i) = std::max(opt.fl_E, I_E(k,j,i));

      // Neutrino current evolution is linearly implicit; assemble directly
      // Note: could be moved outside loop but more convenient to put here
      const Real dotFv = Assemble::sc_dot_dense_sp__(I_F_d, sp_v_u, k, j, i);

      const Real I_J = Assemble::sc_J__(
        W2, dotFv, I_E, drf.sp_v_u, drf.sp_P_dd,
        k, j, i
      );

      const Real WnGam = O_nG(k,j,i) + dt * R_nG(k,j,i);
      const Real I_Gam = Assemble::sc_G_(
        W, I_E(k,j,i), I_J, dotFv,
        opt.fl_E, opt.fl_J, opt.eps_E
      );

      I_nG(k,j,i) = WnGam / (
        1.0 - dt * sc_alpha(k,j,i) * sc_kap_a_0(k,j,i) / I_Gam
      );


      // dump some information ------------------------------------------------

      if (opt.verbose_iter_P)
      {
        if (e_abs_cur >= e_P_abs_tol)
        {
          std::cout << "M1::CalcImplicitUpdatePicardMinerboPC:\n";
          std::cout << "Tol. not achieved: " << e_abs_cur << "\n";
          std::cout << "(pit,rit): " << pit << "," << rit << "\n";
          std::cout << "chi: " << sc_chi(k,j,i) << "\n";
        }
      }

    }

    // Compute closure based on updated solution
    CalcClosure(u_cur);

  }

}


// ----------------------------------------------------------------------------
// Function to update the state vector
void M1::CalcUpdate(Real const dt,
                    AthenaArray<Real> & u_pre,
                    AthenaArray<Real> & u_cur,
		                AthenaArray<Real> & u_inh)
{

  if (opt.value_inject)
  {
    std::cout << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "No sources-----------------------------------------------\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "\n";

    pm1->StatePrintPoint(0, 0,
                         mbi.kl, mbi.ju-1, mbi.iu-1, false);
    std::cout << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "AddGRSources---------------------------------------------\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "\n";

    u_inh.ZeroClear();
    AddGRSources(u_cur, u_inh);

    SetVarAliasesLab(u_inh, lab);
    pm1->StatePrintPoint(0, 0,
                         mbi.kl, mbi.ju-1, mbi.iu-1, false);

    std::cout << "\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "AddMatterSources-----------------------------------------\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "\n";

    u_inh.ZeroClear();
    AddMatterSources(u_cur, u_inh);

    SetVarAliasesLab(u_inh, lab);
    pm1->StatePrintPoint(0, 0,
                         mbi.kl, mbi.ju-1, mbi.iu-1, true);
  }


  AddGRSources(u_cur, u_inh);

  switch (opt.integration_strategy)
  {
    case (opt_integration_strategy::full_explicit):
    {
      AddMatterSources(u_pre, u_inh);
      CalcExplicitUpdate(dt, u_pre, u_cur, u_inh);
      break;
    }
    case (opt_integration_strategy::semi_implicit_PicardFrozenP):
    {
      CalcImplicitUpdatePicardFrozenP(dt, u_pre, u_cur, u_inh);
      break;
    }
    case (opt_integration_strategy::semi_implicit_PicardMinerboP):
    {
      CalcImplicitUpdatePicardMinerboP(dt, u_pre, u_cur, u_inh);
      break;
    }
    case (opt_integration_strategy::semi_implicit_PicardMinerboPC):
    {
      CalcImplicitUpdatePicardMinerboPC(dt, u_pre, u_cur, u_inh);
      break;
    }
    default:
    {
      assert(false);
      std::exit(0);
    }
  }

}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//