// c++
#include <iostream>

// Athena++ headers
#include "m1_calc_closure.hpp"
#include "m1_utils.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

// ============================================================================
namespace M1::Closures {
// ============================================================================

void InfoDump(M1 * pm1, const int ix_g, const int ix_s,
              const int k, const int j, const int i)
{
  std::cout << "(k,j,i) " << k << "," << j << "," << i << "\n";
  std::cout << "sc_W "    << pm1->fidu.sc_W(  k,j,i) << "\n";
  std::cout << "sc_xi "   << pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i) << "\n";
  std::cout << "sc_chi "  << pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i) << "\n";
  std::cout << "sc_E "    << pm1->lab.sc_E(ix_g,ix_s)(k,j,i) << "\n";

  std::cout << "sp_F_d:" << "\n";

  for (int a=0; a<N; ++a)
  {
    std::cout << "a " << a << " "
              << pm1->lab.sp_F_d(ix_g,ix_s)(a,k,j,i) << "\n";
  }

  std::cout << "sc_J:" << pm1->rad.sc_J(ix_g,ix_s)(k,j,i) << "\n";
  std::cout << "sc_H_t:" << pm1->rad.sc_H_t(ix_g,ix_s)(k,j,i) << "\n";

  std::cout << "sp_H_d:" << "\n";

  for (int a=0; a<N; ++a)
  {
    std::cout << "a " << a << " "
              << pm1->rad.sp_H_d(ix_g,ix_s)(a,k,j,i) << "\n";
  }

  std::cout << "sp_P_dd:" << "\n";

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    std::cout << "(a,b) " << a << "," << b << " "
              << pm1->lab_aux.sp_P_dd(ix_g,ix_s)(a,b,k,j,i) << "\n";
  }
}

// P_{i j} in Eq.(15) of [1] - works with densitized variables
//
void ClosureThin(M1 * pm1,
                 const Real weight,
                 const int ix_g,
                 const int ix_s,
                 AT_N_sym & sp_P_dd_,
                 const int k, const int j,
                 const int il, const int iu)
{
  AT_N_vec & sp_F_d  = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_E    = pm1->lab.sc_E(  ix_g,ix_s);

  // point to scratch
  AT_C_sca & sc_norm2_F_ = pm1->scratch.sc_A_;

  Assemble::sp_norm2_(sc_norm2_F_,
                      sp_F_d,
                      pm1->geom.sp_g_uu,
                      k, j, il, iu);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    const Real F2 = sc_norm2_F_(i);
    const Real fac = (F2 > 0) ? sc_E(k,j,i) / F2
                              : 0.0;
    sp_P_dd_(a,b,i) = weight * fac * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
  }
}

void ClosureThin(M1 * pm1,
                 AT_N_sym & sp_P_dd_,
                 const AT_C_sca & sc_E,
                 const AT_N_vec & sp_F_d,
                 const int k, const int j, const int i)
{
  const Real nF2 = Assemble::sp_norm2__(sp_F_d, pm1->geom.sp_g_uu, k, j, i);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    const Real fac = (nF2 > 0) ? sc_E(k,j,i) / nF2
                               : 0.0;
    sp_P_dd_(a,b,i) = fac * sp_F_d(a,k,j,i) * sp_F_d(b,k,j,i);
  }
}

void SetClosureThin(M1 * pm1,
                    const Real weight,
                    const int ix_g,
                    const int ix_s,
                    AT_N_sym & sp_P_dd)
{
  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_N_sym & sp_P_dd_ = pm1->scratch.sp_sym_A_;

  M1_GLOOP2(k,j)
  {
    ClosureThin(pm1, weight, ix_g, ix_s, sp_P_dd_, k, j, il, iu);
    Assemble::ScratchToDense(sp_P_dd, sp_P_dd_, k, j, il, iu);
  }
}

// Function sets:
// lab_aux.sp_P_dd += wei * P_dd (thick)
inline void ClosureThick(M1 * pm1,
                         const Real weight,
                         const int ix_g,
                         const int ix_s,
                         AT_N_sym & sp_P_dd_,
                         const int k, const int j,
                         const int il, const int iu)
{

  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);

  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);
  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);

  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;
  AT_C_sca & sc_W   = pm1->fidu.sc_W;

  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;

  // point to available scratches
  AT_C_sca & sc_dot_Fv_ = pm1->scratch.sc_A_;
  AT_C_sca & W2_        = pm1->scratch.sc_B_;

  // --------------------------------------------------------------------------
  Assemble::sc_dot_dense_sp_(sc_dot_Fv_,
                             sp_F_d,
                             sp_v_u,
                             k, j, il, iu);

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    W2_(i) = SQR(sc_W(k,j,i));
  }

  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sc_J(k,j,i) = 3.0 / (2.0 * W2_(i) + 1.0) * (
      (2.0 * W2_(i) - 1.0) * sc_E(k,j,i) -
      2.0 * W2_(i) * sc_dot_Fv_(i)
    );
  }

  for (int a=0; a<N; ++a)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_H_d(a,k,j,i) = (
      1.0 / sc_W(k,j,i) * sp_F_d(a,k,j,i) +
      sc_W(k,j,i) * sp_v_d(a,k,j,i) / (2.0 * W2_(i) + 1) *
        (
          (4.0 * W2_(i) + 1) * sc_dot_Fv_(i) -
          4.0 * W2_(i) * sc_E(k,j,i)
        )
    );
  }

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_P_dd_(a,b,i) = weight * (
      4.0 * ONE_3RD * W2_(i) * sc_J(k,j,i) *
      sp_v_d(a,k,j,i) * sp_v_d(b,k,j,i) +
      sc_W(k,j,i) * (
        sp_v_d(a,k,j,i) * sp_H_d(b,k,j,i) +
        sp_v_d(b,k,j,i) * sp_H_d(a,k,j,i)
      ) +
      ONE_3RD * sc_J(k,j,i) * sp_g_dd(a,b,k,j,i)
    );
  }

}

void ClosureThick(M1 * pm1,
                  AT_N_sym & sp_P_dd_,
                  const Real dotFv,
                  const AT_C_sca & sc_E,
                  const AT_N_vec & sp_F_d,
                  const int k, const int j, const int i)
{
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;
  AT_C_sca & sc_W   = pm1->fidu.sc_W;

  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;

  // --------------------------------------------------------------------------

  const Real W    = sc_W(k,j,i);
  const Real oo_W = 1.0 / W;
  const Real W2   = SQR(W);

  const Real J_tk = 3.0 / (2.0 * W2 + 1.0) * (
    (2.0 * W2 - 1.0) * sc_E(k,j,i) - 2.0 * W2 * dotFv
  );

  const Real fac_H_tk = W /  (2.0 * W2 + 1.0) * (
    (4.0 * W2 + 1.0) * dotFv - 4.0 * W2 * sc_E(k,j,i)
  );

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    const Real H_a_tk = oo_W * sp_F_d(a,k,j,i) +
                        fac_H_tk * sp_v_d(a,k,j,i);
    const Real H_b_tk = oo_W * sp_F_d(b,k,j,i) +
                        fac_H_tk * sp_v_d(b,k,j,i);

    sp_P_dd_(a,b,i) = (
      4.0 * ONE_3RD * W2 * J_tk * sp_v_d(a,k,j,i) * sp_v_d(b,k,j,i) +
      W * (sp_v_d(a,k,j,i) * H_b_tk +
           sp_v_d(b,k,j,i) * H_a_tk) +
      ONE_3RD * J_tk * sp_g_dd(a,b,k,j,i)
    );
  }

}

void SetClosureThick(M1 * pm1,
                     const Real weight,
                     const int ix_g,
                     const int ix_s,
                     AT_N_sym & sp_P_dd)
{
  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_N_sym & sp_P_dd_ = pm1->scratch.sp_sym_B_;

  M1_GLOOP2(k,j)
  {
    ClosureThick(pm1, weight, ix_g, ix_s, sp_P_dd_, k, j, il, iu);
    Assemble::ScratchToDense(sp_P_dd, sp_P_dd_, k, j, il, iu);
  }
}
// ============================================================================
} // namespace M1::Closures
// ============================================================================

// ============================================================================
namespace M1::Closures::Minerbo {
// ============================================================================

Real R(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  // assemble P_dd
  drf->sc_xi( k,j,i) = xi;
  drf->sc_chi(k,j,i) = chi(xi);

  sp_P_dd(
    drf->sp_P_dd, drf->sc_chi, drf->sp_P_tn_dd_, drf->sp_P_tk_dd_,
    k, j, i, i
  );

  const Real W  = drf->sc_W(k,j,i);
  const Real W2 = SQR(W);

  drf->dotFv = Assemble::sc_dot_dense_sp__(
    drf->sp_F_d,
    drf->sp_v_u,
    k, j, i
  );

  // assemble J
  drf->sc_J(k,j,i) = Assemble::sc_J__(
    W2, drf->dotFv, drf->sc_E, drf->sp_v_u, drf->sp_P_dd,
    k, j, i
  );

  // drf->sc_J(k,j,i) = std::max(drf->sc_J(k,j,i), 1e-14);

  // assemble H_t
  drf->sc_H_t(k,j,i) = Assemble::sc_H_t__(
    W, drf->dotFv, drf->sc_E, drf->sc_J,
    k, j, i
  );

  // assemble H_d
  Assemble::sp_H_d__(drf->sp_H_d, W, drf->sc_J, drf->sp_F_d,
                     drf->sp_v_d, drf->sp_v_u, drf->sp_P_dd,
                     k, j, i);

  // assemble sc_H_st
  Real sc_H2_st = Assemble::sc_H2_st__(
    drf->sc_H_t, drf->sp_H_d, drf->sp_g_uu,
    k, j, i
  );

  // assemble R
  // const Real R_ = (std::abs(drf->sc_E(k,j,i)) > 0)
  //   ? (SQR(xi) * SQR(drf->sc_J(k,j,i)) - sc_H2_st) / drf->sc_E(k,j,i)
  //   : 0.0;

  const Real R_ = (SQR(xi * drf->sc_J(k,j,i)) - sc_H2_st);

  drf->Z_xi_im2 = drf->Z_xi_im1;
  drf->Z_xi_im1 = drf->Z_xi_i;
  drf->Z_xi_i = R_;

  drf->xi_im2 = drf->xi_im1;
  drf->xi_im1 = drf->xi_i;
  drf->xi_i   = xi;

  return R_;
}

Real dR(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  const Real W  = drf->sc_W(k,j,i);
  const Real W2 = SQR(W);

  // derivative of Z_xi
  Real dJ (0);
  Real dHH (0);

  const Real J  = drf->sc_J(  k,j,i);
  const Real Hn = drf->sc_H_t(k,j,i);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    drf->sp_dP_dd_(a,b,i) = 3.0 / 5.0 * (
      drf->sp_P_tn_dd_(a,b,i) - drf->sp_P_tk_dd_(a,b,i)
    ) * xi * (2.0 - xi + 4.0 * SQR(xi));
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dJ += drf->sp_v_u(a,k,j,i) *
          drf->sp_v_u(b,k,j,i) *
          drf->sp_dP_dd_(a,b,i);
  }

  dJ = W * dJ;

  const Real dHn = -W * dJ;

  for (int a=0; a<N; ++a)
  {
    drf->sp_dH_d_(a,i) = drf->sp_v_d(a,k,j,i) * dJ;
    for (int b=0; b<N; ++b)
    {
      drf->sp_dH_d_(a,i) += drf->sp_v_u(b,k,j,i) * drf->sp_dP_dd_(a,b,i);
    }
    drf->sp_dH_d_(a,i) = -W * drf->sp_dH_d_(a,i);
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dHH += drf->sp_g_uu(a,b,k,j,i) *
           drf->sp_dH_d_(a,i) *
           drf->sp_H_d(b,k,j,i);
  }

  const Real dR_ = 2 * xi * SQR(J) + 2 * J * SQR(xi) * dJ + 2 * Hn * dHn - dHH;
  drf->dZ_xi_im2 = drf->Z_xi_im1;
  drf->dZ_xi_im1 = drf->Z_xi_i;
  drf->dZ_xi_i = dR_;

  return dR_;
}

/*
Real d_R(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  // assemble P_dd
  drf->sc_xi( k,j,i) = xi;
  drf->sc_chi(k,j,i) = chi(xi);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    drf->sp_P_dd(a,b,k,j,i) = 0.5 * (
      (3.0 * drf->sc_chi(k,j,i) - 1.0) * drf->sp_P_tn_dd_(a,b,i) +
      3.0 * (1.0 - drf->sc_chi(k,j,i)) * drf->sp_P_tk_dd_(a,b,i)
    );
  }

  const Real W  = drf->sc_W(k,j,i);
  const Real W2 = SQR(W);

  Real dotFv = Assemble::sc_dot_dense_sp__(
    drf->sp_F_d,
    drf->sp_v_u,
    k, j, i
  );

  // assemble J
  drf->sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, drf->sc_E, drf->sp_v_u, drf->sp_P_dd,
    k, j, i
  );

  // drf->sc_J(k,j,i) = std::max(drf->sc_J(k,j,i), 1e-14);

  // assemble H_t
  drf->sc_H_t(k,j,i) = Assemble::sc_H_t__(
    W, dotFv, drf->sc_E, drf->sc_J,
    k, j, i
  );

  // assemble H_d
  Assemble::sp_H_d__(drf->sp_H_d, W, drf->sc_J, drf->sp_F_d,
                     drf->sp_v_d, drf->sp_v_u, drf->sp_P_dd,
                     k, j, i);

  // assemble sc_H_st
  Real sc_H2_st = Assemble::sc_H2_st__(
    drf->sc_H_t, drf->sp_H_d, drf->sp_g_uu,
    k, j, i
  );

  // assemble Zxi
  return (SQR(xi * drf->sc_J(k,j,i)) - sc_H2_st);
}

Real d_dR(Real xi, void *par)
{
  DataRootfinder * drf = reinterpret_cast<DataRootfinder*>(par);

  const int i = drf->i;
  const int j = drf->j;
  const int k = drf->k;

  // assemble P_dd
  drf->sc_xi( k,j,i) = xi;
  drf->sc_chi(k,j,i) = chi(xi);

  const Real W  = drf->sc_W(k,j,i);
  const Real W2 = SQR(W);

  // derivative of Z_xi
  Real dJ (0);
  Real dHH (0);

  const Real J  = drf->sc_J(  k,j,i);
  const Real Hn = drf->sc_H_t(k,j,i);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    drf->sp_dP_dd_(a,b,i) = 3.0 / 5.0 * (
      drf->sp_P_tn_dd_(a,b,i) - drf->sp_P_tk_dd_(a,b,i)
    ) * xi * (2.0 - xi + 4.0 * SQR(xi));
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dJ += drf->sp_v_u(a,k,j,i) *
          drf->sp_v_u(b,k,j,i) *
          drf->sp_dP_dd_(a,b,i);
  }

  dJ = W * dJ;

  const Real dHn = -W * dJ;

  for (int a=0; a<N; ++a)
  {
    drf->sp_dH_d_(a,i) = drf->sp_v_d(a,k,j,i) * dJ;
    for (int b=0; b<N; ++b)
    {
      drf->sp_dH_d_(a,i) += drf->sp_v_u(b,k,j,i) * drf->sp_dP_dd_(a,b,i);
    }
    drf->sp_dH_d_(a,i) = -W * drf->sp_dH_d_(a,i);
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dHH += drf->sp_g_uu(a,b,k,j,i) *
           drf->sp_dH_d_(a,i) *
           drf->sp_H_d(b,k,j,i);
  }

  return (
    2 * xi * SQR(J) + 2 * J * SQR(xi) * dJ + 2 * Hn * dHn - dHH
  );
}

void d_RdR(Real xi, void *par, Real *R_, Real *dR_)
{
  *R_  = d_R(xi, par);
  *dR_ = d_dR(xi, par);
}
*/

void AddClosure(M1 * pm1,
                const Real weight,
                const int ix_g,
                const int ix_s,
                AT_N_sym & sp_P_dd)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_chi = pm1->lab_aux.sc_chi(ix_g,ix_s);
  AT_C_sca & sc_xi  = pm1->lab_aux.sc_xi( ix_g,ix_s);

  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
  AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);

  // required geometric quantities
  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
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

  // setup GSL ----------------------------------------------------------------
  gsl_set_error_handler_off();

  gsl_function R_;
  R_.function = &R;
  R_.params = &drf;

  gsl_root_fsolver * gsl_solver = gsl_root_fsolver_alloc(
    gsl_root_fsolver_brent
  );

  auto gsl_err_kill = [&](const int status)
  {
    std::ostringstream msg;
    msg << "M1::Closures::Minerbo::AddClosure unexpected error: ";
    msg << status;

    std::cout << msg.str().c_str() << std::endl;

    ATHENA_ERROR(msg);
  };

  auto gsl_err_warn = [&]()
  {
    std::ostringstream msg;
    msg << "M1::Closures::Minerbo::AddClosure maxiter=";
    msg << pm1->opt.max_iter_C << " exceeded";
    std::cout << msg.str().c_str() << std::endl;
  };
  // --------------------------------------------------------------------------

  // main loop
  // int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    Closures::ClosureThin( pm1, 1.0, ix_g, ix_s, sp_P_tn_dd_, k, j, il, iu);
    Closures::ClosureThick(pm1, 1.0, ix_g, ix_s, sp_P_tk_dd_, k, j, il, iu);

    M1_GLOOP1(i)
    {
      // nop if mask has not been set
      if (!pm1->MaskGet(k,j,i))
      {
        continue;
      }

      drf.i = i;
      drf.j = j;
      drf.k = k;

      // // Check limits
      // const Real xi_old = drf.sc_xi(k,j,i);

      // Real e_abs_cur = std::abs(R(xi_min, &drf));
      // if (e_abs_cur < pm1->opt.eps_C)
      // {
      //   continue;
      // }

      // e_abs_cur = std::abs(R(xi_max, &drf));
      // if (e_abs_cur < pm1->opt.eps_C)
      // {
      //   continue;
      // }

      int status = gsl_root_fsolver_set(gsl_solver, &R_, xi_min, xi_max);

      /*
      typedef struct
        {
          double a, b, c, d, e;
          double fa, fb, fc;
        }
      brent_state_t;

      brent_state_t * state = static_cast<brent_state_t*>(gsl_solver->state);
      */

      switch (status)
      {
        case (GSL_EINVAL):  // bracketing failed (revert to thin closure)
        {
          Assemble::ScratchToDense(sp_P_dd, sp_P_tn_dd_, k, j, i, i);
          break;
        }
        case (0):
        {
          // root-finding loop
          Real loc_xi_min = xi_min;
          Real loc_xi_max = xi_max;

          status = GSL_CONTINUE;
          for (int iter=1;
               iter<=pm1->opt.max_iter_C && status == GSL_CONTINUE;
               ++iter)
          {
            status = gsl_root_fsolver_iterate(gsl_solver);

            if (status != GSL_SUCCESS)
            {
              break;
            }
            else if (status)
            {
              sp_P_tn_dd_.array().print_all();
              sp_P_tk_dd_.array().print_all();

              Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

              gsl_err_kill(status);
            }

            loc_xi_min = gsl_root_fsolver_x_lower(gsl_solver);
            loc_xi_max = gsl_root_fsolver_x_upper(gsl_solver);

            status = gsl_root_test_interval(
              loc_xi_min, loc_xi_max, pm1->opt.eps_C, 0
            );

            // if (std::abs(drf.Z_xi_i) < 1e-30)
            // {
            //   status = GSL_SUCCESS;
            //   break;
            // }


            // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-2)
            // {
            //   status = GSL_SUCCESS;
            //   break;
            // }

            // if (sc_E(k,j,i) < 1e-13)
            // {
            //   status = GSL_SUCCESS;
            //   R_.function(xi_min, &drf);
            //   break;
            // }

            if (0) // (iter > 0)
            {
              const Real p0 = drf.xi_im2;
              const Real p1 = drf.xi_im1;
              const Real p2 = drf.xi_i;

              const Real pe = p0-SQR(p1-p0)/(p2-2*p1+p0);

              // loc_xi_min = (SIGN(drf.Z_xi_i) == SIGN(drf.Z_xi_im2)) ? pe : loc_xi_min;
              // loc_xi_max = (SIGN(drf.Z_xi_i) == SIGN(drf.Z_xi_im1)) ? pe : loc_xi_max;
              // loc_xi_min = pe;
              // loc_xi_max = pe;
              if ((loc_xi_min<pe) && (pe<loc_xi_max))
              {
                // std::cout << "accepted" << std::endl;
                // R(pe, &drf);
                drf.xi_i = pe;

                // loc_xi_min = (SIGN(drf.Z_xi_i) == SIGN(drf.Z_xi_im1)) ? pe : loc_xi_min;
                // loc_xi_max = (SIGN(drf.Z_xi_i) == SIGN(drf.Z_xi_im2)) ? pe : loc_xi_max;

                const Real dx = (loc_xi_max - loc_xi_min) * 1e-3;
                // gsl_solver->x_lower = pe-dx;
                // gsl_solver->x_upper = pe+dx;
                gsl_root_fsolver_set(gsl_solver, &R_, pe-dx, pe+dx);

                if (0) // (iter > 10)
                {
                  std::cout << "=======\n";
                  std::cout << loc_xi_min << "\n";
                  std::cout << pe << "\n";
                  std::cout << loc_xi_max << "\n";

                  std::cout << "Z_xi: \n";
                  std::cout << drf.Z_xi_i << "\n";
                  std::cout << drf.Z_xi_im1 << "\n";
                  std::cout << drf.Z_xi_im2 << "\n";
                  std::cout << "=======\n";


                  // std::exit(0);

                }

              }

            }

            if (0) // (iter > 30)
            {
              std::cout << loc_xi_min << "\n";
              std::cout << loc_xi_max << "\n";
              std::cout << "Z_xi: \n";
              std::cout << drf.Z_xi_i << "\n";
              std::cout << drf.Z_xi_im1 << "\n";
              std::cout << drf.Z_xi_im2 << "\n";
              std::cout << "xi: \n";
              std::cout << drf.xi_i << "\n";
              std::cout << drf.xi_im1 << "\n";
              std::cout << drf.xi_im2 << "\n";

              const Real p0 = drf.xi_im2;
              const Real p1 = drf.xi_im1;
              const Real p2 = drf.xi_i;

              const Real ex = p0-SQR(p1-p0)/(p2-2*p1+p0);

              std::cout << ex << std::endl;
              std::cout << R(ex, &drf) << std::endl;

              Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
              std::exit(0);
            }
            // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-5)
            // {
            //   status = GSL_SUCCESS;
            //   break;
            // }


            // if (iter > 0.5 * static_cast<Real>(pm1->opt.max_iter_C))
            // {
            //   std::cout << "iter slow w/" << std::endl;
            //   std::cout << pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i) << std::endl;

            //   std::cout << R_.function(xi_min, &drf) << std::endl;
            //   std::cout << R_.function(gsl_root_fsolver_root(gsl_solver), &drf) << std::endl;

            // }

            // if (status == GSL_SUCCESS)
            // {
            //   std::cout << iter << std::endl;
            // }

            // iter_tot++;
          }

          if ((status != GSL_SUCCESS) && pm1->opt.verbose_iter_C)
          {
            gsl_err_warn();
          }

          // Final call of R to update sp_P_dd with new root info.
          R_.function(gsl_root_fsolver_root(gsl_solver), &drf);

          break;
        }
        default:
        {
          Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
          gsl_err_kill(status);
        }
      }
    }
  }

  // const int nn = pm1->mbi.nn1 * pm1->mbi.nn2 * pm1->mbi.nn3;
  // std::cout << static_cast<Real>(iter_tot) / nn << std::endl;

  // cleanup ------------------------------------------------------------------
  gsl_root_fsolver_free(gsl_solver);
  // TODO: bug - this can't be deactivated properly with OMP
  // gsl_set_error_handler(NULL);  // restore default handler
}

void AddClosureP(M1 * pm1,
                 const Real weight,
                 const int ix_g,
                 const int ix_s,
                 AT_N_sym & sp_P_dd)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  // fiducial quantities
  AT_C_sca & sc_W   = pm1->fidu.sc_W;
  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;

  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_chi = pm1->lab_aux.sc_chi(ix_g,ix_s);
  AT_C_sca & sc_xi  = pm1->lab_aux.sc_xi( ix_g,ix_s);

  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
  AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);

  // required geometric quantities
  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
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

  std::array<Real, 1> iI_xi;

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    Closures::ClosureThin( pm1, 1.0, ix_g, ix_s, sp_P_tn_dd_, k, j, il, iu);
    Closures::ClosureThick(pm1, 1.0, ix_g, ix_s, sp_P_tk_dd_, k, j, il, iu);

    M1_GLOOP1(i)
    {
      const int iter_max_C = pm1->opt.max_iter_C;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = pm1->opt.max_iter_C_rst;  // max restarts
      Real w_opt = pm1->opt.w_opt_ini_C;  // underrelaxation factor
      Real e_C_abs_tol = pm1->opt.eps_C;
      // maximum error amplification factor between iters.
      Real fac_PA = pm1->opt.fac_amp_C;

      // retain values for potential restarts
      iI_xi[0] = sc_xi(k,j,i);

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      // solver loop ----------------------------------------------------------
      drf.i = i;
      drf.j = j;
      drf.k = k;

      const Real W = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);

      do
      {
        pit++;

        // Check limits
        // const Real xi_old = sc_xi(k,j,i);

        // if ((pit == 1) && (rit == 0))
        // {
        //   const Real err_l = std::abs(R(xi_min, &drf));
        //   const Real err_r = std::abs(R(xi_max, &drf));
        //   const Real err_m = std::abs(R(xi_old, &drf));

        //   Real xi = (err_l < err_r) ? xi_min : xi_max;
        //   xi = (err_m < std::min(err_l, err_r)) ? xi_old : xi;

        //   if (std::min(err_m, std::min(err_l, err_r)) < e_C_abs_tol)
        //   {
        //     e_abs_cur = std::min(err_m, std::min(err_l, err_r));
        //     std::abs(R(xi, &drf));
        //     break;
        //   }
        // }

        // const Real xi = xi_old;
        // sc_E(k,j,i) = std::max(sc_E(k,j,i), 1e-14);

        Real Z_xi = R(sc_xi(k,j,i), &drf);

        // if (std::abs(Z_xi) < e_C_abs_tol)
        // {
        //   break;
        // }

        Real sc_xi_can = sc_xi(k,j,i) - w_opt * Z_xi;
        // enforce non-negative values
        // sc_xi(k,j,i) = std::min(std::max(sc_xi_can, xi_min), xi_max);
        sc_xi(k,j,i) = sc_xi_can;

        if (sc_xi_can < xi_min)
        {
          e_abs_cur = std::abs(xi_max - drf.xi_im1);
          e_abs_old = e_abs_cur;

          sc_xi_can = xi_min;
          R(sc_xi_can, &drf);
          pit = 0;
        }
        else if (sc_xi_can > xi_max)
        {
          e_abs_cur = std::abs(xi_max - drf.xi_im1);
          e_abs_old = e_abs_cur;

          // e_abs_cur = std::numeric_limits<Real>::infinity();
          // e_abs_old = e_abs_cur;

          sc_xi_can = xi_max;
          R(sc_xi_can, &drf);
          pit = 0;
        }
        else
        {
          sc_xi(k,j,i) = sc_xi_can;
          e_abs_cur = std::abs(drf.xi_i - drf.xi_im1);
        }


        // e_abs_cur = std::abs(Z_xi);
        // e_abs_cur = std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1));
        // e_abs_cur = std::abs(drf.xi_i - drf.xi_im2);

        // if((sc_xi_can < xi_min)||(sc_xi_can > xi_max))
        // {
        //   sc_xi(k,j,i) = std::min(std::max(sc_xi_can, xi_min), xi_max);
        //   R(sc_xi(k,j,i), &drf);
        //   e_abs_cur = 0;
        //   break;
        // }

        // if (pit > 1)
        // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-5)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }

        // if (pit > 3)
        // {
        //   const Real p0 = drf.xi_im1;
        //   const Real p1 = drf.xi_i;
        //   const Real p2 = sc_xi(k,j,i);

        //   const Real pe = p0-SQR(p1-p0)/(p2-2*p1+p0);
        //   sc_xi(k,j,i) = pe;

        // }

        // e_abs_cur = std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1));

        e_abs_cur = std::abs(drf.xi_i - drf.xi_im1);

        // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-5)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }


        if ((e_abs_cur > fac_PA * e_abs_old))
        {
          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          sc_xi(k,j,i) = iI_xi[0];

          // restart iteration
          e_abs_old = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::Closures::Minerbo::AddClosureP max restarts exceeded.";
            std::cout << msg.str().c_str() << "\n";
            std::cout << e_abs_cur << "\n";
            std::cout << sc_xi_can << "\n";
            std::cout << Z_xi << "\n";
            std::cout << w_opt << "\n";
            std::cout << "more info: \n";
            std::cout << drf.xi_i << "\n";
            std::cout << drf.xi_im1 << "\n";
            std::cout << drf.xi_im2 << "\n";
            std::cout << "Z_xi: \n";
            std::cout << drf.Z_xi_i << "\n";
            std::cout << drf.Z_xi_im1 << "\n";
            std::cout << drf.Z_xi_im2 << "\n";
            Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
            std::exit(0);
          }
        }
        else
        {
          e_abs_old = e_abs_cur;
          sc_chi(k,j,i) = chi(sc_xi(k,j,i));

          /*
          if ((sc_xi(k,j,i) < xi_min))
          {
            sc_xi(k,j,i) = xi_min;
            sc_chi(k,j,i) = chi(sc_xi(k,j,i));

            for (int a=0; a<N; ++a)
            for (int b=a; b<N; ++b)
            {
              sp_P_dd(a,b,k,j,i) = sp_P_tk_dd_(a,b,i);
            }
          }
          else if ((sc_xi(k,j,i) > xi_max))
          {
            sc_xi(k,j,i) = xi_max;
            sc_chi(k,j,i) = chi(sc_xi(k,j,i));

            for (int a=0; a<N; ++a)
            for (int b=a; b<N; ++b)
            {
              sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
            }
          }
          */


          if (pm1->opt.reset_thin)
          {
            if ((sc_xi(k,j,i) < xi_min) || (sc_xi(k,j,i) > xi_max))
            {
              sc_xi(k,j,i) = xi_max;
              sc_chi(k,j,i) = chi(sc_xi(k,j,i));

              for (int a=0; a<N; ++a)
              for (int b=a; b<N; ++b)
              {
                sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
              }

              e_abs_cur = 0.0; // forces iter breakout
            }
          }

        }

      } while ((pit < iter_max_C) && (e_abs_cur >= e_C_abs_tol));



      if ((sc_xi(k,j,i) < xi_min) || sc_xi(k,j,i) > xi_max)
      {
        std::cout << "M1::Closures::Minerbo::AddClosureP:\n";
        std::cout << "outside [0,1] \n";
        std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
        std::cout << pit << "," << rit << "," << e_abs_cur << "\n";
        std::cout << e_abs_cur << "\n";
        std::cout << w_opt << "\n";
        std::cout << "more info: \n";
        std::cout << drf.xi_i << "\n";
        std::cout << drf.xi_im1 << "\n";
        std::cout << drf.xi_im2 << "\n";
        std::cout << 1 - (drf.xi_i + 1) / (drf.xi_im1 + 1) << "\n";
        std::cout << "Z_xi: \n";
        std::cout << drf.Z_xi_i << "\n";
        std::cout << drf.Z_xi_im1 << "\n";
        std::cout << drf.Z_xi_im2 << "\n";

        Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

        std::cout << sc_xi(k,j,i) << "\n";
        std::cout << iI_xi[0] << "\n";

        std::exit(0);
      }


      iter_tot += pit;

      if ((e_abs_cur > e_C_abs_tol) && pm1->opt.verbose_iter_C)
      {
        std::cout << "M1::Closures::Minerbo::AddClosureP:\n";
        std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
        std::cout << pit << "," << rit << "," << e_abs_cur << "\n";
        std::cout << k << "," << j << "," << i << "\n";
        std::cout << e_abs_cur << "\n";
        std::cout << w_opt << "\n";
        std::cout << "more info: \n";
        std::cout << drf.xi_i << "\n";
        std::cout << drf.xi_im1 << "\n";
        std::cout << drf.xi_im2 << "\n";
        std::cout << 1 - (drf.xi_i + 1) / (drf.xi_im1 + 1) << "\n";
        std::cout << "Z_xi: \n";
        std::cout << drf.Z_xi_i << "\n";
        std::cout << drf.Z_xi_im1 << "\n";
        std::cout << drf.Z_xi_im2 << "\n";

        Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        std::exit(0);
      }
    }


  }

  // const int nn = pm1->mbi.nn1 * pm1->mbi.nn2 * pm1->mbi.nn3;
  // std::cout << static_cast<Real>(iter_tot) / nn << std::endl;

}

void AddClosureN(M1 * pm1,
                 const Real weight,
                 const int ix_g,
                 const int ix_s,
                 AT_N_sym & sp_P_dd)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  // fiducial quantities
  AT_C_sca & sc_W   = pm1->fidu.sc_W;
  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;

  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_chi = pm1->lab_aux.sc_chi(ix_g,ix_s);
  AT_C_sca & sc_xi  = pm1->lab_aux.sc_xi( ix_g,ix_s);

  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
  AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);

  // required geometric quantities
  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
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

  std::array<Real, 1> iI_xi;

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    Closures::ClosureThin( pm1, 1.0, ix_g, ix_s, sp_P_tn_dd_, k, j, il, iu);
    Closures::ClosureThick(pm1, 1.0, ix_g, ix_s, sp_P_tk_dd_, k, j, il, iu);

    M1_GLOOP1(i)
    {
      const int iter_max_C = pm1->opt.max_iter_C;
      int pit = 0;     // iteration counter
      int rit = 0;     // restart counter
      const int iter_max_R = pm1->opt.max_iter_C_rst;  // max restarts
      Real w_opt = pm1->opt.w_opt_ini_C;  // underrelaxation factor
      Real e_C_abs_tol = pm1->opt.eps_C;
      // maximum error amplification factor between iters.
      Real fac_PA = pm1->opt.fac_amp_C;

      // retain values for potential restarts
      iI_xi[0] = sc_xi(k,j,i);

      Real e_abs_old = std::numeric_limits<Real>::infinity();
      Real e_abs_cur = 0;

      // solver loop ----------------------------------------------------------
      drf.i = i;
      drf.j = j;
      drf.k = k;

      const Real W = sc_W(k,j,i);
      const Real oo_W = 1.0 / W;
      const Real W2   = SQR(W);

      Real dotFv = Assemble::sc_dot_dense_sp__(sp_F_d, sp_v_u, k, j, i);

      do
      {
        pit++;

        // const Real xi_old = sc_xi(k,j,i);

        // if ((pit == 1) && (rit == 0))
        // {
        //   const Real err_l = std::abs(R(xi_min, &drf));
        //   const Real err_r = std::abs(R(xi_max, &drf));
        //   const Real err_m = std::abs(R(xi_old, &drf));

        //   Real xi = (err_l < err_r) ? xi_min : xi_max;
        //   xi = (err_m < std::min(err_l, err_r)) ? xi_old : xi;

        //   if (std::min(err_m, std::min(err_l, err_r)) < e_C_abs_tol)
        //   {
        //     e_abs_cur = std::min(err_m, std::min(err_l, err_r));
        //     std::abs(R(xi, &drf));
        //     break;
        //   }
        // }

        // const Real xi = xi_old;

        // sc_E(k,j,i) = std::max(sc_E(k,j,i), 1e-12);

        const Real xi  = sc_xi(k,j,i);
        Real Z_xi = R(xi, &drf);

        // if (std::abs(Z_xi) < 1e-20)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }

        Real dZ_xi = dR(xi, &drf);

        // Apply Newton iterate with fallback
        Real D = 0;
        Real sc_xi_can = std::numeric_limits<Real>::infinity();


        if ((std::abs(dZ_xi) > pm1->opt.eps_C_N * std::abs(Z_xi)) && (rit == 0))
        {
          // Newton
          D = Z_xi / dZ_xi;
          sc_xi_can = sc_xi(k,j,i) - D;
          // if (pit == 1)
          // {
          // }
          // else
          // {

          // }

          // if ((sc_xi_can < xi_min) || (sc_xi_can > xi_max))
          // {
          //   // Picard
          //   D = w_opt * Z_xi;
          //   sc_xi_can = sc_xi(k,j,i) - D;
          // }
        }
        else
        {
          // Picard
          D = w_opt * Z_xi;
          sc_xi_can = sc_xi(k,j,i) - D;

          // sc_xi_can = std::min(std::max(sc_xi_can, xi_min), xi_max);
          // sc_xi(k,j,i) = xi_max;
          // R(xi_max, &drf);
          // D = 0;
          // Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
          // std::exit(0);

          // if (sc_xi_can < xi_min)
          // {
          //   sc_xi_can = xi_min;
          //   R(sc_xi_can, &drf);
          //   pit = 0;
          // }
          // else if (sc_xi_can > xi_max)
          // {
          //   sc_xi_can = xi_max;
          //   R(sc_xi_can, &drf);
          //   pit = 0;
          // }
        }

        // enforce non-negative values
        // sc_xi(k,j,i) = std::min(std::max(sc_xi_can, xi_min), xi_max);


        // sc_xi(k,j,i) = std::min(std::abs(sc_xi_can), xi_max);
        sc_xi(k,j,i) = sc_xi_can;

        // if (sc_xi(k,j,i) < xi_min)
        // {
        //   sc_xi(k,j,i) = xi_min;
        //   R(sc_xi(k,j,i), &drf);
        //   pit = 0;
        // }

        // if (sc_xi(k,j,i) > xi_max)
        // {
        //   sc_xi(k,j,i) = xi_max;
        //   R(sc_xi(k,j,i), &drf);
        //   pit = 0;
        // }

        // if (sc_xi(k,j,i) > xi_max)
        // {
        //   sc_xi(k,j,i) = xi_max;
        //   R(sc_xi(k,j,i), &drf);
        //   e_abs_cur = 0;
        //   break;
        // }

        // R(sc_xi(k,j,i), &drf);

        // scale error tol by step
        // e_abs_cur = std::abs(D);
        e_abs_cur = std::abs(D / w_opt);


        // e_abs_cur = std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1));

        // if (std::abs(1 - (drf.xi_i + 1) / (drf.xi_im1 + 1)) < 1e-3)
        // {
        //   e_abs_cur = 0;
        //   break;
        // }

        if ((e_abs_cur > fac_PA * e_abs_old) && (pit > 30))
        {
          std::cout << pit << std::endl;
          std::cout << e_abs_cur << std::endl;
          std::cout << e_abs_old << std::endl;
          std::cout << Z_xi << std::endl;
          std::cout << D << std::endl;
          std::cout << dZ_xi << std::endl;
          std::cout << sc_xi_can << std::endl;
          std::cout << sc_xi(k,j,i) << std::endl;
          Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

          std::cout << drf.xi_i << "\n";
          std::cout << drf.xi_im1 << "\n";
          std::cout << drf.xi_im2 << "\n";
          std::exit(0);

          // halve underrelaxation and recover old values
          w_opt = w_opt / 2;
          sc_xi(k,j,i) = iI_xi[0];

          // restart iteration
          e_abs_old = std::numeric_limits<Real>::infinity();
          pit = 0;
          rit++;

          if (rit > iter_max_R)
          {
            std::ostringstream msg;
            msg << "M1::Closures::Minerbo::AddClosureN max restarts exceeded.";
            std::cout << msg.str().c_str() << std::endl;
            Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
            std::exit(0);
          }
        }
        else
        {
          e_abs_old = e_abs_cur;
          sc_chi(k,j,i) = chi(sc_xi(k,j,i));

          if (pm1->opt.reset_thin)
          {
            if ((sc_xi(k,j,i) < xi_min) || (sc_xi(k,j,i) > xi_max))
            {
              sc_xi(k,j,i) = xi_max;
              sc_chi(k,j,i) = chi(sc_xi(k,j,i));

              for (int a=0; a<N; ++a)
              for (int b=a; b<N; ++b)
              {
                sp_P_dd(a,b,k,j,i) = sp_P_tn_dd_(a,b,i);
              }

              e_abs_cur = 0.0; // forces iter breakout
            }
          }

        }

      } while ((pit < iter_max_C) && (e_abs_cur >= e_C_abs_tol));

      iter_tot += pit;

      if ((e_abs_cur > e_C_abs_tol) && pm1->opt.verbose_iter_C)
      {
        std::cout << "M1::Closures::Minerbo::AddClosureN:\n";
        std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
        std::cout << pit << "," << rit << "," << e_abs_cur << "\n";
        Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        // std::exit(0);
      }

      // sc_xi(k,j,i) = std::min(std::max(sc_xi(k,j,i), xi_min), xi_max);

    }


  }

  // const int nn = pm1->mbi.nn1 * pm1->mbi.nn2 * pm1->mbi.nn3;
  // std::cout << static_cast<Real>(iter_tot) / nn << std::endl;

}

/*
void AddClosureN_(M1 * pm1,
                  const Real weight,
                  const int ix_g,
                  const int ix_s,
                  AT_N_sym & sp_P_dd)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int il = 0;
  const int iu = pm1->mbi.nn1-1;

  AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
  AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

  AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
  AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
  AT_C_sca & sc_chi = pm1->lab_aux.sc_chi(ix_g,ix_s);
  AT_C_sca & sc_xi  = pm1->lab_aux.sc_xi( ix_g,ix_s);

  AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);
  AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
  AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);

  // point to scratches
  AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
  AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
  AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
  AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

  // initialize struct for root-finding
  DataRootfinder drf {
    sp_g_uu,
    sc_E,
    sp_F_d,
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

  // setup GSL ----------------------------------------------------------------
  gsl_set_error_handler_off();

  gsl_function_fdf RdR_;
  RdR_.f      = &d_R;
  RdR_.df     = &d_dR;
  RdR_.fdf    = &d_RdR;
  RdR_.params = &drf;

  gsl_root_fdfsolver * gsl_solver = gsl_root_fdfsolver_alloc(
    gsl_root_fdfsolver_steffenson
  );

  auto gsl_err_kill = [&](const int status)
  {
    std::ostringstream msg;
    msg << "M1::Closures::Minerbo::AddClosureN unexpected error: ";
    msg << status;

    std::cout << msg.str().c_str() << std::endl;

    ATHENA_ERROR(msg);
  };

  auto gsl_err_warn = [&]()
  {
    std::ostringstream msg;
    msg << "M1::Closures::Minerbo::AddClosureN maxiter=";
    msg << pm1->opt.max_iter_C << " exceeded";
    std::cout << msg.str().c_str() << std::endl;
  };
  // --------------------------------------------------------------------------

  // main loop
  // int iter_tot = 0;

  M1_GLOOP2(k,j)
  {
    Closures::ClosureThin( pm1, 1.0, ix_g, ix_s, sp_P_tn_dd_, k, j, il, iu);
    Closures::ClosureThick(pm1, 1.0, ix_g, ix_s, sp_P_tk_dd_, k, j, il, iu);

    M1_GLOOP1(i)
    {
      const int iter_max_C = pm1->opt.max_iter_C;
      int pit = 0;     // iteration counter
      Real e_C_abs_tol = pm1->opt.eps_C;

      Real xi = sc_xi(k,j,i);
      Real xi_;

      int status = gsl_root_fdfsolver_set(gsl_solver, &RdR_, xi);

      // solver loop ----------------------------------------------------------
      drf.i = i;
      drf.j = j;
      drf.k = k;

      Real e_abs_cur = std::numeric_limits<Real>::infinity();

      do
      {
        pit++;
        // Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        // std::cout << RdR_.f(xi, &drf) << "\n";
        // std::cout << "etc" << "\n";

        status = gsl_root_fdfsolver_iterate(gsl_solver);

        if (status != GSL_SUCCESS)
        {
          break;
        }

        xi_ = xi;
        // status = gsl_root_test_delta(xi, xi_, e_C_abs_tol, 0);
        status = gsl_root_test_residual(RdR_.f(xi, &drf), e_C_abs_tol);
        // Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        // std::cout << RdR_.f(xi, &drf) << "\n";

        // std::cout << status << "\n";
        // std::cout << "etc" << "\n";

        // std::exit(0);

      }  while ((status == GSL_CONTINUE) && (pit < iter_max_C));

      sc_xi(k,j,i) = xi;

      if (status != GSL_SUCCESS)
      {
        // Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
        // std::cout << status;
        // std::exit(0);
        // gsl_err_warn();
      }


      // switch (status)
      // {
      //   case (GSL_EINVAL):  // bracketing failed (revert to thin closure)
      //   {
      //     Assemble::ScratchToDense(sp_P_dd, sp_P_tn_dd_, k, j, i, i);
      //     break;
      //   }
      //   case (0):
      //   {
      //     // root-finding loop
      //     Real loc_xi_min = xi_min;
      //     Real loc_xi_max = xi_max;

      //     status = GSL_CONTINUE;
      //     for (int iter=1;
      //          iter<=pm1->opt.max_iter_C && status == GSL_CONTINUE;
      //          ++iter)
      //     {
      //       status = gsl_root_fsolver_iterate(gsl_solver);

      //       if (status != GSL_SUCCESS)
      //       {
      //         break;
      //       }
      //       else if (status)
      //       {
      //         sp_P_tn_dd_.array().print_all();
      //         sp_P_tk_dd_.array().print_all();

      //         Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);

      //         gsl_err_kill(status);
      //       }

      //       loc_xi_min = gsl_root_fsolver_x_lower(gsl_solver);
      //       loc_xi_max = gsl_root_fsolver_x_upper(gsl_solver);

      //       status = gsl_root_test_interval(
      //         loc_xi_min, loc_xi_max, pm1->opt.eps_C, 0
      //       );
      //     }

      //     if (status != GSL_SUCCESS)
      //     {
      //       gsl_err_warn();
      //     }

      //     // Final call of R to update sp_P_dd with new root info.
      //     R_.function(gsl_root_fsolver_root(gsl_solver), &drf);

      //     break;
      //   }
      //   default:
      //   {
      //     Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
      //     gsl_err_kill(status);
      //   }
      // }

    }
  }

  // cleanup ------------------------------------------------------------------
  gsl_root_fdfsolver_free(gsl_solver);
  // TODO: bug - this can't be deactivated properly with OMP
  // gsl_set_error_handler(NULL);  // restore default handler
}
*/

// ============================================================================
} // namespace M1::Closures::Minerbo
// ============================================================================


// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Computes the closure on a mesh block
void M1::CalcClosure(AthenaArray<Real> & u)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    pm1->lab_aux.sp_P_dd(ix_g,ix_s).ZeroClear();

    switch (opt.closure_variety)
    {
      case (opt_closure_variety::thin):
      {
        Closures::SetClosureThin(this,
                                 1.0, ix_g, ix_s,
                                 lab_aux.sp_P_dd(ix_g,ix_s));
        lab_aux.sc_chi(ix_g,ix_s).Fill(1.0);
        lab_aux.sc_xi( ix_g,ix_s).Fill(1.0);
        break;
      }
      case (opt_closure_variety::thick):
      {
        Closures::SetClosureThick(this,
                                  1.0, ix_g, ix_s,
                                  lab_aux.sp_P_dd(ix_g,ix_s));
        lab_aux.sc_chi(ix_g,ix_s).Fill(ONE_3RD);
        lab_aux.sc_xi( ix_g,ix_s).Fill(0.0);
        break;
      }
      case (opt_closure_variety::Minerbo):
      {
        Closures::Minerbo::AddClosure(this,
                                      1.0, ix_g, ix_s,
                                      lab_aux.sp_P_dd(ix_g,ix_s));

        /*
        AT_C_sca xi_(mbi.nn3, mbi.nn2, mbi.nn1);
        AT_C_sca chi_(mbi.nn3, mbi.nn2, mbi.nn1);
        M1_ILOOP3(k,j,i)
        {
          xi_(k,j,i)  = lab_aux.sc_xi(ix_g,ix_s)(k,j,i);
          chi_(k,j,i) = lab_aux.sc_chi(ix_g,ix_s)(k,j,i);
        }

        lab_aux.sc_xi(ix_g,ix_s).Fill(0.9);
        Closures::Minerbo::AddClosureN(this,
                                       1.0, ix_g, ix_s,
                                       lab_aux.sp_P_dd(ix_g,ix_s));
        Real err (0);
        M1_ILOOP3(k,j,i)
        {
          Real err_cur = std::abs(chi_(k,j,i)-lab_aux.sc_chi(ix_g,ix_s)(k,j,i));
          err += err_cur;
          if (lab.sc_E(ix_g,ix_s)(k,j,i) >  1e-6)
          if (err_cur > 1e-4)
          {
            AT_N_sym & sp_g_dd = pm1->geom.sp_g_dd;
            AT_N_sym & sp_g_uu = pm1->geom.sp_g_uu;

            // point to scratches
            AT_N_vec & sp_dH_d_    = pm1->scratch.sp_vec_A_;
            AT_N_sym & sp_P_tn_dd_ = pm1->scratch.sp_sym_A_;
            AT_N_sym & sp_P_tk_dd_ = pm1->scratch.sp_sym_B_;
            AT_N_sym & sp_dP_dd_   = pm1->scratch.sp_sym_C_;

            // initialize struct for root-finding
            Closures::Minerbo::DataRootfinder drf {
              sp_g_uu,
              pm1->lab.sc_E(  ix_g,ix_s),
              pm1->lab.sp_F_d(ix_g,ix_s),
              lab_aux.sp_P_dd(ix_g,ix_s),
              pm1->lab_aux.sc_chi(ix_g,ix_s),
              pm1->lab_aux.sc_xi( ix_g,ix_s),
              pm1->rad.sc_J(  ix_g,ix_s),
              pm1->rad.sc_H_t(ix_g,ix_s),
              pm1->rad.sp_H_d(ix_g,ix_s),
              pm1->fidu.sc_W,
              pm1->fidu.sp_v_u,
              pm1->fidu.sp_v_d,
              sp_dH_d_,
              sp_P_tn_dd_,
              sp_P_tk_dd_,
              sp_dP_dd_
            };

            drf.sp_P_tn_dd_ = sp_P_tn_dd_;
            drf.sp_P_tk_dd_ = sp_P_tk_dd_;

            Closures::ClosureThin( pm1, 1.0, ix_g, ix_s, sp_P_tn_dd_, k, j, i, i);
            Closures::ClosureThick(pm1, 1.0, ix_g, ix_s, sp_P_tk_dd_, k, j, i, i);

            drf.i = i;
            drf.j = j;
            drf.k = k;


            // pm1->lab.sc_E(  ix_g,ix_s)(k,j,i) = 0;
            std::cout << err_cur << std::endl;
            std::cout << xi_(k,j,i) << std::endl;
            std::cout << chi_(k,j,i) << std::endl;

            std::cout << "AddClosureP ------------\n";
            Closures::Minerbo::AddClosureP(this,
                                           1.0, ix_g, ix_s,
                                           lab_aux.sp_P_dd(ix_g,ix_s));


            Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);
            std::cout << Closures::Minerbo::R(
              pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i), &drf) << "\n";



            std::cout << "AddClosure ------------\n";
            Closures::Minerbo::AddClosure(this,
                                          1.0, ix_g, ix_s,
                                          lab_aux.sp_P_dd(ix_g,ix_s));

            Closures::InfoDump(pm1, ix_g, ix_s, k, j, i);



            std::cout << pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i) << "\n";
            std::cout << Closures::Minerbo::R(
              pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i), &drf) << "\n";
            std::cout << Closures::Minerbo::R(1., &drf) << "\n";
            std::cout << Closures::Minerbo::R(0.5, &drf) << "\n";
            std::cout << Closures::Minerbo::R(0., &drf) << "\n";

            if (0) {
              AT_N_vec & sp_H_d_     = pm1->scratch.sp_vec_;

              const Real W = fidu.sc_W(k,j,i);
              const Real oo_W = 1.0 / W;
              const Real W2   = SQR(W);

              Real dotFv (0);
              for (int a=0; a<N; ++a)
              {
                dotFv += lab.sp_F_d(ix_g,ix_s)(a,k,j,i) * fidu.sp_v_u(a,k,j,i);
              }

              const Real E = lab.sc_E(ix_g,ix_s)(k,j,i);
              const Real xi  = lab_aux.sc_xi(ix_g,ix_s)(k,j,i);
              const Real chi_ = Closures::Minerbo::chi(xi);

              for (int a=0; a<N; ++a)
              for (int b=a; b<N; ++b)
              {
                lab_aux.sp_P_dd(ix_g,ix_s)(a,b,k,j,i) = (
                  0.5 * (3.0 * chi_ - 1.0) * sp_P_tn_dd_(a,b,i) +
                  0.5 * 3.0 * (1.0 - chi_) * sp_P_tk_dd_(a,b,i)
                );
              }

              // assemble zero functional
              Real dotPvv (0);
              for (int a=0; a<N; ++a)
              for (int b=0; b<N; ++b)
              {
                dotPvv += lab_aux.sp_P_dd(ix_g,ix_s)(a,b,k,j,i) * fidu.sp_v_u(a,k,j,i) * fidu.sp_v_u(b,k,j,i);
              }

              const Real J_fac = E - 2.0 * dotFv + dotPvv;
              const Real J     = W2 * J_fac;
              const Real Hn    = W * (E - J - dotFv);

              for (int a=0; a<N; ++a)
              {
                Real dotPv (0);
                for (int b=0; b<N; ++b)
                {
                  dotPv += lab_aux.sp_P_dd(ix_g,ix_s)(a,b,k,j,i) * fidu.sp_v_u(b,k,j,i);
                }

                sp_H_d_(a,i) = W * (lab.sp_F_d(ix_g,ix_s)(a,k,j,i) - J * fidu.sp_v_d(a,k,j,i) - dotPv);
              }

              Real Z_xi = SQR(xi) * SQR(J) + SQR(Hn);
              for (int a=0; a<N; ++a)
              for (int b=0; b<N; ++b)
              {
                Z_xi -= sp_g_uu(a,b,k,j,i) * sp_H_d_(a,i) * sp_H_d_(b,i);
              }
              std::cout << Z_xi << "\n";
            }


            std::exit(0);
          }

        }
        */
        // if (err > 1e-4)
        //   std::cout << err << std::endl;

        break;
      }
      case (opt_closure_variety::MinerboP):
      {
        // lab_aux.sc_xi(ix_g,ix_s).ZeroClear();
        Closures::Minerbo::AddClosureP(this,
                                       1.0, ix_g, ix_s,
                                       lab_aux.sp_P_dd(ix_g,ix_s));
        break;
      }
      case (opt_closure_variety::MinerboN):
      {
        // lab_aux.sc_xi(ix_g,ix_s).ZeroClear();
        Closures::Minerbo::AddClosureN(this,
                                       1.0, ix_g, ix_s,
                                       lab_aux.sp_P_dd(ix_g,ix_s));
        break;
      }
      default:
      {
        assert(false);
        std::exit(0);
      }
    }
  }

  return;
}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, P_dd) to (J, H_d)
void M1::CalcFiducialFrame(AthenaArray<Real> & u)
{
  vars_Lab U_n { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U_n);

  AT_C_sca & sc_W = pm1->fidu.sc_W;
  AT_N_vec & sp_v_u = pm1->fidu.sp_v_u;
  AT_N_vec & sp_v_d = pm1->fidu.sp_v_d;

  // point to scratches
  AT_C_sca & dotFv_ = pm1->scratch.sc_A_;
  AT_C_sca & sc_W2_ = pm1->scratch.sc_B_;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d  = U_n.sp_F_d(ix_g,ix_s);
    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_H_t = pm1->rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d = pm1->rad.sp_H_d(ix_g,ix_s);
    AT_C_sca & sc_J   = pm1->rad.sc_J(  ix_g,ix_s);

    M1_GLOOP2(k,j)
    {
      dotFv_.ZeroClear();

      for (int a=0; a<N; ++a)
      M1_GLOOP1(i)
      {
        dotFv_(i) += sp_F_d(a,k,j,i) * sp_v_u(a,k,j,i);
      }

      M1_GLOOP1(i)
      {
        sc_W2_(i) = SQR(sc_W(k,j,i));
      }

      // assemble J
      M1_GLOOP1(i)
      {
        sc_J(k,j,i) = (
          sc_E(k,j,i) - 2.0 * dotFv_(i)
        );
      }

      for (int a=0; a<N; ++a)
      for (int b=0; b<N; ++b)
      M1_GLOOP1(i)
      {
        sc_J(k,j,i) += (
          sp_P_dd(a,b,k,j,i) *
          sp_v_u(a,k,j,i) *
          sp_v_u(b,k,j,i)
        );
      }

      M1_GLOOP1(i)
      {
        sc_J(k,j,i) = sc_W2_(i) * sc_J(k,j,i);
      }

      // assemble H_t
      M1_GLOOP1(i)
      {
        sc_H_t(k,j,i) = sc_W(k,j,i) * (
          sc_E(k,j,i) - sc_J(k,j,i) - dotFv_(i)
        );
      }

      // assemble H_d
      for (int a=0; a<N; ++a)
      {
        M1_GLOOP1(i)
        {
          sp_H_d(a,k,j,i) = (
            sp_F_d(a,k,j,i) - sc_J(k,j,i) * sp_v_d(a,k,j,i)
          );
        }

        for (int b=0; b<N; ++b)
        M1_GLOOP1(i)
        {
          sp_H_d(a,k,j,i) -= (
            sp_v_u(b,k,j,i) * sp_P_dd(b,a,k,j,i)
          );
        }

        M1_GLOOP1(i)
        {
          sp_H_d(a,k,j,i) = sc_W(k,j,i) * sp_H_d(a,k,j,i);
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