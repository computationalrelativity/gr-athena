// c++
#include <functional>
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_closure.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

// ============================================================================
namespace M1::Closures {
// ============================================================================

ClosureMetaVector ConstructClosureMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s)
{
  using namespace std::placeholders;

  ClosureMetaVector C {
    pm1,
    ix_g,
    ix_s,
    // geometric quantities
    pm1.geom.sp_g_dd,
    pm1.geom.sp_g_uu,
    // fiducial quantities
    pm1.fidu.sp_v_d,
    pm1.fidu.sp_v_u,
    pm1.fidu.sc_W,
    // state-vector dependent
    vlab.sc_E(  ix_g,ix_s),
    vlab.sp_F_d(ix_g,ix_s),
    // group-dependent, but common
    pm1.lab_aux.sp_P_dd(ix_g,ix_s),
    pm1.lab_aux.sc_chi( ix_g,ix_s),
    pm1.lab_aux.sc_xi(  ix_g,ix_s),
    // Lagrangian frame
    pm1.rad.sc_J(  ix_g,ix_s),
    pm1.rad.sc_H_t(ix_g,ix_s),
    pm1.rad.sp_H_d(ix_g,ix_s),
    // scratches
    pm1.scratch.sp_vec_A_, // sp_dH_d_
    pm1.scratch.sp_sym_A_, // sp_P_tn_dd_
    pm1.scratch.sp_sym_B_, // sp_P_tk_dd_
    pm1.scratch.sp_sym_C_, // sp_dP_dd_
  };

  // store limiting cases for fallback --------------------------------------
  C.ClosureThin = [&](const int k, const int j, const int i)
  {
    const bool populate_scratch = false;
    ClosureThin(pm1, C, k, j, i, populate_scratch);
    C.sc_xi( k,j,i) = 1.0;
    C.sc_chi(k,j,i) = 1.0;
  };

  C.ClosureThick = [&](const int k, const int j, const int i)
  {
    const bool populate_scratch = false;
    ClosureThick(pm1, C, k, j, i, populate_scratch);
    C.sc_xi( k,j,i) = 0.0;
    C.sc_chi(k,j,i) = ONE_3RD;
  };

  // select closure ---------------------------------------------------------
  switch (pm1.opt.closure_variety)
  {
    case (M1::opt_closure_variety::thin):
    {
      C.Closure = C.ClosureThin;
      break;
    }
    case (M1::opt_closure_variety::thick):
    {
      C.Closure = C.ClosureThick;
      break;
    }
    case (M1::opt_closure_variety::MinerboP):
    {
      C.Closure = [&](const int k, const int j, const int i)
      {
        Minerbo::ClosureMinerboPicard(pm1, C, k, j, i);
      };
      break;
    }
    case (M1::opt_closure_variety::Minerbo):
    {
      C.Closure = [&](const int k, const int j, const int i)
      {
        Minerbo::gsl::ClosureMinerboBrent(pm1, C, k, j, i);
      };
      break;
    }
    case (M1::opt_closure_variety::MinerboN):
    {
      C.Closure = [&](const int k, const int j, const int i)
      {
        Minerbo::gsl::ClosureMinerboNewton(pm1, C, k, j, i);
        // Minerbo::ClosureMinerboNewton(*this, C, k, j, i);
      };
      break;
    }
    default:
    {
      assert(false);
      std::exit(0);
    }
  }

  // store limiting cases for fallback --------------------------------------
  C.ClosureThin = [&](const int k, const int j, const int i)
  {
    const bool populate_scratch = false;
    ClosureThin(pm1, C, k, j, i, populate_scratch);
    C.sc_xi( k,j,i) = 1.0;
    C.sc_chi(k,j,i) = 1.0;
  };

  C.ClosureThick = [&](const int k, const int j, const int i)
  {
    const bool populate_scratch = false;
    ClosureThick(pm1, C, k, j, i, populate_scratch);
    C.sc_xi( k,j,i) = 0.0;
    C.sc_chi(k,j,i) = ONE_3RD;
  };

  return C;
}


// P_{i j} in Eq.(15) of [1] - works with densitized variables
void ClosureThin(M1 & pm1,
                 ClosureMetaVector & C,
                 const int k, const int j, const int i,
                 const bool populate_scratch)
{
  const Real nF2 = Assemble::sp_norm2__(C.sp_F_d, C.sp_g_uu, k, j, i);

  auto val_P_ab = [&](const int a, const int b)
  {
    const Real fac = (nF2 > 0) ? C.sc_E(k,j,i) / nF2
                               : 0.0;
    return fac * C.sp_F_d(a,k,j,i) * C.sp_F_d(b,k,j,i);

  };

  if (populate_scratch)
  {
    for (int a=0; a<N; ++a)
    for (int b=a; b<N; ++b)
    {
      C.sp_P_tn_dd_(a,b,i) = val_P_ab(a,b);
    }
  }
  else
  {
    for (int a=0; a<N; ++a)
    for (int b=a; b<N; ++b)
    {
      C.sp_P_dd(a,b,k,j,i) = val_P_ab(a,b);
    }
  }
}

void ClosureThick(M1 & pm1,
                  ClosureMetaVector & C,
                  const int k, const int j, const int i,
                  const bool populate_scratch)
{
  const Real W    = C.sc_W(k,j,i);
  const Real oo_W = 1.0 / W;
  const Real W2   = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d,
                                                 C.sp_v_u,
                                                 k, j, i);

  Real J_tk = 3.0 / (2.0 * W2 + 1.0) * (
    (2.0 * W2 - 1.0) * C.sc_E(k,j,i) - 2.0 * W2 * dotFv
  );

  const Real fac_H_tk = W /  (2.0 * W2 + 1.0) * (
    (4.0 * W2 + 1.0) * dotFv - 4.0 * W2 * C.sc_E(k,j,i)
  );

  auto val_P_ab = [&](const int a, const int b)
  {
    const Real H_a_tk = oo_W * C.sp_F_d(a,k,j,i) +
                        fac_H_tk * C.sp_v_d(a,k,j,i);
    const Real H_b_tk = oo_W * C.sp_F_d(b,k,j,i) +
                        fac_H_tk * C.sp_v_d(b,k,j,i);

    return (
      4.0 * ONE_3RD * W2 * J_tk * C.sp_v_d(a,k,j,i) * C.sp_v_d(b,k,j,i) +
      W * (C.sp_v_d(a,k,j,i) * H_b_tk +
           C.sp_v_d(b,k,j,i) * H_a_tk) +
      ONE_3RD * J_tk * C.sp_g_dd(a,b,k,j,i)
    );
  };

  if (populate_scratch)
  {
    for (int a=0; a<N; ++a)
    for (int b=a; b<N; ++b)
    {
      C.sp_P_tk_dd_(a,b,i) = val_P_ab(a,b);
    }
  }
  else
  {
    for (int a=0; a<N; ++a)
    for (int b=a; b<N; ++b)
    {
      C.sp_P_dd(a,b,k,j,i) = val_P_ab(a,b);
    }
  }
}

// ============================================================================
} // namespace M1::Closures
// ============================================================================

// ============================================================================
namespace M1::Closures::Minerbo {
// ============================================================================

bool EnforceClosureLimits(
  M1 & pm1, ClosureMetaVector & C,
  const int k, const int j, const int i,
  const bool compute_limiting_P_dd)
{
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  if ((C.sc_xi(k,j,i) < xi_min) || (C.sc_xi(k,j,i) > xi_max))
  {
    C.sc_xi(k,j,i) = std::max(
      std::min(xi_max, C.sc_xi(k,j,i)), xi_min
    );
    C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));

    if (compute_limiting_P_dd)
    {
      const bool populate_scratch = true;
      ClosureThin( pm1, C, k, j, i, populate_scratch);
      ClosureThick(pm1, C, k, j, i, populate_scratch);
    }

    sp_P_dd__(C.sp_P_dd, C.sc_chi, C.sp_P_tn_dd_, C.sp_P_tk_dd_, k, j, i);

    return true;
  }

  return false;
}

Real Z_xi(
  const Real xi,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i,
  const bool compute_limiting_P_dd)
{
  // assemble P_dd
  C.sc_xi( k,j,i) = xi;
  C.sc_chi(k,j,i) = chi(xi);

  if (compute_limiting_P_dd)
  {
    const bool populate_scratch = true;
    ClosureThin( pm1, C, k, j, i, populate_scratch);
    ClosureThick(pm1, C, k, j, i, populate_scratch);
  }

  sp_P_dd__(C.sp_P_dd, C.sc_chi, C.sp_P_tn_dd_, C.sp_P_tk_dd_, k, j, i);

  const Real W  = C.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(
    C.sp_F_d,
    C.sp_v_u,
    k, j, i
  );

  // assemble J
  C.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, C.sc_E, C.sp_v_u, C.sp_P_dd,
    k, j, i
  );

  // assemble H_t
  C.sc_H_t(k,j,i) = Assemble::sc_H_t__(
    W, dotFv, C.sc_E, C.sc_J,
    k, j, i
  );

  // assemble H_d
  Assemble::sp_H_d__(C.sp_H_d, W, C.sc_J, C.sp_F_d,
                     C.sp_v_d, C.sp_v_u, C.sp_P_dd,
                     k, j, i);

  // assemble sc_H_st
  const Real sc_H2_st = Assemble::sc_H2_st__(
    C.sc_H_t, C.sp_H_d, C.sp_g_uu,
    k, j, i
  );

  // const Real Z_ = (std::abs(drf->sc_E(k,j,i)) > 0)
  //   ? (SQR(xi) * SQR(drf->sc_J(k,j,i)) - sc_H2_st) / drf->sc_E(k,j,i)
  //   : 0.0;

  const Real Z_ = (SQR(xi * C.sc_J(k,j,i)) - sc_H2_st);

  return Z_;
}

Real dZ_xi(
  const Real xi,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i,
  const bool compute_limiting_P_dd)
{
  if (compute_limiting_P_dd)
  {
    const bool populate_scratch = true;
    ClosureThin( pm1, C, k, j, i, populate_scratch);
    ClosureThick(pm1, C, k, j, i, populate_scratch);
  }

  const Real W  = C.sc_W(k,j,i);
  const Real W2 = SQR(W);

  // derivative of Z_xi
  Real dJ (0);
  Real dHH (0);

  const Real J  = C.sc_J(  k,j,i);
  const Real Hn = C.sc_H_t(k,j,i);

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
    C.sp_dP_dd_(a,b,i) = 3.0 / 5.0 * (
      C.sp_P_tn_dd_(a,b,i) - C.sp_P_tk_dd_(a,b,i)
    ) * xi * (2.0 - xi + 4.0 * SQR(xi));
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dJ += C.sp_v_u(a,k,j,i) *
          C.sp_v_u(b,k,j,i) *
          C.sp_dP_dd_(a,b,i);
  }

  dJ = W * dJ;

  const Real dHn = -W * dJ;

  for (int a=0; a<N; ++a)
  {
    C.sp_dH_d_(a,i) = C.sp_v_d(a,k,j,i) * dJ;
    for (int b=0; b<N; ++b)
    {
      C.sp_dH_d_(a,i) += C.sp_v_u(b,k,j,i) * C.sp_dP_dd_(a,b,i);
    }
    C.sp_dH_d_(a,i) = -W * C.sp_dH_d_(a,i);
  }

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dHH += C.sp_g_uu(a,b,k,j,i) *
           C.sp_dH_d_(a,i) *
           C.sp_H_d(b,k,j,i);
  }

  return 2 * xi * SQR(J) + 2 * J * SQR(xi) * dJ + 2 * Hn * dHn - dHH;
}

void ClosureMinerboPicard(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int iter_max_C = pm1.opt.max_iter_C;
  int iter     = 0;     // iteration counter
  int iter_rst = 0;     // restart counter
  const int iter_max_R = pm1.opt.max_iter_C_rst;  // max restarts
  Real w_opt = pm1.opt.w_opt_ini_C;  // underrelaxation factor
  Real e_C_abs_tol = pm1.opt.eps_C;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1.opt.fac_amp_C;

  C.FallbackStore(k, j, i);

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  // solver loop ----------------------------------------------------------
  const Real W    = C.sc_W(k,j,i);
  const Real oo_W = 1.0 / W;
  const Real W2   = SQR(W);

  do
  {
    iter++;

    populate_scratch = false;
    const Real Z_ = Z_xi(C.sc_xi(k,j,i), pm1, C, k, j, i, populate_scratch);

    Real sc_xi_can = C.sc_xi(k,j,i) - w_opt * Z_;

    e_abs_cur = std::abs(sc_xi_can - C.sc_xi(k,j,i));
    C.sc_xi(k,j,i) = sc_xi_can;

    bool compute_limiting_P_dd = false;
    const bool ecl = Closures::Minerbo::EnforceClosureLimits(
      pm1, C, k, j, i, compute_limiting_P_dd
    );

    if ((e_abs_cur > fac_PA * e_abs_old))
    {
      // halve underrelaxation and recover old values
      w_opt = w_opt / 2;
      C.Fallback(k, j, i);

      // restart iteration
      e_abs_old = std::numeric_limits<Real>::infinity();
      iter = 0;
      ++iter_rst;

      if (iter_rst > iter_max_R)
      {
        std::ostringstream msg;
        msg << "ClosureMinerboPicard max restarts exceeded.";
        std::cout << msg.str().c_str() << "\n";
        pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);
      }
    }
    else
    {
      e_abs_old = e_abs_cur;
      C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));
    }

  } while ((iter < iter_max_C) && (e_abs_cur >= e_C_abs_tol));

  if ((e_abs_cur > e_C_abs_tol) && pm1.opt.verbose_iter_C)
  {
    std::cout << "ClosureMinerboPicard:\n";
    std::cout << "Tol. not achieved: (pit,rit,e_abs_cur) ";
    std::cout << iter << "," << iter_rst << "," << e_abs_cur << "\n";
    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);
  }
}

void ClosureMinerboNewton(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int iter_max_C = pm1.opt.max_iter_C;
  int iter = 0;     // iteration counter
  Real w_opt = pm1.opt.w_opt_ini_C;  // underrelaxation factor
  Real e_C_abs_tol = pm1.opt.eps_C;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1.opt.fac_amp_C;

  auto is_valid = [&](const Real xi)
  {
    return (xi >= xi_min) && (xi <= xi_max);
  };

  C.FallbackStore(k, j, i);

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  // solver loop ----------------------------------------------------------
  const Real W    = C.sc_W(k,j,i);
  const Real oo_W = 1.0 / W;
  const Real W2   = SQR(W);

  bool revert_Picard = false;

  do
  {
    iter++;

    populate_scratch = false;
    const Real Z_ = Z_xi(C.sc_xi(k,j,i), pm1, C, k, j, i, populate_scratch);

    Real sc_xi_can = C.sc_xi(k,j,i) - w_opt * Z_;

    // break-fast?
    if ((std::abs(C.sc_xi(k,j,i) - sc_xi_can) < e_C_abs_tol))
    {
      break;
    }

    const Real dZ_ = dZ_xi(C.sc_xi(k,j,i), pm1, C, k, j, i, populate_scratch);

    // Apply Newton iterate with fallback
    Real D = 0;

    if ((std::abs(dZ_) > pm1.opt.eps_C_N))
    {
      // Newton
      D = Z_ / dZ_;
      sc_xi_can = C.sc_xi(k,j,i) - D;
    }

    bool compute_limiting_P_dd = false;
    const bool ecl = Closures::Minerbo::EnforceClosureLimits(
      pm1, C, k, j, i, compute_limiting_P_dd
    );

    if (ecl)
    {
      revert_Picard = true;
      break;
    }

    e_abs_cur = std::abs(C.sc_xi(k,j,i) - sc_xi_can);

    if ((e_abs_cur > fac_PA * e_abs_old))
    {
      // stagnated
      revert_Picard = true;
      break;
    }
    else
    {
      e_abs_old = e_abs_cur;
      C.sc_xi(k,j,i) = sc_xi_can;
      C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));
    }

  } while ((iter < iter_max_C) &&
           (e_abs_cur >= e_C_abs_tol) &&
           !revert_Picard);

  if (revert_Picard || !is_valid(C.sc_xi(k,j,i)))
  {
    C.Fallback(k, j, i);
    ClosureMinerboPicard(pm1, C, k, j, i);
  }

}

// ============================================================================
} // namespace M1::Closures::Minerbo
// ============================================================================

// ============================================================================
namespace M1::Closures::Minerbo::gsl {
// ============================================================================

struct gsl_params {
  M1 & pm1;
  ClosureMetaVector & C;

  const int i;
  const int j;
  const int k;
};

Real Z_xi(Real xi, void * par_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  const bool compute_limiting_P_dd = false;
  return ::M1::Closures::Minerbo::Z_xi(xi, par->pm1, par->C, k, j, i,
                                       compute_limiting_P_dd);
}

Real dZ_xi(Real xi, void * par_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  const bool compute_limiting_P_dd = false;
  return ::M1::Closures::Minerbo::dZ_xi(xi, par->pm1, par->C, k, j, i,
                                        compute_limiting_P_dd);
}

void ZdZ_xi(Real xi, void * par_, Real * Z, Real * dZ)
{
  *Z  = Z_xi( xi, par_);
  *dZ = dZ_xi(xi, par_);
}

void ClosureMinerboBrent(M1 & pm1,
                         ClosureMetaVector & C,
                         const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // select function & solver -------------------------------------------------
  struct gsl_params par = {pm1, C, i, j, k};

  gsl_function Z_;
  Z_.function = &Z_xi;
  Z_.params = &par;

  auto gsl_err_kill = [&](const int status)
  {
    std::ostringstream msg;
    msg << "ClosureMinerboBrent unexpected error: ";
    msg << status;
    std::cout << msg.str().c_str() << std::endl;

    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
  };

  auto gsl_err_warn = [&]()
  {
    std::ostringstream msg;
    msg << "ClosureMinerboBrent maxiter=";
    msg << pm1.opt.max_iter_C << " exceeded";
    std::cout << msg.str().c_str() << std::endl;
  };
  // --------------------------------------------------------------------------

  const bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  int gsl_status = gsl_root_fsolver_set(pm1.gsl_brent_solver,
                                        &Z_, xi_min, xi_max);

  /*
  typedef struct
    {
      double a, b, c, d, e;
      double fa, fb, fc;
    }
  brent_state_t;

  brent_state_t * state = static_cast<brent_state_t*>(gsl_solver->state);
  */

  switch (gsl_status)
  {
    case (GSL_EINVAL):  // bracketing failed (revert to thin closure)
    {
      Assemble::PointToDense(C.sp_P_dd, C.sp_P_tn_dd_, k, j, i);
      break;
    }
    case (0):
    {
      // root-finding loop
      Real loc_xi_min = xi_min;
      Real loc_xi_max = xi_max;

      gsl_status = GSL_CONTINUE;
      int iter = 0;
      do
      {
        iter++;
        gsl_status = gsl_root_fsolver_iterate(pm1.gsl_brent_solver);

        if (gsl_status != GSL_SUCCESS)
        {
          break;
        }
        else if (gsl_status)
        {
          C.sp_P_tn_dd_.array().print_all();
          C.sp_P_tk_dd_.array().print_all();

          pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);

          gsl_err_kill(gsl_status);
        }

        loc_xi_min = gsl_root_fsolver_x_lower(pm1.gsl_brent_solver);
        loc_xi_max = gsl_root_fsolver_x_upper(pm1.gsl_brent_solver);

        gsl_status = gsl_root_test_interval(
          loc_xi_min, loc_xi_max, pm1.opt.eps_C, 0
        );

      } while (iter<=pm1.opt.max_iter_C && gsl_status == GSL_CONTINUE);

      if ((gsl_status != GSL_SUCCESS) && pm1.opt.verbose_iter_C)
      {
        gsl_err_warn();
      }

      // Update P_dd with root
      C.sc_xi( k,j,i) = pm1.gsl_brent_solver->root;
      C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));

      sp_P_dd__(C.sp_P_dd, C.sc_chi, C.sp_P_tn_dd_, C.sp_P_tk_dd_, k, j, i);

      break;
    }
    default:
    {
      // unknown status code
      pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);
      gsl_err_kill(gsl_status);
    }
  }


}

void ClosureMinerboNewton(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // select function & solver -------------------------------------------------
  struct gsl_params par = {pm1, C, i, j, k};

  gsl_function_fdf Z_;
  Z_.f   = &Z_xi;
  Z_.df  = &dZ_xi;
  Z_.fdf = &ZdZ_xi;
  Z_.params = &par;

  auto gsl_err_kill = [&](const int status)
  {
    std::ostringstream msg;
    msg << "ClosureMinerboNewton unexpected error: ";
    msg << status;
    std::cout << msg.str().c_str() << std::endl;

    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
  };

  auto gsl_err_warn = [&]()
  {
    std::ostringstream msg;
    msg << "ClosureMinerboNewton maxiter=";
    msg << pm1.opt.max_iter_C << " exceeded";
    std::cout << msg.str().c_str() << std::endl;
  };
  // --------------------------------------------------------------------------

  const bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  int gsl_status = gsl_root_fdfsolver_set(pm1.gsl_newton_solver,
                                          &Z_, C.sc_xi(k,j,i));

  bool revert_brent = false;

  switch (gsl_status)
  {
    case (GSL_EINVAL):  // bracketing failed (revert to thin closure)
    {
      Assemble::PointToDense(C.sp_P_dd, C.sp_P_tn_dd_, k, j, i);
      break;
    }
    case (0):
    {
      // root-finding loop
      Real loc_xim1 = C.sc_xi(k,j,i);
      Real loc_xi   = C.sc_xi(k,j,i);

      gsl_status = GSL_CONTINUE;
      int iter = 0;
      do
      {
        iter++;
        gsl_status = gsl_root_fdfsolver_iterate(pm1.gsl_newton_solver);

        if (gsl_status != GSL_SUCCESS)
        {
          break;
        }

        loc_xim1 = loc_xi;
        loc_xi = gsl_root_fdfsolver_root(pm1.gsl_newton_solver);

        gsl_status = gsl_root_test_delta(
          loc_xim1, loc_xi, pm1.opt.eps_C, 0
        );

        C.sc_xi(k,j,i) = loc_xi;

        bool compute_limiting_P_dd = false;
        const bool ecl = Closures::Minerbo::EnforceClosureLimits(
          pm1, C, k, j, i, compute_limiting_P_dd
        );

      } while (iter<=pm1.opt.max_iter_C && gsl_status == GSL_CONTINUE);

      if ((gsl_status != GSL_SUCCESS) && pm1.opt.verbose_iter_C)
      {
        gsl_err_warn();
      }

      // Update P_dd with root
      C.sc_xi( k,j,i) = pm1.gsl_newton_solver->root;
      C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));

      sp_P_dd__(C.sp_P_dd, C.sc_chi, C.sp_P_tn_dd_, C.sp_P_tk_dd_, k, j, i);

      break;
    }
    default:
    {
      revert_brent = true;
    }
  }

  if (revert_brent)
  {
    ClosureMinerboBrent(pm1, C, k, j, i);
    return;
  }

}

// ============================================================================
} // namespace M1::Closures::Minerbo::gsl
// ============================================================================


// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
// Computes the closure on a mesh block
void M1::CalcClosure(AthenaArray<Real> & u)
{
  using namespace Closures;

  vars_Lab U { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U);

  // fix ranges ---------------------------------------------------------------
  const int IL = 0;
  const int IU = mbi.nn1 - 1;

  const int JL = 0;
  const int JU = mbi.nn2 - 1;

  const int KL = 0;
  const int KU = mbi.nn3 - 1;

  // dispatch closure strategy ------------------------------------------------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    ClosureMetaVector C = ConstructClosureMetaVector(*this, U, ix_g, ix_s);

    for (int k=KL; k<=KU; ++k)
    for (int j=JL; j<=JU; ++j)
    for (int i=IL; i<=IU; ++i)
    if (pm1->MaskGet(k,j,i))
    {
      // NonFiniteToZero(*this, C, k, j, i);
      C.Closure(k,j,i);
    }
  }
  return;
}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, P_dd) to (J, H_d)
//
// This isn't needed for the core algorithm but may be for data-dumping
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