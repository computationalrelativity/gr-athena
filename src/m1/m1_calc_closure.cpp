// c++
#include <functional>
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_closure.hpp"
#include "m1_calc_update.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <ostream>

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
  switch (pm1.opt_closure.variety)
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
    case (M1::opt_closure_variety::MinerboB):
    {
      C.Closure = [&](const int k, const int j, const int i)
      {
        Minerbo::ClosureMinerboBisection(pm1, C, k, j, i);
      };
      break;
    }
    case (M1::opt_closure_variety::MinerboN):
    {
      C.Closure = [&](const int k, const int j, const int i)
      {
        // Minerbo::gsl::ClosureMinerboNewton(pm1, C, k, j, i);
        Minerbo::ClosureMinerboNewton(pm1, C, k, j, i);
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

  dJ = W2 * dJ;

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

  return 2 * xi * SQR(J) + 2 * J * SQR(xi) * dJ + 2 * Hn * dHn - 2 * dHH;
}

// Check for thin / thick limits where bracket does not change sign.
//
// At such a limit fix appropriate P_dd
bool MinerboFallbackLimits(M1 & pm1,
                           ClosureMetaVector & C,
                           const int k, const int j, const int i,
                           const bool populate_scratch)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  Bracket & br_xi = C.br_xi;
  br_xi.a = 0.0;
  br_xi.b = 1.0;

  Bracket & br_Z = C.br_Z;

  br_Z.a = Z_xi(br_xi.a, pm1, C, k, j, i, populate_scratch);
  br_Z.b = Z_xi(br_xi.b, pm1, C, k, j, i, populate_scratch);

  bool limit_fallback = false;

  auto revert_tn = [&]()
  {
    C.sc_xi(k,j,i)  = 1;
    C.sc_chi(k,j,i) = 1;
    Assemble::PointToDense(C.sp_P_dd, C.sp_P_tn_dd_, k, j, i);
    limit_fallback = true;
  };

  auto revert_tk = [&]()
  {
    C.sc_xi(k,j,i)  = 0;
    C.sc_chi(k,j,i) = ONE_3RD;
    Assemble::PointToDense(C.sp_P_dd, C.sp_P_tk_dd_, k, j, i);
    limit_fallback = true;
  };

  if (!(br_Z.sign_change()))
  {
    if (pm1.opt_closure.fallback_thin)
    {
      // force this if desired
      revert_tn();
    }
    else if (std::abs(br_Z.a) < std::abs(br_Z.b))
    {
      // revert thick
      revert_tk();
    }
    else
    {
      // revert thin (N.B lands here when Z_a=Z_b=c)
      revert_tn();
    }
  }

  // if (std::abs(br_Z.a) < std::abs(br_Z.b))
  // {
  //   if (std::abs(br_Z.a / C.sc_E(k,j,i)) < pm1.opt_closure.eps_Z_o_E)
  //   {
  //     revert_tk();
  //   }
  // }
  // else
  // {
  //   if (std::abs(br_Z.b / C.sc_E(k,j,i)) < pm1.opt_closure.eps_Z_o_E)
  //   {
  //     revert_tn();
  //   }
  // }

  if (C.sc_xi(k,j,i) <= pm1.opt_closure.fac_Z_o_E)
  {
    if (std::abs(br_Z.a / C.sc_E(k,j,i)) < pm1.opt_closure.eps_Z_o_E)
    {
      revert_tk();
    }
  }
  else if (C.sc_xi(k,j,i) >= 1-pm1.opt_closure.fac_Z_o_E)
  {
    if (std::abs(br_Z.b / C.sc_E(k,j,i)) < pm1.opt_closure.eps_Z_o_E)
    {
      revert_tn();
    }
  }


  return limit_fallback;
}

void ClosureMinerboPicard(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i)
{
  const Real INF = std::numeric_limits<Real>::infinity();

  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  const int iter_max = pm1.opt_closure.iter_max;
  int iter     = 0;     // iteration counter
  int iter_rst = 0;     // restart counter
  const int iter_max_R = pm1.opt_closure.iter_max_rst;  // max restarts
  Real w_opt = pm1.opt_closure.w_opt_ini;  // underrelaxation factor
  Real err_max = pm1.opt_closure.eps_tol;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1.opt_closure.fac_err_amp;

  C.FallbackStore(k, j, i);

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  Real err_old = INF;
  Real err_cur = 0;

  // attempt to accelerate with neighbor-data?
  if (pm1.opt_closure.use_Neighbor)
  {
    if ((i > pm1.mbi.il) && C.sc_xi(k,j,i-1))
    {
      C.sc_xi(k,j,i) = C.sc_xi(k,j,i-1);
    }
  }

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

    err_cur = std::abs(sc_xi_can - C.sc_xi(k,j,i));
    C.sc_xi(k,j,i) = sc_xi_can;

    bool compute_limiting_P_dd = false;
    const bool ecl = Closures::Minerbo::EnforceClosureLimits(
      pm1, C, k, j, i, compute_limiting_P_dd
    );

    if ((err_cur > fac_PA * err_old))
    {
      // halve underrelaxation and recover old values
      w_opt = w_opt / 2;
      C.Fallback(k, j, i);

      // restart iteration
      err_old = INF;
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
      err_old = err_cur;
      C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));
    }

  } while ((iter < iter_max) && (err_cur >= err_max));

  if ((err_cur > err_max) && pm1.opt_closure.verbose)
  {
    std::cout << "ClosureMinerboPicard:\n";
    std::cout << "Tol. not achieved: (pit,rit,err_cur) ";
    std::cout << iter << "," << iter_rst << "," << err_cur << "\n";
    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);
  }
}

void ClosureMinerboBisection(M1 & pm1,
                             ClosureMetaVector & C,
                             const int k, const int j, const int i)
{
  const Real INF = std::numeric_limits<Real>::infinity();

  const int iter_max = pm1.opt_closure.iter_max;
  int iter = 0;     // iteration counter
  Real w_opt = pm1.opt_closure.w_opt_ini;  // underrelaxation factor
  Real err_tol = pm1.opt_closure.eps_tol;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1.opt_closure.fac_err_amp;

  auto sign_change = [](const Real a, const Real b)
  {
    return a * b < 0;
  };

  C.FallbackStore(k, j, i);

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  Real err_cur = INF;

  // solver loop ----------------------------------------------------------
  populate_scratch = false;

  Bracket & br_Z  = C.br_Z;
  Bracket & br_xi = C.br_xi;

  // admissible limits and check whether at one
  br_xi.a = 0;
  br_xi.b = 1.0;

  const bool at_limit = MinerboFallbackLimits(
    pm1, C, k, j, i, populate_scratch
  );

  if (at_limit)
  {
    return;
  }

  // not at limit, continue with solver
  Real xi_i = br_xi.midpoint();

  do
  {
    iter++;

    Real Z_i = Z_xi(xi_i, pm1, C, k, j, i, populate_scratch);

    if (sign_change(br_Z.a, Z_i))
    {
      br_xi.b = xi_i;
      br_Z.b  = Z_i;
    }
    else
    {
      br_xi.a = xi_i;
      br_Z.a  = Z_i;
    }

    // new mid-point
    xi_i = br_xi.midpoint();

    err_cur = std::abs(1 - chi(br_xi.a) / chi(br_xi.b));
  } while ((iter < iter_max) &&
           (err_cur >= err_tol));

  C.sc_xi(k,j,i)  = xi_i;
  C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));
  sp_P_dd__(C.sp_P_dd, C.sc_chi, C.sp_P_tn_dd_, C.sp_P_tk_dd_, k, j, i);

  if (pm1.opt_closure.verbose &&
      ((err_cur > err_tol) || (iter == iter_max)))
  {
    std::cout << "ClosureMinerboBisection:\n";
    std::cout << "max_iter hit / tol. not achieved: (iter,err_cur) ";
    std::cout << iter << "," << err_cur << "\n";
  }

}

void ClosureMinerboNewton(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i)
{
  const Real INF = std::numeric_limits<Real>::infinity();

  const int iter_max = pm1.opt_closure.iter_max;
  int iter = 0;     // iteration counter
  Real err_tol = pm1.opt_closure.eps_tol;

  auto sign_change = [](const Real a, const Real b)
  {
    return a * b < 0;
  };

  C.FallbackStore(k, j, i);

  // main loop ----------------------------------------------------------------
  int iter_tot = 0;

  bool populate_scratch = true;
  ClosureThin( pm1, C, k, j, i, populate_scratch);
  ClosureThick(pm1, C, k, j, i, populate_scratch);

  Real err_cur = INF;

  // solver loop ----------------------------------------------------------
  populate_scratch = false;

  Bracket & br_Z  = C.br_Z;
  Bracket & br_xi = C.br_xi;

  // admissible limits and check whether at one
  br_xi.a = 0;
  br_xi.b = 1.0;

  const bool at_limit = MinerboFallbackLimits(
    pm1, C, k, j, i, populate_scratch
  );

  if (at_limit)
  {
    return;
  }

  // not at limit, continue with solver
  Real xi_i = C.sc_xi(k,j,i);
  // Real xi_i = br_xi.midpoint();

  // attempt to accelerate with neighbor-data?
  if (pm1.opt_closure.use_Neighbor)
  {
    if ((i > pm1.mbi.il) && br_xi.bounds_strict(C.sc_xi(k,j,i-1)))
    {
      xi_i = C.sc_xi(k,j,i-1);
    }
  }

  Real xi_im1 = INF;
  Real xi_im2 = INF;

  bool accept_newton = true;
  Real Z_i   = Z_xi(xi_i, pm1, C, k, j, i, populate_scratch);
  Real Z_im1 = INF;
  Real Z_im2 = INF;

  Real dZ_i   = INF;
  Real dZ_im1 = INF;
  Real dZ_im2 = INF;

  auto seq3_push = [](const Real push, Real & a_i, Real & a_im1, Real & a_im2)
  {
    a_im2 = a_im1;
    a_im1 = a_i;
    a_i   = push;
  };


  // TODO:
  // Use the three initial points with a fit to get a first zero guess?

  // #pragma omp critical
  // {
  //   std::cout << xi_i << std::endl;
  //   std::cout << (Z_i * br_xi.a - xi_i * br_Z.a) / (Z_i - br_Z.a) << std::endl;
  //   std::cout << (Z_i * br_xi.b - xi_i * br_Z.b) / (Z_i - br_Z.b) << std::endl;
  //   std::exit(0);
  // }

  // Real xi_l = (Z_i * br_xi.a - xi_i * br_Z.a) / (Z_i - br_Z.a);
  // Real xi_r = (Z_i * br_xi.b - xi_i * br_Z.b) / (Z_i - br_Z.b);

  // if ((br_xi.a < xi_l) && (xi_l < xi_i))
  // {
  //   seq3_push(
  //     Z_xi(xi_l, pm1, C, k, j, i, populate_scratch),
  //     Z_i, Z_im1, Z_im2
  //   );
  //   seq3_push(
  //     xi_l, xi_i, xi_im1, xi_im2
  //   );
  // }
  // else if ((xi_i < xi_r) && (xi_r < br_xi.b))
  // {
  //   seq3_push(
  //     Z_xi(xi_r, pm1, C, k, j, i, populate_scratch),
  //     Z_i, Z_im1, Z_im2
  //   );
  //   seq3_push(
  //     xi_r, xi_i, xi_im1, xi_im2
  //   );
  // }

  int nfail_newton    = 0;
  int nfail_ostrowski = 0;
  int lf_newton = -1;

  Real dbg_br_Z_a = br_Z.a;
  Real dbg_br_Z_b = br_Z.b;

  Real xi_c, dZ_c, Y_i;

  do
  {
    iter++;

    if (!accept_newton)
    {
      seq3_push(
        Z_xi(xi_i, pm1, C, k, j, i, populate_scratch),
        Z_i, Z_im1, Z_im2
      );
    }

    if (1) // (accept_newton)
    {
      if (!pm1.opt_closure.use_Ostrowski)
      {
        if (iter == 1)
          dZ_c = dZ_xi(xi_i, pm1, C, k, j, i, populate_scratch);
        else
          dZ_c = (Z_i - Z_im1) / (xi_i - xi_im1);  // secant
        xi_c = xi_i - Z_i / dZ_c;
      }
      else
      {
        // Ostrowski 4th order refinement
        dZ_c = dZ_xi(xi_i, pm1, C, k, j, i, populate_scratch);

        Real xi_n = xi_i - Z_i / dZ_c;

        Y_i = Z_xi(xi_n, pm1, C, k, j, i, populate_scratch);
        Real xi_o = xi_i - Z_i / dZ_c * (Y_i - Z_i) / (2.0 * Y_i - Z_i);

        if (br_xi.bounds(xi_o))
        {
          xi_c = xi_o;
        }
        else if (br_xi.bounds(xi_n))
        {
          ++nfail_ostrowski;
          xi_c = xi_n;
        }
        else if (iter > 1)
        {
          ++nfail_ostrowski;
          ++nfail_newton;

          // secant fall-back
          dZ_c = (Z_i - Z_im1) / (xi_i - xi_im1);
          const Real xi_s = xi_i - Z_i / dZ_c;
          if (br_xi.bounds(xi_s))
          {
            xi_c = xi_s;
          }
        }


        // Real xi_e = (Z_i * xi_i - xi_c * Z_im1) / (Z_i - Z_im1);

        // if (br_xi.bounds(xi_e))
        // {
        //   xi_c = xi_e;
        // }

      }

      // accept_newton = br_xi.bounds(xi_c);
      accept_newton = br_xi.bounds(xi_c) &&  // Numerical recipes slope check
                      (std::abs(2.0 * Z_i) < std::abs(dZ_c * (xi_i-xi_im1)));

      seq3_push(
        dZ_c,
        dZ_i, dZ_im1, dZ_im2
      );
    }

    // try to get a better estimate
    if (0) // (accept_newton && (iter >= 2))
    {
      // Real xi_e = (Y_i * xi_i - xi_c * Z_im1) / (Y_i - Z_im1);
      Real xi_e = (Z_i * xi_i - xi_c * Z_im1) / (Z_i - Z_im1);

      if (br_xi.bounds(xi_e))
      {
        xi_c = xi_e;
      }
    }

    if (accept_newton)
    {
      // avoid a function eval. if possible
      err_cur = std::abs(1 - chi(xi_i) / chi(xi_c));
      if (err_cur < err_tol)
      {
        break;
      }

      // need fcn at updated location
      seq3_push(
        Z_xi(xi_c, pm1, C, k, j, i, populate_scratch),
        Z_i, Z_im1, Z_im2
      );

      seq3_push(
        xi_c, xi_i, xi_im1, xi_im2
      );
    }

    if (sign_change(br_Z.a, Z_i))
    {
      br_xi.b = xi_i;
      br_Z.b  = Z_i;
    }
    else
    {
      br_xi.a = xi_i;
      br_Z.a  = Z_i;
    }

    // fallback to bisection estimate
    if (!accept_newton)
    {
      lf_newton = iter;

      Real xi_e = -1;

      if (iter > 1)
      {
        xi_e = (Z_i * xi_im1 - xi_i * Z_im1) / (Z_i - Z_im1);
      }

      // // xi_e = br_xi.squeeze(xi_e);
      // // if (xi_e == 1)
      // //   break;

      // // if (xi_e == 0)
      // //   break;

      if (br_xi.bounds(xi_e))
      {
        xi_c = xi_e;
      }
      else
      {
        xi_c = br_xi.midpoint();
      }

      // xi_c = br_xi.midpoint();
      seq3_push(
        xi_c,
        xi_i, xi_im1, xi_im2
      );
    }

    err_cur = std::abs(1 - chi(xi_im1) / chi(xi_i));
    err_cur = std::min(err_cur, std::abs(1 - chi(br_xi.a) / chi(br_xi.b)));

  } while ((iter < iter_max) && (err_cur >= err_tol));

  if (0) //  (pm1.opt_closure.verbose)
  {
    if ((iter > 10) || (nfail_newton > 1))
    #pragma omp critical
    {
      std::cout << iter << std::endl;
      std::cout << "i_c "<< xi_c << "\n";
      std::cout << "i_0 "<< xi_i << "\n";
      std::cout << "im1 "<< xi_im1 << "\n";
      std::cout << "im2 "<< xi_im2 << "\n";
      std::cout << "chi_i_c "<< chi(xi_c) << "\n";
      std::cout << "chi_i_0 "<< chi(xi_i) << "\n";
      std::cout << "chi_im1 "<< chi(xi_im1) << "\n";
      std::cout << "chi_im2 "<< chi(xi_im2) << "\n";
      std::cout << "Zi_0 "<< Z_i / C.sc_E(k,j,i) << "\n";
      std::cout << "Zim1 "<< Z_im1 / C.sc_E(k,j,i) << "\n";
      std::cout << "Zim2 "<< Z_im2 / C.sc_E(k,j,i) << "\n";

      std::cout << "dbg_br_Z_a "<< dbg_br_Z_a << "\n";
      std::cout << "dbg_br_Z_b "<< dbg_br_Z_b << "\n";

      std::cout << "br_xi.a "<< br_xi.a << "\n";
      std::cout << "br_xi.b "<< br_xi.b << "\n";

      std::cout << "br_chi.a "<< chi(br_xi.a) << "\n";
      std::cout << "br_chi.b "<< chi(br_xi.b) << "\n";

      std::cout << "C.sc_xi(k,j,i) " << C.sc_xi(k,j,i) << "\n";
      std::cout << "C.sc_E(k,j,i) " << C.sc_E(k,j,i) << "\n";
      std::cout << "e_abs_cur " << err_cur << "\n";

      std::cout << accept_newton << "\n";
      std::cout << nfail_newton << std::endl;
      std::cout << nfail_ostrowski << std::endl;
      std::cout << lf_newton << std::endl;

      std::exit(0);
    }
    // pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);
  }

  C.sc_xi(k,j,i)  = xi_i;
  C.sc_chi(k,j,i) = chi(C.sc_xi(k,j,i));
  sp_P_dd__(C.sp_P_dd, C.sc_chi, C.sp_P_tn_dd_, C.sp_P_tk_dd_, k, j, i);

  if (pm1.opt_closure.verbose &&
      ((err_cur > err_tol) || (iter == iter_max)))
  {
    std::cout << "ClosureMinerboNewton:\n";
    std::cout << "max_iter hit / tol. not achieved: (iter,err_cur) ";
    std::cout << iter << "," << err_cur << "\n";
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
    #pragma omp critical
    {
      std::ostringstream msg;
      msg << "ClosureMinerboBrent unexpected error: ";
      msg << status;
      std::cout << msg.str().c_str() << std::endl;
    }
    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
  };

  auto gsl_err_warn = [&]()
  {
    #pragma omp critical
    {
      std::ostringstream msg;
      msg << "ClosureMinerboBrent maxiter=";
      msg << pm1.opt_closure.iter_max << " exceeded";
      std::cout << msg.str().c_str() << std::endl;
    }
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
    case (GSL_EINVAL):  // bracketing failed
    {
      if (pm1.opt_closure.fallback_thin)
      {
        Assemble::PointToDense(C.sp_P_dd, C.sp_P_tn_dd_, k, j, i);
      }
      else
      {
        const bool populate_scratch = false;
        MinerboFallbackLimits(pm1, C, k, j, i, populate_scratch);
      }
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
          loc_xi_min, loc_xi_max, pm1.opt_closure.eps_tol, 0
        );

      } while (iter<=pm1.opt_closure.iter_max && gsl_status == GSL_CONTINUE);

      if ((gsl_status != GSL_SUCCESS) && pm1.opt_closure.verbose)
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
    msg << pm1.opt_closure.iter_max << " exceeded";
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
          loc_xim1, loc_xi, pm1.opt_closure.eps_tol, 0
        );

        C.sc_xi(k,j,i) = loc_xi;

        bool compute_limiting_P_dd = false;
        const bool ecl = Closures::Minerbo::EnforceClosureLimits(
          pm1, C, k, j, i, compute_limiting_P_dd
        );

      } while (iter<=pm1.opt_closure.iter_max && gsl_status == GSL_CONTINUE);

      if ((gsl_status != GSL_SUCCESS) && pm1.opt_closure.verbose)
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

  // dispatch closure strategy ------------------------------------------------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    ClosureMetaVector C = ConstructClosureMetaVector(*this, U, ix_g, ix_s);

    M1_FLOOP3(k,j,i)
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