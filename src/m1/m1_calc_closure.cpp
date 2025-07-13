// c++
#include <functional>
#include <iostream>
#include <algorithm>

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
namespace M1 {
// ============================================================================

// ============================================================================
namespace Closures {
// ============================================================================

static const Real XI_MIN = 0.0;
static const Real XI_MAX = 1.0;

// ============================================================================
namespace EddingtonFactors {
// ============================================================================

void ThinLimit(Real & xi, Real & chi)
{
  xi = 1.0;
  chi = 1.0;
}

void ThickLimit(Real & xi, Real & chi)
{
  xi = 0.0;
  chi = ONE_3RD;
}

void Minerbo(Real & xi, Real & chi)
{
  const Real xi2 = SQR(xi);
  chi = ONE_3RD + xi2 / 15.0 * (6.0 - 2.0 * xi + 6 * xi2);
}

void Kershaw(Real & xi, Real & chi)
{
  const Real xi2 = SQR(xi);
  chi = ONE_3RD * (1.0 + 2.0 * xi2);
}

// General interface ----------------------------------------------------------
void Compute(M1 & pm1, Real & xi, Real & chi)
{
  typedef M1::opt_closure_variety ocv;

  switch (pm1.opt_closure.variety)
  {
    case (ocv::thin):
    {
      ThinLimit(xi, chi);
      break;
    }
    case (ocv::thick):
    {
      ThickLimit(xi, chi);
      break;
    }
    case (ocv::Minerbo):
    {
      Minerbo(xi, chi);
      break;
    }
    case (ocv::Kershaw):
    {
      Kershaw(xi, chi);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

// ============================================================================
namespace D1 {
// ============================================================================

// First derivatives (wrt. xi);
// chi=chi(xi) -> chi'(xi)

void ThinLimit(Real & xi, Real & chi)
{
  xi = 0.0;
  chi = 0.0;
}

void ThickLimit(Real & xi, Real & dchi_dxi)
{
  xi = 0.0;
  dchi_dxi = 0.0;
}

void Minerbo(Real & xi, Real & dchi_dxi)
{
  dchi_dxi = (2.0 / 5.0) * xi * (
    2.0 + xi * (4.0 * xi - 1.0)
  );
}

void Kershaw(Real & xi, Real & dchi_dxi)
{
  dchi_dxi = 4.0 * ONE_3RD * xi;
}

// General interface ----------------------------------------------------------
void Compute(M1 & pm1, Real & xi, Real & dchi_dxi)
{
  typedef M1::opt_closure_variety ocv;

  switch (pm1.opt_closure.variety)
  {
    case (ocv::thin):
    {
      ThinLimit(xi, dchi_dxi);
      break;
    }
    case (ocv::thick):
    {
      ThickLimit(xi, dchi_dxi);
      break;
    }
    case (ocv::Minerbo):
    {
      Minerbo(xi, dchi_dxi);
      break;
    }
    case (ocv::Kershaw):
    {
      Kershaw(xi, dchi_dxi);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}
// ============================================================================
} // namespace M1::Closures::EddingtonFactors::D1
// ============================================================================

// ============================================================================
} // namespace M1::Closures::EddingtonFactors
// ============================================================================

void ClosureMetaVector::Closure(const int k, const int j, const int i)
{
  // do not call solver for exact prescriptions -------------------------------
  typedef M1::opt_closure_variety ocv;
  if ((pm1.opt_closure.variety != ocv::thin) &&
      (pm1.opt_closure.variety != ocv::thick))
  {
    // solve for xi root if required: -----------------------------------------
    typedef solvers::status status_sol;
    status_sol ss = status_sol::success;

    switch (pm1.opt_closure.method)
    {
      case (M1::opt_closure_method::none):
      {
        break;
      }
      case (M1::opt_closure_method::gsl_Brent):
      {
        ss = solvers::gsl_Brent(pm1, *this, k, j, i);
        break;
      }
      case (M1::opt_closure_method::gsl_Newton):
      {
        ss = solvers::gsl_Newton(pm1, *this, k, j, i);
        break;
      }
      case (M1::opt_closure_method::custom_NB):
      {
        ss = solvers::custom_NB(pm1, *this, k, j, i);
        break;
      }
      case (M1::opt_closure_method::custom_NAB):
      {
        ss = solvers::custom_NAB(pm1, *this, k, j, i);
        break;
      }
      case (M1::opt_closure_method::custom_ONAB):
      {
        ss = solvers::custom_ONAB(pm1, *this, k, j, i);
        break;
      }
      default:
      {
        assert(false);
      }
    }

    // deal with failures / reversion -----------------------------------------
    if (ss != status_sol::success)
    // if ((ss == status_sol::fail_unknown) ||
    //     (ss == status_sol::fail_bracket) ||
    //     (ss == status_sol::fail_value))
    {
      // std::printf("%d\n", static_cast<int>(ss));
      solvers::Fallback_Xi_Chi_Limits(pm1, *this, k, j, i);
      return;
    }
  }

  // compute Eddington factor -------------------------------------------------
  EddingtonFactors::Compute(pm1, sc_xi(k,j,i), sc_chi(k,j,i));
}

ClosureMetaVector ConstructClosureMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s)
{
  return ClosureMetaVector {
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
    pm1.lab_aux.sc_chi( ix_g,ix_s),
    pm1.lab_aux.sc_xi(  ix_g,ix_s),
    // Lagrangian frame
    pm1.rad.sc_J(  ix_g,ix_s),
    pm1.rad.st_H_u(ix_g,ix_s),
  };
}

// ============================================================================
namespace solvers {
// ============================================================================

// functional to find root of
Real Z_xi(
  const Real xi__,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i)
{
  C.sc_xi(k,j,i) = xi__;
  EddingtonFactors::Compute(pm1, C.sc_xi(k,j,i), C.sc_chi(k,j,i));

  // Prepare H^alpha

  // We write:
  // sc_J = J_0
  // st_H^alpha = H_n n^alpha + H_v v^alpha + H_F F^alpha
  Real J_0, H_n, H_v, H_F;

  Assemble::Frames::ToFiducialExpansionCoefficients(
    pm1,
    J_0, H_n, H_v, H_F,
    C.sc_chi, C.sc_E, C.sp_F_d,
    k, j, i
  );

  // Projections
  const Real dotFv = Assemble::sc_dot_dense_sp__(
    C.sp_F_d,
    C.sp_v_u,
    k, j, i
  );

  const Real dotFF = Assemble::sp_norm2__(
    C.sp_F_d,
    pm1.geom.sp_g_uu,
    k,j,i
  );

  const Real dotvv = Assemble::sp_norm2__(
    pm1.fidu.sp_v_d,
    pm1.geom.sp_g_uu,
    k,j,i
  );

  const Real st_H_norm_2__ = (
    dotFF * SQR(H_F) - SQR(H_n) + 2 * dotFv * H_F * H_v + SQR(H_v) * dotvv
  );

  // Functional
  // N.B. perform a further rescaling
  const Real OO_E2 = OO(SQR(std::max(C.sc_E(k,j,i), pm1.opt.fl_E)));
  return OO_E2 * (
    st_H_norm_2__ - SQR(xi__ * J_0)
  );
}

void ZdZ_xi(
  Real & Z__,
  Real & dZ__,
  const Real xi__,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i)
{
  C.sc_xi(k,j,i) = xi__;
  EddingtonFactors::Compute(pm1, C.sc_xi(k,j,i), C.sc_chi(k,j,i));

  Real loc_xi__ = xi__;
  Real d_chi_dxi__;
  EddingtonFactors::D1::Compute(pm1, loc_xi__, d_chi_dxi__);

  // Prepare H^alpha

  // We write:
  // sc_J = J_0
  // st_H^alpha = H_n n^alpha + H_v v^alpha + H_F F^alpha
  Real J_0, H_n, H_v, H_F;
  Real dJ_dchi_0, dH_dchi_n, dH_dchi_v, dH_dchi_F;

  Assemble::Frames::D1::ToFiducialExpansionCoefficients(
    pm1,
    J_0, H_n, H_v, H_F,
    dJ_dchi_0, dH_dchi_n, dH_dchi_v, dH_dchi_F,
    C.sc_chi, C.sc_E, C.sp_F_d,
    k, j, i
  );

  // Projections
  const Real dotFv = Assemble::sc_dot_dense_sp__(
    C.sp_F_d,
    C.sp_v_u,
    k, j, i
  );

  const Real dotFF = Assemble::sp_norm2__(
    C.sp_F_d,
    pm1.geom.sp_g_uu,
    k,j,i
  );

  const Real dotvv = Assemble::sp_norm2__(
    pm1.fidu.sp_v_d,
    pm1.geom.sp_g_uu,
    k,j,i
  );

  const Real st_H_norm_2__ = (
    dotFF * SQR(H_F) - SQR(H_n) + 2 * dotFv * H_F * H_v + SQR(H_v) * dotvv
  );

  const Real dst_H_norm_2_dxi__ = 2.0 * d_chi_dxi__ * (
    dotFF * H_F * dH_dchi_F -
    H_n * dH_dchi_n +
    dotFv * (dH_dchi_F * H_v + H_F * dH_dchi_v)  +
    H_v * dH_dchi_v * dotvv
  );

  // // Functional
  const Real OO_E2 = OO(SQR(std::max(C.sc_E(k,j,i), pm1.opt.fl_E)));
  Z__ = OO_E2 * (
    st_H_norm_2__ - SQR(xi__ * J_0)
  );

  dZ__ = OO_E2 * (
    dst_H_norm_2_dxi__ -
    2.0 * (xi__ * J_0) * (J_0 + xi__ * dJ_dchi_0 * d_chi_dxi__)
  );
}

Real dZ_xi(
  const Real xi__,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i)
{
  Real Z__, dZ__;
  ZdZ_xi(Z__, dZ__, xi__, pm1, C, k, j, i);
  return dZ__;
}

// bool Enforce_Xi_Limits(Real & xi)
// {
//   if ((xi < XI_MIN) || (xi > XI_MAX))
//   {
//     xi = std::max(std::min(xi, XI_MAX), XI_MIN);
//     return true;
//   }
//   return false;
// }

bool Enforce_Xi_Limits(Real & xi)
{
  Real restricted_xi = std::clamp(xi, XI_MIN, XI_MAX);
  if (restricted_xi != xi)
  {
    xi = restricted_xi;
    return true;
  }
  return false;
}

void Fallback_Xi_Chi_Limits(M1 & pm1, ClosureMetaVector & C,
                            const int k, const int j, const int i)
{
  using namespace EddingtonFactors;

  // Admissible ranges for xi entering Eddington factor chi(xi)
  utils::Bracket br_xi;
  br_xi.a = XI_MIN;
  br_xi.b = XI_MAX;

  utils::Bracket br_Z;

  br_Z.a = Z_xi(br_xi.a, pm1, C, k, j, i);
  br_Z.b = Z_xi(br_xi.b, pm1, C, k, j, i);

  if (!(br_Z.sign_change()))
  {
    if (pm1.opt_closure.fallback_thin)
    {
      ThinLimit(C.sc_xi(k,j,i), C.sc_chi(k,j,i));
    }
  }

  if (std::abs(br_Z.a) < std::abs(br_Z.b))
  {
    // if (std::abs(br_Z.a / C.sc_E(k,j,i)) < pm1.opt_closure.eps_Z_o_E)
    // {
    //   revert_tk();
    // }
    ThickLimit(C.sc_xi(k,j,i), C.sc_chi(k,j,i));
  }
  else
  {
    // if (std::abs(br_Z.b / C.sc_E(k,j,i)) < pm1.opt_closure.eps_Z_o_E)
    // {
    //   revert_tn();
    // }
    ThinLimit(C.sc_xi(k,j,i), C.sc_chi(k,j,i));
  }

  /*
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
  */
}


// gsl specific ---------------------------------------------------------------
struct gsl_params {
  M1 & pm1;
  ClosureMetaVector & C;

  const int i;
  const int j;
  const int k;
};

Real gsl_Z_xi(Real xi, void * par_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  return Z_xi(xi, par->pm1, par->C, k, j, i);
}

Real gsl_dZ_xi(Real xi, void * par_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  return dZ_xi(xi, par->pm1, par->C, k, j, i);
}

void gsl_ZdZ_xi(Real xi, void * par_, Real * Z, Real * dZ)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  ZdZ_xi(*Z, *dZ, xi, par->pm1, par->C, k, j, i);
}

status gsl_Brent(M1 & pm1,
                 ClosureMetaVector & C,
                 const int k, const int j, const int i)
{
  // select function & solver -------------------------------------------------
  struct gsl_params par = {pm1, C, i, j, k};

  gsl_function Z_;
  Z_.function = &gsl_Z_xi;
  Z_.params = &par;

  auto gsl_err_kill = [&](const int status)
  {
    std::ostringstream msg;
    msg << "gsl_Brent unexpected error: ";
    msg << status;

    pm1.StatePrintPoint(
      msg.str(),
      C.ix_g, C.ix_s, k, j, i, true
    );
  };

  auto gsl_err_warn = [&]()
  {
    #pragma omp critical
    {
      std::ostringstream msg;
      msg << "gsl_Brent maxiter=";
      msg << pm1.opt_closure.iter_max << " exceeded";
      std::cout << msg.str().c_str() << std::endl;
    }
  };
  // --------------------------------------------------------------------------

  int gsl_status = gsl_root_fsolver_set(pm1.gsl_brent_solver,
                                        &Z_,
                                        XI_MIN,
                                        XI_MAX);

  /*
  typedef struct
    {
      double a, b, c, d, e;
      double fa, fb, fc;
    }
  brent_state_t;

  brent_state_t * state = static_cast<brent_state_t*>(gsl_solver->state);
  */

  status cur_status = status::success;

  switch (gsl_status)
  {
    case (GSL_EINVAL): // invalid function or bad initial guess
    {
      cur_status = status::fail_value;
      break;
    }
    case (0):
    {
      // root-finding loop
      Real loc_xi_min;
      Real loc_xi_max;

      gsl_status = GSL_CONTINUE;
      int iter = 0;
      do
      {
        iter++;
        gsl_status = gsl_root_fsolver_iterate(pm1.gsl_brent_solver);

        if (gsl_status != GSL_SUCCESS)
        {
          cur_status = status::fail_unknown;
          break;
        }

        C.sc_xi(k,j,i) = pm1.gsl_brent_solver->root;

        loc_xi_min = gsl_root_fsolver_x_lower(pm1.gsl_brent_solver);
        loc_xi_max = gsl_root_fsolver_x_upper(pm1.gsl_brent_solver);

        gsl_status = gsl_root_test_interval(
          loc_xi_min,
          loc_xi_max,
          pm1.opt_closure.eps_tol,
          0
        );
      } while (iter<=pm1.opt_closure.iter_max &&
               gsl_status == GSL_CONTINUE);


      if (gsl_status != GSL_SUCCESS)
      {
        if (pm1.opt_closure.verbose)
        {
          gsl_err_warn();
        }
        cur_status = status::fail_tolerance_not_met;
      }

      break;
    }
    default:
    {
      cur_status = status::fail_unknown;
    }
  }

  return cur_status;
}

status gsl_Newton(M1 & pm1,
                  ClosureMetaVector & C,
                  const int k, const int j, const int i)
{
  // select function & solver -------------------------------------------------
  struct gsl_params par = {pm1, C, i, j, k};

  gsl_function_fdf Z_;
  Z_.f   = &gsl_Z_xi;
  Z_.df  = &gsl_dZ_xi;
  Z_.fdf = &gsl_ZdZ_xi;
  Z_.params = &par;

  auto gsl_err_warn = [&]()
  {
    #pragma omp critical
    {
      std::ostringstream msg;
      msg << "gsl_Newton maxiter=";
      msg << pm1.opt_closure.iter_max << " exceeded";
      std::cout << msg.str().c_str() << std::endl;
    }
  };
  // --------------------------------------------------------------------------

  int gsl_status = gsl_root_fdfsolver_set(pm1.gsl_newton_solver,
                                          &Z_,
                                          C.sc_xi(k,j,i));

  status cur_status = status::success;

  switch (gsl_status)
  {
    case (GSL_EINVAL):  // bracketing failed
    {
      cur_status = status::fail_bracket;
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
          cur_status = status::fail_unknown;
          break;
        }

        C.sc_xi(k,j,i) = pm1.gsl_newton_solver->root;

        loc_xim1 = loc_xi;
        loc_xi = gsl_root_fdfsolver_root(pm1.gsl_newton_solver);

        gsl_status = gsl_root_test_delta(
          loc_xi, loc_xim1,
          pm1.opt_closure.eps_tol,
          0
        );

      } while (iter<=pm1.opt_closure.iter_max &&
               gsl_status == GSL_CONTINUE);

      break;
    }
    default:
    {
      cur_status = status::fail_unknown;
    }
  }

  if (gsl_status != GSL_SUCCESS)
  {
    cur_status = status::fail_tolerance_not_met;
  }

  const bool limits_enforced = Enforce_Xi_Limits(C.sc_xi(k,j,i));

  if ((cur_status != status::success) ||
      limits_enforced)
  {
    if (pm1.opt_closure.fallback_brent)
    {
      return gsl_Brent(pm1, C, k, j, i);
    }

    if (pm1.opt_closure.verbose)
    {
      gsl_err_warn();
    }
    cur_status = status::fail_tolerance_not_met;
  }

  return cur_status;
}

status custom_NB(
    M1 &pm1,
    ClosureMetaVector &C,
    const int k, const int j, const int i)
{
  using namespace EddingtonFactors;

  const Real tol = pm1.opt_closure.eps_tol;
  const int max_iter = pm1.opt_closure.iter_max;

  Real &xi = C.sc_xi(k,j,i);
  Real Z, dZ;
  int iter = 0;
  bool converged = false;

  // Newton-Raphson phase
  while (iter < max_iter)
  {
    ZdZ_xi(Z, dZ, xi, pm1, C, k, j, i);

    if (std::abs(dZ) < 1e-14)
      break;

    Real dx = Z / dZ;
    xi -= dx;

    if (std::abs(dx) < tol)
    {
      converged = true;
      break;
    }

    if (Enforce_Xi_Limits(xi))
      break;

    ++iter;
  }

  // Bisection fallback (always with XI_MIN/XI_MAX)
  if (!converged)
  {
    Real a = XI_MIN;
    Real b = XI_MAX;

    Real fa = Z_xi(a, pm1, C, k, j, i);
    Real fb = Z_xi(b, pm1, C, k, j, i);

    if (fa * fb > 0.0)
    {
      // No sign change; cannot bracket
      return status::fail_bracket;
    }

    Real lo = a, hi = b;
    for (int b_iter = 0; b_iter < max_iter; ++b_iter)
    {
      Real mid = 0.5 * (lo + hi);
      Real fmid = Z_xi(mid, pm1, C, k, j, i);

      if (std::abs(fmid) < tol || (hi - lo) < tol)
      {
        xi = mid;
        converged = true;
        break;
      }

      if (fa * fmid < 0.0)
      {
        hi = mid;
        fb = fmid;
      }
      else
      {
        lo = mid;
        fa = fmid;
      }
    }

    if (!converged)
    {
      if (pm1.opt_closure.verbose)
      {
        #pragma omp critical
        std::cerr << "gsl_Newton bisection fallback max_iter exceeded\n";
      }
      return status::fail_tolerance_not_met;
    }
  }

  if (Enforce_Xi_Limits(xi))
  {
    // Solution outside [0,1]
    return status::fail_value;
  }

  return status::success;
}

status custom_NAB(
    M1 &pm1,
    ClosureMetaVector &C,
    const int k, const int j, const int i)
{
  using namespace EddingtonFactors;

  Real &xi = C.sc_xi(k,j,i);
  const Real tol = pm1.opt_closure.eps_tol;
  const int max_iter = pm1.opt_closure.iter_max;

  // --- Newton-Raphson Phase ---
  int iter = 0;
  bool newton_success = false;

  while (iter < max_iter)
  {
    Real Z, dZ;
    ZdZ_xi(Z, dZ, xi, pm1, C, k, j, i);

    if (std::abs(dZ) < 1e-14)
      break;

    Real dx = Z / dZ;
    xi -= dx;

    if (std::abs(dx) < tol)
    {
      newton_success = true;
      break;
    }

    if (Enforce_Xi_Limits(xi))
      break;

    ++iter;
  }

  if (newton_success)
    return status::success;

  // --- Anderson-Björck False Position Fallback ---
  Real a = XI_MIN, b = XI_MAX;
  Real fa = Z_xi(a, pm1, C, k, j, i);
  Real fb = Z_xi(b, pm1, C, k, j, i);

  if (fa * fb > 0.0)
  {
    // No sign change; cannot bracket
    return status::fail_bracket;
  }

  Real xi_a = a, xi_b = b;
  Real f_a = fa, f_b = fb;

  int fallback_iter = 0;
  while (fallback_iter < max_iter)
  {
    Real xi_r = xi_b - f_b * (xi_b - xi_a) / (f_b - f_a);
    Real f_r = Z_xi(xi_r, pm1, C, k, j, i);

    if (std::abs(f_r) < tol || std::abs(xi_b - xi_a) < tol)
    {
      xi = xi_r;
      return status::success;
    }

    if (f_a * f_r < 0.0)
    {
      xi_b = xi_r;
      f_b = f_r;
      f_a *= 0.5; // Anderson-Björck damping
    }
    else
    {
      xi_a = xi_r;
      f_a = f_r;
      f_b *= 0.5;
    }

    ++fallback_iter;
  }

  if (pm1.opt_closure.verbose)
  {
    #pragma omp critical
    std::cerr << "custom_Newton Anderson-Bjorck: max_iter exceeded\n";
  }

  return status::fail_tolerance_not_met;
}

status custom_ONAB(
    M1 &pm1,
    ClosureMetaVector &C,
    const int k, const int j, const int i)
{
  using namespace EddingtonFactors;

  Real &xi = C.sc_xi(k, j, i);
  const Real tol = pm1.opt_closure.eps_tol;
  const int max_iter = pm1.opt_closure.iter_max;

  int iter = 0;
  bool converged = false;

  while (iter < max_iter)
  {
    Real f_x, df_x;
    ZdZ_xi(f_x, df_x, xi, pm1, C, k, j, i);  // f(xi), f'(xi)

    if (std::abs(df_x) < 1e-14)
      break;

    Real y = xi - f_x / df_x;
    Real f_y = Z_xi(y, pm1, C, k, j, i);

    Real den_diff = f_x - 2.0 * f_y;

    if (std::abs(den_diff) < 1e-14)
      break;

    Real dx = (f_y / df_x) * (f_x / den_diff);
    xi = y - dx;

    if (Enforce_Xi_Limits(xi))
      break;

    if (std::abs(dx) < tol)
    {
      converged = true;
      break;
    }

    ++iter;
  }

  if (converged)
    return status::success;

  // If Ostrowski did not converge, fallback
  return custom_NAB(pm1, C, k, j, i);
}

// ============================================================================
} // namespace M1::Closures::solvers
// ============================================================================

// ============================================================================
} // namespace M1::Closures
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

    // BD: TODO - fix limits on loops
    M1_FLOOP3(k,j,i)
    if (pm1->MaskGet(k,j,i))
    {
      // NonFiniteToZero(*this, C, k, j, i);
      if (pm1->opt_solver.equilibrium_use_thick &&
          pm1->IsEquilibrium(k,j,i))
      {
        // set directly, only 1 rep of closures
        Closures::EddingtonFactors::ThickLimit(
          pm1->lab_aux.sc_xi(ix_g,ix_s)(k,j,i),
          pm1->lab_aux.sc_chi(ix_g,ix_s)(k,j,i)
        );

        // M1::opt_closure_variety opt_cl_variety = pm1->opt_closure.variety;
        // pm1->opt_closure.variety = M1::opt_closure_variety::thick;

        // C.Closure(k,j,i);

        // pm1->opt_closure.variety = opt_cl_variety;
      }
      else
      {
        C.Closure(k,j,i);
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
