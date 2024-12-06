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

// ============================================================================
} // namespace M1::Closures::EddingtonFactors
// ============================================================================

void ClosureMetaVector::Closure(const int k, const int j, const int i)
{
  // solve for xi root if required: -------------------------------------------
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
      // BD: TODO - add me
      assert(false);
    }
    default:
    {
      assert(false);
    }
  }

  // deal with failures / reversion -------------------------------------------
  if (ss != status_sol::success)
  {
    solvers::Fallback_Xi_Chi_Limits(pm1, *this, k, j, i);
    return;
  }

  // compute Eddington factor -------------------------------------------------
  using namespace EddingtonFactors;
  typedef M1::opt_closure_variety ocv;

  Real &xi__  = sc_xi(k,j,i);
  Real &chi__ = sc_chi(k,j,i);

  switch (pm1.opt_closure.variety)
  {
    case (ocv::thin):
    {
      ThinLimit(xi__, chi__);
      break;
    }
    case (ocv::thick):
    {
      ThickLimit(xi__, chi__);
      break;
    }
    case (ocv::Minerbo):
    {
      Minerbo(xi__, chi__);
      break;
    }
    case (ocv::Kershaw):
    {
      Kershaw(xi__, chi__);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

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
    pm1.lab_aux.sc_chi( ix_g,ix_s),
    pm1.lab_aux.sc_xi(  ix_g,ix_s),
    // Lagrangian frame
    pm1.rad.sc_J(  ix_g,ix_s),
    pm1.rad.st_H_u(ix_g,ix_s),
  };

  return C;
}

// ============================================================================
namespace solvers {
// ============================================================================

// functional to find root of
Real Z_xi(
  const Real xi,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i)
{
  // assemble P_dd
  C.sc_xi( k,j,i) = xi;

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
  const Real Z_ = (
    st_H_norm_2__ - SQR(xi * J_0)
  );
  // N.B. could perform a further rescaling by an "ad-hoc" factor of
  // SQR(std::max(C.sc_E(k,j,i), pm1.opt.eps_E));

  return Z_;
}

bool Enforce_Xi_Limits(Real & xi)
{
  if ((xi < XI_MIN) || (xi > XI_MAX))
  {
    xi = std::max(std::min(xi, XI_MAX), XI_MIN);
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

  switch (gsl_status)
  {
    case (GSL_EINVAL):  // bracketing failed
    {
      return status::fail_bracket;
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

        /*
        if (gsl_status != GSL_SUCCESS)
        {
          break;
        }
        else if (gsl_status)
        {
          pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);

          gsl_err_kill(gsl_status);
        }
        */

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

      C.sc_xi(k,j,i) = pm1.gsl_brent_solver->root;

      if (gsl_status != GSL_SUCCESS)
      {
        if (pm1.opt_closure.verbose)
        {
          gsl_err_warn();
        }
        return status::fail_tolerance_not_met;
      }

      break;
    }
    default:
    {
      // unknown status code / generic failure
      gsl_err_kill(gsl_status);
    }
  }

  return status::success;
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
      C.Closure(k,j,i);
    }
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//