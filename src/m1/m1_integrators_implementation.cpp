// C++ standard headers
// Athena++ headers
#include "m1.hpp"
#include "m1_integrators.hpp"
#include "m1_set_equilibrium.hpp"
#include "m1_sources.hpp"
#include "m1_utils.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>
#include <iomanip>
#include <limits>


// ============================================================================
namespace M1::Integrators {
// ============================================================================

// ============================================================================
namespace Explicit {
// ============================================================================

void StepExplicit_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,         // current (target) step
  const Update::StateMetaVector & P,   // previous step data
  const Update::StateMetaVector & I,   // inhomogeneity
  const Update::SourceMetaVector & S,  // carries matter source contribution
  const int k, const int j, const int i)
{
  using namespace Update;

  // C := P
  Copy_E_F_d(C, P, k, j, i);

  // C := P + dt * (I + S)
  InPlaceScalarMulAdd_E_F_d(dt, C, I, k, j, i);
  InPlaceScalarMulAdd_E_F_d(dt, C, S, k, j, i);

  EnforcePhysical_E_F_d(pm1, C, k, j, i);
}

void StepExplicit_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,         // current (target) step
  const Update::StateMetaVector & P,   // previous step data
  const Update::StateMetaVector & I,   // inhomogeneity
  const Update::SourceMetaVector & S,  // carries matter source contribution
  const int k, const int j, const int i)
{
  using namespace Update;

  // C := P
  Copy_nG(C, P, k, j, i);

  // C = P + dt * (I + S)
  InPlaceScalarMulAdd_nG(dt, C, I, k, j, i);
  InPlaceScalarMulAdd_nG(dt, C, S, k, j, i);

  EnforcePhysical_nG(pm1, C, k, j, i);
}

void PrepareApproximateFirstOrder_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i)
{
  using namespace Sources;

  // ------------------------------------------------------------------------
  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*) _without_ matter sources
  SetMatterSourceZero(S, k, j, i);

  // Explicit step, no sources; applies floors internally
  StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);

  // Compute closure & construct fiducial frame:
  CL_C.Closure(k, j, i);

  // Evolve fiducial frame; prepare (J, H^alpha):
  // We write:
  // sc_J = J_0
  // st_H = H_n n^alpha + H_v v^alpha + H_F F^alpha
  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const AT_N_vec & sp_v_d = pm1.fidu.sp_v_d;
  const AT_N_vec & sp_v_u = pm1.fidu.sp_v_u;

  const AT_C_sca & sc_alpha  = pm1.geom.sc_alpha;
  const AT_N_vec & sp_beta_u = pm1.geom.sp_beta_u;

  Real J_0, H_n, H_v, H_F;

  Assemble::Frames::ToFiducialExpansionCoefficients(
    pm1,
    J_0, H_n, H_v, H_F,
    C.sc_chi, C.sc_E, C.sp_F_d,
    k, j, i
  );

  // J_0 = std::max(
  //   (
  //     (J_0 * W + dt * C.sc_eta(k,j,i)) /
  //     (W + dt * C.sc_kap_a(k,j,i))
  //   ),
  //   pm1.opt.fl_J
  // );

  J_0 = (J_0 * W + dt * C.sc_eta(k,j,i)) /
        (W + dt * C.sc_kap_a(k,j,i));

  const Real fac_ev_H = W / (
    W + dt * (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i))
  );

  H_n *= fac_ev_H;
  H_v *= fac_ev_H;
  H_F *= fac_ev_H;

  // Transform back, assuming thick closure:

  // Contract on ST idx:
  // v_a H^a = v_i beta^i H^0 + v_i H^i
  //         = v^a H_a
  //         = v^i H_i
  // where:
  // H^a = H_n n^a + H_v v^a + H_F F^a
  // H_i = H_n n_i + H_v v_i + H_F F_i
  //     =    0    + H_v v_i + H_F F_i

  Real dotHv (0);
  for (int a=0; a<N; ++a)
  {
    dotHv += sp_v_u(a,k,j,i) * (
      H_v * sp_v_d(a,k,j,i) +
      H_F * C.sp_F_d(a,k,j,i)
    );
  }

  const Real dotHn = -H_n;

  C.sc_E(k,j,i) = (
    4.0 * ONE_3RD * W2 * J_0 +
    2.0 * W * dotHv -
    ONE_3RD * J_0
  );

  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = (
      4.0 * ONE_3RD * W2 * J_0 * sp_v_d(a,k,j,i) +
      W * (
        H_v * sp_v_d(a,k,j,i) +
        H_F * C.sp_F_d(a,k,j,i)
      ) -
      W * dotHn * sp_v_d(a,k,j,i)
    );
  }

  Closures::EddingtonFactors::ThickLimit(
    CL_C.sc_xi(k,j,i), CL_C.sc_chi(k,j,i)
  );

  // Ensure physical state
  EnforcePhysical_E_F_d(pm1, C, k, j, i);
}

// ============================================================================
} // namespace M1::Integrators::Explicit
// ============================================================================

// ============================================================================
namespace Implicit {
// ============================================================================

void StepImplicitPrepareInitialGuess(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i)
{
  using namespace Explicit;
  using namespace Sources;

  // Retain values for potential restarts
  C.FallbackStore(k, j, i);

  // prepare initial guess for iteration --------------------------------------
  // See \S3.2.4 of [1]

  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*) _without_ matter sources
  SetMatterSourceZero(S, k, j, i);

  // try a nearest-neighbour guess?
  if (pm1.opt_solver.use_Neighbor &&
      (i > pm1.mbi.il))
  {
    C.sc_E(k,j,i) = C.sc_E(k,j,i-1);
    for (int a=0; a<N; ++a)
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i-1);

    CL_C.Closure(k, j, i);
  }
  // Previous state as guess
  // {
  //   C.sc_E(k,j,i) = P.sc_E(k,j,i);
  //   for (int a=0; a<N; ++a)
  //     C.sp_F_d(a,k,j,i) = P.sp_F_d(a,k,j,i);

  //   CL_C.Closure(k, j, i);
  // }
  else
  {
    // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*)
    // With approximate solution of _implicit_ system at O(v)
    PrepareApproximateFirstOrder_E_F_d(pm1, dt, C, P, I, S, CL_C, k, j, i);
  }
  // --------------------------------------------------------------------------
}

// ============================================================================
namespace System {
// ============================================================================

inline void Z_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  const Update::SourceMetaVector & S, // carries matter source contribution
  const int k, const int j, const int i)
{
  const Real WE = P.sc_E(k,j,i) + dt * I.sc_E(k,j,i);
  const Real ZE = C.sc_E(k,j,i) - dt * S.sc_E(k,j,i) - WE;
  C.Z_E[0] = ZE;

  for (int a=0; a<N; ++a)
  {
    const Real WF_d = P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(a,k,j,i);
    const Real ZF_d = C.sp_F_d(a,k,j,i) - dt * S.sp_F_d(a,k,j,i) - WF_d;
    C.Z_F_d[a] = ZF_d;
  }
}

// ============================================================================
} // namespace M1::Integrators::Implicit::System
// ============================================================================

// ============================================================================
namespace gsl {
// ============================================================================

struct gsl_params
{
  M1 & pm1;
  Real dt;
  Update::StateMetaVector & C;
  const Update::StateMetaVector & P;
  const Update::StateMetaVector & I;
  Update::SourceMetaVector & S;

  // For Jacobian-based methods
  AA & J;

  const int i;
  const int j;
  const int k;
};

// convenience maps -----------------------------------------------------------

// populate StateMetaVector with gsl_vector
inline void gsl_V2T_E_F_d(Update::StateMetaVector & T,
                          const gsl_vector *U,
                          const int k, const int j, const int i)
{
  T.sc_E(k,j,i) = U->data[0];
  for (int a=0; a<N; ++a)
  {
    T.sp_F_d(a,k,j,i) = U->data[1+a];
  }
}

// populate gsl_vector with StateMetaVector
inline void gsl_T2V_E_F_d(gsl_vector *U,
                          const Update::StateMetaVector & T,
                          const int k, const int j, const int i)
{
  U->data[0] = T.sc_E(k,j,i);
  for (int a=0; a<N; ++a)
  {
    U->data[1+a] = T.sp_F_d(a,k,j,i);
  }
}

// populate gsl_vector with StateMetaVector.Z array
// (The latter is the zero we seek)
inline void gsl_TZ2V_E_F_d(gsl_vector *U,
                           const Update::StateMetaVector & T)
{
  U->data[0] = T.Z_E[0];
  for (int a=0; a<N; ++a)
  {
    U->data[1+a] = T.Z_F_d[a];
  }
}

// wrapper for system to solve
int Z_E_F_d(const gsl_vector *U, void * par_, gsl_vector *Z)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  M1 & pm1 = par->pm1;
  const Real dt = par->dt;
  Update::StateMetaVector & C = par->C;
  const Update::StateMetaVector & P = par->P;
  const Update::StateMetaVector & I = par->I;

  Update::SourceMetaVector & S = par->S;

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  gsl_V2T_E_F_d(C, U, k, j, i);                   // U->C

  Sources::PrepareMatterSource_E_F_d(pm1, C, S, k, j, i);
  System::Z_E_F_d(pm1, dt, C, P, I, S, k, j, i);  // updates C.Z_E, C.Z_F_d

  gsl_TZ2V_E_F_d(Z, C);                           // C.Z -> Z

  return GSL_SUCCESS;
}

int dZ_E_F_d(const gsl_vector *U, void * par_, gsl_matrix *J_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  M1 & pm1 = par->pm1;
  const Real dt = par->dt;
  Update::StateMetaVector & C = par->C;
  const Update::StateMetaVector & P = par->P;
  const Update::StateMetaVector & I = par->I;

  Update::SourceMetaVector & S = par->S;

  AA & J = par->J;

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  // Prepare Jacobian for sources
  // BD: TODO - refactor switch for other closures...
  switch (pm1.opt_closure.variety)
  {
    default:
    {
      ::M1::Sources::Minerbo::PrepareMatterSourceJacobian_E_F_d(
        pm1, dt, J, C, k, j, i
      );
    }
  }

  // Need Jacobian for Z; not sources:
  const int N_SYS = J.GetDim1();
  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    J(a,b) = (a==b) - dt * J(a,b);
  }

  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    gsl_matrix_set(J_, a, b, J(a, b));
  }

  return GSL_SUCCESS;
}

int ZdZ_E_F_d(const gsl_vector *U,
              void * par_,
              gsl_vector *Z,
              gsl_matrix *J)
{
  // BD: TODO - recycle dotFv etc
  Z_E_F_d(U, par_, Z);
  dZ_E_F_d(U, par_, J);
  return GSL_SUCCESS;
}

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (FD approximation for Jacobian)
void StepImplicitHybrids(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i)
{
  using namespace Sources;

  StepImplicitPrepareInitialGuess(pm1, dt, C, P, I, S, CL_C, k, j, i);

  // GSL specific -------------------------------------------------------------
  {
    const size_t N_SYS = 1 + N;

    // values to iterate (seed with initial guess)
    gsl_vector *U_i = gsl_vector_alloc(N_SYS);
    gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

    // select function & solver -----------------------------------------------
    AA J_;  // unused in this method

    struct gsl_params par = {pm1, dt, C, P, I, S, J_, i, j, k};
    gsl_multiroot_function mrf = {&Z_E_F_d, N_SYS, &par};
    gsl_multiroot_fsolver *slv = gsl_multiroot_fsolver_alloc(
      gsl_multiroot_fsolver_hybrids,
      N_SYS);

    int gsl_status = gsl_multiroot_fsolver_set(slv, &mrf, U_i);

    // do we even need to iterate or is the estimate sufficient?
    // gsl_status = gsl_multiroot_test_residual(slv->f,
    //                                          pm1.opt_solver.eps_a_tol);
    // gsl_status = gsl_multiroot_test_delta(slv->dx, slv->x,
    //                                       pm1.opt_solver.eps_a_tol,
    //                                       pm1.opt_solver.eps_r_tol);
    // solver loop ------------------------------------------------------------
    int iter = 0;

    // if (gsl_status!=GSL_SUCCESS)
    do
    {
      iter++;
      gsl_status = gsl_multiroot_fsolver_iterate(slv);

      // break on issue with solver
      if (gsl_status)
      {
        break;
      }

      gsl_V2T_E_F_d(C, slv->x, k, j, i);  // U_i->C

      // Ensure update preserves energy non-negativity
      EnforcePhysical_E_F_d(pm1, C, k, j, i);

      // compute closure with updated state
      CL_C.Closure(k,j,i);

      // Updated iterated vector
      gsl_T2V_E_F_d(slv->x, C, k, j, i);  // C->U_i
      // gsl_status = gsl_multiroot_test_residual(slv->f,
      //                                          pm1.opt_solver.eps_a_tol);
      gsl_status = gsl_multiroot_test_delta(slv->dx, slv->x,
                                            pm1.opt_solver.eps_a_tol,
                                            pm1.opt_solver.eps_r_tol);
    }
    while (gsl_status == GSL_CONTINUE && iter < pm1.opt_solver.iter_max);

    bool revert_thick = false;

    if ((gsl_status == GSL_ETOL)     ||
        (gsl_status == GSL_EMAXITER) ||
        (iter >= pm1.opt_solver.iter_max))
    {
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        std::cout << "Warning: StepImplicitHybrids: ";
        std::cout << "GSL_ETOL || GSL_EMAXITER\n";
        std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
      }

      revert_thick = pm1.opt_solver.thick_tol;
    }
    else if ((gsl_status == GSL_ENOPROG) || (gsl_status == GSL_ENOPROGJ))
    {
      // Iteration stagnated; revert to thick-limit
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        if (pm1.pmy_block->IsPhysicalIndex_cc(k, j, i))
        {
          std::cout << "Warning: StepImplicitHybrids: ";
          std::cout << "GSL_ENOPROG || GSL_ENOPROGJ : iter " << iter << " \n";
          std::cout << "sc_chi : " << C.sc_chi(k,j,i) << " ";
          std::cout << "@ (k, j, i): " << k << ", " << j << ", " << i << "\n";
          std::cout << "opt_closure.variety = " << static_cast<int>(pm1.opt_closure.variety) << "\n";
        }
      }

      if (pm1.opt_solver.verbose)
      if (pm1.opt_closure.variety == M1::opt_closure_variety::thick)
      {
        C.Fallback(k, j, i);

        std::ostringstream msg;
        msg << "StepImplicitHybrids failure: ";
        msg << gsl_status << " " << iter;

        pm1.StatePrintPoint(msg.str(), C.ix_g, C.ix_s, k, j, i, true);
      }

      revert_thick = pm1.opt_solver.thick_tol;
    }
    else if ((gsl_status != GSL_SUCCESS))
    {
      // Failure on thick-limit reversion; print & kill
      std::ostringstream msg;
      msg << "StepImplicitHybrids failure: ";
      msg << gsl_status << " " << iter;

      pm1.StatePrintPoint(msg.str(), C.ix_g, C.ix_s, k, j, i, true);
    }

    if (revert_thick &&
        (pm1.opt_closure.variety != M1::opt_closure_variety::thick))
    {
      M1::opt_closure_variety opt_cl = pm1.opt_closure.variety;
      pm1.opt_closure.variety = M1::opt_closure_variety::thick;

      C.Fallback(k, j, i);

      StepImplicitHybrids(
        pm1,
        dt,
        C,
        P,
        I,
        S,
        CL_C,
        k,
        j,
        i
      );

      pm1.opt_closure.variety = opt_cl;
    }

    // cleanup ----------------------------------------------------------------
    gsl_multiroot_fsolver_free(slv);
    gsl_vector_free(U_i);
  }

  // Ensure update preserves energy non-negativity
  EnforcePhysical_E_F_d(pm1, C, k, j, i);
}

void StepImplicitHybridsJ(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i)
{

  // Extract source mask value for potential short-circuit treatment ----------
  // DEBUG:
  // typedef M1::evolution_strategy::opt_source_treatment ost;
  // if (ost::set_zero ==  pm1.ev_strat.masks.source_treatment(C.ix_g,C.ix_s,k,j,i))
  // {
  //   std::cout << static_cast<int>(pm1.opt_solver.solvers.equilibrium) << "\n";
  //   pm1.StatePrintPoint("Debug SemiImplicit", C.ix_g, C.ix_s, k, j, i, true);
  // }
  // --------------------------------------------------------------------------

  using namespace Implicit;
  using namespace Sources;

  StepImplicitPrepareInitialGuess(pm1, dt, C, P, I, S, CL_C, k, j, i);

  // GSL specific -------------------------------------------------------------
  {
    const size_t N_SYS = 1 + N;

    // values to iterate (seed with initial guess)
    gsl_vector *U_i = gsl_vector_alloc(N_SYS);
    gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

    // select function & solver -----------------------------------------------
    AA J(N_SYS,N_SYS);

    struct gsl_params par = {pm1, dt, C, P, I, S, J, i, j, k};

    gsl_multiroot_function_fdf mrf = {&Z_E_F_d,
                                      &dZ_E_F_d,
                                      &ZdZ_E_F_d,
                                      N_SYS,
                                      &par};
    gsl_multiroot_fdfsolver *slv = gsl_multiroot_fdfsolver_alloc(
      gsl_multiroot_fdfsolver_hybridsj,
      N_SYS);


    int gsl_status = gsl_multiroot_fdfsolver_set(slv, &mrf, U_i);

    // do we even need to iterate or is the estimate sufficient?
    // gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_a_tol);

    // gsl_status = gsl_multiroot_test_delta(slv->dx, slv->x,
    //                                       pm1.opt_solver.eps_a_tol,
    //                                       pm1.opt_solver.eps_r_tol);

    // solver loop ------------------------------------------------------------
    int iter = 0;

    Real xi_avg = CL_C.sc_xi(k,j,i);
    Real xi_min = std::numeric_limits<Real>::infinity();

    // if (gsl_status!=GSL_SUCCESS)
    do
    {
      iter++;
      gsl_status = gsl_multiroot_fdfsolver_iterate(slv);

      // break on issue with solver
      if (gsl_status)
      {
        break;
      }

      gsl_V2T_E_F_d(C, slv->x, k, j, i);  // U_i->C

      // Ensure update preserves energy non-negativity
      EnforcePhysical_E_F_d(pm1, C, k, j, i);
      /*
      if (iter > 10 && (C.sc_E(k,j,i) < 0))
      {
        std::ostringstream msg;
        msg << "Panic: 0 > C.sc_E(k,j,i) = ";
        msg << std::setprecision(3);
        msg << C.sc_E(k,j,i);
        pm1.StatePrintPoint(msg.str(),
                            C.ix_g, C.ix_s, k, j, i, true);
      }

      if (C.sc_E(k,j,i) < 0)
      {
        C.sc_E(k,j,i) = 0;
        for (int a=0; a<N; ++a)
        {
          C.sp_F_d(a,k,j,i) = 0;
        }
      }
      */

      // compute closure with updated state
      CL_C.Closure(k,j,i);

      // Updated iterated vector
      gsl_T2V_E_F_d(slv->x, C, k, j, i);  // C->U_i
      // gsl_status = gsl_multiroot_test_residual(slv->f,
      //                                          pm1.opt_solver.eps_a_tol);

      gsl_status = gsl_multiroot_test_delta(slv->dx, slv->x,
                                            pm1.opt_solver.eps_a_tol,
                                            pm1.opt_solver.eps_r_tol);

      xi_avg += CL_C.sc_xi(k,j,i);
      xi_min = std::min(CL_C.sc_xi(k,j,i), xi_min);
      // if ((xi_min < 1e-6) && pm1.opt_closure.variety != M1::opt_closure_variety::thick)
      //   gsl_status = GSL_ENOPROG;
    }
    while (gsl_status == GSL_CONTINUE && iter < pm1.opt_solver.iter_max);

    bool revert_thick = false;

    if ((gsl_status == GSL_ETOL)     ||
        (gsl_status == GSL_EMAXITER) ||
        (iter >= pm1.opt_solver.iter_max))
    {
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        std::cout << "Warning: StepImplicitHybridsJ: ";
        std::cout << "GSL_ETOL || GSL_EMAXITER\n";
        std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
      }

      revert_thick = pm1.opt_solver.thick_tol;
    }
    else if ((gsl_status == GSL_ENOPROG) || (gsl_status == GSL_ENOPROGJ))
    {
      // Iteration stagnated; revert to thick-limit
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        std::cout << "Warning: StepImplicitHybridsJ: ";
        std::cout << "GSL_ENOPROG || GSL_ENOPROGJ (" << gsl_status;
        std::cout << "): iter " << iter << " \n";
        std::cout << "sc_chi : " << C.sc_chi(k,j,i) << " ";
        std::printf("|.-1/3|=%.3g ", std::abs(C.sc_chi(k,j,i) - ONE_3RD));
        // std::cout << "xi_avg : " << xi_avg / (iter + 1) << " ";
        // std::cout << "xi_min : " << xi_min << " ";
        // std::cout << "xi_min : " << xi_min << " ";
        std::cout << "@ (k, j, i): " << k << ", " << j << ", " << i << "\n";
      }

      if (pm1.opt_solver.verbose)
      if (pm1.opt_closure.variety == M1::opt_closure_variety::thick)
      {
        /*
        if (pm1.ev_strat.masks.source_treatment(k,j,i) ==
            M1::evolution_strategy::opt_source_treatment::set_zero)
        {


          using namespace Update;
          using namespace Sources;
          using namespace Closures;

          using namespace Explicit;

          // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*)
          SetMatterSourceZero(S, k, j, i);
          StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);

          // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
          CL_C.Closure(k, j, i);
        }
        */

        std::ostringstream msg;
        msg << "StepImplicitHybridsJ failure: ";
        msg << gsl_status << " " << iter;

        pm1.StatePrintPoint(msg.str(), C.ix_g, C.ix_s, k, j, i, false);
      }

      // retry thick
      revert_thick = pm1.opt_solver.thick_npg;
    }
    else if ((gsl_status != GSL_SUCCESS))
    {
      // Failure on thick-limit reversion; print & kill
      std::ostringstream msg;
      msg << "StepImplicitHybridsJ failure: ";
      msg << gsl_status << " " << iter;

      pm1.StatePrintPoint(msg.str(), C.ix_g, C.ix_s, k, j, i, true);
    }

    if (revert_thick &&
        (pm1.opt_closure.variety != M1::opt_closure_variety::thick))
    {
      M1::opt_closure_variety opt_cl = pm1.opt_closure.variety;
      pm1.opt_closure.variety = M1::opt_closure_variety::thick;

      C.Fallback(k, j, i);

      StepImplicitHybridsJ(
        pm1,
        dt,
        C,
        P,
        I,
        S,
        CL_C,
        k,
        j,
        i
      );

      pm1.opt_closure.variety = opt_cl;
   }

    // cleanup ----------------------------------------------------------------
    gsl_multiroot_fdfsolver_free(slv);
    gsl_vector_free(U_i);
  }

  // Ensure update preserves energy non-negativity
  EnforcePhysical_E_F_d(pm1, C, k, j, i);
}

// ============================================================================
} // namespace M1::Integrators::Implicit::gsl
// ============================================================================

// ----------------------------------------------------------------------------
// Implicit update strategy for nG component of state vector
void SolveImplicitNeutrinoCurrent(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  const int k, const int j, const int i)
{
  // Extract source mask value for potential short-circuit treatment ----------
  typedef M1::evolution_strategy::opt_source_treatment ost;
  const bool non_zero_src = !(
    ost::set_zero ==  pm1.GetMaskSourceTreatment(0,0,k,j,i)
  );
  // --------------------------------------------------------------------------

  AT_C_sca & sc_alpha = pm1.geom.sc_alpha;
  AT_C_sca & sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g;

  const Real sc_G__ = Assemble::Frames::ToFiducial(
    pm1,
    C.sc_J, C.st_H_u,
    C.sc_chi, C.sc_E, C.sp_F_d,
    k, j, i
  );

  // BD: I.sc_nG contains flux div, but not \alpha \sqrt \eta^0
  const Real src_term = non_zero_src * (
    sc_alpha(k,j,i) * sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i)
  );

  const Real WnGam = P.sc_nG(k,j,i) + dt * (src_term + I.sc_nG(k,j,i));

  C.sc_nG(k,j,i) = WnGam / (
    1.0 + dt * sc_alpha(k,j,i) * C.sc_kap_a_0(k,j,i) / sc_G__
  );

  // Ensure update preserves non-negativity
  EnforcePhysical_nG(pm1, C, k, j, i);

  // Derived quantities & source retention
  C.sc_n(k,j,i) = C.sc_nG(k,j,i) / sc_G__;
  {
    const Real src_term_2 = non_zero_src * (
      sc_alpha(k,j,i) * C.sc_kap_a_0(k,j,i) * C.sc_n(k,j,i)
    );

    S.sc_nG(k,j,i) = src_term - src_term_2;
  }
}

// ============================================================================
} // namespace M1::Integrators::Implicit
// ============================================================================

// ============================================================================
} // namespace M1::Integrators
// ============================================================================


//
// :D
//