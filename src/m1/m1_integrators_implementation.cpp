// C++ standard headers
// Athena++ headers
#include "m1.hpp"
#include "m1_calc_update.hpp"
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

void Advect_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,         // current (target) step
  const Update::StateMetaVector & P,   // previous step data
  const Update::StateMetaVector & I,   // inhomogeneity
  Update::SourceMetaVector & S,        // treated 0, set as S <- P + dt * I
  const int k, const int j, const int i)
{
  using namespace Update;

  // C := P
  Copy_E_F_d(C, P, k, j, i);

  // C := P + dt * (I + S)
  InPlaceScalarMulAdd_E_F_d(dt, C, I, k, j, i);
  EnforcePhysical_E_F_d(pm1, C, k, j, i);

  Copy_E_F_d(S, C, k, j, i);
}

void Advect_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,         // current (target) step
  const Update::StateMetaVector & P,   // previous step data
  const Update::StateMetaVector & I,   // inhomogeneity
  Update::SourceMetaVector & S,        // treated 0, set as S <- P + dt * I
  const int k, const int j, const int i)
{
  using namespace Update;

  // C := P
  Copy_nG(C, P, k, j, i);

  // C = P + dt * (I + S)
  InPlaceScalarMulAdd_nG(dt, C, I, k, j, i);
  EnforcePhysical_nG(pm1, C, k, j, i);

  Copy_nG(S, C, k, j, i);
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

  // Explicit step, no sources; applies floors internally
  Advect_E_F_d(pm1, dt, C, P, I, S, k, j, i);

  // Compute closure & construct fiducial frame:
  CL_C.Closure(k, j, i);

  // Evolve fiducial frame; prepare (J, H^alpha):
  // We write:
  // sc_J   = J_0
  // st_H^a = H_n n^a + H_v v^a + H_F F^a
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

  // populate spatial projection of fluid frame rad. flux.
  AT_N_vec & sp_H_d_ = pm1.scratch.sp_vec_A_;
  for (int a=0; a<N; ++a)
  {
    sp_H_d_(a,i) = H_v * sp_v_d(a,k,j,i) + H_F * C.sp_F_d(a,k,j,i);
  }
  // --------------------------------------------------------------------------


  // Evolve fluid frame quantities
  const Real sqrt_det_g__ = pm1.geom.sc_sqrt_det_g(k,j,i);
  const Real kap_as = C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i);

  J_0 = (J_0 * W + dt * sqrt_det_g__ * C.sc_eta(k,j,i)) /
        (W + dt * C.sc_kap_a(k,j,i));

  for (int a=0; a<N; ++a)
  {
    sp_H_d_(a,i) = W * sp_H_d_(a,i) / (W + dt * kap_as);
  }

  // This follows from the orthogonality relation H_a u^a = 0
  H_n = 0.0;
  for (int a=0; a<N; ++a)
  {
    H_n += sp_v_u(a,k,j,i) * sp_H_d_(a,i);
  }

  // Project back assuming thick limit ----------------------------------------
  Closures::EddingtonFactors::ThickLimit(
    CL_C.sc_xi(k,j,i), CL_C.sc_chi(k,j,i)
  );

  // Use thick regime expressions:
  C.sc_E(k,j,i) = (
    ONE_3RD * (4.0 * W2 - 1.0) * J_0 +
    2.0 * W * H_n
  );

  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = (
      4.0 * ONE_3RD * W2 * J_0 * sp_v_d(a,k,j,i) +
      W * (H_n * sp_v_d(a,k,j,i) + sp_H_d_(a,i))
    );
  }

  // Ensure physical state ----------------------------------------------------
  EnforcePhysical_E_F_d(pm1, C, k, j, i);

  // Update sources to contain (C - (P + dt * I)) / dt ------------------------
  // This can be done due to advection above.
  using namespace Update;

  const Real oo_dt = OO(dt);
  InPlaceScalarMul_E_F_d(-oo_dt, S, k, j, i);
  InPlaceScalarMulAdd_E_F_d(oo_dt, S, C, k, j, i);
}

// ============================================================================
} // namespace M1::Integrators::Explicit
// ============================================================================

// ============================================================================
namespace Implicit {
// ============================================================================

// Prepares initial guess. If "accurate-enough" returns true to short-circuit
// implicit solver where this is used
bool StepImplicitPrepareInitialGuess(
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
  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*)
  // With approximate solution of _implicit_ system at O(v)
  PrepareApproximateFirstOrder_E_F_d(pm1, dt, C, P, I, S, CL_C, k, j, i);
  // --------------------------------------------------------------------------

  const bool eql_short_circuit = (
    !pm1.opt_solver.equilibrium_src_E_F_d &&
    pm1.IsEquilibrium(C.ix_s,k,j,i)
  );

  if (eql_short_circuit)
  {
    return true;
  }

  // Some other criteria indicating "accurate-enough" go here
  // ...

  return false;
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

  const bool need_implicit = !StepImplicitPrepareInitialGuess(
    pm1, dt, C, P, I, S, CL_C, k, j, i
  );

  // GSL specific -------------------------------------------------------------
  if (need_implicit)
  {
    const size_t N_SYS = 1 + N;

    // Reuse pre-allocated solver and vector from M1 class
    gsl_vector *U_i = pm1.gsl_U_i;
    gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

    // select function & solver -----------------------------------------------
    AA _J;  // unused in this method

    struct gsl_params par = {pm1, dt, C, P, I, S, _J,
                             i, j, k};
    gsl_multiroot_function mrf = {&Z_E_F_d, N_SYS, &par};
    gsl_multiroot_fsolver *slv = pm1.gsl_hybrids_solver;

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

      revert_thick = pm1.opt_solver.thick_npg;
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

    // Ensure update preserves energy non-negativity
    EnforcePhysical_E_F_d(pm1, C, k, j, i);
  }
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
  using namespace Implicit;
  using namespace Sources;

  const bool need_implicit = !StepImplicitPrepareInitialGuess(
    pm1, dt, C, P, I, S, CL_C, k, j, i
  );

  // GSL specific -------------------------------------------------------------
  if (need_implicit)
  {
    const size_t N_SYS = 1 + N;

    // Reuse pre-allocated solver and vector from M1 class
    gsl_vector *U_i = pm1.gsl_U_i;
    gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

    // select function & solver -----------------------------------------------
    AA J(N_SYS,N_SYS);

    struct gsl_params par = {pm1, dt, C, P, I, S, J,
                             i, j, k};

    gsl_multiroot_function_fdf mrf = {&Z_E_F_d,
                                      &dZ_E_F_d,
                                      &ZdZ_E_F_d,
                                      N_SYS,
                                      &par};
    gsl_multiroot_fdfsolver *slv = pm1.gsl_hybridsj_solver;


    int gsl_status = gsl_multiroot_fdfsolver_set(slv, &mrf, U_i);

    // do we even need to iterate or is the estimate sufficient?
    // gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_a_tol);

    // gsl_status = gsl_multiroot_test_delta(slv->dx, slv->x,
    //                                       pm1.opt_solver.eps_a_tol,
    //                                       pm1.opt_solver.eps_r_tol);

    // solver loop ------------------------------------------------------------
    int iter = 0;


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

    // Ensure update preserves energy non-negativity
    EnforcePhysical_E_F_d(pm1, C, k, j, i);
  }
}

// ============================================================================
} // namespace M1::Integrators::Implicit::gsl
// ============================================================================

// ============================================================================
namespace custom {
// ============================================================================

// 4x4 LU decomposition with partial pivoting + forward/back substitution.
// Solves A * x = b in-place: on return, b contains the solution x.
// A is modified (overwritten with L\U factors); diagonal entries are
// replaced by their reciprocals during factorization so back-substitution
// uses only multiplications (no divisions).
// Returns true on success, false if the matrix is singular.
//
// Singularity criterion: a pivot is treated as zero when
//   |pivot| < eps_mach * max_abs
// where max_abs tracks the largest absolute element seen during
// factorization. This is much more robust than an exact == 0.0 test
// when the Jacobian has O(1e+8) entries alongside O(1e-8) entries.
static inline bool solve_4x4_LU_pp(
  Real (&A)[4][4],
  Real (&b)[4])
{
  constexpr int n = 4;
  constexpr Real eps_mach = std::numeric_limits<Real>::epsilon();
  int piv[n]; // pivot indices

  // Track the largest absolute element seen during factorization for
  // relative singularity threshold
  Real max_abs = 0.0;

  // LU factorization with partial pivoting -----------------------------------
  for (int k = 0; k < n; ++k)
  {
    // Find pivot: row with largest |A[r][k]| for r in [k, n)
    int p = k;
    Real max_val = std::abs(A[k][k]);
    for (int r = k + 1; r < n; ++r)
    {
      const Real val = std::abs(A[r][k]);
      if (val > max_val)
      {
        max_val = val;
        p = r;
      }
    }

    piv[k] = p;

    // Update running max of all absolute values seen
    if (max_val > max_abs) max_abs = max_val;

    // Singular check: pivot is effectively zero relative to matrix scale
    if (max_val < eps_mach * max_abs)
    {
      return false;
    }

    // Swap rows k and p (in both A and b)
    if (p != k)
    {
      for (int c = 0; c < n; ++c)
      {
        std::swap(A[k][c], A[p][c]);
      }
      std::swap(b[k], b[p]);
    }

    // Store reciprocal pivot on the diagonal - eliminates 4 divisions
    // from the back-substitution phase
    const Real inv_pivot = OO(A[k][k]);
    A[k][k] = inv_pivot;

    // Eliminate below pivot
    for (int r = k + 1; r < n; ++r)
    {
      const Real factor = A[r][k] * inv_pivot;
      A[r][k] = factor;  // store L factor in lower triangle
      for (int c = k + 1; c < n; ++c)
      {
        A[r][c] -= factor * A[k][c];
      }
      b[r] -= factor * b[k];
    }
  }

  // Back substitution (U * x = b') ------------------------------------------
  // Diagonal entries already hold 1/U[k][k], so use multiply instead of divide
  for (int k = n - 1; k >= 0; --k)
  {
    for (int c = k + 1; c < n; ++c)
    {
      b[k] -= A[k][c] * b[c];
    }
    b[k] *= A[k][k];  // A[k][k] holds reciprocal pivot
  }

  return true;
}

// Hand-written 4x4 Newton solver for the M1 implicit system (E, F_x, F_y, F_z).
//
// This is a drop-in replacement for StepImplicitHybridsJ that eliminates:
// - Per-cell heap allocation (AA J -> stack Real[4][4])
// - GSL function pointer dispatch overhead
// - GSL data marshalling between gsl_vector/gsl_matrix and native types
// - Redundant dot product / Lorentz factor computations via DotProductCache
//
// Convergence criterion matches gsl_multiroot_test_delta:
//   |dU[a]| < eps_a + eps_r * |U[a]|  for all a in {E, F_x, F_y, F_z}
//
// Fallback logic is identical to StepImplicitHybridsJ:
//   1. On max iterations / stagnation -> revert to thick-limit closure
//   2. On thick-limit failure -> fatal error with StatePrintPoint
void StepImplicitCustomN(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i)
{
  using namespace Implicit;
  using namespace Sources;

  // Prepare initial guess (O(v) approximate solution) and check if
  // equilibrium short-circuit applies
  const bool need_implicit = !StepImplicitPrepareInitialGuess(
    pm1, dt, C, P, I, S, CL_C, k, j, i
  );

  if (need_implicit)
  {
    constexpr int N_SYS = 4;  // = 1 + N (energy + 3 spatial flux components)

    const Real eps_a = pm1.opt_solver.eps_a_tol;
    const Real eps_r = pm1.opt_solver.eps_r_tol;
    const int iter_max = pm1.opt_solver.iter_max;

    // Hoist iteration-invariant constants outside the Newton loop:
    //   WE    = P.sc_E  + dt * I.sc_E    (scalar)
    //   WF_d  = P.sp_F_d + dt * I.sp_F_d (3-vector)
    // These are the "previous + inhomogeneity" terms that don't change
    // between Newton iterations (only C changes).
    const Real WE = P.sc_E(k,j,i) + dt * I.sc_E(k,j,i);
    Real WF_d[N];
    for (int a = 0; a < N; ++a)
    {
      WF_d[a] = P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(a,k,j,i);
    }

    // Source values from the last Newton iteration - needed by downstream
    // code (Limiter::Apply, CheckPhysicalFallback, GR coupling) which
    // reads S.sc_E / S.sp_F_d after this solver returns.
    Real S_E_final = 0.0;
    Real S_F_d_final[N] = {0.0, 0.0, 0.0};

    // Newton iteration -------------------------------------------------------
    int iter = 0;
    bool converged = false;
    bool solver_failed = false;

    do
    {
      iter++;

      // Build DotProductCache once per iteration - shared between source
      // and Jacobian evaluations, eliminating ~27 redundant loads and
      // ~31 redundant FLOPs per iteration.
      auto cache = Assemble::Frames::make_cache(
        pm1, C.sc_E, C.sp_F_d, k, j, i
      );

      // Fused evaluation: sources + residual + Z-Jacobian in a single pass.
      // This replaces the old 3-call sequence:
      //   1. PrepareMatterSource_E_F_d (sources)
      //   2. System::Z_E_F_d (residual)
      //   3. ZJacobian_sc_E_sp_F_d_raw (Jacobian)
      // All intermediates (d_th/d_tk, opacities, alpha, expansion coefficients,
      // F^a raised with metric) are computed once and shared.
      Real Z_vec[N_SYS];
      Real ZJ[N_SYS][N_SYS];

      switch (pm1.opt_closure.variety)
      {
        default:
        {
          Assemble::Frames::sources_and_ZJacobian_sc_E_sp_F_d(
            pm1,
            S_E_final, S_F_d_final,  // output: source terms
            Z_vec,                     // output: residual
            ZJ,                        // output: Z-Jacobian
            dt,
            WE, WF_d,                 // pre-hoisted iteration-invariant terms
            C.sc_chi, C.sc_E, C.sp_F_d,
            C.sc_eta, C.sc_kap_a, C.sc_kap_s,
            cache,
            k, j, i
          );
        }
      }

      // Solve ZJ * dU = Z via 4x4 LU with partial pivoting.
      // On return, Z_vec contains the Newton step dU.
      const bool lu_ok = solve_4x4_LU_pp(ZJ, Z_vec);

      if (!lu_ok)
      {
        // Singular Jacobian - cannot proceed
        solver_failed = true;
        break;
      }

      // Apply Newton update: U <- U - dU
      C.sc_E(k,j,i) -= Z_vec[0];
      for (int a = 0; a < N; ++a)
      {
        C.sp_F_d(a,k,j,i) -= Z_vec[1 + a];
      }

      // Enforce physicality (non-negative energy, causality)
      EnforcePhysical_E_F_d(pm1, C, k, j, i);

      // Recompute closure with updated state
      CL_C.Closure(k, j, i);

      // Convergence test (matches gsl_multiroot_test_delta semantics):
      //   |dU[a]| < eps_a + eps_r * |U[a]|  for all a
      converged = true;
      {
        // Check energy component
        const Real U_0 = C.sc_E(k,j,i);
        if (std::abs(Z_vec[0]) >= eps_a + eps_r * std::abs(U_0))
        {
          converged = false;
        }

        // Check flux components
        for (int a = 0; a < N; ++a)
        {
          const Real U_a = C.sp_F_d(a,k,j,i);
          if (std::abs(Z_vec[1 + a]) >= eps_a + eps_r * std::abs(U_a))
          {
            converged = false;
          }
        }
      }

    }
    while (!converged && !solver_failed && iter < iter_max);

    // Write source values from the last Newton iteration to S so that
    // downstream code (Limiter::Apply, CheckPhysicalFallback, GR-evolution
    // coupling) can read them. This is required: those routines access
    // S.sc_E and S.sp_F_d after this solver returns.
    S.sc_E(k,j,i) = S_E_final;
    for (int a = 0; a < N; ++a)
    {
      S.sp_F_d(a,k,j,i) = S_F_d_final[a];
    }

    // Failure handling (mirrors StepImplicitHybridsJ) ------------------------
    bool revert_thick = false;

    if (!converged && !solver_failed && iter >= iter_max)
    {
      // Max iterations reached without convergence
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        std::cout << "Warning: StepImplicitCustomN: ";
        std::cout << "MAXITER (iter=" << iter << ")\n";
        std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
      }

      revert_thick = pm1.opt_solver.thick_tol;
    }
    else if (solver_failed)
    {
      // Singular Jacobian or other solver breakdown
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        std::cout << "Warning: StepImplicitCustomN: ";
        std::cout << "solver failure (singular Jacobian or stagnation)";
        std::cout << ": iter " << iter << " \n";
        std::cout << "sc_chi : " << C.sc_chi(k,j,i) << " ";
        std::printf("|.-1/3|=%.3g ", std::abs(C.sc_chi(k,j,i) - ONE_3RD));
        std::cout << "@ (k, j, i): " << k << ", " << j << ", " << i << "\n";
      }

      if (pm1.opt_solver.verbose)
      if (pm1.opt_closure.variety == M1::opt_closure_variety::thick)
      {
        std::ostringstream msg;
        msg << "StepImplicitCustomN failure: ";
        msg << "singular " << iter;

        pm1.StatePrintPoint(msg.str(), C.ix_g, C.ix_s, k, j, i, false);
      }

      // retry thick
      revert_thick = pm1.opt_solver.thick_npg;
    }

    if (revert_thick &&
        (pm1.opt_closure.variety != M1::opt_closure_variety::thick))
    {
      M1::opt_closure_variety opt_cl = pm1.opt_closure.variety;
      pm1.opt_closure.variety = M1::opt_closure_variety::thick;

      C.Fallback(k, j, i);

      StepImplicitCustomN(
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
    else if (!converged && !revert_thick && !solver_failed)
    {
      // Non-convergence without revert option: fatal error
      std::ostringstream msg;
      msg << "StepImplicitCustomN failure: ";
      msg << "no convergence " << iter;

      pm1.StatePrintPoint(msg.str(), C.ix_g, C.ix_s, k, j, i, true);
    }

    // Note: No final EnforcePhysical_E_F_d needed here. The last Newton
    // iteration already called EnforcePhysical_E_F_d (line above closure
    // recomputation), and the revert_thick path handles its own enforcement
    // via the recursive StepImplicitCustomN call.
  }
}

// ============================================================================
} // namespace M1::Integrators::Implicit::custom
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
  using namespace Explicit;

  // sources either always on, or, if we are at equilibrium they are off
  const bool non_zero_src = (pm1.opt_solver.equilibrium_src_nG)
    ? true
    : !pm1.IsEquilibrium(C.ix_s, k, j, i);

  AT_C_sca & sc_alpha = pm1.geom.sc_alpha;
  AT_C_sca & sc_sqrt_det_g = pm1.geom.sc_sqrt_det_g;

  const Real sc_G__ = Assemble::Frames::ToFiducial(
    pm1,
    C.sc_J, C.st_H_u,
    C.sc_chi, C.sc_E, C.sp_F_d,
    k, j, i
  );

  // const Real num = (
  //   P.sc_nG(k,j,i) + dt * (
  //     I.sc_nG(k,j,i) +
  //     non_zero_src * sc_alpha(k,j,i) * (
  //       sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i)
  //     )
  //   )
  // );

  // store advected state (P + dt * I) in S
  Advect_nG(pm1, dt, C, P, I, S, k, j, i);

  const Real num = (
    S.sc_nG(k,j,i) + dt * (
      non_zero_src * sc_alpha(k,j,i) * (
        sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i)
      )
    )
  );

  const Real den = (
    1.0 + dt * non_zero_src * sc_alpha(k,j,i) * (
      C.sc_kap_a_0(k,j,i) / sc_G__
    )
  );

  C.sc_nG(k,j,i) = num / den;

  // Ensure update preserves non-negativity
  EnforcePhysical_nG(pm1, C, k, j, i);

  // Derived quantities & source retention
  C.sc_n(k,j,i) = C.sc_nG(k,j,i) / sc_G__;

  // S.sc_nG(k,j,i) = non_zero_src * sc_alpha(k,j,i) * (
  //   sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i) -
  //   C.sc_kap_a_0(k,j,i) * C.sc_n(k,j,i)
  // );

  // Update sources to contain (C - (P + dt * I)) / dt ------------------------
  // This can be done due to advection above.
  if (non_zero_src)
  {
    const Real oo_dt = OO(dt);
    InPlaceScalarMul_nG(-oo_dt, S, k, j, i);
    InPlaceScalarMulAdd_nG(oo_dt, S, C, k, j, i);
  }
  else
  {
    S.sc_nG(k,j,i) = 0;
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