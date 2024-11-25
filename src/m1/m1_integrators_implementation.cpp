// C++ standard headers
// Athena++ headers
#include "m1.hpp"
#include "m1_integrators.hpp"
#include "m1_set_equilibrium.hpp"
#include "m1_sources.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>


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
}

void PrepareApproximateFirstOrder_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & V, // state to utilize
  const int k, const int j, const int i)
{
  // construct fiducial frame quantities (tilde of [1])
  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(
    V.sp_F_d,
    pm1.fidu.sp_v_u,
    k, j, i
  );

  // assemble J
  V.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, V.sc_E, pm1.fidu.sp_v_u, V.sp_P_dd,
    k, j, i
  );

  // assemble H_d
  Assemble::sp_H_d__(V.sp_H_d, W, V.sc_J, V.sp_F_d,
                     pm1.fidu.sp_v_d, pm1.fidu.sp_v_u, V.sp_P_dd,
                     k, j, i);

  // propagate fiducial frame quantities (hat of [1])
  V.sc_J(k,j,i) = (
    (V.sc_J(k,j,i) * W + dt * V.sc_eta(k,j,i)) /
    (W + dt * V.sc_kap_a(k,j,i))
  );

  for (int a=0; a<N; ++a)
  {
    V.sp_H_d(a,k,j,i) = W * V.sp_H_d(a,k,j,i) / (
      W + dt * (V.sc_kap_a(k,j,i) + V.sc_kap_s(k,j,i))
    );
  }

  // transform to Eulerian frame assuming _thick_ limit closure
  const Real dotHv = Assemble::sc_dot_dense_sp__(
    V.sp_H_d,
    pm1.fidu.sp_v_u,
    k, j, i
  );

  // assemble initial guess
  V.sc_E(k,j,i) = ONE_3RD * (
    4.0 * W2 - 1.0
  ) * V.sc_J(k,j,i) + 2.0 * W * dotHv;

  for (int a=0; a<N; ++a)
  {
    V.sp_F_d(a,k,j,i) = (
      4.0 * ONE_3RD * W2 * V.sc_J(k,j,i) * pm1.fidu.sp_v_d(a,k,j,i) +
      W * (V.sp_H_d(a,k,j,i) + dotHv * pm1.fidu.sp_v_d(a,k,j,i))
    );
  }

  EnforcePhysical_E_F_d(pm1, V, k, j, i);
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

 // prepare initial guess for iteration --------------------------------------
  // See \S3.2.4 of [1]

  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*) _without_ matter sources
  SetMatterSourceZero(S, k, j, i);

  // try a nearest-neighbour guess?
  if (pm1.opt_solver.use_Neighbor &&
      (i > pm1.mbi.il) && (j > pm1.mbi.jl))
  {
    C.sc_E(k,j,i) = C.sc_E(k,j,i-1);
    for (int a=0; a<N; ++a)
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i-1);

    CL_C.Closure(k, j, i);

    // PrepareApproximateFirstOrder_E_F_d(pm1, dt, C, k, j, i);
    C.FallbackStore(k, j, i);
  }
  else
  {
    // Explicit step, no sources; applies floors internally
    StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);


    // PrepareMatterSource_E_F_d(pm1, P, S, k, j, i);
    // StepExplicit_E_F_d(pm1, dt, C, P, I, S, k, j, i);
    // SetMatterSourceZero(S, k, j, i);

    // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
    CL_C.Closure(k, j, i);

    // Prepare state as approximate solution of _implicit_ system at O(v)
    PrepareApproximateFirstOrder_E_F_d(pm1, dt, C, k, j, i);

    // BD: TODO - use source masking
    // PrepareMatterSource_E_F_d(pm1, C, S, k, j, i);


    // BD: TODO - application of this here causes issues?
    // Compute (pm1 storage) (sp_P_dd, ...) based on implicit appr.
    // CL_C.ClosureThick(k, j, i);

    CL_C.ClosureThick(k, j, i);

    // retain values for potential restarts
    C.FallbackStore(k, j, i);

    // This is done within the system Z functional
    // PrepareMatterSource_E_F_d(pm1, C, S, k, j, i);
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
    gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);

    // solver loop ------------------------------------------------------------
    int iter = 0;

    if (gsl_status!=GSL_SUCCESS)
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
      gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);
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

      revert_thick = false;
    }
    else if ((gsl_status == GSL_ENOPROG) || (gsl_status == GSL_ENOPROGJ))
    {
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

      revert_thick = true;
    }
    else if ((gsl_status != GSL_SUCCESS))
    {
      #pragma omp critical
      {
        std::cout << "StepImplicitHybrids failure: ";
        std::cout << gsl_status << "\n";
        std::cout << iter << std::endl;
      }
      pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
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

  // Deal with neutrinos (nG, n)
  SolveImplicitNeutrinoCurrent(pm1, dt, C, P, I, S, k, j, i);
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
    gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);

    // solver loop ------------------------------------------------------------
    int iter = 0;

    if (gsl_status!=GSL_SUCCESS)
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
      gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);
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

      revert_thick = false;
    }
    else if ((gsl_status == GSL_ENOPROG) || (gsl_status == GSL_ENOPROGJ))
    {
      if (pm1.opt_solver.verbose)
      #pragma omp critical
      {
        std::cout << "Warning: StepImplicitHybridsJ: ";
        std::cout << "GSL_ENOPROG || GSL_ENOPROGJ : iter " << iter << " \n";
        std::cout << "sc_chi : " << C.sc_chi(k,j,i) << " ";
        std::cout << "@ (k, j, i): " << k << ", " << j << ", " << i << "\n";
      }

      revert_thick = true;
    }
    else if ((gsl_status != GSL_SUCCESS))
    {
      #pragma omp critical
      {
        std::cout << "StepImplicitHybridsJ failure: ";
        std::cout << gsl_status << "\n";
        std::cout << iter << std::endl;
      }
      pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
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

  // Deal with neutrinos (nG, n)
  SolveImplicitNeutrinoCurrent(pm1, dt, C, P, I, S, k, j, i);
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
  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  C.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, C.sc_E, pm1.fidu.sp_v_u, C.sp_P_dd,
    k, j, i
  );

  // BD: I.sc_nG contains flux div, but not \alpha \sqrt \eta^0
  const Real src_term = pm1.geom.sc_alpha(k,j,i) *
    pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i);

  const Real WnGam = P.sc_nG(k,j,i) + dt * (src_term + I.sc_nG(k,j,i));

  const Real C_Gam = Assemble::sc_G__(
    W, C.sc_E(k,j,i), C.sc_J(k,j,i), dotFv,
    pm1.opt.fl_E, pm1.opt.fl_J, pm1.opt.eps_E
  );

  C.sc_nG(k,j,i) = WnGam / (
    1.0 + dt * pm1.geom.sc_alpha(k,j,i) * C.sc_kap_a_0(k,j,i) / C_Gam
  );

  C.sc_n(k,j,i) = C.sc_nG(k,j,i) / C_Gam;

  // Retain source information
  {
    const Real src_term_2 = pm1.geom.sc_alpha(k,j,i) *
      C.sc_kap_a_0(k,j,i) * C.sc_n(k,j,i);

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