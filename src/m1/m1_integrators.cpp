// C++ standard headers
// Athena++ headers
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

inline void ExplicitIntegration(
  M1 & pm1_,
  const Real dt,
  Update::StateMetaVector & C,
  const Update::StateMetaVector & P,
  const Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k,
  const int j,
  const int i)
{
  using namespace Update;
  using namespace Sources;
  using namespace Closures;

  using namespace Explicit;

  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*)
  PrepareMatterSource_E_F_d(pm1_, P, S, k, j, i);
  StepExplicit_E_F_d(pm1_, dt, C, P, I, S, k, j, i);

  // Compute (pm1 storage) (sp_P_dd, ...) based on (sc_E*, sp_F_d*)
  CL.Closure(k, j, i);

  // Evolve (sc_nG, ) -> (sc_nG*, ): sources use (sc_E, sp_F_d)
  if (!pm1_.opt_solver.solver_explicit_nG)
  {
    // Evolve (sc_nG, ) -> (sc_nG*, ) with semi-implicit
    // Prepares also (sc_n*, ) and S
    Implicit::SolveImplicitNeutrinoCurrent(pm1_, dt, C, P, I, S, k, j, i);
  }
  else
  {
    PrepareMatterSource_nG(pm1_, P, S, k, j, i);
    StepExplicit_nG(pm1_, dt, C, P, I, S, k, j, i);

    // We now have nG*, it is useful to immediately construct n*
    Prepare_n_from_nG(pm1_, C, k, j, i);
  }
}

inline void ExplicitApproximateSemiImplicitIntegration(
  M1 & pm1_,
  const Real dt,
  Update::StateMetaVector & C,
  const Update::StateMetaVector & P,
  const Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k,
  const int j,
  const int i)
{
  using namespace Update;
  using namespace Sources;
  using namespace Closures;

  using namespace Explicit;

  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*)
  // With approximate solution of _implicit_ system at O(v)
  PrepareApproximateFirstOrder_E_F_d(pm1_, dt, C, P, I, S, CL, k, j, i);

  // Use previous state to construct source
  PrepareMatterSource_E_F_d(pm1_, P, S, k, j, i);

  if (!pm1_.opt_solver.solver_explicit_nG)
  {
    // Evolve (sc_nG, ) -> (sc_nG*, ) with semi-implicit
    // Prepares also (sc_n*, ) and S
    Implicit::SolveImplicitNeutrinoCurrent(pm1_, dt, C, P, I, S, k, j, i);
  }
  else
  {
    PrepareMatterSource_nG(pm1_, P, S, k, j, i);
    StepExplicit_nG(pm1_, dt, C, P, I, S, k, j, i);

    // We now have nG*, it is useful to immediately construct n*
    Prepare_n_from_nG(pm1_, C, k, j, i);
  }
}

inline void SemiImplicitHybridsIntegration(
  M1 & pm1_,
  const Real dt,
  Update::StateMetaVector & C,
  const Update::StateMetaVector & P,
  const Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k,
  const int j,
  const int i)
{
  using namespace Update;
  using namespace Sources;
  using namespace Closures;

  using namespace Explicit;
  using namespace Implicit;
  using namespace Implicit::gsl;

  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*) with semi-implicit
  StepImplicitHybrids(pm1_,
                      dt, C, P, I, S,
                      CL,
                      k, j, i);

  if (!pm1_.opt_solver.solver_explicit_nG)
  {
    // Evolve (sc_nG, ) -> (sc_nG*, ) with semi-implicit
    // Prepares also (sc_n*, ) and S
    SolveImplicitNeutrinoCurrent(pm1_, dt, C, P, I, S, k, j, i);
  }
  else
  {
    PrepareMatterSource_nG(pm1_, P, S, k, j, i);
    StepExplicit_nG(pm1_, dt, C, P, I, S, k, j, i);

    // We now have nG*, it is useful to immediately construct n*
    Prepare_n_from_nG(pm1_, C, k, j, i);
  }
}

inline void SemiImplicitHybridsJIntegration(
  M1 & pm1_,
  const Real dt,
  Update::StateMetaVector & C,
  const Update::StateMetaVector & P,
  const Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k,
  const int j,
  const int i)
{
  using namespace Update;
  using namespace Sources;
  using namespace Closures;

  using namespace Explicit;
  using namespace Implicit;
  using namespace Implicit::gsl;

  // Evolve (sc_E, sp_F_d) -> (sc_E*, sp_F_d*) with semi-implicit
  StepImplicitHybridsJ(pm1_,
                       dt, C, P, I, S,
                       CL,
                       k, j, i);

  if (!pm1_.opt_solver.solver_explicit_nG)
  {
    // Evolve (sc_nG, ) -> (sc_nG*, ) with semi-implicit
    // Prepares also (sc_n*, ) and S
    SolveImplicitNeutrinoCurrent(pm1_, dt, C, P, I, S, k, j, i);
  }
  else
  {
    PrepareMatterSource_nG(pm1_, P, S, k, j, i);
    StepExplicit_nG(pm1_, dt, C, P, I, S, k, j, i);

    // We now have nG*, it is useful to immediately construct n*
    Prepare_n_from_nG(pm1_, C, k, j, i);
  }
}

// ----------------------------------------------------------------------------
inline void DispatchIntegrationMethodImplementation(
  M1 & pm1_,
  const Real dt,
  M1::opt_integration_strategy & ois,
  const int k,
  const int j,
  const int i,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::StateMetaVector & I,
  Update::SourceMetaVector & S,
  Closures::ClosureMetaVector & CL)
{
  switch (ois)
  {
    case (M1::opt_integration_strategy::do_nothing):
    {
      break;
    }
    case (M1::opt_integration_strategy::full_explicit):
    {
      ExplicitIntegration(pm1_, dt, C, P, I, S, CL, k, j, i);
      break;
    }
    case (M1::opt_integration_strategy::explicit_approximate_semi_implicit):
    {
      ExplicitApproximateSemiImplicitIntegration(
        pm1_, dt, C, P, I, S, CL, k, j, i
      );
      break;
    }
    case (M1::opt_integration_strategy::semi_implicit_Hybrids):
    {
      SemiImplicitHybridsIntegration(
        pm1_, dt, C, P, I, S, CL, k, j, i
      );
      break;
    }
    case (M1::opt_integration_strategy::semi_implicit_HybridsJ):
    {
      SemiImplicitHybridsJIntegration(
        pm1_, dt, C, P, I, S, CL, k, j, i
      );
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void DispatchIntegrationMethod(
  M1 & pm1_,
  const Real dt,
  M1::vars_Lab & U_C,        // current (target) step
  const M1::vars_Lab & U_P,  // previous step data
  const M1::vars_Lab & U_I,  // inhomogeneity
  M1::vars_Source & U_S,     // carries matter source contribution
  const int kl, const int ku,
  const int jl, const int ju,
  const int il, const int iu
)
{
  using namespace Update;
  using namespace Sources;
  using namespace Closures;
  using namespace Equilibrium;

  using namespace Explicit;
  using namespace Implicit;

  // Work-around for ptr to utilize e.g. M1_ILOOP macro
  M1 * pm1 = &pm1_;

  for (int ix_g=0; ix_g<pm1_.N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1_.N_SPCS; ++ix_s)
  {
    // For quick reference XMetaVector should correspond to:
    //
    // C: new state   ~ U_A*
    // P: prior state ~ U_A
    // I: inh         ~ -div F_A + G_A
    //
    // S: sources     ~ S_A[U_A*]  (or S_A[U_A] for explicit)
    //
    // CL_P, CL_C: closure ~ CL[U_A], CL[U_A*].
    // During construction of closure from indicated state-vector:
    // Data is written to internal {sp_P_dd, sc_chi, sc_xi} & Lag. frame

    // const_cast here: internal pm1_ state is modified, but state-vector is not
    // in subsequent function calls
    M1::vars_Lab& U_P_ = const_cast<M1::vars_Lab&>(U_P);
    M1::vars_Lab& U_I_ = const_cast<M1::vars_Lab&>(U_I);

    StateMetaVector C = ConstructStateMetaVector(pm1_, U_C,  ix_g, ix_s);
    StateMetaVector P = ConstructStateMetaVector(pm1_, U_P_, ix_g, ix_s);
    StateMetaVector I = ConstructStateMetaVector(pm1_, U_I_, ix_g, ix_s);

    SourceMetaVector S = ConstructSourceMetaVector(pm1_, U_S, ix_g, ix_s);

    // For use in computation of closure based on P or C state
    ClosureMetaVector CL_P = ConstructClosureMetaVector(pm1_, U_P_, ix_g, ix_s);
    ClosureMetaVector CL_C = ConstructClosureMetaVector(pm1_, U_C,  ix_g, ix_s);

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    for (int i=il; i<=iu; ++i)
    if (pm1->MaskGet(k, j, i) && pm1->MaskGetHybridize(ix_s, k, j, i))
    {
      // switch to different solver based on solution regime ------------------
      M1::opt_integration_strategy opt_is;
      M1::M1::t_sln_r opt_reg = pm1->GetMaskSolutionRegime(ix_g, ix_s, k, j, i);

      bool use_eql_n_nG = false;
      bool use_eql_E_F_d = false;
      M1::opt_closure_variety opt_cl_variety = pm1->opt_closure.variety;

      switch (opt_reg)
      {
        case M1::t_sln_r::noop:
        {
          opt_is = M1::opt_integration_strategy::do_nothing;
          break;
        }
        case M1::t_sln_r::non_stiff:
        {
          // std::printf("DEBUG: non-stiff @ (%d, %d; %d, %d, %d)\n",
          //             ix_g, ix_s, k, j, i);
          opt_is = pm1->opt_solver.solvers.non_stiff;
          break;
        }
        case M1::t_sln_r::stiff:
        {
          // std::printf("DEBUG: stiff @ (%d, %d; %d, %d, %d)\n",
          //             ix_g, ix_s, k, j, i);
          opt_is = pm1->opt_solver.solvers.stiff;
          break;
        }
        case M1::t_sln_r::scattering:
        {
          // std::printf("DEBUG: scattering @ (%d, %d; %d, %d, %d)\n",
          //             ix_g, ix_s, k, j, i);
          opt_is = pm1->opt_solver.solvers.scattering;
          break;
        }
        case M1::t_sln_r::equilibrium:
        {
          // std::printf("DEBUG: equilibrium @ (%d, %d; %d, %d, %d)\n",
          //             ix_g, ix_s, k, j, i);

          if (pm1->opt_solver.equilibrium_E_F_d)
          {
            use_eql_E_F_d = true;
          }

          // Optionally flag solution for n directly from equilibrium;
          // remainder of (E,F_d) state-vector takes prescribed method
          if (pm1->opt_solver.equilibrium_n_nG)
          {
            use_eql_n_nG = true;
          }

          if (pm1->opt_solver.equilibrium_use_thick)
          {
            pm1->opt_closure.variety = M1::opt_closure_variety::thick;
          }

          opt_is = pm1->opt_solver.solvers.equilibrium;
          break;
        }
        case M1::t_sln_r::equilibrium_wr:
        {
          opt_is = M1::opt_integration_strategy::do_nothing;
          break;
        }
        default:
        {
          assert(false);
        }
      }

      // call suitable solver -------------------------------------------------
      DispatchIntegrationMethodImplementation(
        pm1_, dt, opt_is,
        k, j, i,
        C, P, I, S, CL_C);

      // Additional equilibrium logic -----------------------------------------

      // Overall algorithm:
      // - Zero all sources
      // - Need: S ~ U^new-U^* so retain previous contribution S <- -U^*
      // - Explicit evolution of (E,F_d) in absence of sources
      // - Set U: nG at equilibrium based on updated (E,F_d) fid. & avg eps
      // - Finalize sources: S = U_New - U^*
      // - Evolve (N,E,F_d) explicitly

      if (use_eql_E_F_d &&
          use_eql_n_nG)
      {
        // N.B: will over-write what was computed for (n,nG,E,F_d) in C
        SetEquilibrium_E_F_d_n_nG(*pm1, dt, C, P, I, S, CL_C, k, j, i);
      }
      else if (use_eql_n_nG)
      {
        // N.B: will over-write what was computed for (n,nG,E,F_d) in C
        SetEquilibrium_n_nG(*pm1, dt, C, P, I, S, CL_C, k, j, i);
      }

      // revert to originally selected closure for next point -----------------
      if (pm1->opt_solver.equilibrium_use_thick)
      {
        pm1->opt_closure.variety = opt_cl_variety;
      }

    }
  }

}

// ============================================================================
} // namespace M1::Integrators
// ============================================================================


//
// :D
//