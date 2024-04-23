// C++ standard headers
#include <iostream>

// Athena++ headers
#include "m1.hpp"
#include "m1_calc_closure.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_update.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>

// ============================================================================
namespace M1::Update::System {
// ============================================================================

inline void Z_E_F_d(
  M1 & pm1,
  const Real dt,
  const Real dotPvv,
  const StateMetaVector & P, // previous step data
  StateMetaVector & C,       // current step
  const StateMetaVector & I, // inhomogeneity
  const int k, const int j, const int i)
{
  const Real W    = pm1.fidu.sc_W(k,j,i);
  const Real W2   = SQR(W);

  const Real kap_as = P.sc_kap_a(k,j,i) + P.sc_kap_s(k,j,i);

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  Real S_fac (0);
  S_fac = pm1.geom.sc_sqrt_det_g(k,j,i) * P.sc_eta(k,j,i);
  S_fac += P.sc_kap_s(k,j,i) * W2 * (
    C.sc_E(k,j,i) - 2.0 * dotFv + dotPvv
  );

  const Real S1 = pm1.geom.sc_alpha(k,j,i) * W * (
    kap_as * (dotFv - C.sc_E(k,j,i)) + S_fac
  );

  const Real WE = P.sc_E(k,j,i) + dt * I.sc_E(k,j,i);
  const Real ZE = C.sc_E(k,j,i) - dt * S1 - WE;
  C.Z_E[0] = ZE;

  for (int a=0; a<N; ++a)
  {
    Real dotPv (0);
    for (int b=0; b<N; ++b)
    {
      dotPv += P.sp_P_dd(a,b,k,j,i) * pm1.fidu.sp_v_u(b,k,j,i);
    }

    const Real S1pk = pm1.geom.sc_alpha(k,j,i) * W * (
      kap_as * (dotPv - C.sp_F_d(a,k,j,i)) +
      S_fac * pm1.fidu.sp_v_d(a,k,j,i)
    );

    const Real WF_d = P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(a,k,j,i);
    const Real ZF_d = C.sp_F_d(a,k,j,i) - dt * S1pk - WF_d;
    C.Z_F_d[a] = ZF_d;
  }
}

inline void Z_E_F_d(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P, // previous step data
  StateMetaVector & C,       // current step
  const StateMetaVector & I, // inhomogeneity
  const int k, const int j, const int i)
{
  const Real dotPvv = Assemble::sc_ddot_dense_sp__(pm1.fidu.sp_v_u, C.sp_P_dd,
                                                   k, j, i);

  Z_E_F_d(pm1, dt, dotPvv, P, C, I, k, j, i);
}


// ============================================================================
} // namespace M1::Update::System
// ============================================================================


// ============================================================================
namespace M1::Update {
// ============================================================================

StateMetaVector ConstructStateMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s)
{
  return StateMetaVector {
    pm1,
    ix_g,
    ix_s,
    // state-vector dependent
    vlab.sc_nG( ix_g,ix_s),
    vlab.sc_E(  ix_g,ix_s),
    vlab.sp_F_d(ix_g,ix_s),
    // group-dependent, but common
    pm1.lab_aux.sc_n(   ix_g,ix_s),
    pm1.lab_aux.sc_chi( ix_g,ix_s),
    pm1.lab_aux.sc_xi(  ix_g,ix_s),
    pm1.lab_aux.sp_P_dd(ix_g,ix_s),
    // Lagrangian frame
    pm1.rad.sc_J(  ix_g,ix_s),
    pm1.rad.sc_H_t(ix_g,ix_s),
    pm1.rad.sp_H_d(ix_g,ix_s),
    // opacities
    pm1.radmat.sc_eta_0(  ix_g,ix_s),
    pm1.radmat.sc_kap_a_0(ix_g,ix_s),
    pm1.radmat.sc_eta(  ix_g,ix_s),
    pm1.radmat.sc_kap_a(ix_g,ix_s),
    pm1.radmat.sc_kap_s(ix_g,ix_s)
  };
}

Closures::Minerbo::DataRootfinder PopulateDataRootfinder(
  M1 & pm1, const StateMetaVector & sv)
{
  return Closures::Minerbo::DataRootfinder {
    pm1.geom.sp_g_uu,
    sv.sc_E,
    sv.sp_F_d,
    sv.sp_P_dd,
    sv.sc_chi,
    sv.sc_xi,
    sv.sc_J,
    sv.sc_H_t,
    sv.sp_H_d,
    pm1.fidu.sc_W,
    pm1.fidu.sp_v_u,
    pm1.fidu.sp_v_d,
    pm1.scratch.sp_vec_A_, // sp_dH_d_,
    pm1.scratch.sp_sym_A_, // sp_P_tn_dd_,
    pm1.scratch.sp_sym_B_, // sp_P_tk_dd_,
    pm1.scratch.sp_sym_C_  // sp_dP_dd_
  };
}

void AddSourceMatter(
  M1 & pm1,
  const StateMetaVector & C,  // state to utilize
  StateMetaVector & I,        // add source here
  const int k, const int j, const int i)
{

  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d,
                                                 pm1.fidu.sp_v_u,
                                                 k, j, i);

  const Real G = Assemble::sc_G__(
    W, C.sc_E(k,j,i), I.sc_J(k,j,i), dotFv,
    pm1.opt.fl_E, pm1.opt.fl_J, pm1.opt.eps_E
  );

  I.sc_n(k,j,i) = C.sc_nG(k,j,i) / G;

  I.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, C.sc_E, pm1.fidu.sp_v_u, C.sp_P_dd,
    k, j, i
  );

  I.sc_H_t(k,j,i) = Assemble::sc_H_t__(
    W, dotFv, C.sc_E, I.sc_J,
    k, j, i
  );

  Assemble::sp_H_d__(
    I.sp_H_d, W, I.sc_J, C.sp_F_d,
    pm1.fidu.sp_v_d, pm1.fidu.sp_v_u, C.sp_P_dd,
    k, j, i
  );

  // add to inhomogeneity
  I.sc_nG(k,j,i) += pm1.geom.sc_alpha(k,j,i) * (
    pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i) -
    C.sc_kap_a_0(k,j,i) * I.sc_n(k,j,i)
  );

  I.sc_E(k,j,i) += pm1.geom.sc_alpha(k,j,i) * (
    (pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta(k,j,i) -
     C.sc_kap_a(k,j,i) * I.sc_J(k,j,i)) * W -
    (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i)) * I.sc_H_t(k,j,i)
  );

  for (int a=0; a<N; ++a)
  {
    I.sp_F_d(a,k,j,i) += pm1.geom.sc_alpha(k,j,i) * (
      (pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta(k,j,i) -
       C.sc_kap_a(k,j,i) * I.sc_J(k,j,i)) * W * pm1.fidu.sp_v_d(a,k,j,i) -
      (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i)) * I.sp_H_d(a,k,j,i)
    );
  }
}

// time-integration strategies ------------------------------------------------

// Neutrino current evolution is linearly implicit; assemble nG directly
void SolveImplicitNeutrinoCurrent(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  const int k, const int j, const int i)
{
  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  const Real C_J = Assemble::sc_J__(
    W2, dotFv, C.sc_E, pm1.fidu.sp_v_u, C.sp_P_dd,
    k, j, i
  );

  const Real WnGam = P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i);
  const Real C_Gam = Assemble::sc_G__(
    W, C.sc_E(k,j,i), C_J, dotFv,
    pm1.opt.fl_E, pm1.opt.fl_J, pm1.opt.eps_E
  );

  C.sc_nG(k,j,i) = WnGam / (
    1.0 - dt * pm1.geom.sc_alpha(k,j,i) * C.sc_kap_a_0(k,j,i) / C_Gam
  );
}

// Evolve explicit part of system; optionally suppress nG evo.
void StepExplicit(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,   // previous step data
  StateMetaVector & C,         // current step
  const StateMetaVector & I,   // inhomogeneity
  const bool explicit_step_nG, // evolve nG?
  const int k, const int j, const int i)
{
  if (explicit_step_nG)
  {
    C.sc_nG(k,j,i) = P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i);
  }

  C.sc_E( k,j,i) = P.sc_E( k,j,i) + dt * I.sc_E(k,j,i);

  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(a,k,j,i);
  }

  ApplyFloors(pm1, C, k, j, i);
  EnforceCausality(pm1, C, k, j, i);
}

void StepImplicitPicardFrozenP(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  const int k, const int j, const int i)
{
  // Iterate (S_1, S_{1+k})
  const int iter_max_P = pm1.opt.max_iter_P;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1.opt.max_iter_P_rst;  // maximum number of restarts
  Real w_opt = pm1.opt.w_opt_ini;  // underrelaxation factor
  Real e_P_abs_tol = pm1.opt.eps_P_abs_tol;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1.opt.fac_amp_P;

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // explicit update ------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

  // loop-lift contraction
  Real dotPvv (0);
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dotPvv += P.sp_P_dd(a,b,k,j,i) *
              pm1.fidu.sp_v_u(a,k,j,i) *
              pm1.fidu.sp_v_u(b,k,j,i);
  }

  // solver loop ----------------------------------------------------------
  do
  {
    pit++;

    // state-vector non-linear subsystem --------------------------------------
    System::Z_E_F_d(pm1, dt, dotPvv, P, C, I, k, j, i);

    C.sc_E(k,j,i) = C.sc_E(k,j,i) - w_opt * C.Z_E[0];
    e_abs_cur = std::abs(C.Z_E[0]);

    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) - w_opt * C.Z_F_d[a];
      e_abs_cur = std::max(std::abs(C.Z_F_d[a]), e_abs_cur);
    }

    // Ensure update preserves energy non-negativity --------------------------
    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // scale error tol by step
    // e_abs_cur = w_opt * e_abs_cur;

    if (e_abs_cur > fac_PA * e_abs_old)
    {
      // halve underrelaxation and recover old values
      w_opt = w_opt / 2;
      C.Fallback(k, j, i);
      StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

      // restart iteration
      e_abs_old = std::numeric_limits<Real>::infinity();
      pit = 0;
      rit++;

      if (rit > iter_max_R)
      {
        std::ostringstream msg;
        msg << "StepImplicitPicardFrozenP max restarts exceeded.";
        std::cout << msg.str().c_str() << std::endl;
        std::exit(0);
      }
    }
    else
    {
      e_abs_old = e_abs_cur;
    }

  } while ((pit < iter_max_P) && (e_abs_cur >= e_P_abs_tol));

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // dump some information ------------------------------------------------
  if (pm1.opt.verbose_iter_P)
  {
    if (e_abs_cur >= e_P_abs_tol)
    {
      std::cout << "StepImplicitPicardFrozenP:\n";
      std::cout << "Tol. not achieved: " << e_abs_cur << "\n";
      std::cout << "(pit,rit): " << pit << "," << rit << "\n";
      std::cout << "chi: " << C.sc_chi(k,j,i) << "\n";
      std::cout << "(k,j,i): " << k << "," << j << "," << i << "\n";
    }
  }

}

void StepImplicitPicardMinerboPC(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  Closures::Minerbo::DataRootfinder & D,  // for closure
  const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // Iterate (S_1, S_{1+k})
  const int iter_max_P = pm1.opt.max_iter_P;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1.opt.max_iter_P_rst;  // maximum number of restarts
  Real w_opt = pm1.opt.w_opt_ini;  // underrelaxation factor
  Real e_P_abs_tol = pm1.opt.eps_P_abs_tol;
  Real e_C_abs_tol = pm1.opt.eps_C;
  // maximum error amplification factor between iters.
  Real fac_PA   = pm1.opt.fac_amp_P;
  Real fac_PA_C = pm1.opt.fac_amp_C;

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  Real e_abs_old_C = std::numeric_limits<Real>::infinity();
  Real e_abs_cur_C = 0;

  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // explicit update ------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

  // Minerbo assembly ---------------------------------------------------
  D.k = k;
  D.j = j;
  D.i = i;

  const Real xi = C.sc_xi(k,j,i);
  C.sc_chi(k,j,i) = Closures::Minerbo::chi(C.sc_xi(k,j,i));

  Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                            k, j, i);

  Closures::ClosureThin(&pm1, D.sp_P_tn_dd_, C.sc_E, C.sp_F_d, k, j, i);
  Closures::ClosureThick(
    &pm1,
    D.sp_P_tk_dd_,
    dotFv,
    C.sc_E,
    C.sp_F_d,
    k, j, i);

  Closures::Minerbo::sp_P_dd__(C.sp_P_dd,
                               C.sc_chi,
                               D.sp_P_tn_dd_,
                               D.sp_P_tk_dd_,
                               k, j, i);

  if ((C.sc_xi(k,j,i) < xi_min) || (C.sc_xi(k,j,i) > xi_max))
  {
    C.sc_xi(k,j,i) = std::max(
      std::min(xi_max, C.sc_xi(k,j,i)), xi_min
    );
    D.sc_chi(k,j,i) = Closures::Minerbo::chi(C.sc_xi(k,j,i));

    Closures::Minerbo::sp_P_dd__(C.sp_P_dd,
                                  C.sc_chi,
                                  D.sp_P_tn_dd_,
                                  D.sp_P_tk_dd_,
                                  k, j, i);
  }

  // loop-lift contraction
  Real dotPvv (0);
  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dotPvv += C.sp_P_dd(a,b,k,j,i) *
              pm1.fidu.sp_v_u(a,k,j,i) *
              pm1.fidu.sp_v_u(b,k,j,i);
  }

  // solver loop ----------------------------------------------------------
  do
  {
    pit++;

    // state-vector non-linear subsystem --------------------------------------
    System::Z_E_F_d(pm1, dt, dotPvv, P, C, I, k, j, i);

    C.sc_E(k,j,i) = C.sc_E(k,j,i) - w_opt * C.Z_E[0];
    e_abs_cur = std::abs(C.Z_E[0]);

    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) - w_opt * C.Z_F_d[a];
      e_abs_cur = std::max(std::abs(C.Z_F_d[a]), e_abs_cur);
    }

    // Ensure update preserves energy non-negativity --------------------------
    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // scale error tol by step
    // e_abs_cur = w_opt * e_abs_cur;

    if (e_abs_cur > fac_PA * e_abs_old)
    {
      // halve underrelaxation and recover old values
      w_opt = w_opt / 2;
      C.Fallback(k, j, i);
      StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

      if (rit > iter_max_R)
      {
        std::ostringstream msg;
        msg << "StepImplicitPicardPC max restarts exceeded.";
        std::cout << msg.str().c_str() << std::endl;
        Closures::InfoDump(&pm1, C.ix_g, C.ix_s, k, j, i);
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

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // dump some information ------------------------------------------------
  if (pm1.opt.verbose_iter_P)
  {
    if (e_abs_cur >= e_P_abs_tol)
    {
      std::cout << "StepImplicitPicardPC:\n";
      std::cout << "Tol. not achieved: " << e_abs_cur << "\n";
      std::cout << "(pit,rit): " << pit << "," << rit << "\n";
      std::cout << "chi: " << C.sc_chi(k,j,i) << "\n";
      std::cout << "(k,j,i): " << k << "," << j << "," << i << "\n";
    }
  }

}

void StepImplicitPicardMinerboP(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  Closures::Minerbo::DataRootfinder & D,  // for closure
  const int k, const int j, const int i)
{
  // Admissible ranges for xi entering Eddington factor chi(xi)
  const Real xi_min = 0.0;
  const Real xi_max = 1.0;

  // Iterate (S_1, S_{1+k})
  const int iter_max_P = pm1.opt.max_iter_P;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1.opt.max_iter_P_rst;  // maximum number of restarts
  Real w_opt = pm1.opt.w_opt_ini;  // underrelaxation factor
  Real e_P_abs_tol = pm1.opt.eps_P_abs_tol;
  Real e_C_abs_tol = pm1.opt.eps_C;
  // maximum error amplification factor between iters.
  Real fac_PA   = pm1.opt.fac_amp_P;
  Real fac_PA_C = pm1.opt.fac_amp_C;

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  Real e_abs_old_C = std::numeric_limits<Real>::infinity();
  Real e_abs_cur_C = 0;

  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // explicit update ------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

  // solver loop --------------------------------------------------------------
  D.i = i;
  D.j = j;
  D.k = k;

  do
  {
    pit++;

    // Minerbo assembly -------------------------------------------------------
    const Real xi = C.sc_xi(k,j,i);
    C.sc_chi(k,j,i) = Closures::Minerbo::chi(C.sc_xi(k,j,i));

    Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                             k, j, i);

    Closures::ClosureThin(&pm1, D.sp_P_tn_dd_, C.sc_E, C.sp_F_d, k, j, i);
    Closures::ClosureThick(
      &pm1,
      D.sp_P_tk_dd_,
      dotFv,
      C.sc_E,
      C.sp_F_d,
      k, j, i);

    Closures::Minerbo::sp_P_dd__(C.sp_P_dd,
                                 C.sc_chi,
                                 D.sp_P_tn_dd_,
                                 D.sp_P_tk_dd_,
                                 k, j, i);

    // assemble zero functional
    const Real Z_xi = Closures::Minerbo::R(xi, &D);

    C.sc_xi(k,j,i) = std::abs(C.sc_xi(k,j,i) - w_opt * Z_xi);
    e_abs_cur_C = std::abs(Z_xi);

    if ((C.sc_xi(k,j,i) < xi_min) || (C.sc_xi(k,j,i) > xi_max))
    {
      C.sc_xi(k,j,i) = std::max(
        std::min(xi_max, C.sc_xi(k,j,i)), xi_min
      );
      D.sc_chi(k,j,i) = Closures::Minerbo::chi(C.sc_xi(k,j,i));

      Closures::Minerbo::sp_P_dd__(C.sp_P_dd,
                                   C.sc_chi,
                                   D.sp_P_tn_dd_,
                                   D.sp_P_tk_dd_,
                                   k, j, i);
    }

    // if iteration pushes outside admissible range reset
    /*
    if (pm1.opt.reset_thin)
    {
      if ((C.sc_xi(k,j,i) < xi_min) || (C.sc_xi(k,j,i) > xi_max))
      {
        C.sc_xi(k,j,i) = xi_max;
        D.sc_chi(k,j,i) = Closures::Minerbo::chi(C.sc_xi(k,j,i));

        for (int a=0; a<N; ++a)
        for (int b=a; b<N; ++b)
        {
          C.sp_P_dd(a,b,k,j,i) = D.sp_P_tn_dd_(a,b,i);
        }

        // e_abs_old = std::numeric_limits<Real>::infinity();
        // e_abs_cur = std::numeric_limits<Real>::infinity();
      }
      e_abs_cur_C = 0.0;
    }
    */

    // state-vector non-linear subsystem --------------------------------------
    System::Z_E_F_d(pm1, dt, P, C, I, k, j, i);

    C.sc_E(k,j,i) = C.sc_E(k,j,i) - w_opt * C.Z_E[0];
    e_abs_cur = std::abs(C.Z_E[0]);

    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) - w_opt * C.Z_F_d[a];
      e_abs_cur = std::max(std::abs(C.Z_F_d[a]), e_abs_cur);
    }

    // Ensure update preserves energy non-negativity --------------------------
    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // scale error tol by step
    // e_abs_cur = w_opt * e_abs_cur;

    if (e_abs_cur > fac_PA * e_abs_old)
    {
      // halve underrelaxation and recover old values
      w_opt = w_opt / 2;
      C.Fallback(k, j, i);
      StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

      if (rit > iter_max_R)
      {
        std::ostringstream msg;
        msg << "StepImplicitPicardMinerboP max restarts exceeded.";
        std::cout << msg.str().c_str() << std::endl;
        Closures::InfoDump(&pm1, C.ix_g, C.ix_s, k, j, i);
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

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // dump some information ----------------------------------------------------
  if (pm1.opt.verbose_iter_P)
  {
    if (e_abs_cur >= e_P_abs_tol)
    {
      std::cout << "StepImplicitPicardMinerboP:\n";
      std::cout << "Tol. not achieved: " << e_abs_cur << "\n";
      std::cout << "(pit,rit): " << pit << "," << rit << "\n";
      std::cout << "chi: " << C.sc_chi(k,j,i) << "\n";
      std::cout << "(k,j,i): " << k << "," << j << "," << i << "\n";
    }
  }

}

// ============================================================================
} // namespace M1::Update
// ============================================================================

// ============================================================================
namespace M1::Update::gsl {
// ============================================================================

struct rparams {
  M1 & pm1;
  Real dt;
  const StateMetaVector & P;
  StateMetaVector & C;
  const StateMetaVector & I;

  const int i;
  const int j;
  const int k;
};

int Z_E_F_d(const gsl_vector *U, void * par_, gsl_vector *Z)
{
  rparams * par = static_cast<rparams*>(par_);

  M1 & pm1 = par->pm1;
  const Real dt = par->dt;
  const StateMetaVector & P = par->P;
  StateMetaVector & C = par->C;
  const StateMetaVector & I = par->I;

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  C.sc_E(k,j,i) = U->data[0];
  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = U->data[1+a];
  }

  System::Z_E_F_d(pm1, dt, P, C, I, k, j, i);

  Z->data[0] = C.Z_E[0];
  for (int a=0; a<N; ++a)
  {
    Z->data[1+a] = C.Z_F_d[a];
  }

  return GSL_SUCCESS;
}

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (FD approximation for Jacobian)
void StepImplicitMinerboHybrids(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  Closures::Minerbo::DataRootfinder & D,  // for closure
  SystemSolverSettings & slv_set,
  const int k, const int j, const int i)
{
  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // explicit update ----------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, explicit_step_nG, k, j, i);

  // GSL specific -------------------------------------------------------------
  const size_t N_SYS = 1 + N;

  // solver initial guess
  Real U_0[N_SYS];

  U_0[0] = C.sc_E(k,j,i);
  for (int a=0; a<N; ++a)
  {
    U_0[1+a] = C.sp_F_d(a,k,j,i);
  }

  // values to iterate
  gsl_vector *U_i = gsl_vector_alloc(N_SYS);

  for (int n=0; n<N_SYS; ++n)
  {
    U_i->data[n] = U_0[n];
  }

  // select function & solver -------------------------------------------------
  struct rparams par = {pm1, dt, P, C, I,
                        i, j, k};
  gsl_multiroot_function mrf = {&Z_E_F_d, N_SYS, &par};
  gsl_multiroot_fsolver *slv = gsl_multiroot_fsolver_alloc(
    gsl_multiroot_fsolver_hybrids,
    N_SYS);

  int gsl_status = gsl_multiroot_fsolver_set(slv, &mrf, U_i);

  // solver loop --------------------------------------------------------------
  int iter = 0;
  do
    {
      iter++;
      gsl_status = gsl_multiroot_fsolver_iterate(slv);

      // break on issue with solver
      if (gsl_status)
        break;

      // Ensure update preserves energy non-negativity
      ApplyFloors(pm1, C, k, j, i);
      EnforceCausality(pm1, C, k, j, i);

      gsl_status = gsl_multiroot_test_residual(slv->f, slv_set.tol_abs);
    }
  while (gsl_status == GSL_CONTINUE && iter < slv_set.iter_max);

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // cleanup ------------------------------------------------------------------
  gsl_multiroot_fsolver_free(slv);
  gsl_vector_free(U_i);

}

// ============================================================================
} // namespace M1::Update::gsl
// ============================================================================


// ============================================================================
namespace M1 {
// ============================================================================

void DebugValueInject(M1 & pm1,
                      AthenaArray<Real> & u_pre,
                      AthenaArray<Real> & u_cur,
		                  AthenaArray<Real> & u_inh);

// ----------------------------------------------------------------------------
// Function to update the state vector
void M1::CalcUpdate(Real const dt,
                    AthenaArray<Real> & u_pre,
                    AthenaArray<Real> & u_cur,
		                AthenaArray<Real> & u_inh)
{
  using namespace Update;
  using namespace Update::gsl;

  if (opt.value_inject)
  {
    DebugValueInject(*this, u_pre, u_cur, u_inh);
  }

  // setup aliases ------------------------------------------------------------

  vars_Lab U_P { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_P);

  vars_Lab U_C { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_C);

  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, U_I);

  // dispatch integration strategy --------------------------------------------
  SystemSolverSettings slv_set = ConstructSystemSolverSettings(*this);

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(*this, U_C, ix_g, ix_s);
    StateMetaVector P = ConstructStateMetaVector(*this, U_P, ix_g, ix_s);
    StateMetaVector I = ConstructStateMetaVector(*this, U_I, ix_g, ix_s);

    switch (opt.integration_strategy)
    {
      case (opt_integration_strategy::full_explicit):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          ::M1::Update::AddSourceMatter(*this, C, I, k, j, i);
          const bool explicit_step_nG = true;
          StepExplicit(*this, dt, P, C, I, explicit_step_nG, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_PicardFrozenP):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitPicardFrozenP(*this, dt, P, C, I, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_PicardMinerboP):
      {
        using Closures::Minerbo::DataRootfinder;

        DataRootfinder D = PopulateDataRootfinder(*this, C);

        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitPicardMinerboP(*this, dt, P, C, I, D, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_PicardMinerboPC):
      {
        using Closures::Minerbo::DataRootfinder;

        DataRootfinder D = PopulateDataRootfinder(*this, C);

        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitPicardMinerboPC(*this, dt, P, C, I, D, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_Hybrids):
      {
        using Closures::Minerbo::DataRootfinder;

        DataRootfinder D = PopulateDataRootfinder(*this, C);

        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitMinerboHybrids(*this, dt, P, C, I, D, slv_set, k, j, i);
        }
        break;
      }
      default:
      {
        assert(false);
        std::exit(0);
      }
    }
  }

}

void DebugValueInject(M1 & pm1,
                      AthenaArray<Real> & u_pre,
                      AthenaArray<Real> & u_cur,
		                  AthenaArray<Real> & u_inh)
{
  std::cout << "\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << "No sources-----------------------------------------------\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << "\n";

  pm1.StatePrintPoint(0, 0,
                      pm1.mbi.kl, pm1.mbi.ju-1, pm1.mbi.iu-1, false);
  std::cout << "\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << "AddSourceGR----------------------------------------------\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << "\n";

  u_inh.ZeroClear();
  pm1.AddSourceGR(u_cur, u_inh);

  pm1.SetVarAliasesLab(u_inh, pm1.lab);
  pm1.StatePrintPoint(0, 0,
                      pm1.mbi.kl, pm1.mbi.ju-1, pm1.mbi.iu-1, false);

  std::cout << "\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << "AddMatterSources-----------------------------------------\n";
  std::cout << "---------------------------------------------------------\n";
  std::cout << "\n";

  u_inh.ZeroClear();
  pm1.AddSourceMatter(u_cur, u_inh);

  pm1.SetVarAliasesLab(u_inh, pm1.lab);
  pm1.StatePrintPoint(0, 0,
                      pm1.mbi.kl, pm1.mbi.ju-1, pm1.mbi.iu-1, true);
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//