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

// Source term Jacobian in terms of \sqrt{g} densitized variables.
// Here P_dd is considered independent of E, F_d
//
// See also 3.2 of [1].
inline void dZ_E_F_d_FrozenP(
  M1 & pm1,
  const Real dt,
  AA & J,                    // Storage for Jacobian
  const StateMetaVector & C, // current step
  const int k, const int j, const int i)
{
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);
  const Real W3 = W * W2;

  const Real kap_as = (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i));
  const Real kap_s = C.sc_kap_s(k,j,i);

  const Real alpha = pm1.geom.sc_alpha(k,j,i);

  // J(I,J) ~ D[S_I,(E,F_d)_J]

  // sc_Stil_1 ----------------------------------------------------------------

  // D_E
  J(0,0) = alpha * W * (
    -kap_as + kap_s * W2
  );

  const Real fac_J_0a = alpha * W * (
    kap_as - 2.0 * kap_s * W2
  );

  // D_F_d
  for (int a=0; a<N; ++a)
  {
    J(0,1+a) = fac_J_0a * pm1.fidu.sp_v_u(a,k,j,i);
  }

  // sc_Stil_1pa --------------------------------------------------------------
  for (int a=0; a<N; ++a)
  {
    // D_E
    J(1+a,0) = alpha * kap_s * W3 * pm1.fidu.sp_v_d(a,k,j,i);

    // D_F_d
    for (int b=0; b<N; ++b)
    {
      J(1+a,1+b) = -alpha * W * (
        (a == b) * kap_as +
        2.0 * kap_s * W2 * pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i)
      );
    }
  }

  // Actually need Jacobian for Z and not the sources -------------------------
  const int N_SYS = J.GetDim1();
  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    J(a,b) = (a==b) - dt * J(a,b);
  }

}

// Source term Jacobian in terms of \sqrt{g} densitized variables.
//
// See also 3.2 of [1].
inline void dZ_E_F_d_Minerbo(
  M1 & pm1,
  const Real dt,
  AA & J,                    // Storage for Jacobian
  const StateMetaVector & C, // current step
  const int k, const int j, const int i)
{
  const Real W = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);
  const Real W3 = W * W2;

  const Real kap_as = (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i));
  const Real kap_s = C.sc_kap_s(k,j,i);

  const Real alpha = pm1.geom.sc_alpha(k,j,i);

  // P_dd thick (tk) and thin (tn) factors
  const Real d_tk = 3.0 * 0.5 * (1.0 - C.sc_chi(k,j,i));
  const Real d_tn = 1.0 - d_tk;

  const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                                 k, j, i);

  const Real dotvv = Assemble::sc_dot_dense_sp__(pm1.fidu.sp_v_d,
                                                 pm1.fidu.sp_v_u,
                                                 k, j, i);

  Assemble::sp_d_to_u_(&pm1, pm1.scratch.sp_F_u_, C.sp_F_d, k, j, i, i);
  Real dotFF = 0;

  for (int a=0; a<N; ++a)
  {
    dotFF += pm1.scratch.sp_F_u_(a,i) * C.sp_F_d(a,k,j,i);
  }

  // J(I,J) ~ D[S_I,(E,F_d)_J]

  // sc_Stil_1 ----------------------------------------------------------------

  // D_E
  J(0,0) = alpha * W * (
    -kap_as + kap_s * W2
  );

  const Real fac_J_0a = alpha * W * (
    kap_as - 2.0 * kap_s * W2
  );

  // D_F_d
  for (int a=0; a<N; ++a)
  {
    J(0,1+a) = fac_J_0a * pm1.fidu.sp_v_u(a,k,j,i);
  }

  // sc_Stil_1pa --------------------------------------------------------------
  for (int a=0; a<N; ++a)
  {
    // D_E
    J(1+a,0) = alpha * kap_s * W3 * pm1.fidu.sp_v_d(a,k,j,i);

    // D_F_d
    for (int b=0; b<N; ++b)
    {
      J(1+a,1+b) = -alpha * W * (
        (a == b) * kap_as +
        2.0 * kap_s * W2 * pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i)
      );
    }
  }

  // thin correction to Jacobian ----------------------------------------------
  if (dotFF > 0)
  {
    J(0,0) += d_tn * alpha * W3 * kap_s * SQR(dotFv) / dotFF;

    for (int b=0; b<N; ++b)
    {
      J(0,1+b) += d_tn * 2.0 * alpha * dotFv * C.sc_E(k,j,i) * kap_s * W3 * (
        -dotFv * pm1.scratch.sp_F_u_(b,i) +
        dotFF * pm1.fidu.sp_v_u(b,k,j,i)
      ) / SQR(dotFF);
    }

    for (int a=0; a<N; ++a)
    {
      J(1+a,0) += d_tn * alpha * dotFv * W * (
        C.sp_F_d(a,k,j,i) * kap_as +
        W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)
      ) / dotFF;

      for (int b=0; b<N; ++b)
      {
        J(1+a,1+b) += d_tn * alpha * C.sc_E(k,j,i) * W * (
          -2.0 * dotFv * pm1.scratch.sp_F_u_(b,i) *
          (C.sp_F_d(a,k,j,i) * kap_as +
           W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)) +
          dotFF * ((a==b) * dotFv * kap_as +
                   pm1.fidu.sp_v_u(b,k,j,i) * (
                    C.sp_F_d(a,k,j,i) * kap_as +
                    2.0 * W2 * dotFv * kap_s * pm1.fidu.sp_v_d(a,k,j,i)
                   ))
        ) / SQR(dotFF);
      }
    }
  }

  // thick correction to Jacobian ---------------------------------------------
  J(0,0) += d_tk * alpha * dotvv * kap_s * W3 * (
    -1.0 + (2.0 - 4.0 * dotvv) * W2
  ) / (1.0 + 2.0 * W2);

  for (int b=0; b<N; ++b)
  {
    J(0,1+b) += d_tk * 2.0 * alpha * dotvv * kap_s * W3 *
                pm1.fidu.sp_v_u(b,k,j,i) * (
                  1.0 + (1.0 + dotvv) * W2
                ) / (1.0 + 2.0 * W2);
  }

  for (int a=0; a<N; ++a)
  {
    J(1+a,0) += -d_tk * alpha * pm1.fidu.sp_v_d(a,k,j,i) * W * (
      1.0 + (-2.0 + 4.0 * dotvv) * W2
    ) * (
      kap_as + dotvv * kap_s * W2
    ) / (1.0 + 2.0 * W2);

    for (int b=0; b<N; ++b)
    {
      J(1+a,1+b) += d_tk * alpha * W * (
        (a == b) * dotvv * kap_as +
        pm1.fidu.sp_v_u(b,k,j,i) * pm1.fidu.sp_v_d(a,k,j,i) * (
          kap_as + 2.0 * dotvv * kap_as * W2 +
          2.0 * dotvv * kap_s * W2 * (1.0 + (1.0 + dotvv) * W2)
        ) / (1.0 + 2.0 * W2)
      );
    }
  }

  // Actually need Jacobian for Z and not the sources -------------------------
  const int N_SYS = J.GetDim1();
  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    J(a,b) = (a==b) - dt * J(a,b);
  }

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
    pm1.radmat.sc_kap_s(ix_g,ix_s),
    // averages
    pm1.radmat.sc_avg_nrg(ix_g,ix_s)
  };
}

SourceMetaVector ConstructSourceMetaVector(
  M1 & pm1, M1::vars_Source & vsrc,
  const int ix_g, const int ix_s)
{
  return SourceMetaVector {
    pm1,
    ix_g,
    ix_s,
    vsrc.sc_S0(  ix_g,ix_s),
    vsrc.sc_S1(  ix_g,ix_s),
    vsrc.sp_S1_d(ix_g,ix_s),
  };
}

void AddSourceMatter(
  M1 & pm1,
  const StateMetaVector & C,  // state to utilize
  StateMetaVector & I,        // add source here
  SourceMetaVector & S,       // add matter coupling terms here
  const int k, const int j, const int i)
{
  const Real W  = pm1.fidu.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real kap_as = C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i);

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

  AT_C_sca & dotFv_  = pm1.scratch.sc_A_;
  AT_C_sca & dotPvv_ = pm1.scratch.sc_B_;
  AT_N_vec & dotPv_d_  = pm1.scratch.sp_vec_A_;

  dotFv_(i) = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u, k, j, i);

  for (int a=0; a<N; ++a)
  for (int b=0; b<N; ++b)
  {
    dotPv_d_(a,i) += C.sp_P_dd(a,b,k,j,i) * pm1.fidu.sp_v_u(b,k,j,i);
  }

  for (int a=0; a<N; ++a)
  {
    dotPvv_(i) += dotPv_d_(a,i) * pm1.fidu.sp_v_u(a,k,j,i);
  }

  // add to inhomogeneity
  Real S0 = pm1.geom.sc_alpha(k,j,i) * (
    pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i) -
    C.sc_kap_a_0(k,j,i) * I.sc_n(k,j,i)
  );

  I.sc_nG(k,j,i) += S0;
  S.sc_S0(k,j,i) = S0;

  Real S1 = pm1.geom.sc_alpha(k,j,i) * (
    (pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta(k,j,i) -
     C.sc_kap_a(k,j,i) * I.sc_J(k,j,i)) * W -
    (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i)) * I.sc_H_t(k,j,i)
  );
  I.sc_E(k,j,i) += S1;
  S.sc_S1(k,j,i) = S1;

  for (int a=0; a<N; ++a)
  {
    Real S1_d = pm1.geom.sc_alpha(k,j,i) * (
      (pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta(k,j,i) -
       C.sc_kap_a(k,j,i) * I.sc_J(k,j,i)) * W * pm1.fidu.sp_v_d(a,k,j,i) -
      (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i)) * I.sp_H_d(a,k,j,i)
    );
    I.sp_F_d(a,k,j,i) += S1_d;
    S.sp_S1_d(a,k,j,i) = S1_d;

  }
}

void AssembleAverages(
  M1 & pm1,
  const StateMetaVector & C,  // state to utilize
  const bool recompute_n,
  const int k, const int j, const int i)
{
  if (recompute_n)
  {

    const Real W  = pm1.fidu.sc_W(k,j,i);
    const Real W2 = SQR(W);

    const Real dotFv = Assemble::sc_dot_dense_sp__(C.sp_F_d, pm1.fidu.sp_v_u,
                                                  k, j, i);

    C.sc_J(k,j,i) = Assemble::sc_J__(
      W2, dotFv, C.sc_E, pm1.fidu.sp_v_u, C.sp_P_dd,
      k, j, i
    );

    const Real C_Gam = Assemble::sc_G__(
      W, C.sc_E(k,j,i), C.sc_J(k,j,i), dotFv,
      pm1.opt.fl_E, pm1.opt.fl_J, pm1.opt.eps_E
    );

    C.sc_n(k,j,i) = C.sc_nG(k,j,i) / C_Gam;
  }

  C.sc_avg_nrg(k,j,i) = (C.sc_n(k,j,i) > 0) ? C.sc_J(k,j,i) / C.sc_n(k,j,i)
                                            : 0.0;
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

  C.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, C.sc_E, pm1.fidu.sp_v_u, C.sp_P_dd,
    k, j, i
  );


  // BD: I.sc_nG contains flux div, but not \alpha \sqrt \eta^0

  // const Real WnGam = P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i);

  const Real S_0 = pm1.geom.sc_alpha(k,j,i) *
    pm1.geom.sc_sqrt_det_g(k,j,i) * C.sc_eta_0(k,j,i);

  const Real WnGam = P.sc_nG(k,j,i) + dt * (S_0 + I.sc_nG(k,j,i));

  const Real C_Gam = Assemble::sc_G__(
    W, C.sc_E(k,j,i), C.sc_J(k,j,i), dotFv,
    pm1.opt.fl_E, pm1.opt.fl_J, pm1.opt.eps_E
  );

  C.sc_nG(k,j,i) = WnGam / (
    1.0 + dt * pm1.geom.sc_alpha(k,j,i) * C.sc_kap_a_0(k,j,i) / C_Gam
  );

  C.sc_n(k,j,i) = C.sc_nG(k,j,i) / C_Gam;
}

// Evolve explicit part of system; optionally suppress nG evo.
void StepExplicit(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,   // previous step data
  StateMetaVector & C,         // current step
  const StateMetaVector & I,   // inhomogeneity
  SourceMetaVector & S,
  const bool explicit_step_nG, // evolve nG?
  const int k, const int j, const int i)
{
  if (explicit_step_nG)
  {
    C.sc_nG(k,j,i) = P.sc_nG(k,j,i) + dt * I.sc_nG(k,j,i);
    S.sc_S0(k,j,i) *= dt;
  }

  C.sc_E( k,j,i) = P.sc_E( k,j,i) + dt * I.sc_E(k,j,i);
  S.sc_S1(k,j,i) *= dt;

  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = P.sp_F_d(a,k,j,i) + dt * I.sp_F_d(a,k,j,i);
    S.sp_S1_d(a,k,j,i) *= dt;
  }

  // NonFiniteToZero(pm1, C, k, j, i);
  ApplyFloors(pm1, C, k, j, i);
  EnforceCausality(pm1, C, k, j, i);
}

// Can solve the implicit system approxmately at O(v).
// This can be used as an initial guess for implicit integration [1].
void StepApproximateFirstOrder(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k, const int j, const int i)
{
  // explicit step
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

  CL.Closure(k,j,i);

  // construct fiducial frame quantities (tilde of [1])
  const Real W  = CL.sc_W(k,j,i);
  const Real W2 = SQR(W);

  const Real dotFv = Assemble::sc_dot_dense_sp__(
    C.sp_F_d,
    C.pm1.fidu.sp_v_u,
    k, j, i
  );

  // assemble J
  C.sc_J(k,j,i) = Assemble::sc_J__(
    W2, dotFv, C.sc_E, C.pm1.fidu.sp_v_u, C.sp_P_dd,
    k, j, i
  );

  // assemble H_d
  Assemble::sp_H_d__(C.sp_H_d, W, C.sc_J, C.sp_F_d,
                     C.pm1.fidu.sp_v_d, C.pm1.fidu.sp_v_u, C.sp_P_dd,
                     k, j, i);

  // propagate fiducial frame quantities (hat of [1])
  C.sc_J(k,j,i) = (
    (C.sc_J(k,j,i) * W + dt * C.sc_eta(k,j,i)) /
    (W + dt * C.sc_kap_a(k,j,i))
  );

  for (int a=0; a<N; ++a)
  {
    C.sp_H_d(a,k,j,i) = W * C.sp_H_d(a,k,j,i) / (
      W + dt * (C.sc_kap_a(k,j,i) + C.sc_kap_s(k,j,i))
    );
  }

  // transform to Eulerian frame assuming _thick_ limit closure
  const Real dotHv = Assemble::sc_dot_dense_sp__(
    C.sp_H_d,
    C.pm1.fidu.sp_v_u,
    k, j, i
  );

  // assemble initial guess
  C.sc_E(k,j,i) = ONE_3RD * (
    4.0 * W2 - 1.0
  ) * C.sc_J(k,j,i) + 2.0 * W * dotHv;

  for (int a=0; a<N; ++a)
  {
    C.sp_F_d(a,k,j,i) = (
      4.0 * ONE_3RD * W2 * C.sc_J(k,j,i) * C.pm1.fidu.sp_v_d(a,k,j,i) +
      W * (C.sp_H_d(a,k,j,i) + dotHv * C.pm1.fidu.sp_v_d(a,k,j,i))
    );
  }

  // NonFiniteToZero(pm1, C, k, j, i);
  ApplyFloors(pm1, C, k, j, i);
  EnforceCausality(pm1, C, k, j, i);

  // CL.ClosureThick(k,j,i);
}

void StepImplicitPicardFrozenP(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  SourceMetaVector & S,
  const int k, const int j, const int i)
{
  // Iterate (S_1, S_{1+k})
  const int iter_max = pm1.opt_solver.iter_max;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1.opt_solver.iter_max_rst;  // maximum restarts
  Real w_opt = pm1.opt_solver.w_opt_ini;  // underrelaxation factor
  Real err_tol = pm1.opt_solver.eps_tol;
  // maximum error amplification factor between iters.
  Real fac_PA = pm1.opt_solver.fac_err_amp;

  Real err_old = std::numeric_limits<Real>::infinity();
  Real err_cur = 0;

  // explicit update ----------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // loop-lift contraction
  const Real dotPvv = Assemble::sc_ddot_dense_sp__(pm1.fidu.sp_v_u, C.sp_P_dd,
                                                   k, j, i);
  // solver loop --------------------------------------------------------------
  do
  {
    pit++;

    // state-vector non-linear subsystem --------------------------------------
    System::Z_E_F_d(pm1, dt, dotPvv, P, C, I, k, j, i);

    C.sc_E(k,j,i) = C.sc_E(k,j,i) - w_opt * C.Z_E[0];
    err_cur = std::abs(C.Z_E[0]);

    for (int a=0; a<N; ++a)
    {
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i) - w_opt * C.Z_F_d[a];
      err_cur = std::max(std::abs(C.Z_F_d[a]), err_cur);
    }

    // Ensure update preserves energy non-negativity --------------------------
    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // scale error tol by step
    // err_cur = w_opt * err_cur;

    if (err_cur > fac_PA * err_old)
    {
      // halve underrelaxation and recover old values
      w_opt = w_opt / 2;
      C.Fallback(k, j, i);
      StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

      // restart iteration
      err_old = std::numeric_limits<Real>::infinity();
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
      err_old = err_cur;
    }

  } while ((pit < iter_max) && (err_cur >= err_tol));

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // dump some information ------------------------------------------------
  if (pm1.opt_solver.verbose)
  {
    if (err_cur >= err_tol)
    {
      std::cout << "StepImplicitPicardFrozenP:\n";
      std::cout << "Tol. not achieved: " << err_cur << "\n";
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
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k, const int j, const int i)
{
  // Iterate (S_1, S_{1+k})
  const int iter_max = pm1.opt_solver.iter_max;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1.opt_solver.iter_max_rst;  // maximum restarts

  Real w_opt = pm1.opt_closure.w_opt_ini;  // underrelaxation factor
  Real e_P_abs_tol = pm1.opt_solver.eps_tol;
  Real e_C_abs_tol = pm1.opt_closure.eps_tol;
  // maximum error amplification factor between iters.
  Real fac_PA   = pm1.opt_solver.fac_err_amp;
  Real fac_PA_C = pm1.opt_closure.fac_err_amp;

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  Real e_abs_old_C = std::numeric_limits<Real>::infinity();
  Real e_abs_cur_C = 0;

  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // explicit update ----------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

  // Minerbo assembly ---------------------------------------------------------
  bool compute_limiting_P_dd = true;
  const Real Z_xi = Closures::Minerbo::Z_xi(
    C.sc_xi(k,j,i), pm1, CL, k, j, i, compute_limiting_P_dd);

  C.sc_xi(k,j,i) = std::abs(C.sc_xi(k,j,i) - w_opt * Z_xi);
  e_abs_cur_C = std::abs(Z_xi);

  compute_limiting_P_dd = false;
  Closures::Minerbo::EnforceClosureLimits(pm1, CL, k, j, i,
                                          compute_limiting_P_dd);

  // loop-lift contraction
  const Real dotPvv = Assemble::sc_ddot_dense_sp__(pm1.fidu.sp_v_u, C.sp_P_dd,
                                                   k, j, i);

  // solver loop --------------------------------------------------------------
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
      StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

      if (rit > iter_max_R)
      {
        std::ostringstream msg;
        msg << "StepImplicitPicardPC max restarts exceeded.";
        std::cout << msg.str().c_str() << std::endl;
        pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
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

  } while ((pit < iter_max) &&
            (e_abs_cur >= e_P_abs_tol));

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // dump some information ------------------------------------------------
  if (pm1.opt_solver.verbose)
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
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k, const int j, const int i)
{
  // Iterate (S_1, S_{1+k})
  const int iter_max = pm1.opt_solver.iter_max;
  int pit = 0;     // iteration counter
  int rit = 0;     // restart counter
  const int iter_max_R = pm1.opt_solver.iter_max_rst;  // maximum restarts

  Real w_opt = pm1.opt_closure.w_opt_ini;  // underrelaxation factor
  Real e_P_abs_tol = pm1.opt_solver.eps_tol;
  Real e_C_abs_tol = pm1.opt_closure.eps_tol;
  // maximum error amplification factor between iters.
  Real fac_PA   = pm1.opt_solver.fac_err_amp;
  Real fac_PA_C = pm1.opt_closure.fac_err_amp;

  Real e_abs_old = std::numeric_limits<Real>::infinity();
  Real e_abs_cur = 0;

  Real e_abs_old_C = std::numeric_limits<Real>::infinity();
  Real e_abs_cur_C = 0;

  // explicit update ----------------------------------------------------------
  const bool explicit_step_nG = false;
  StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

  // retain values for potential restarts
  C.FallbackStore(k, j, i);

  // solver loop --------------------------------------------------------------
  int iter_closure_freeze = 0;
  do
  {
    pit++;

    // Minerbo assembly -------------------------------------------------------
    if (iter_closure_freeze < iter_max)  // TODO: pass this as param
    {
      bool compute_limiting_P_dd = true;
      const Real Z_xi = Closures::Minerbo::Z_xi(
        C.sc_xi(k,j,i), pm1, CL, k, j, i, compute_limiting_P_dd);

      C.sc_xi(k,j,i) = std::abs(C.sc_xi(k,j,i) - w_opt * Z_xi);
      e_abs_cur_C = std::abs(Z_xi);

      compute_limiting_P_dd = false;
      const bool ecl = Closures::Minerbo::EnforceClosureLimits(
        pm1, CL, k, j, i, compute_limiting_P_dd
      );

      if (ecl)
        ++iter_closure_freeze;
    }

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
      StepExplicit(pm1, dt, P, C, I, S, explicit_step_nG, k, j, i);

      if (rit > iter_max_R)
      {
        std::ostringstream msg;
        msg << "StepImplicitPicardMinerboP max restarts exceeded.";
        std::cout << msg.str().c_str() << std::endl;
        pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, false);
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

  } while ((pit < iter_max) &&
            (e_abs_cur >= e_P_abs_tol));

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // dump some information ----------------------------------------------------
  if (pm1.opt_solver.verbose)
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

void SourceLimitPropagated(
  M1 * pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL)
{
  /*
  // C is new, P is previous, I is inh / src
  const Real source_limiter = pm1->opt.source_limiter;

  // C.sc_E( k,j,i) = P.sc_E( k,j,i) + dt * I.sc_E(k,j,i);


  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  M1_ILOOP3(k,j,i)
  if (pm1->MaskGet(k, j, i))
  {
    // Real Estar = P.sc_E( k,j,i) + dt * I.sc_E(k,j,i);
    // Real Enew = C.sc_E(k,j,i);
    // Real Estar = P.sc_E( k,j,i);

    Real DrE = dt * I.sc_E(k,j,i);
    Real DrFx = dt * I.sp_F_d(0,k,j,i);
    Real DrFy = dt * I.sp_F_d(1,k,j,i);
    Real DrFz = dt * I.sp_F_d(2,k,j,i);


  }
  */
}


// ============================================================================
} // namespace M1::Update
// ============================================================================

// ============================================================================
namespace M1::Update::gsl {
// ============================================================================

struct gsl_params {
  M1 & pm1;
  Real dt;
  const StateMetaVector & P;
  StateMetaVector & C;
  const StateMetaVector & I;

  // For Jacobian-based methods
  AA & J;

  const int i;
  const int j;
  const int k;
};

// convenience maps
inline void gsl_V2T_E_F_d(StateMetaVector & T,
                          const gsl_vector *U,
                          const int k, const int j, const int i)
{
  T.sc_E(k,j,i) = U->data[0];
  for (int a=0; a<N; ++a)
  {
    T.sp_F_d(a,k,j,i) = U->data[1+a];
  }
}

inline void gsl_T2V_E_F_d(gsl_vector *U,
                          const StateMetaVector & T,
                          const int k, const int j, const int i)
{
  U->data[0] = T.sc_E(k,j,i);
  for (int a=0; a<N; ++a)
  {
    U->data[1+a] = T.sp_F_d(a,k,j,i);
  }
}

// update from U based on T.Z (std::array)
inline void gsl_TZ2V_E_F_d(gsl_vector *U,
                           const StateMetaVector & T)
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
  const StateMetaVector & P = par->P;
  StateMetaVector & C = par->C;
  const StateMetaVector & I = par->I;

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  gsl_V2T_E_F_d(C, U, k, j, i);                // U->C

  System::Z_E_F_d(pm1, dt, P, C, I, k, j, i);  // updates C.Z_E, C.Z_F_d
  gsl_TZ2V_E_F_d(Z, C);                        // C.Z -> Z

  return GSL_SUCCESS;
}

int dZ_E_F_d_FrozenP(const gsl_vector *U, void * par_, gsl_matrix *J_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  M1 & pm1 = par->pm1;
  const Real dt = par->dt;
  StateMetaVector & C = par->C;

  AA & J = par->J;

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  System::dZ_E_F_d_FrozenP(pm1, dt, J, C, k, j, i);

  const int N_SYS = J.GetDim1();

  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    gsl_matrix_set(J_, a, b, J(a, b));
  }

  return GSL_SUCCESS;
}

int ZdZ_E_F_d_FrozenP(const gsl_vector *U, void * par_,
                      gsl_vector *Z, gsl_matrix *J)
{
  Z_E_F_d(U, par_, Z);
  dZ_E_F_d_FrozenP(U, par_, J);
  return GSL_SUCCESS;
}

int dZ_E_F_d_Minerbo(const gsl_vector *U, void * par_, gsl_matrix *J_)
{
  gsl_params * par = static_cast<gsl_params*>(par_);

  M1 & pm1 = par->pm1;
  const Real dt = par->dt;
  StateMetaVector & C = par->C;

  AA & J = par->J;

  const int i = par->i;
  const int j = par->j;
  const int k = par->k;

  System::dZ_E_F_d_Minerbo(pm1, dt, J, C, k, j, i);

  const int N_SYS = J.GetDim1();

  for (int a=0; a<N_SYS; ++a)
  for (int b=0; b<N_SYS; ++b)
  {
    gsl_matrix_set(J_, a, b, J(a, b));
  }

  return GSL_SUCCESS;
}

int ZdZ_E_F_d_Minerbo(const gsl_vector *U, void * par_,
                      gsl_vector *Z, gsl_matrix *J)
{
  Z_E_F_d(U, par_, Z);
  dZ_E_F_d_Minerbo(U, par_, J);
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
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k, const int j, const int i)
{
  // prepare initial guess for iteration --------------------------------------
  // See \S3.2.4 of [1]
  StepApproximateFirstOrder(pm1, dt, P, C, I, S, CL, k, j, i);

  // retain values for potential restarts
  C.FallbackStore(k, j, i);
  // --------------------------------------------------------------------------

  // GSL specific -------------------------------------------------------------
  const size_t N_SYS = 1 + N;

  // values to iterate (seed with initial guess)
  gsl_vector *U_i = gsl_vector_alloc(N_SYS);
  gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

  // select function & solver -------------------------------------------------
  AA J_;  // unused in this method

  struct gsl_params par = {pm1, dt, P, C, I, J_, i, j, k};
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
    {
      break;
    }

    // Ensure update preserves energy non-negativity
    gsl_V2T_E_F_d(C, slv->x, k, j, i);  // U_i->C

    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // compute closure with updated state
    CL.Closure(k,j,i);

    // Updated iterated vector
    gsl_T2V_E_F_d(slv->x, C, k, j, i);  // C->U_i
    gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);
  }
  while (gsl_status == GSL_CONTINUE && iter < pm1.opt_solver.iter_max);

  if ((gsl_status == GSL_ETOL)     ||
      (gsl_status == GSL_EMAXITER) ||
      (iter >= pm1.opt_solver.iter_max))
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitMinerboHybrids: ";
      std::cout << "GSL_ETOL || GSL_EMAXITER\n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }
  }
  else if ((gsl_status == GSL_ENOPROG) || (gsl_status == GSL_ENOPROGJ))
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitMinerboHybrids: ";
      std::cout << "GSL_ENOPROG || GSL_ENOPROGJ : iter " << iter << " \n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }

    // C.Fallback(k, j, i);
  }
  else if ((gsl_status != GSL_SUCCESS))
  {
    #pragma omp critical
    {
      std::cout << "StepImplicitMinerboHybrids failure: ";
      std::cout << gsl_status << "\n";
      std::cout << iter << std::endl;
    }
    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
  }

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // cleanup ------------------------------------------------------------------
  gsl_multiroot_fsolver_free(slv);
  gsl_vector_free(U_i);
}

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (Analytical Jacobian fixed P_dd)
void StepImplicitHybridsJFrozenP(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k, const int j, const int i)
{
  // prepare initial guess for iteration --------------------------------------
  // See \S3.2.4 of [1]
  StepApproximateFirstOrder(pm1, dt, P, C, I, S, CL, k, j, i);

  // retain values for potential restarts
  C.FallbackStore(k, j, i);
  // --------------------------------------------------------------------------

  // GSL specific -------------------------------------------------------------
  const size_t N_SYS = 1 + N;

  // values to iterate (seed with initial guess)
  gsl_vector *U_i = gsl_vector_alloc(N_SYS);
  gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

  // select function & solver -------------------------------------------------
  AA J(N_SYS,N_SYS);

  struct gsl_params par = {pm1, dt, P, C, I, J, i, j, k};
  gsl_multiroot_function_fdf mrf = {&Z_E_F_d,
                                    &dZ_E_F_d_FrozenP,
                                    &ZdZ_E_F_d_FrozenP,
                                    N_SYS, &par};
  gsl_multiroot_fdfsolver *slv = gsl_multiroot_fdfsolver_alloc(
    gsl_multiroot_fdfsolver_hybridsj,
    N_SYS);

  int gsl_status = gsl_multiroot_fdfsolver_set(slv, &mrf, U_i);

  // solver loop --------------------------------------------------------------
  int iter = 0;
  do
  {
    iter++;
    gsl_status = gsl_multiroot_fdfsolver_iterate(slv);

    // break on issue with solver
    if (gsl_status)
    {
      break;
    }

    // Ensure update preserves energy non-negativity
    gsl_V2T_E_F_d(C, slv->x, k, j, i);  // U_i->C

    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // Updated iterated vector
    gsl_T2V_E_F_d(slv->x, C, k, j, i);  // C->U_i
    gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);
  }
  while (gsl_status == GSL_CONTINUE && iter < pm1.opt_solver.iter_max);

  if ((gsl_status == GSL_ETOL)     ||
      (gsl_status == GSL_EMAXITER) ||
      (iter >= pm1.opt_solver.iter_max))
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitHybridsJFrozenP: ";
      std::cout << "GSL_ETOL || GSL_EMAXITER\n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }
  }
  else if ((gsl_status == GSL_ENOPROG) || (gsl_status == GSL_ENOPROGJ))
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitHybridsJFrozenP: ";
      std::cout << "GSL_ENOPROG || GSL_ENOPROGJ : iter " << iter << " \n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }

    C.Fallback(k, j, i);
  }
  else if ((gsl_status != GSL_SUCCESS))
  {
    #pragma omp critical
    {
      std::cout << "StepImplicitMinerboHybridsJ failure: ";
      std::cout << gsl_status << "\n";
      std::cout << iter << std::endl;
    }
    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
  }

  // compute closure with updated state
  CL.Closure(k,j,i);

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // cleanup ------------------------------------------------------------------
  gsl_multiroot_fdfsolver_free(slv);
  gsl_vector_free(U_i);
}

// ----------------------------------------------------------------------------
// Implicit update strategy for state vector (Analytical Jacobian fixed P_dd)
void StepImplicitHybridsJMinerbo(
  M1 & pm1,
  const Real dt,
  const StateMetaVector & P,  // previous step data
  StateMetaVector & C,        // current step
  const StateMetaVector & I,  // inhomogeneity
  SourceMetaVector & S,
  Closures::ClosureMetaVector & CL,
  const int k, const int j, const int i)
{
  // prepare initial guess for iteration --------------------------------------
  // See \S3.2.4 of [1]

  if (pm1.opt_solver.use_Neighbor && (i > pm1.mbi.il))
  {
    C.sc_E(k,j,i) = C.sc_E(k,j,i-1);
    for (int a=0; a<N; ++a)
      C.sp_F_d(a,k,j,i) = C.sp_F_d(a,k,j,i-1);
  }
  else
  {
    StepApproximateFirstOrder(pm1, dt, P, C, I, S, CL, k, j, i);
    // StepExplicit(pm1, dt, P, C, I, false, k, j, i);
    // StepImplicitHybridsJFrozenP(pm1, dt, P, C, I, CL, k, j, i);
  }

  // retain values for potential restarts
  C.FallbackStore(k, j, i);
  // --------------------------------------------------------------------------

  // GSL specific -------------------------------------------------------------
  const size_t N_SYS = 1 + N;

  // values to iterate (seed with initial guess)
  gsl_vector *U_i = gsl_vector_alloc(N_SYS);
  gsl_T2V_E_F_d(U_i, C, k, j, i);                  // C->U_i

  // select function & solver -------------------------------------------------
  AA J(N_SYS,N_SYS);

  struct gsl_params par = {pm1, dt, P, C, I, J, i, j, k};
  gsl_multiroot_function_fdf mrf = {&Z_E_F_d,
                                    &dZ_E_F_d_Minerbo,
                                    &ZdZ_E_F_d_Minerbo,
                                    N_SYS, &par};
  gsl_multiroot_fdfsolver *slv = gsl_multiroot_fdfsolver_alloc(
    gsl_multiroot_fdfsolver_hybridsj,
    N_SYS);

  int gsl_status = gsl_multiroot_fdfsolver_set(slv, &mrf, U_i);
  // do we even need to iterate or is the estimate sufficient?
  gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);

  // solver loop --------------------------------------------------------------
  int iter = 0;
  do
  {
    if (gsl_status != GSL_CONTINUE)
    {
      break;
    }

    iter++;
    gsl_status = gsl_multiroot_fdfsolver_iterate(slv);

    // break on issue with solver
    if (gsl_status)
    {
      break;
    }

    // Ensure update preserves energy non-negativity
    gsl_V2T_E_F_d(C, slv->x, k, j, i);  // U_i->C

    NonFiniteToZero(pm1, C, k, j, i);
    ApplyFloors(pm1, C, k, j, i);
    EnforceCausality(pm1, C, k, j, i);

    // compute closure with updated state
    CL.Closure(k,j,i);

    // Updated iterated vector
    gsl_T2V_E_F_d(slv->x, C, k, j, i);  // C->U_i
    gsl_status = gsl_multiroot_test_residual(slv->f, pm1.opt_solver.eps_tol);
  }
  while (gsl_status == GSL_CONTINUE && iter < pm1.opt_solver.iter_max);

  if ((gsl_status == GSL_ETOL)     ||
      (gsl_status == GSL_EMAXITER) ||
      (iter >= pm1.opt_solver.iter_max))
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitHybridsJMinerbo: ";
      std::cout << "GSL_ETOL || GSL_EMAXITER\n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }
  }
  else if (gsl_status == GSL_ENOPROG)
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitHybridsJMinerbo: ";
      std::cout << "GSL_ENOPROG : iter " << iter << "\n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }

    // C.Fallback(k, j, i);
  }
  else if (gsl_status == GSL_ENOPROGJ)
  {
    if (pm1.opt_solver.verbose)
    #pragma omp critical
    {
      std::cout << "Warning: StepImplicitHybridsJMinerbo: ";
      std::cout << "GSL_ENOPROGJ : iter " << iter << "\n";
      std::cout << "sc_chi : " << C.sc_chi(k,j,i) << "\n";
    }

    // C.Fallback(k, j, i);
  }
  else if ((gsl_status != GSL_SUCCESS))
  {
    #pragma omp critical
    {
      std::cout << "StepImplicitHybridsJMinerbo failure: ";
      std::cout << gsl_status << "\n";
      std::cout << iter << std::endl;
    }
    pm1.StatePrintPoint(C.ix_g, C.ix_s, k, j, i, true);
  }

  // Neutrino current evolution
  SolveImplicitNeutrinoCurrent(pm1, dt, P, C, I, k, j, i);

  // cleanup ------------------------------------------------------------------
  gsl_multiroot_fdfsolver_free(slv);
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
		                  AthenaArray<Real> & u_inh,
                      AthenaArray<Real> & u_src);

// ----------------------------------------------------------------------------
// Function to update the state vector
void M1::CalcUpdate(Real const dt,
                    AthenaArray<Real> & u_pre,
                    AthenaArray<Real> & u_cur,
		                AthenaArray<Real> & u_inh,
                    AthenaArray<Real> & u_src)
{
  using namespace Update;
  using namespace Update::gsl;

  using namespace Closures;

  if (opt.value_inject)
  {
    DebugValueInject(*this, u_pre, u_cur, u_inh, u_src);
  }

  // setup aliases ------------------------------------------------------------

  vars_Lab U_P { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_pre, U_P);

  vars_Lab U_C { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_cur, U_C);

  vars_Lab U_I { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u_inh, U_I);

  vars_Source U_S { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesSource(u_src, U_S);


  // dispatch integration strategy --------------------------------------------
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    StateMetaVector C = ConstructStateMetaVector(*this, U_C, ix_g, ix_s);
    StateMetaVector P = ConstructStateMetaVector(*this, U_P, ix_g, ix_s);
    StateMetaVector I = ConstructStateMetaVector(*this, U_I, ix_g, ix_s);

    SourceMetaVector S = ConstructSourceMetaVector(*this, U_S, ix_g, ix_s);

    // Apply limits to current state vector enforcing causality
    // warning: without this there are issues for E>|F|_\gamma in e.g.
    // thin-closure

    if (opt.enforce_causality)
    {
      M1_GLOOP3(k,j,i)
      if (MaskGet(k, j, i))
      {
        EnforceCausality(*this, C, k, j, i);
      }
    }

    ClosureMetaVector CL = ConstructClosureMetaVector(*this, U_C, ix_g, ix_s);

    switch (opt_solver.strategy)
    {
      case (opt_integration_strategy::full_explicit):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          ::M1::Update::AddSourceMatter(*this, C, I, S, k, j, i);
          const bool explicit_step_nG = true;
          StepExplicit(*this, dt, P, C, I, S, explicit_step_nG, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_PicardFrozenP):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitPicardFrozenP(*this, dt, P, C, I, S, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_PicardMinerboP):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitPicardMinerboP(*this, dt, P, C, I, S, CL, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_PicardMinerboPC):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitPicardMinerboPC(*this, dt, P, C, I, S, CL, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_Hybrids):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          /*
          // non-stiff limit
          if ((dt * C.sc_kap_a(k,j,i) < 1) &&
              (dt * C.sc_kap_s(k,j,i) < 1))
          {
            ::M1::Update::AddSourceMatter(*this, C, I, k, j, i);
            const bool explicit_step_nG = true;
            StepExplicit(*this, dt, P, C, I, explicit_step_nG, k, j, i);
          }
          else
          {
            StepImplicitMinerboHybrids(*this, dt, P, C, I, CL,
                                       k, j, i);
          }
          */

          StepImplicitMinerboHybrids(*this, dt, P, C, I, S, CL,
                                     k, j, i);

        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_HybridsJFrozenP):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          StepImplicitHybridsJFrozenP(
            *this, dt, P, C, I, S, CL, k, j, i);
        }
        break;
      }
      case (opt_integration_strategy::semi_implicit_HybridsJMinerbo):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          // // non-stiff limit
          // if ((dt * C.sc_kap_a(k,j,i) < 1) &&
          //     (dt * C.sc_kap_s(k,j,i) < 1))
          // {
          //   ::M1::Update::AddSourceMatter(*this, C, I, k, j, i);
          //   const bool explicit_step_nG = true;
          //   StepExplicit(*this, dt, P, C, I, explicit_step_nG, k, j, i);
          // }
          // else
          // {
          //   StepImplicitHybridsJMinerbo(
          //     *this, dt, P, C, I, CL, k, j, i);
          // }

          StepImplicitHybridsJMinerbo(
            *this, dt, P, C, I, S, CL, k, j, i);
        }
        break;
      }
      // switching between regimes
      case (opt_integration_strategy::auto_esi_HybridsJMinerbo):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          // non-stiff limit
          if ((dt * C.sc_kap_a(k,j,i) < 1) &&
              (dt * C.sc_kap_s(k,j,i) < 1))
          {
            // ::M1::Update::AddSourceMatter(*this, C, I, S, k, j, i);
            // const bool explicit_step_nG = true;
            // StepExplicit(*this, dt, P, C, I, S, explicit_step_nG, k, j, i);
            StepImplicitHybridsJMinerbo(
              *this, dt, P, C, I, S, CL, k, j, i);
          }
          else
          {
            StepImplicitHybridsJMinerbo(
              *this, dt, P, C, I, S, CL, k, j, i);
          }
        }
        break;
      }
      case (opt_integration_strategy::auto_esi_PicardMinerboP):
      {
        M1_ILOOP3(k,j,i)
        if (MaskGet(k, j, i))
        {
          // non-stiff limit
          if ((dt * C.sc_kap_a(k,j,i) < 1) &&
              (dt * C.sc_kap_s(k,j,i) < 1))
          {
            ::M1::Update::AddSourceMatter(*this, C, I, S, k, j, i);
            const bool explicit_step_nG = true;
            StepExplicit(*this, dt, P, C, I, S, explicit_step_nG, k, j, i);
          }
          else
          {
            StepImplicitPicardMinerboP(*this, dt, P, C, I, S, CL, k, j, i);
          }
        }
        break;
      }
      default:
      {
        assert(false);
        std::exit(0);
      }
    }

    // apply limiter to sources -----------------------------------------------
    // ::M1::Update::SourceLimitPropagated(this, dt, P, C, I, S, CL);
    // ------------------------------------------------------------------------

    // deal with averages

    // BD: TODO - should always br needed due to internal lab_aux.sc_n
    //            calculation?

    const bool recompute_n = opt_solver.strategy ==
                             opt_integration_strategy::full_explicit;

    M1_ILOOP3(k,j,i)
    if (MaskGet(k, j, i))
    {
      AssembleAverages(*this, C, recompute_n, k, j, i);
    }
  }

}

void DebugValueInject(M1 & pm1,
                      AthenaArray<Real> & u_pre,
                      AthenaArray<Real> & u_cur,
		                  AthenaArray<Real> & u_inh,
                      AthenaArray<Real> & u_src)
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
  pm1.AddSourceMatter(u_cur, u_inh, u_src);

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