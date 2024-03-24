// C headers

// C++ headers
// ..

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../m1/m1.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

// ============================================================================

namespace {
// ============================================================================

int RefinementCondition(MeshBlock *pmb);

void InitM1Advection(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  const Real b_x_0 = pin->GetReal("problem", "b_x_0");
  const Real f_x_0 = pin->GetReal("problem", "f_x_0");
  const Real abs_v = pin->GetReal("problem", "abs_v");

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
    M1::AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & sc_nG  = pm1->lab.sc_nG( ix_g,ix_s);

    sc_E.ZeroClear();
    sp_F_d.ZeroClear();
    sc_nG.ZeroClear();

    M1_GLOOP3(k,j,i)
    {
      sc_E(k,j,i)     = (pm1->mbi.x1(i) < b_x_0) ? 1.0 : 0.0;
      sp_F_d(0,k,j,i) = sc_E(k,j,i);
    }
  }

  // Initialize fiducial velocity
  pm1->fidu.sp_v_u.ZeroClear();

  const Real W = 1.0 / std::sqrt(1 - SQR(abs_v));

  M1_GLOOP3(k,j,i)
  {
    pm1->hydro.sc_W(k,j,i)          = W;
    pm1->hydro.sp_w_util_u(0,k,j,i) = W * abs_v;
  }

  // assemble sp_v_u, sp_v_d etc.
  pm1->CalcFiducialVelocity();
}

void InitM1Diffusion(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  const Real b_x_a = pin->GetReal("problem", "b_x_a");
  const Real b_x_b = pin->GetReal("problem", "b_x_b");

  const Real rho   = pin->GetReal("problem", "rho");
  const Real kap_s = pin->GetReal("problem", "kap_s");

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
    M1::AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & sc_nG  = pm1->lab.sc_nG( ix_g,ix_s);

    M1::AT_C_sca & sc_kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s);

    sc_E.ZeroClear();
    sp_F_d.ZeroClear();
    sc_nG.ZeroClear();

    M1_GLOOP3(k,j,i)
    {
      const bool nz = (pm1->mbi.x1(i) >= b_x_a) && (pm1->mbi.x1(i) <= b_x_b);
      sc_E(k,j,i) = nz ? 1.0 : 0.0;
    }

    sc_kap_s.Fill(kap_s);
  }

  // Initialize fiducial velocity
  pm1->fidu.sp_v_u.ZeroClear();
  pm1->hydro.sc_w_rho.Fill(rho);
}

void InitM1DiffusionMovingMedium(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  const Real abs_v = pin->GetReal("problem", "abs_v");

  const Real rho   = pin->GetReal("problem", "rho");
  const Real kap_s = pin->GetReal("problem", "kap_s");

  // Initialize fiducial velocity
  pm1->fidu.sp_v_u.ZeroClear();

  const Real W = 1.0 / std::sqrt(1 - SQR(abs_v));

  M1_GLOOP3(k,j,i)
  {
    pm1->hydro.sc_W(k,j,i)          = W;
    pm1->hydro.sp_w_util_u(0,k,j,i) = W * abs_v;
  }

  pm1->hydro.sc_w_rho.Fill(rho);
  // assemble sp_v_u, sp_v_d etc.
  pm1->CalcFiducialVelocity();


  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
    M1::AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & sc_nG  = pm1->lab.sc_nG( ix_g,ix_s);

    M1::AT_C_sca & sc_kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s);

    sc_E.ZeroClear();
    sp_F_d.ZeroClear();
    sc_nG.ZeroClear();

    M1_GLOOP3(k,j,i)
    {
      sc_E(k,j,i) = std::exp(-9.0 * SQR(pm1->mbi.x1(i)));
      const Real J = 3.0 * sc_E(k,j,i) / (4.0 * SQR(W) - 1.0);

      sp_F_d(0,k,j,i) = 4.0 * ONE_3RD * J * SQR(W) * pm1->fidu.sp_v_d(0,k,j,i);
    }

    sc_kap_s.Fill(kap_s);
  }


}

void BCOutFlowInnerX1(MeshBlock *pmb,
                      Coordinates *pco,
                      Real time, Real dt,
                      int il, int iu,
                      int jl, int ju,
                      int kl, int ku, int ngh)
{
  // Warning: u gets called with ph->w.
  M1::M1 * pm1 = pmb->pm1;

  // Are we being called when required by M1?
  if (!pm1->enable_user_bc)
  {
    return;
  }

  M1::M1::vars_Lab U {
    {pm1->N_GRPS,pm1->N_SPCS},
    {pm1->N_GRPS,pm1->N_SPCS},
    {pm1->N_GRPS,pm1->N_SPCS}
  };
  pm1->SetVarAliasesLab(pm1->storage.u, U);

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
    M1::AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for (int i=1; i<=ngh; ++i)
      {
        U_nG(k,j,il-i) = U_nG(k,j,il);
        U_E( k,j,il-i) = U_E(k,j,il);
      }

      for (int a=0; a<M1::N; ++a)
      for (int i=1; i<=ngh; ++i)
      {
        U_F_d(a,k,j,il-i) = U_F_d(a,k,j,il);
      }
    }
  }

}

void BCShadowInnerX1(MeshBlock *pmb,
                     Coordinates *pco,
                     Real time, Real dt,
                     int il, int iu,
                     int jl, int ju,
                     int kl, int ku, int ngh)
{
  // Warning: u gets called with ph->w.
  M1::M1 * pm1 = pmb->pm1;

  // Are we being called when required by M1?
  if (!pm1->enable_user_bc)
  {
    return;
  }

  M1::M1::vars_Lab U {
    {pm1->N_GRPS,pm1->N_SPCS},
    {pm1->N_GRPS,pm1->N_SPCS},
    {pm1->N_GRPS,pm1->N_SPCS}
  };
  pm1->SetVarAliasesLab(pm1->storage.u, U);

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
    M1::AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for (int i=1; i<=ngh; ++i)
      {
        U_nG(k,j,il-i) = U_nG(k,j,il);
        U_E( k,j,il-i) = (std::abs(pm1->mbi.x2(j)) <= 1.1) ? 1.0 : 0;

        U_F_d(0,k,j,il-i) = (std::abs(pm1->mbi.x2(j)) <= 1.1) ? 1.0 : 0;
      }

      for (int a=1; a<M1::N; ++a)
      for (int i=1; i<=ngh; ++i)
      {
        U_F_d(a,k,j,il-i) = U_F_d(a,k,j,il);
      }
    }
  }

}

void InitM1Shadow(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  const Real abs_v = pin->GetReal("problem", "abs_v");

  const Real rho    = pin->GetReal("problem", "rho");
  const Real kap_a  = pin->GetReal("problem", "kap_a");
  const Real kap_s  = pin->GetReal("problem", "kap_s");

  const Real R_star = pin->GetReal("problem", "R_star");

  // Initialize fiducial velocity
  pm1->fidu.sp_v_u.ZeroClear();

  const Real W = 1.0 / std::sqrt(1 - SQR(abs_v));

  M1_GLOOP3(k,j,i)
  {
    pm1->hydro.sc_W(k,j,i)          = W;
    pm1->hydro.sp_w_util_u(0,k,j,i) = W * abs_v;
  }

  pm1->hydro.sc_w_rho.Fill(rho);
  // assemble sp_v_u, sp_v_d etc.
  pm1->CalcFiducialVelocity();

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
    M1::AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & sc_nG  = pm1->lab.sc_nG( ix_g,ix_s);

    M1::AT_C_sca & sc_kap_a = pm1->radmat.sc_kap_a(ix_g,ix_s);
    M1::AT_C_sca & sc_kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s);

    sc_E.ZeroClear();
    sp_F_d.ZeroClear();
    sc_nG.ZeroClear();

    sc_kap_s.Fill(kap_s);
    sc_kap_a.ZeroClear();

    M1_GLOOP3(k,j,i)
    {
      if ((SQR(pm1->mbi.x1(i)) + SQR(pm1->mbi.x2(j))) < R_star)
      {
        sc_kap_a(k,j,i) = kap_a;
      }
    }
  }


}

void InitM1SphereRadAbs(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  const Real abs_v = pin->GetReal("problem", "abs_v");

  const Real rho   = pin->GetReal("problem", "rho");
  const Real eta   = pin->GetReal("problem", "eta");
  const Real kap_a = pin->GetReal("problem", "kap_a");
  const Real kap_s = pin->GetReal("problem", "kap_s");

  const Real R_star = pin->GetReal("problem", "R_star");

  // Initialize fiducial velocity
  pm1->fidu.sp_v_u.ZeroClear();

  const Real W = 1.0 / std::sqrt(1 - SQR(abs_v));

  M1_GLOOP3(k,j,i)
  {
    pm1->hydro.sc_W(k,j,i)          = W;
    pm1->hydro.sp_w_util_u(0,k,j,i) = W * abs_v;
  }

  pm1->hydro.sc_w_rho.Fill(rho);
  // assemble sp_v_u, sp_v_d etc.
  pm1->CalcFiducialVelocity();

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
    M1::AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & sc_nG  = pm1->lab.sc_nG( ix_g,ix_s);

    M1::AT_C_sca & sc_eta   = pm1->radmat.sc_eta(  ix_g,ix_s);
    M1::AT_C_sca & sc_kap_a = pm1->radmat.sc_kap_a(ix_g,ix_s);
    M1::AT_C_sca & sc_kap_s = pm1->radmat.sc_kap_s(ix_g,ix_s);

    sc_E.ZeroClear();
    sp_F_d.ZeroClear();
    sc_nG.ZeroClear();

    sc_kap_s.Fill(kap_s);
    sc_kap_a.ZeroClear();

    M1_GLOOP3(k,j,i)
    {
      if ((SQR(pm1->mbi.x1(i)) + SQR(pm1->mbi.x2(j))) < R_star)
      {
        sc_E(k,j,i)     = 1.0;
        sc_eta(k,j,i)   = eta;
        sc_kap_a(k,j,i) = kap_a;
      }
    }
  }


}

void InitM1Zero(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & sc_E   = pm1->lab.sc_E(  ix_g,ix_s);
    M1::AT_N_vec & sp_F_d = pm1->lab.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & sc_nG  = pm1->lab.sc_nG( ix_g,ix_s);

    sc_E.ZeroClear();
    sp_F_d.ZeroClear();
    sc_nG.ZeroClear();
  }

  // Initialize fiducial velocity
  pm1->fidu.sp_v_u.ZeroClear();

  // assemble sp_v_u, sp_v_d etc.
  pm1->CalcFiducialVelocity();
}

// For debugging purpose
void InitM1ValueInject(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  // inject debugging values at a node
  const int I = pm1->mbi.iu-1;
  const int J = pm1->mbi.ju-1;
  const int K = pm1->mbi.kl;

  // geom -------------------------------------------------------------------
  pm1->geom.sc_alpha( K,J,I) = 1.23;

  for (int a=0; a<M1::N; ++a)
  {
    pm1->geom.sp_beta_u(a,K,J,I) = (a + 1.0) / 10.0;
  }

  for (int a=0; a<M1::N; ++a)
  for (int b=a; b<M1::N; ++b)
  {
    pm1->geom.sp_g_dd(a,b,K,J,I) = (a==b) + 0.1 * (a + b) + 0.1;
    pm1->geom.sp_K_dd(a,b,K,J,I) = -0.1*pm1->geom.sp_g_dd(a,b,K,J,I);
  }

  for (int a=0; a<M1::N; ++a)
  {
    pm1->geom.sp_dalpha_d(a,K,J,I) = 1.2 + a * 10;
  }


  for (int a=0; a<M1::N; ++a)
  for (int b=0; b<M1::N; ++b)
  {
    pm1->geom.sp_dbeta_du(a,b,K,J,I) = 1.2 + a * 10 + b;
  }


  for (int c=0; c<M1::N; ++c)
  for (int a=0; a<M1::N; ++a)
  for (int b=a; b<M1::N; ++b)
  {
    pm1->geom.sp_dg_ddd(c,a,b,K,J,I) = 1.2 + 10 * c + (a==b) + 0.1 * (a + b);
  }

  // fidu -------------------------------------------------------------------
  for (int a=0; a<M1::N; ++a)
  {
    pm1->hydro.sp_w_util_u(a,K,J,I) = -(a + 1.0) / 12;
  }

  // radiation --------------------------------------------------------------
  pm1->lab.sc_nG(0,0)(K,J,I) = 2.1;
  pm1->lab.sc_E(0,0)(K,J,I) = 1.1;

  for (int a=0; a<M1::N; ++a)
  {
    pm1->lab.sp_F_d(0,0)(a,K,J,I) = (a+1) * 2.2;
  }

  pm1->DerivedGeometry(pm1->geom, pm1->scratch);

  // opacities --------------------------------------------------------------
  pm1->radmat.sc_eta_0(0,0)(K,J,I)   = 1.3;
  pm1->radmat.sc_kap_a_0(0,0)(K,J,I) = 0.7;

  pm1->radmat.sc_eta(0,0)(K,J,I)   = 1.1;
  pm1->radmat.sc_kap_a(0,0)(K,J,I) = 0.9;
  pm1->radmat.sc_kap_s(0,0)(K,J,I) = 2.3;
}

// ============================================================================
} // namespace
// ============================================================================

// ============================================================================

namespace Kerr {
// ============================================================================

// Enable M1 if outside BH
void SetBHMask(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  const Real R_BH = pin->GetReal("problem", "R_BH");

  M1_GLOOP3(k,j,i)
  {
    const Real x1 = pm1->mbi.x1(i);
    const Real x2 = pm1->mbi.x2(j);
    const Real x3 = pm1->mbi.x3(k);
    pm1->MaskSet(SQR(x1) + SQR(x2) + SQR(x3) > SQR(R_BH),
                 k, j, i);
  }
}

// Setup the background metric
void SetADMKerrSchild(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  // required geometric quantities
  M1::AT_C_sca & sc_alpha      = pm1->geom.sc_alpha;
  M1::AT_C_sca & sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g;
  M1::AT_N_vec & sp_beta_u     = pm1->geom.sp_beta_u;
  M1::AT_N_sym & sp_g_dd       = pm1->geom.sp_g_dd;
  M1::AT_N_sym & sp_K_dd       = pm1->geom.sp_K_dd;

  std::array<Real, M1::N> l;

  // M1_GLOOP3(k,j,i)
  for (int k=0; k<pm1->mbi.nn3-1; ++k)
  for (int j=0; j<pm1->mbi.nn2-1; ++j)
  for (int i=0; i<pm1->mbi.nn1-1; ++i)
  {
    const Real x1 = pm1->mbi.x1(i);
    const Real x2 = pm1->mbi.x2(j);
    const Real x3 = pm1->mbi.x3(k);

    const Real r  = std::sqrt(SQR(x1) + SQR(x2) + SQR(x3));

    l[0] = x1 / r;
    l[1] = x2 / r;
    l[2] = x3 / r;

    sc_alpha(k,j,i) = std::pow(1.0 + 2.0 / r, -0.5);

    for (int a=0; a<M1::N; ++a)
    {
      sp_beta_u(a,k,j,i) = 2.0 / r * SQR(sc_alpha(k,j,i)) * l[a];

      for (int b=0; b<M1::N; ++b)
      {
        const Real delta_ab = (a==b) ? 1.0 : 0.0;

        sp_g_dd(a,b,k,j,i) = delta_ab + 2.0 / r * l[a] * l[b];
        sp_K_dd(a,b,k,j,i) = 2.0 * sc_alpha(k,j,i) / SQR(r) * (
          delta_ab - (2.0 + 1.0 / r) * l[a] * l[b]
        );
      }
    }
  }


}

void BCInnerX1(MeshBlock *pmb,
               Coordinates *pco,
               Real time, Real dt,
               int il, int iu,
               int jl, int ju,
               int kl, int ku, int ngh)
{
  // Warning: u gets called with ph->w.
  M1::M1 * pm1 = pmb->pm1;

  // Are we being called when required by M1?
  if (!pm1->enable_user_bc)
  {
    return;
  }

  const Real beam_A = 0.25;
  const Real beam_y = 3.5;
  // const Real beam_y = 102.5;

  // required geometric quantities
  M1::AT_C_sca & sc_alpha      = pm1->geom.sc_alpha;
  M1::AT_C_sca & sc_sqrt_det_g = pm1->geom.sc_sqrt_det_g;
  M1::AT_N_vec & sp_beta_u     = pm1->geom.sp_beta_u;
  M1::AT_N_vec & sp_beta_d     = pm1->geom.sp_beta_d;
  M1::AT_N_sym & sp_g_dd       = pm1->geom.sp_g_dd;
  M1::AT_N_sym & sp_K_dd       = pm1->geom.sp_K_dd;

  // scratches
  M1::AT_N_vec & sp_F_u_       = pm1->scratch.sp_vec_A_;

  M1::M1::vars_Lab U {
    {pm1->N_GRPS,pm1->N_SPCS},
    {pm1->N_GRPS,pm1->N_SPCS},
    {pm1->N_GRPS,pm1->N_SPCS}
  };
  pm1->SetVarAliasesLab(pm1->storage.u, U);

  for (int ix_g=0; ix_g<pm1->N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<pm1->N_SPCS; ++ix_s)
  {
    M1::AT_C_sca & U_nG   = U.sc_nG( ix_g,ix_s);
    M1::AT_N_vec & U_F_d  = U.sp_F_d(ix_g,ix_s);
    M1::AT_C_sca & U_E    = U.sc_E(  ix_g,ix_s);

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for (int i=1; i<=ngh; ++i)
      {
        if (pm1->MaskGet(k,j,il-i) &&
            (pm1->mbi.x2(j) >= beam_y-beam_A) &&
            (pm1->mbi.x2(j) <= beam_y+beam_A) &&
            (pm1->mbi.x3(k) >= -beam_A) &&
            (pm1->mbi.x3(k) <= +beam_A))
        {
          U_nG(k,j,il-i) = sc_sqrt_det_g(k,j,il-i);
          U_E( k,j,il-i) = sc_sqrt_det_g(k,j,il-i);

          Real beta2 (0);
          for (int a=0; a<M1::N; ++a)
          {
            beta2 += sp_beta_d(a,k,j,il-i) * sp_beta_u(a,k,j,il-i);
          }

          // const Real fac_x1 = U_E(k,j,il-i) / sp_g_dd(0,0,k,j,il-i) * (
          //   -sp_beta_d(0,k,j,il-i) +
          //   std::sqrt(
          //     SQR(sp_beta_d(0,k,j,il-i)) -
          //     (beta2 - 0.99 * SQR(sc_alpha(k,j,il-i)))
          //   )
          // );

          const Real fac_x1 = U_E(k,j,il-i) / sp_g_dd(0,0,k,j,il-i) * (
            -sp_beta_d(0,k,j,il-i) +
            std::sqrt(
              SQR(sp_beta_d(0,k,j,il-i)) -
              sp_g_dd(0,0,k,j,il-i) * (beta2 - 0.99 * SQR(sc_alpha(k,j,il-i)))
            )
          );

          sp_F_u_(0,il-i) = (
            fac_x1 + sp_beta_u(0,k,j,il-i) * U_E(k,j,il-i)
          ) / sc_alpha(k,j,il-i);

          for (int a=1; a<M1::N; ++a)
          {
            sp_F_u_(a,il-i) = sp_beta_u(a,k,j,il-i) *
                              U_E(k,j,il-i) / sc_alpha(k,j,il-i);
          }

          for (int a=0; a<M1::N; ++a)
          {
            U_F_d(a,k,j,il-i) = 0;
            for (int b=0; b<M1::N; ++b)
            {
              U_F_d(a,k,j,il-i) += sp_F_u_(b,il-i) * sp_g_dd(a,b,k,j,il-i);
            }
          }

        }
        else
        {
          U_nG(k,j,il-i) = 0.0;
          U_E( k,j,il-i) = 0.0;

          for (int a=0; a<M1::N; ++a)
          {
            U_F_d(a,k,j,il-i) = 0.0;
          }

        }
      }
    }
  }

}

void InitM1FixedBackgroundBeam(MeshBlock *pmb, ParameterInput *pin)
{
  M1::M1 * pm1 = pmb->pm1;

  SetADMKerrSchild(pmb, pin);

    // sliced quantities

    // required geometric quantities
    M1::AT_C_sca & sl_alpha      = pm1->geom.sc_alpha;
    M1::AT_C_sca & sl_sqrt_det_g = pm1->geom.sc_sqrt_det_g;
    M1::AT_N_vec & sl_beta_u     = pm1->geom.sp_beta_u;
    M1::AT_N_vec & sl_beta_d     = pm1->geom.sp_beta_d;
    M1::AT_N_sym & sl_g_dd       = pm1->geom.sp_g_dd;
    M1::AT_N_sym & sl_K_dd       = pm1->geom.sp_K_dd;

    // scratch quantities
    M1::AT_C_sca alpha_(pm1->mbi.nn1);

    M1::AT_N_vec beta_u_(pm1->mbi.nn1);

    M1::AT_N_sym g_dd_(pm1->mbi.nn1);
    M1::AT_N_sym K_dd_(pm1->mbi.nn1);

    M1::AT_N_D1sca dalpha_d_(pm1->mbi.nn1);

    M1::AT_N_D1vec dbeta_du_(pm1->mbi.nn1);

    M1::AT_N_D1sym dg_ddd_(pm1->mbi.nn1);

    if (1) {
      // M1: on CC --------------------------------------------------------------
      // int IL, IU, JL, JU, KL, KU;
      // pmb->pcoord->GetGeometricFieldCCIdxRanges(
      //   IL, IU,
      //   JL, JU,
      //   KL, KU);

      const int IL = pmb->is, IU = pmb->ie;
      const int JL = pmb->js, JU = pmb->je;
      const int KL = pmb->ks, KU = pmb->ke;


      for (int k=KL; k<=KU; ++k)
      for (int j=JL; j<=JU; ++j)
      {

        for(int a=0; a<NDIM; ++a)
        {
          pmb->pcoord->GetGeometricFieldDerCC(dalpha_d_, sl_alpha,  a, k, j);
          pmb->pcoord->GetGeometricFieldDerCC(dbeta_du_, sl_beta_u, a, k, j);
          pmb->pcoord->GetGeometricFieldDerCC(dg_ddd_,   sl_g_dd,   a, k, j);
        }

        // Copy to dense storage ------------------------------------------------

        for(int c = 0; c < NDIM; ++c)
        #pragma omp simd
        for (int i=IL; i<=IU; ++i)
        {
          pm1->geom.sp_dalpha_d(c,k,j,i) = dalpha_d_(c,i);
        }

        for(int c = 0; c < NDIM; ++c)
        for(int a = 0; a < NDIM; ++a)
        #pragma omp simd
        for (int i=IL; i<=IU; ++i)
        {
          pm1->geom.sp_dbeta_du(c,a,k,j,i) = dbeta_du_(c,a,i);
        }


        for(int c = 0; c < NDIM; ++c)
        for(int a = 0; a < NDIM; ++a)
        for(int b = a; b < NDIM; ++b)
        #pragma omp simd
        for (int i=IL; i<=IU; ++i)
        {
          pm1->geom.sp_dg_ddd(c,a,b,k,j,i) = dg_ddd_(c,a,b,i);
        }

      }
    }


  pm1->DerivedGeometry(pm1->geom, pm1->scratch);


  SetBHMask(pmb, pin);

}

// ============================================================================
} // namespace Kerr
// ============================================================================


void Mesh::InitUserMeshData(ParameterInput *pin)
{
  std::string m1_test = pin->GetOrAddString("problem", "test", "advection");

  if (m1_test == "shadow")
  {
    // EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BCOutFlowInnerX1);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BCShadowInnerX1);
  }
  else if (m1_test == "Kerr_fixed_background_beam")
  {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1,
                               Kerr::BCInnerX1);
  }

  if (adaptive)
  {
    EnrollUserRefinementCondition(RefinementCondition);
  }
  return;
}


void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  std::string m1_test = pin->GetOrAddString("problem", "test", "advection");

  if (m1_test == "advection")
  {
    InitM1Advection(this, pin);
  }
  else if (m1_test == "diffusion")
  {
    InitM1Diffusion(this, pin);
  }
  else if (m1_test == "diffusion_moving_medium")
  {
    InitM1DiffusionMovingMedium(this, pin);
  }
  else if (m1_test == "shadow")
  {
    InitM1Shadow(this, pin);
  }
  else if (m1_test == "sphere_radabs")
  {
    InitM1SphereRadAbs(this, pin);
  }
  else if (m1_test == "zero")
  {
    InitM1Zero(this, pin);
  }
  else if (m1_test == "Kerr_fixed_background_beam")
  {
    Kerr::InitM1FixedBackgroundBeam(this, pin);
  }
  else if (m1_test == "value_inject")
  {
    InitM1ValueInject(this, pin);
  }

}

// ============================================================================
namespace {  // impl. details
// ============================================================================

int RefinementCondition(MeshBlock *pmb)
{
  return -1;
}

// ============================================================================
} // namespace
// ============================================================================

//
// :D
//