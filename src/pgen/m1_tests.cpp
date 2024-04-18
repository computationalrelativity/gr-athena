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
                      AthenaArray<Real> &u,
                      FaceField &b,
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
                     AthenaArray<Real> &u,
                     FaceField &b,
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

// ============================================================================
} // namespace
// ============================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  std::string m1_test = pin->GetOrAddString("problem", "test", "advection");

  if (m1_test == "shadow")
  {
    // EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BCOutFlowInnerX1);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BCShadowInnerX1);
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