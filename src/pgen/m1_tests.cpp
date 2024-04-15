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


// ============================================================================
} // namespace
// ============================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
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