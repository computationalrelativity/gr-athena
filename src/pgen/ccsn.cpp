#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../parameter_input.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../utils/utils.hpp"
#include "../z4c/ahf.hpp"
#include "../z4c/z4c.hpp"
#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif  // M1_ENABLED
#if MAGNETIC_FIELDS_ENABLED
#include "../field/seed_magnetic_field.hpp"
#endif

// Configuration checking
#if not FLUID_ENABLED
#error "This problem generator requires fluid (configure with -f)."
#endif

// BD: TODO- use internal factor (from EoS)
#define UDENS (6.1762691458861632e+17)

using namespace gra::aliases;

//========================================================================================
// Utilities
//========================================================================================
namespace
{

static const int IYE = 0;  // species IDX in pscalars->r/s

class StellarProfile
{
  public:
  enum vars
  {
    alp      = 0,
    gxx      = 1,
    mass     = 2,
    vel      = 3,
    rho      = 4,
    temp     = 5,
    ye       = 6,
    press    = 7,
    e        = 8,
    s        = 9,
    om       = 10,
    ab       = 11,
    phi      = 12,
    phiZ     = 13,
    num_vars = 14,
  };

  public:
  StellarProfile(ParameterInput* pin);
  ~StellarProfile();

  Real Eval(int var, Real r) const;

  private:
  int siz;
  Real* pr;
  Real* pvars[num_vars];
  Real* Phi;
};

// Deleptonization scheme by astro-ph/0504072, updated fits from 1701.02752
class Deleptonization
{
  public:
  Deleptonization(ParameterInput* pin);
  Real Ye_of_rho(Real rho) const;

  private:
  Real log10_rho1;
  Real log10_rho2;
  Real Ye_2;
  Real Ye_c;
  Real Ye_H;
};

Deleptonization* pdelept = nullptr;
enum class opt_deleptonization_method
{
  Liebendoerfer,
  Simple,
  None
};
opt_deleptonization_method opt_dlp_mtd_;

bool opt_update_conserved = false;
bool opt_update_entropy   = true;

// various parameters seeded from input file
Real opt_E_nu_avg;
Real opt_rho_trap;
Real opt_rho_cut;
Real opt_Omega_0;
Real opt_Omega_A;
Real opt_B0_amp;
Real opt_B0_rad;

// additional scalar dumps
Real MassPerMeshBlock(MeshBlock* pmb, int iout);
Real MaxMassInCell(MeshBlock* pmb, int iout);

// purely for convenience (avoid parsing sim out logs)
Real MaxLevel(MeshBlock* pmb, int iout);

// rotation laws for progenitor
Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A);
// Rotation law motivated by GRB progenitor
// https://arxiv.org/abs/1012.1853
// https://arxiv.org/abs/astro-ph/0508175
// This implementation is taken from Zelmani:
// https://bitbucket.org/zelmani/zelmani/
//   src/master/ZelmaniStarMapper/src/StarMapper_Map1D3D.F90
Real OmegaGRB_lam(Real rad, Real drtrans, Real rfe);
Real OmegaGRB(Real rad,
              Real Omega_0,
              Real Omega_A,
              Real rfe,
              Real drtrans,
              Real dropfac);

// refinement
Real opt_delta_min_m;
Real opt_delta_max_m;

enum class opt_refinement_method
{
  none,
  MassPerMeshBlock,
  MaxMassInCell,
  MaxMassInCellTracker
};
opt_refinement_method opt_refm_;

int RefinementCondition(MeshBlock* pmb);

bool interpolate_Phi;
bool interpolate_temp;
bool interpolate_offset_r;
int SP_NVARS = StellarProfile::num_vars;

// field data dumped (as user_out)
struct user_dumps
{
  enum
  {
    RefinementCondition,
    EntropyPerbaryon,
    N
  };
};

Real D_max_last     = -std::numeric_limits<Real>::infinity();
int D_max_steps_inc = 0;
int D_max_steps_dec = 0;

// ---------------------------------------------------------------------------
// Local MPI reduction helpers for bounce detection
// ---------------------------------------------------------------------------

// Return max of ph->u(IDN,...) over all rank-local MeshBlocks, then MPI-reduce
Real GlobalMaxConservedDensity(Mesh* pm)
{
  Real local_max = -std::numeric_limits<Real>::infinity();
  MeshBlock* pmb = pm->pblock;
  while (pmb != nullptr)
  {
    Hydro* ph = pmb->phydro;
    CC_ILOOP3(k, j, i)
    {
      local_max = std::max(local_max, ph->u(IDN, k, j, i));
    }
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  Real global_max;
  MPI_Allreduce(
    &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global_max;
#else
  return local_max;
#endif
}

// Return max of ph->w(IDN,...) over all rank-local MeshBlocks, then MPI-reduce
Real GlobalMaxPrimitiveDensity(Mesh* pm)
{
  Real local_max = -std::numeric_limits<Real>::infinity();
  MeshBlock* pmb = pm->pblock;
  while (pmb != nullptr)
  {
    Hydro* ph = pmb->phydro;
    CC_ILOOP3(k, j, i)
    {
      local_max = std::max(local_max, ph->w(IDN, k, j, i));
    }
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  Real global_max;
  MPI_Allreduce(
    &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global_max;
#else
  return local_max;
#endif
}

// ---------------------------------------------------------------------------
// Break-hook for bounce short-circuit
// Registered via EnrollUserMainLoopBreak; called from main.cpp each cycle.
// ---------------------------------------------------------------------------
bool BounceShortCircuit(Mesh* pm, ParameterInput* pin)
{
  if (!pin->GetOrAddBoolean("problem", "detect_bounce", false))
    return false;

  const bool post_bounce_short_circuit =
    pin->GetOrAddBoolean("problem", "post_bounce_short_circuit", false);
  if (post_bounce_short_circuit)
  {
    // Reset short-circuit for next restart
    pin->SetBoolean("problem", "post_bounce_short_circuit", false);

    // Disable future detection
    pin->SetBoolean("problem", "detect_bounce", false);

    if (Globals::my_rank == 0)
      std::cout << "Writing post_bounce restart..." << std::endl;

    // Store restart tag for main.cpp to write a tagged restart after breaking
    pin->SetString("problem", "restart_tag", "post_bounce");
    return true;
  }

  return false;
}

}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can
//  also be used to initialize variables which are global to (and therefore can
//  be passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput* pin)
{
  if (adaptive == true)
    EnrollUserRefinementCondition(RefinementCondition);

  // Register bounce break hook
  EnrollUserMainLoopBreak(BounceShortCircuit);

  // interpolate Phi? ---------------------------------------------------------
  interpolate_Phi = pin->GetOrAddBoolean("problem", "interpolate_Phi", false);
  if (!interpolate_Phi)
  {
    SP_NVARS = SP_NVARS - 1;
  }
  interpolate_temp =
    pin->GetOrAddBoolean("problem", "interpolate_temp", false);
  interpolate_offset_r =
    pin->GetOrAddBoolean("problem", "interpolate_offset_r", false);

  // Select deleptonization method --------------------------------------------
  std::ostringstream msg;

  {
    static const std::map<std::string, opt_deleptonization_method> opt_lep{
      { "Liebendoerfer", opt_deleptonization_method::Liebendoerfer },
      { "Simple", opt_deleptonization_method::Simple },
      { "None", opt_deleptonization_method::None }
    };

    auto itr = opt_lep.find(
      pin->GetOrAddString("problem", "deleptonization_method", "Simple"));

    if (itr != opt_lep.end())
    {
      opt_dlp_mtd_ = itr->second;
    }
    else
    {
      msg << "problem/deleptonization_method unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  opt_update_conserved =
    pin->GetOrAddBoolean("problem", "update_conserved", false);

  opt_update_entropy = pin->GetOrAddBoolean("problem", "update_entropy", true);

  // --------------------------------------------------------------------------

  // other params
  const Real INF = std::numeric_limits<Real>::infinity();

  opt_E_nu_avg = pin->GetOrAddReal("problem", "E_nu_avg", 10.0);
  opt_rho_trap = pin->GetOrAddReal("problem", "rho_trap", 1e12) / UDENS;
  opt_rho_cut  = pin->GetOrAddReal("problem", "rho_cut", -INF);
  opt_Omega_0  = pin->GetOrAddReal("problem", "Omega_0", 0.0);
  opt_Omega_A  = pin->GetOrAddReal("problem", "Omega_A", 0.0);
  opt_B0_amp   = pin->GetOrAddReal("problem", "B0_amp", 0.0);
  opt_B0_rad   = pin->GetOrAddReal("problem", "B0_rad", 0.0);

  // refinement strategy ------------------------------------------------------

  static const std::map<std::string, opt_refinement_method> opt_ref{
    { "none", opt_refinement_method::none },
    { "MassPerMeshBlock", opt_refinement_method::MassPerMeshBlock },
    { "MaxMassInCell", opt_refinement_method::MaxMassInCell },
    { "MaxMassInCellTracker", opt_refinement_method::MaxMassInCellTracker }
  };

  auto itr_ref =
    opt_ref.find(pin->GetOrAddString("problem", "refinement_method", "none"));

  if (itr_ref != opt_ref.end())
  {
    opt_refm_ = itr_ref->second;
  }
  else
  {
    msg << "problem/refinement_method unknown" << std::endl;
    ATHENA_ERROR(msg);
  }

  opt_delta_min_m = pin->GetOrAddReal("problem", "delta_min_m", -INF);
  opt_delta_max_m = pin->GetOrAddReal("problem", "delta_max_m", INF);
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------

  pdelept = new Deleptonization(pin);

  EnrollUserStandardHydro(pin);
  EnrollUserStandardField(pin);
  EnrollUserStandardZ4c(pin);
  EnrollUserStandardM1(pin);

  // New outputs can now be specified with the form:
  EnrollUserHistoryOutput(
    MaxMassInCell, "max_MassInCell", UserHistoryOperation::max);
  EnrollUserHistoryOutput(
    MassPerMeshBlock, "max_MassPerMB", UserHistoryOperation::max);

  EnrollUserHistoryOutput(MaxLevel, "max_level", UserHistoryOperation::max);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput* pin)
{
  AllocateUserOutputVariables(user_dumps::N);
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput* pin)
{
  MeshBlock* pmb        = this;
  Coordinates* pco      = pmb->pcoord;
  EquationOfState* peos = pmb->peos;
  Hydro* ph             = pmb->phydro;
  PassiveScalars* ps    = pmb->pscalars;

  // refinement condition -----------------------------------------------------
  switch (opt_refm_)
  {
    case opt_refinement_method::MassPerMeshBlock:
    {
      // accumulate on all physical cells
      Real M_loc = 0;
      CC_ILOOP3(k, j, i)
      {
        const Real vol = pco->GetCellVolume(k, j, i);
        M_loc += ph->u(IDN, k, j, i) * vol;
      }

      // dump on all cells
      CC_GLOOP3(k, j, i)
      {
        user_out_var(user_dumps::RefinementCondition, k, j, i) = M_loc;
      }

      break;
    }
    case opt_refinement_method::MaxMassInCell:
    case opt_refinement_method::MaxMassInCellTracker:
    {
      // look in ghost layer also as this affects dynamics on the
      // current MB
      CC_GLOOP3(k, j, i)
      {
        const Real cell_vol  = pco->GetCellVolume(k, j, i);
        const Real cell_mass = ph->u(IDN, k, j, i) * cell_vol;

        user_out_var(user_dumps::RefinementCondition, k, j, i) = cell_mass;
      }
      break;
    }
    case opt_refinement_method::none:
    {
      break;
    }
    default:
    {
      assert(false);
    }
  }

  // EntropyPerBaryon ---------------------------------------------------------
  const Real mb = peos->GetEOS().GetBaryonMass();
  Real Y[MAX_SPECIES]{ 0 };

  CC_GLOOP3(k, j, i)
  {
    const Real rho = ph->w(IDN, k, j, i);
    const Real Y_e = ps->r(IYE, k, j, i);
    Y[IYE]         = Y_e;

    const Real n = rho / mb;
    const Real p = ph->w(IPR, k, j, i);
    const Real T = peos->GetEOS().GetTemperatureFromP(n, p, Y);

    user_out_var(user_dumps::EntropyPerbaryon, k, j, i) =
      (peos->GetEOS().GetEntropyPerBaryon(n, T, Y));
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput* pin)
{
  // Read stellar profile
  StellarProfile* pstar = new StellarProfile(pin);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // Interpolate quantities to the grid
  const Real mb = peos->GetEOS().GetBaryonMass();

  for (int k = 0; k < mbi->nn3; ++k)
  {
    for (int j = 0; j < mbi->nn2; ++j)
    {
      for (int i = 0; i < mbi->nn1; ++i)
      {
        Real const xp      = mbi->x1(i);
        Real const yp      = mbi->x2(j);
        Real const zp      = mbi->x3(k);
        Real const rad     = sqrt(xp * xp + yp * yp + zp * zp);
        Real const rad_cyl = sqrt(xp * xp + yp * yp);

        const Real rho = pstar->Eval(StellarProfile::rho, rad);

        phydro->w(IDN, k, j, i) = rho;

        Real A;
        if (interpolate_Phi)
        {
          const Real Phi = pstar->Eval(StellarProfile::phi, rad);
          A              = 1.0 - 2.0 * Phi;
          alpha(k, j, i) = std::sqrt(1 + 2.0 * Phi);
        }
        else
        {
          alpha(k, j, i) = pstar->Eval(StellarProfile::alp, rad);
          A              = pstar->Eval(StellarProfile::gxx, rad);
        }

        g_dd(2, 2, k, j, i) = g_dd(1, 1, k, j, i) = g_dd(0, 0, k, j, i) = A;

        Real vr = pstar->Eval(StellarProfile::vel, rad);
        Real vx = vr * xp / rad;
        Real vy = vr * yp / rad;
        Real vz = vr * zp / rad;

        // Add rotation
        if (opt_Omega_0 > 0.0 && opt_Omega_A > 0.0)
        {
          Real const Omega = OmegaLaw(rad_cyl, opt_Omega_0, opt_Omega_A);
          vx -= Omega * yp;  // rad_cyl * sinphi;
          vy += Omega * xp;  // rad_cyl * cosphi;
        }

        Real W = 1.0 / sqrt(1.0 - A * (vx * vx + vy * vy + vz * vz));
        phydro->w(IVX, k, j, i) = W * vx;
        phydro->w(IVY, k, j, i) = W * vy;
        phydro->w(IVZ, k, j, i) = W * vz;

        // BD: enforce max species limits? (Cf. tabulated EoS)
        pscalars->r(IYE, k, j, i) =
          std::min(peos->GetEOS().GetMaximumSpeciesFraction(IYE),
                   pstar->Eval(StellarProfile::ye, rad));

        if (interpolate_temp)
        {
          const Real T = pstar->Eval(StellarProfile::temp, rad);
          phydro->derived_ms(IX_T, k, j, i) = T;

          Real Y[] = { 0. };
          Y[0]     = pscalars->r(IYE, k, j, i);

          phydro->w(IPR, k, j, i) = peos->GetEOS().GetPressure(rho / mb, T, Y);
        }
        else
        {
          phydro->w(IPR, k, j, i) = pstar->Eval(StellarProfile::press, rad);
        }

        // introduce cutoff at low density:
        // This relies on primitive flooring procedure
        if ((opt_rho_cut > 0) &&  // ensure we have a non-trivial cut
            (rho < opt_rho_cut))
        {
          phydro->w(IDN, k, j, i) = 0;
        }

        // K_dd : extrinsic curvature
        // Radial-only velocity -> K_ij ~ (r_i * r_j / r^2) * K_rr
        // where K_rr comes from the profile
        // The stellar profile provides K_rr if available
      }
    }
  }

  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);

  bool id_floor_primitives =
    pin->GetOrAddBoolean("problem", "id_floor_primitives", true);

  if (id_floor_primitives)
  {
    for (int k = 0; k < ncells3; ++k)
      for (int j = 0; j < ncells2; ++j)
        for (int i = 0; i < ncells1; ++i)
        {
          PrimHelper::ApplyPrimitiveFloors(
            peos->GetEOS(), phydro->w, pscalars->r, k, j, i);
        }
  }

  // here we have geom & primitive hydro
  // do magnetic and radiation fields, as required:

#if MAGNETIC_FIELDS_ENABLED
  // Use SeedFaceBFromEdgePotential for div(B)=0 guaranteed B-field init
  // A_i model of e.g.
  // https://arxiv.org/abs/1004.2896
  // https://arxiv.org/abs/1403.1230
  if (opt_B0_amp > 0.0 && opt_B0_rad > 0.0)
  {
    const Real B0_amp_local = opt_B0_amp;
    const Real B0_rad_local = opt_B0_rad;

    SeedFaceBFromEdgePotential(
      this,
      [=](Real x,
          Real y,
          Real z,
          Real /*p*/,
          Real /*rho*/,
          Real& Ax,
          Real& Ay,
          Real& Az)
      {
        const Real rad_cyl_sqr = SQR(x) + SQR(y);
        const Real rad_cyl     = std::sqrt(rad_cyl_sqr);
        const Real oo_rad_cyl  = 1.0 / rad_cyl;

        const Real rad    = std::sqrt(rad_cyl_sqr + SQR(z));
        const Real oo_rad = 1.0 / rad;

        // Sph2Cart Jacobian (only azimuthal component contributes)
        const Real dphdx = -y * SQR(oo_rad_cyl);
        const Real dphdy = x * SQR(oo_rad_cyl);
        const Real dphdz = 0.0;

        // A_phi dipole model
        const Real Aph = B0_amp_local * rad_cyl /
                         (std::pow(rad, 3) + std::pow(B0_rad_local, 3));

        Ax = dphdx * Aph;
        Ay = dphdy * Aph;
        Az = dphdz * Aph;
      });
  }
#endif

#if M1_ENABLED
  pm1->UpdateGeometry(pm1->geom, pm1->scratch);
  pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
  pm1->CalcFiducialVelocity();
#endif  // M1_ENABLED

  // Finalize: init geom and (M)HD evolution variables

  peos->PrimitiveToConserved(phydro->w,
                             pscalars->r,
                             pfield->bcc,
                             phydro->u,
                             pscalars->s,
                             pcoord,
                             0,
                             ncells1 - 1,
                             0,
                             ncells2 - 1,
                             0,
                             ncells3 - 1);

  // --------------------------------------------------------------------------
  // The following is now done else-where and is redundant here
  /*
  // Set up ADM matter variables
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);
  */
  // --------------------------------------------------------------------------

  // Cleanup
  delete pstar;
}

namespace
{

void Equilibriate_M1(Mesh* pm, ParameterInput* pin)
{
#if M1_ENABLED
  if (Globals::my_rank == 0)
  {
    std::printf("Imposing M1 equilibrium...\n");
  }

  int nthreads = pm->GetNumMeshThreads();

  // initialize a vector of MeshBlock pointers
  const auto& pmb_array = pm->GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nmb; ++i)
  {
    MeshBlock* pmb = pmb_array[i];
    M1::M1* pm1    = pmb->pm1;

    // update internal representations in M1 class
    pm1->UpdateGeometry(pm1->geom, pm1->scratch);
    pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
    pm1->CalcFiducialVelocity();

    M1::M1::vars_Lab U_C{ { pm1->N_GRPS, pm1->N_SPCS },
                          { pm1->N_GRPS, pm1->N_SPCS },
                          { pm1->N_GRPS, pm1->N_SPCS } };

    pm1->SetVarAliasesLab(pm1->storage.u, U_C);

    M1::M1::vars_Source U_S{ { pm1->N_GRPS, pm1->N_SPCS },
                             { pm1->N_GRPS, pm1->N_SPCS },
                             { pm1->N_GRPS, pm1->N_SPCS } };

    pm1->SetVarAliasesSource(pm1->storage.u_sources, U_S);

    M1_ILOOP3(k, j, i)
    {
      M1::Equilibrium::SetEquilibrium(*pm1, U_C, U_S, k, j, i);
    }
    // --------------------------------------------------------------------
  }
#endif  // M1_ENABLED
}

}  // namespace

void Mesh::UserWorkBeforeLoop(ParameterInput* pin)
{
  // might be useful to be able to inject equilibrium on an rst
  if (pin->GetOrAddBoolean("problem", "inject_equilibrium", false))
  {
    Equilibriate_M1(this, pin);
    pin->SetBoolean("problem", "inject_equilibrium", false);
  }

  if (!pin->GetOrAddBoolean("problem", "detect_bounce", false))
  {
    return;
  }

  // Need to detect bounce. Can use: ------------------------------------------
  // - Local density maximum
  // - Maximum density
  // - Entropy / baryon criterion

  enum class opt_bounce_detection_method
  {
    local_maximum,
    maximum,
    entropy,
    None
  };
  opt_bounce_detection_method opt_bdm;

  std::ostringstream msg;

  static const std::map<std::string, opt_bounce_detection_method> opt_bdm_{
    { "local_maximum", opt_bounce_detection_method::local_maximum },
    { "maximum", opt_bounce_detection_method::maximum },
    { "entropy", opt_bounce_detection_method::entropy },
    { "None", opt_bounce_detection_method::None }
  };

  auto itr = opt_bdm_.find(
    pin->GetOrAddString("problem", "bounce_detection_method", "None"));

  if (itr != opt_bdm_.end())
  {
    opt_bdm = itr->second;
  }
  else
  {
    msg << "problem/bounce_detection_method unknown" << std::endl;
    ATHENA_ERROR(msg);
  }

  Real opt_bdm_rho_min     = 0;
  Real opt_bdm_rho_max     = 0;
  Real opt_bdm_spb_max     = 0;
  Real opt_bdm_r_max       = 0;
  int par_D_max_steps_dec  = 0;
  Real rat_D_max_threshold = 0;

  switch (opt_bdm)
  {
    case opt_bounce_detection_method::local_maximum:
    {
      par_D_max_steps_dec =
        pin->GetOrAddInteger("problem", "D_max_steps_dec", 3);
      rat_D_max_threshold =
        pin->GetOrAddReal("problem", "rat_D_max_threshold", 0.001);
      break;
    }
    case opt_bounce_detection_method::maximum:
    {
      opt_bdm_rho_max = pin->GetOrAddReal("problem", "bdm_rho_max", 2.0e12);
      break;
    }
    case opt_bounce_detection_method::entropy:
    {
      opt_bdm_rho_min = pin->GetOrAddReal("problem", "bdm_rho_min", 1.0e10);
      opt_bdm_spb_max = pin->GetOrAddReal("problem", "bdm_spb_max", 3);
      opt_bdm_r_max   = pin->GetOrAddReal("problem", "bdm_r_max", 30.0);
      break;
    }
    case opt_bounce_detection_method::None:
    {
      break;
    }
    default:
    {
      assert(false);
    }
  }

  // --------------------------------------------------------------------------

  bool at_bounce = false;

  switch (opt_bdm)
  {
    case opt_bounce_detection_method::local_maximum:
    {
      // monitor how extrema is changing during evolution
      // Use local MPI helper instead of presc->GlobalMaximum
      const Real D_max = GlobalMaxConservedDensity(this);

      const bool do_check =
        std::abs(1 - D_max_last / D_max) > rat_D_max_threshold;

      if (do_check)
      {
        if (D_max > D_max_last)
        {
          D_max_steps_inc++;
          D_max_steps_dec = 0;
        }
        else
        {
          D_max_steps_inc = 0;
          D_max_steps_dec++;
        }
        D_max_last = D_max;
      }
      at_bounce = D_max_steps_dec >= par_D_max_steps_dec;

      if (Globals::my_rank == 0)
      {
        std::printf("do_check %d; ", do_check);
        std::printf("D_max_steps_inc %d; ", D_max_steps_inc);
        std::printf("D_max_steps_dec %d; ", D_max_steps_dec);
        std::printf("D_max %.17e\n", D_max);
      }

      break;
    }
    case opt_bounce_detection_method::maximum:
    {
      // Check all blocks- if any have density above rhomax consider @ bounce
      MeshBlock* pmb = pblock;
      Real rho_max   = 0;

      while (pmb != nullptr)
      {
        Hydro* ph = pmb->phydro;

        CC_ILOOP3(k, j, i)
        {
          Real rho_cell = ph->w(IDN, k, j, i) * UDENS;
          rho_max       = std::max(rho_max, rho_cell);
          at_bounce     = at_bounce or (rho_cell > opt_bdm_rho_max);
        }

        pmb = pmb->next;
      }

      if (at_bounce && Globals::my_rank == 0)
      {
        std::printf("rho_max [CGS] %.3e\n", rho_max);
      }

      break;
    }
    case opt_bounce_detection_method::entropy:
    {
      // Check all blocks-
      // If any point is at above rhocut with entropy above cut then @ bounce
      MeshBlock* pmb   = pblock;
      Real rho_trigger = 0;
      Real spb_trigger = 0;

      while (pmb != nullptr)
      {
        Hydro* ph = pmb->phydro;

        CC_ILOOP3(k, j, i)
        {
          Real rho_cur = ph->w(IDN, k, j, i) * UDENS;
          Real spb_cur = ph->derived_ms(IX_SPB, k, j, i);

          // Only consider cells within coordinate radius r < bdm_r_max
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);
          Real r  = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);

          bool triggered = (rho_cur > opt_bdm_rho_min) &&
                           (spb_cur > opt_bdm_spb_max) && (r < opt_bdm_r_max);

          if (triggered && !at_bounce)
          {
            // Save the first triggering cell's values for diagnostics
            rho_trigger = rho_cur;
            spb_trigger = spb_cur;
          }

          at_bounce = at_bounce or triggered;
        }

        pmb = pmb->next;
      }

      if (at_bounce && Globals::my_rank == 0)
      {
        std::printf("rho [CGS] %.3e; spb %.3e\n", rho_trigger, spb_trigger);
      }

      break;
    }
    case opt_bounce_detection_method::None:
    {
      break;
    }
    default:
    {
      assert(false);
    }
  }

    // Synchronize bounce flag across all MPI ranks so post-bounce actions
    // (deleptonization disable, M1 equilibration, restart dump) are
    // consistent.
#ifdef MPI_PARALLEL
  {
    int local_bounce  = at_bounce ? 1 : 0;
    int global_bounce = 0;
    MPI_Allreduce(
      &local_bounce, &global_bounce, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    at_bounce = (global_bounce != 0);
  }
#endif

  if (at_bounce)
  {
    if (Globals::my_rank == 0)
    {
      std::printf("Bounce detected... @ %.13e\n", time);
    }

    pin->SetReal("problem", "t_bounce", time);

    // Enforce M1 equilibrium globally
    const bool equilibriate_post_bounce =
      pin->GetOrAddBoolean("problem", "equilibriate_post_bounce", true);

    if (equilibriate_post_bounce)
    {
      Equilibriate_M1(this, pin);
    }

    // Enable M1 evolution
    pin->SetBoolean("problem", "M1_enabled", true);

    // Disable deleptonization
    opt_dlp_mtd_ = opt_deleptonization_method::None;
    pin->SetString("problem", "deleptonization_method", "None");
    if (Globals::my_rank == 0)
    {
      std::printf("Deleptonization disabled\n");
    }

    // Flag post-bounce phase
    pin->SetBoolean("problem", "post_bounce", true);

    // Flag post-bounce short-circuit (unflagged in main for restart)
    pin->SetBoolean("problem", "post_bounce_short_circuit", true);

    // Disable future detection
    // pin->SetBoolean("problem", "detect_bounce", false);
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop(ParameterInput *pin)
//  \brief Application of deleptonization scheme
//========================================================================================
void Mesh::UserWorkInLoop(ParameterInput* pin)
{
  // Triggered during: GRMHD_Z4c::UserWork;
  MeshBlock* pmb = pblock;
  Coordinates* pco;
  EquationOfState* peos = pblock->peos;
  Hydro* ph;
  PassiveScalars* ps;
  Field* pf;
  Z4c* pz4c;

  const Real E_nu_avg = opt_E_nu_avg;
  const Real rho_trap = opt_rho_trap;

  auto method_Liebendoerfer = [&]()
  {
    while (pmb != nullptr)
    {
      peos = pmb->peos;
      ph   = pmb->phydro;
      ps   = pmb->pscalars;
      pf   = pmb->pfield;
      pco  = pmb->pcoord;
      pz4c = pmb->pz4c;

      AA aux_s;
      aux_s.InitWithShallowSlice(ph->derived_ms, IX_SPB, 1);
      AA aux_T;
      aux_T.InitWithShallowSlice(ph->derived_ms, IX_T, 1);
      AA aux_h;
      aux_h.InitWithShallowSlice(ph->derived_ms, IX_ETH, 1);
      AA aux_e;
      aux_e.InitWithShallowSlice(ph->derived_ms, IX_SEN, 1);

      auto reos     = peos->GetEOS();
      const Real mb = reos.GetBaryonMass();

      AT_N_sca sqrt_detgamma(pz4c->storage.aux_extended,
                             Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma);

      CC_GLOOP3(k, j, i)
      {
        const Real rho = ph->w(IDN, k, j, i);
        Real& Y_e = ps->r(IYE, k, j, i);  // N.B. reference modified->s.v. mod.
        Real Y_old[MAX_SPECIES]{ 0 };
        Real Y_new[MAX_SPECIES]{ 0 };
        Y_old[IYE] = Y_e;

        // Electron fraction update -------------------------------------------
        const Real Y_e_bar   = pdelept->Ye_of_rho(rho);
        const Real delta_Y_e = std::min(0.0, Y_e_bar - Y_e);

        if (delta_Y_e < 0.0)
        {
          Y_e += delta_Y_e;  // N.B: Update directly to state-vector
          Y_new[IYE] = Y_e;  // needed for PrimitiveSolver

          // next need entropy update -----------------------------------------
          const Real n = rho / mb;

          if (opt_update_entropy)
          {
            // mu_nu := mu_e - mu_n + mu_p = mu_l  (CompOSE identity:
            // species potentials are mu_i = B_i*mu_b + Q_i*mu_q + L_i*mu_l;
            // the combination mu_e-mu_n+mu_p = (-mu_q+mu_l)-mu_b+(mu_b+mu_q)
            // = mu_l)
            const Real mu_nu = reos.GetElectronLeptonChemicalPotential(
              n, aux_T(k, j, i), Y_old);

            // adjust entropy based on threshold
            if ((mu_nu > E_nu_avg) &&  // MeV units
                (rho < rho_trap) &&    // code units
                (delta_Y_e < 0.0))
            {
              // aux_s (IX_SPB) only populated on output cycles.
              aux_s(k, j, i) =
                reos.GetEntropyPerBaryon(n, aux_T(k, j, i), Y_old);
              aux_s(k, j, i) -=
                delta_Y_e * (mu_nu - E_nu_avg) / aux_T(k, j, i);
            }
            else
            {
              // as above
              aux_s(k, j, i) =
                reos.GetEntropyPerBaryon(n, aux_T(k, j, i), Y_old);
            }
          }
          else
          {
            // entropy from updated fraction
            aux_s(k, j, i) =
              reos.GetEntropyPerBaryon(n, aux_T(k, j, i), Y_new);
          }

          // update derived hydro quantities
          aux_T(k, j, i) =
            reos.GetTemperatureFromEntropy(n, aux_s(k, j, i), Y_new);
          aux_h(k, j, i) = reos.GetEnthalpy(n, aux_T(k, j, i), Y_new);
          aux_e(k, j, i) =
            reos.GetSpecificInternalEnergy(n, aux_T(k, j, i), Y_new);

          if (opt_update_conserved)
          {
            // extract total energy density
            Real E = reos.GetEnergy(n, aux_T(k, j, i), Y_new);

            // Have new state (s, T, h, E) adjust conserved variables:
            ph->u(IEN, k, j, i) =
              sqrt_detgamma(k, j, i) * E - ph->u(IDN, k, j, i);
            ps->s(IYE, k, j, i) = ph->u(IDN, k, j, i) * Y_e;

            // update complementary primitive variables
            static const int coarse_flag = 0;
            peos->ConservedToPrimitive(ph->u,
                                       ph->w1,
                                       ph->w,
                                       ps->s,
                                       ps->r,
                                       pf->bcc,
                                       pco,
                                       i,
                                       i,
                                       j,
                                       j,
                                       k,
                                       k,
                                       coarse_flag);
          }
          else
          {
            // remaining primitive quantities updated:
            ph->w(IPR, k, j, i) = reos.GetPressure(n, aux_T(k, j, i), Y_new);

            // new conserved variables
            peos->PrimitiveToConserved(
              ph->w, ps->r, pf->bcc, ph->u, ps->s, pco, i, i, j, j, k, k);
          }
        }
      }

      pmb = pmb->next;
    }
  };

  auto method_Simple = [&]()
  {
    Primitive::UnitSystem* us_gs = &Primitive::GeometricSolar;
    Primitive::UnitSystem* us_nu = &Primitive::Nuclear;

    const Real us_fac_MeV2Msun = (us_nu->EnergyConversion(*us_gs)  // MeV->Msun
    );

    while (pmb != nullptr)
    {
      peos = pmb->peos;
      ph   = pmb->phydro;
      ps   = pmb->pscalars;
      pf   = pmb->pfield;
      pco  = pmb->pcoord;
      pz4c = pmb->pz4c;

      // The following are ADM quantities
      AT_N_sca alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);
      AT_N_sca sqrt_detgamma(pz4c->storage.aux_extended,
                             Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma);
      AA aux_W;
      aux_W.InitWithShallowSlice(ph->derived_ms, IX_LOR, 1);

      auto reos     = peos->GetEOS();
      const Real mb = reos.GetBaryonMass();

      CC_GLOOP2(k, j)
      {
        // update matter fields here ------------------------------------------
        CC_GLOOP1(i)
        {
          const Real rho = ph->w(IDN, k, j, i);
          const Real n   = rho / mb;
          Real& tau      = ph->u(IEN, k, j, i);
          const Real Y_e = ps->r(IYE, k, j, i);

          // Electron fraction update
          // -------------------------------------------
          const Real Y_e_bar = pdelept->Ye_of_rho(rho);

          const Real delta_Y_e = std::min(0.0, Y_e_bar - Y_e);

          if ((E_nu_avg > 0))
          {
            // reset electron fraction & update tau variable ------------------
            ps->r(IYE, k, j, i) = Y_e_bar;
            ps->s(IYE, k, j, i) = ph->u(IDN, k, j, i) * Y_e_bar;

            // E_nu_avg [MeV] -> [Msun]
            // Neutrino energy loss: delta_Y_e < 0 from electron capture,
            // so this term is negative, reducing tau.
            if (delta_Y_e < 0)
              tau +=
                (alpha(k, j, i) * sqrt_detgamma(k, j, i) * aux_W(k, j, i) * n *
                 delta_Y_e * (us_fac_MeV2Msun * E_nu_avg));

            static const int coarse_flag = 0;
            peos->ConservedToPrimitive(ph->u,
                                       ph->w1,
                                       ph->w,
                                       ps->s,
                                       ps->r,
                                       pf->bcc,
                                       pco,
                                       i,
                                       i,
                                       j,
                                       j,
                                       k,
                                       k,
                                       coarse_flag);
          }
        }
      }

      pmb = pmb->next;
    }
  };

  // delegate to selected method ----------------------------------------------
  switch (opt_dlp_mtd_)
  {
    case opt_deleptonization_method::Liebendoerfer:
    {
      method_Liebendoerfer();
      break;
    }
    case opt_deleptonization_method::Simple:
    {
      // reset Y_e, update Tau, ContoPrim
      method_Simple();
      break;
    }
    case opt_deleptonization_method::None:
    {
      break;
    }
    default:
    {
      assert(false);
    }
  }

  // Hydro state updated, need to update z4c matter
  {
    // initialize a vector of MeshBlock pointers
    const auto& pmb_array = GetMeshBlocksCached();
    FinalizeZ4cADM_Matter(pmb_array);
  }
}

void Mesh::UserWorkAfterLoop(ParameterInput* pin)
{
  delete pdelept;
  pdelept = nullptr;
}

//========================================================================================
namespace
{
//========================================================================================

StellarProfile::StellarProfile(ParameterInput* pin) : siz(0)
{
  std::string fname = pin->GetString("problem", "progenitor");
  // count number of lines in file --------------------------------------------
  std::ifstream in;
  in.open(fname.c_str());
  std::string line;
  if (!in.is_open())
  {
    std::stringstream msg;
    msg << "### FATAL ERROR problem/progenitor: " << std::string(fname) << " "
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }
  else
  {
    while (std::getline(in, line))
    {
      if (line.find("#") < line.length())
      {
        // line with comment (ignored)
      }
      else
      {
        ++siz;
      }
    }
    in.close();
  }

  // allocate & parse file ----------------------------------------------------
  pr = new Real[siz];
  for (int vi = 0; vi < SP_NVARS; ++vi)
  {
    pvars[vi] = new Real[siz];
  }

  // parse
  in.open(fname.c_str());
  int el = 0;
  while (std::getline(in, line))
  {
    if (line.find("#") < line.length())
    {
      // comment; pass line
    }
    else if (line.find(" ") < line.length())
    {
      // process elements#
      std::vector<std::string> vs;
      tokenize(line, ' ', vs);

      pr[el] = std::stod(vs[0]);

      for (int ix = StellarProfile::mass; ix < SP_NVARS; ++ix)
      {
        pvars[ix][el] = std::stod(vs[ix - StellarProfile::mass + 1]);
      }
      el++;
    }
  }
  in.close();

  // compute derived quantities -----------------------------------------------

  // Compute the metric (Newtonian limit)

  // mass at face 1 (=0 at face 0)
  Real massi = 4.0 / 3.0 * PI * std::pow(pr[1], 3) * pvars[rho][0];
  Phi        = new Real[siz];
  Phi[0]     = 0.0;
  Phi[1]     = massi / pr[1];
  for (int i = 2; i < siz; ++i)
  {
    Real dr = pr[i] - pr[i - 1];
    massi += (4.0 * PI * std::pow(pr[i - 1], 2) * pvars[rho][i - 1]) * dr;
    Phi[i] = Phi[i - 1] + (massi / (std::pow(pr[i], 2))) * dr;
  }
  Real const dPhi = Phi[siz - 1] - massi / pr[siz - 1];
  for (int i = 0; i < siz; ++i)
  {
    Phi[i] -= dPhi;
    pvars[alp][i] = sqrt(1. + 2. * Phi[i]);
    pvars[gxx][i] = 1. - 2. * Phi[i];
  }
}

StellarProfile::~StellarProfile()
{
  delete[] pr;
  for (int vi = 0; vi < SP_NVARS; ++vi)
  {
    delete[] pvars[vi];
  }
  delete[] Phi;
}

Real StellarProfile::Eval(int vi, Real rad) const
{
  assert(vi >= 0 && vi < SP_NVARS);

  Real offset = interpolate_offset_r ? pr[0] / 2.0 : 0.0;

  if (rad <= pr[0] + offset)
  {
    return pvars[vi][0];
  }
  if (rad >= pr[siz - 1] + offset)
  {
    return pvars[vi][siz - 1];
  }

  // Find the interval for interpolation
  int offset_idx = std::lower_bound(pr, pr + siz, rad - offset) - pr - 1;

  Real pr_lower = pr[offset_idx] + offset;
  Real pr_upper = pr[offset_idx + 1] + offset;

  Real lam = (rad - pr_lower) / (pr_upper - pr_lower);

  return pvars[vi][offset_idx] * (1 - lam) + pvars[vi][offset_idx + 1] * lam;
}

Deleptonization::Deleptonization(ParameterInput* pin)
{
  // Default values are SFHo from 1701.02752
  log10_rho1 = pin->GetOrAddReal("deleptonization", "log10_rho1", 7.795);
  log10_rho2 = pin->GetOrAddReal("deleptonization", "log10_rho2", 12.816);
  Ye_2       = pin->GetOrAddReal("deleptonization", "Ye_2", 0.308);
  Ye_c       = pin->GetOrAddReal("deleptonization", "Ye_c", 0.0412);
  Ye_H       = pin->GetOrAddReal("deleptonization", "Ye_H", 0.257);
}

Real Deleptonization::Ye_of_rho(Real rho) const
{
  Real const Ye_1       = 0.5;
  Real const log10_rhoH = 15;

  Real const log10_rho = log10(rho * UDENS);
  Real const x         = std::max(-1.0,
                          std::min(1.0,
                                   (2 * log10_rho - log10_rho2 - log10_rho1) /
                                     (log10_rho2 - log10_rho1)));
  Real const m         = (Ye_H - Ye_2) / (log10_rhoH - log10_rho2);

  if (log10_rho > log10_rho2)
  {
    return Ye_2 + m * (log10_rho - log10_rho2);
  }
  else
  {
    return 0.5 * (Ye_2 + Ye_1) + 0.5 * x * (Ye_2 - Ye_1) +
           Ye_c * (1 - std::abs(x) +
                   4 * std::abs(x) * (std::abs(x) - 0.5) * (std::abs(x) - 1));
  }
}

Real MassPerMeshBlock(MeshBlock* pmb, int iout)
{
  Hydro* ph        = pmb->phydro;
  Coordinates* pco = pmb->pcoord;

  // accumulate on all physical cells
  Real M_loc = 0;
  CC_ILOOP3(k, j, i)
  {
    const Real vol = pco->GetCellVolume(k, j, i);
    M_loc += ph->u(IDN, k, j, i) * vol;
  }
  return M_loc;
}

Real MaxMassInCell(MeshBlock* pmb, int iout)
{
  Hydro* ph        = pmb->phydro;
  Coordinates* pco = pmb->pcoord;

  Real max_mass = -std::numeric_limits<Real>::infinity();

  CC_GLOOP3(k, j, i)
  {
    const Real cell_vol  = pco->GetCellVolume(k, j, i);
    const Real cell_mass = ph->u(IDN, k, j, i) * cell_vol;

    max_mass = std::max(max_mass, cell_mass);
  }
  return max_mass;
}

Real MaxLevel(MeshBlock* pmb, int iout)
{
  return pmb->pmy_mesh->M_info.max_level;
}

int RefinementConditionTracker(MeshBlock* pmb)
{
  Mesh* pmesh                      = pmb->pmy_mesh;
  ExtremaTracker* ptracker_extrema = pmesh->ptracker_extrema;

  int root_level        = ptracker_extrema->root_level;
  int mb_physical_level = pmb->loc.level - root_level;

  // Iterate over refinement levels offered by trackers.
  //
  // By default if a point is not in any sphere, completely de-refine.
  int req_level = 0;

  for (int n = 1; n <= ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained = false;
    int cur_req_level = ptracker_extrema->ref_level(n - 1);

    {
      if (ptracker_extrema->ref_type(n - 1) == 0)
      {
        is_contained = pmb->PointContained(ptracker_extrema->c_x1(n - 1),
                                           ptracker_extrema->c_x2(n - 1),
                                           ptracker_extrema->c_x3(n - 1));
      }
      else if (ptracker_extrema->ref_type(n - 1) == 1)
      {
        is_contained =
          pmb->SphereIntersects(ptracker_extrema->c_x1(n - 1),
                                ptracker_extrema->c_x2(n - 1),
                                ptracker_extrema->c_x3(n - 1),
                                ptracker_extrema->ref_zone_radius(n - 1));
      }
      else if (ptracker_extrema->ref_type(n - 1) == 2)
      {
        // If any excision; activate this refinement
        bool use = false;

        // Get the minimal radius over all apparent horizons
        Real horizon_radius = std::numeric_limits<Real>::infinity();

        for (auto pah_f : pmesh->pah_finder)
        {
          if (not pah_f->IsFound())
            continue;

          if (pah_f->GetHorizonMinRadius() < horizon_radius)
          {
            horizon_radius = pah_f->GetHorizonMinRadius();
          }
          else
          {
            continue;
          }

          // populate the tracker with AHF based information
          ptracker_extrema->ref_zone_radius(n - 1) =
            (pah_f->GetHorizonMinRadius());

          use = true;
        }

        if (use)
        {
          is_contained =
            pmb->SphereIntersects(ptracker_extrema->c_x1(n - 1),
                                  ptracker_extrema->c_x2(n - 1),
                                  ptracker_extrema->c_x3(n - 1),
                                  ptracker_extrema->ref_zone_radius(n - 1));
        }
      }
    }

    if (is_contained)
    {
      req_level = std::max(cur_req_level, req_level);
    }
  }

  if (req_level > mb_physical_level)
  {
    return 1;  // currently too coarse, refine
  }
  else if (req_level == mb_physical_level)
  {
    return 0;  // level satisfied, do nothing
  }

  // otherwise de-refine
  return -1;
}

int RefinementCondition(MeshBlock* pmb)
{
  switch (opt_refm_)
  {
    case opt_refinement_method::MassPerMeshBlock:
    {
      // accumulate on all physical cells
      Real M_loc = MassPerMeshBlock(pmb, 0);

      // mass per MeshBlock exceed refinement par
      if (M_loc > opt_delta_max_m)
      {
        return 1;
      }
      else if (M_loc < opt_delta_min_m)
      {
        return -1;
      }

      break;
    }
    case opt_refinement_method::MaxMassInCell:
    {
      Real max_mass = MaxMassInCell(pmb, 0);

      if (max_mass > opt_delta_max_m)
      {
        return 1;
      }
      else if (max_mass < opt_delta_min_m)
      {
        return -1;
      }

      break;
    }
    case opt_refinement_method::MaxMassInCellTracker:
    {
      //  1: refine
      //  0: do-nothing
      // -1: coarsen
      int ref = 0;

      // Allow tracker to increase refinement level
      int ref_tr = RefinementConditionTracker(pmb);
      ref        = (ref_tr == 1) ? 1 : 0;

      if (ref == 1)
      {
        return 1;
      }

      // tracker didn't increase level:
      Real max_mass = MaxMassInCell(pmb, 0);

      if (max_mass > opt_delta_max_m)
      {
        return 1;
      }
      else if (max_mass < opt_delta_min_m)
      {
        return -1;
      }

      break;
    }
    case opt_refinement_method::none:
    {
      break;
    }
    default:
    {
      assert(false);
    }
  }

  return 0;
}

Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A)
{
  return Omega_0 / (1.0 + SQR(rad / Omega_A));
}

// Rotation law motivated by GRB progenitor
// https://arxiv.org/abs/1012.1853
// https://arxiv.org/abs/astro-ph/0508175
// This implementation is taken from Zelmani:
// https://bitbucket.org/zelmani/zelmani/
//   src/master/ZelmaniStarMapper/src/StarMapper_Map1D3D.F90
Real OmegaGRB_lam(Real rad, Real drtrans, Real rfe)
{
  return 0.5 * (1.0 + tanh((rad - rfe) / drtrans));
}

Real OmegaGRB(Real rad,
              Real Omega_0,
              Real Omega_A,
              Real rfe,
              Real drtrans,
              Real dropfac)
{
  Real const lam       = OmegaGRB_lam(rad, drtrans, rfe);
  Real const Omega_r   = OmegaLaw(rad, Omega_0, Omega_A);
  Real const Omega_rfe = OmegaLaw(rfe, Omega_0, Omega_A);
  Real const fac =
    dropfac / (1 + std::pow(std::abs(rad - rfe) / Omega_A, 1.0 / 3.0));
  return ((1.0 - lam) * Omega_r + lam * Omega_rfe / fac);
}

}  // namespace
