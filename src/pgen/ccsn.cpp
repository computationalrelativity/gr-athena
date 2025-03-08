#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <map>

#include "../athena_aliases.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"
#include "../utils/linear_algebra.hpp"
#include "../z4c/z4c.hpp"
#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif  // M1_ENABLED

// BD: TODO- use internal factor (from EoS)
#define UDENS (6.1762691458861632e+17)

using namespace std;  // WTF
using namespace gra::aliases;

//========================================================================================
// Utilities
//========================================================================================
namespace {

static const int IYE = 0;  // species IDX in pscalars->r/s

void skipline(FILE *fptr);

class StellarProfile
{
  public:
    enum vars
    {
      alp = 0,
      gxx = 1,
      mass = 2,  // second quantity in table
      vel = 3,
      rho = 4,
      temp = 5,
      ye = 6,
      press = 7,
      phi = 8,
      num_vars = 9,
    };

  public:
    StellarProfile(ParameterInput *pin);
    ~StellarProfile();

    Real Eval(int var, Real r) const;

  private:
    int siz;
    Real *pr;
    Real *pvars[num_vars];
    Real *Phi;
};

// Deleptonization scheme by astro-ph/0504072, updated fits from 1701.02752
class Deleptonization
{
  public:
    Deleptonization(ParameterInput *pin);
    Real Ye_of_rho(Real rho) const;

  private:
    Real log10_rho1;
    Real log10_rho2;
    Real Ye_2;
    Real Ye_c;
    Real Ye_H;
};

Deleptonization *pdelept = nullptr;
enum class opt_deleptonization_method { Liebendoerfer, Simple, None };
opt_deleptonization_method opt_dlp_mtd_;

// various parameters seeded from input file
Real opt_E_nu_avg;
Real opt_rho_trap;
Real opt_rho_cut;
Real opt_Omega_0;
Real opt_Omega_A;
Real opt_B0_amp;
Real opt_B0_rad;

// additional scalar dumps
Real MaxRho(MeshBlock *pmb, int iout);
Real MaxTemp(MeshBlock *pmb, int iout);

// rotation laws for progenitor
Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A);
Real OmegaGRB_lam(Real rad, Real drtrans, Real rfe);
Real OmegaGRB(Real rad, Real Omega_0, Real Omega_A,
	      Real rfe,Real drtrans, Real dropfac);

#if MAGNETIC_FIELDS_ENABLED
// Initialize magnetic field
void InitMagneticFields(MeshBlock *pmb, ParameterInput *pin);

// additional scalar dumps for MHD
Real DivBface(MeshBlock *pmb, int iout);
#endif

// refinement
Real opt_delta_min_m;
Real opt_delta_max_m;

enum class opt_refinement_method { none, MassPerMeshBlock, MaxMassInCell };
opt_refinement_method opt_refm_;

int RefinementCondition(MeshBlock *pmb);

bool interpolate_Phi;
int SP_NVARS = StellarProfile::num_vars;

// field data dumped (as user_out)
struct user_dumps
{
  enum {RefinementCondition, EntropyPerbaryon, N};
};


// use for maxima computation
gra::hydro::rescaling::Rescaling * presc;

Real D_max_last = -std::numeric_limits<Real>::infinity();
int D_max_steps_inc = 0;
int D_max_steps_dec = 0;

}  // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also
//  be used to initialize variables which are global to (and therefore can be
//  passed to) other functions in this file.  Called in Mesh constructor.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (adaptive == true) EnrollUserRefinementCondition(RefinementCondition);

  // interpolate Phi? ---------------------------------------------------------
  interpolate_Phi = pin->GetOrAddBoolean("problem", "interpolate_Phi", false);
  if (!interpolate_Phi)
  {
    SP_NVARS = SP_NVARS - 1;
  }

  // Select deleptonization method --------------------------------------------
  std::ostringstream msg;
  std::string dlp_method = pin->GetOrAddString(
    "problem",
    "deleptonization_method",
    "Simple");

  static const std::map<std::string, opt_deleptonization_method> opt_lep
  {
    { "Liebendoerfer", opt_deleptonization_method::Liebendoerfer},
    { "Simple",        opt_deleptonization_method::Simple},
    { "None",          opt_deleptonization_method::None}
  };

  auto itr = opt_lep.find(pin->GetOrAddString("problem",
                                              "deleptonization_method",
                                              "Simple"));

  if (itr != opt_lep.end())
  {
    opt_dlp_mtd_ = itr->second;
  }
  else
  {
    msg << "problem/deleptonization_method unknown" << std::endl;
    ATHENA_ERROR(msg);
  }
  // --------------------------------------------------------------------------

  // other params
  const Real INF = std::numeric_limits<Real>::infinity();

  opt_E_nu_avg = pin->GetOrAddReal("problem", "E_nu_avg", 10.0);
  opt_rho_trap = pin->GetOrAddReal("problem", "rho_trap", 1e12) / UDENS;
  opt_rho_cut = pin->GetOrAddReal("problem", "rho_cut", -INF);
  opt_Omega_0 = pin->GetOrAddReal("problem", "Omega_0", 0.0);
  opt_Omega_A = pin->GetOrAddReal("problem", "Omega_A", 0.0);
  opt_B0_amp = pin->GetOrAddReal("problem", "B0_amp", 0.0);
  opt_B0_rad = pin->GetOrAddReal("problem", "B0_rad", 0.0);

  // refinement strategy ------------------------------------------------------

  static const std::map<std::string, opt_refinement_method> opt_ref
  {
    { "none",             opt_refinement_method::none},
    { "MassPerMeshBlock", opt_refinement_method::MassPerMeshBlock},
    { "MaxMassInCell",    opt_refinement_method::MaxMassInCell}
  };

  auto itr_ref = opt_ref.find(pin->GetOrAddString("problem",
                                                  "refinement_method",
                                                  "none"));

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

  // BD: TODO- cleanup is where?
  pdelept = new Deleptonization(pin);

  // additional scalars dumps
  AllocateUserHistoryOutput(2 + MAGNETIC_FIELDS_ENABLED);

  EnrollUserHistoryOutput(0, MaxRho, "max-rho", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, MaxTemp, "max-temp", UserHistoryOperation::max);

#if MAGNETIC_FIELDS_ENABLED
  EnrollUserHistoryOutput(1 + MAGNETIC_FIELDS_ENABLED,
                          DivBface, "divB", UserHistoryOperation::max);
#endif

}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  AllocateUserOutputVariables(user_dumps::N);
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  MeshBlock * pmb = this;
  Coordinates *pco = pmb->pcoord;
  EquationOfState * peos = pmb->peos;
  Hydro *ph = pmb->phydro;
  PassiveScalars *ps = pmb->pscalars;

  // refinement condition -----------------------------------------------------
  switch (opt_refm_)
  {
    case opt_refinement_method::MassPerMeshBlock:
    {
      // accumulate on all physical cells
      Real M_loc = 0;
      CC_ILOOP3(k,j,i)
      {
        const Real vol = pco->GetCellVolume(k,j,i);
        M_loc += ph->u(IDN,k,j,i) * vol;
      }

      // dump on all cells
      CC_GLOOP3(k, j, i)
      {
        user_out_var(user_dumps::RefinementCondition,k,j,i) = M_loc;
      }

      break;
    }
    case opt_refinement_method::MaxMassInCell:
    {
      // look in ghost layer also as this affects dynamics on the
      // current MB
      CC_GLOOP3(k, j, i)
      {
        const Real cell_vol  = pco->GetCellVolume(k,j,i);
        const Real cell_mass = ph->u(IDN,k,j,i) * cell_vol;

        user_out_var(user_dumps::RefinementCondition,k,j,i) = cell_mass;
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
  Real Y[MAX_SPECIES] { 0 };

  CC_GLOOP3(k, j, i)
  {
    const Real rho = ph->w(IDN,k,j,i);
    const Real Y_e = ps->r(IYE,k,j,i);
    Y[IYE] = Y_e;

    const Real n = rho / mb;
    const Real p = ph->w(IPR,k,j,i);
    const Real T = peos->GetEOS().GetTemperatureFromP(n, p, Y);

    user_out_var(user_dumps::EntropyPerbaryon,k,j,i) = (
      peos->GetEOS().GetEntropyPerBaryon(n, T, Y)
    );
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Read stellar profile
  StellarProfile *pstar = new StellarProfile(pin);

  // container with idx / grids pertaining z4c
  MB_info *mbi = &(pz4c->mbi);

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha( pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // BD: TODO - magnetic field (Cf. how this is injected in TOV/BNS)

  // Interpolate quantities to the grid
  for (int k = 0; k < mbi->nn3; ++k) {
    for (int j = 0; j < mbi->nn2; ++j) {
      for (int i = 0; i < mbi->nn1; ++i) {
        Real const xp = mbi->x1(i);
        Real const yp = mbi->x2(j);
        Real const zp = mbi->x3(k);
        Real const rad = sqrt(xp * xp + yp * yp + zp * zp);
	Real const rad_cyl = sqrt(xp * xp + yp * yp);
	  
        const Real rho = pstar->Eval(StellarProfile::rho, rad);
 
        phydro->w(IDN, k, j, i) = rho;

        Real A;
        if (interpolate_Phi)
        {
          const Real Phi = pstar->Eval(StellarProfile::phi, rad);
          A = 1.0 - 2.0 * Phi;
          alpha(k, j, i) = std::sqrt(1 + Phi);
        }
        else
        {
          alpha(k, j, i) = pstar->Eval(StellarProfile::alp, rad);
          A = pstar->Eval(StellarProfile::gxx, rad);
        }

        g_dd(2, 2, k, j, i) = g_dd(1, 1, k, j, i) = g_dd(0, 0, k, j, i) = A;

        Real vr = pstar->Eval(StellarProfile::vel, rad);
        Real vx = vr * xp / rad;
        Real vy = vr * yp / rad;
        Real vz = vr * zp / rad;

	// Add rotation
	if (opt_Omega_0 > 0.0 && opt_Omega_A > 0.0) {
	  Real const Omega = OmegaLaw(rad_cyl, opt_Omega_0, opt_Omega_A); 
	  vx -= Omega * yp; // rad_cyl * sinphi;
	  vy += Omega * xp; // rad_cyl * cosphi;
	}
	
        Real W = 1.0 / sqrt(1.0 - A * (vx * vx + vy * vy + vz * vz));
        phydro->w(IVX, k, j, i) = W * vx;
        phydro->w(IVY, k, j, i) = W * vy;
        phydro->w(IVZ, k, j, i) = W * vz;

        phydro->w(IPR, k, j, i) = pstar->Eval(StellarProfile::press, rad);

        // BD: enforce max species limits? (Cf. tabulated EoS)
        // std::printf(
        //   "%.3g\n",
        //   peos->GetEOS().GetMaximumSpeciesFraction(IYE)
        // );
        // std::exit(0);
        pscalars->r(IYE,k,j,i) = std::min(
          peos->GetEOS().GetMaximumSpeciesFraction(IYE),
          pstar->Eval(StellarProfile::ye, rad)
        );

        // introduce cutoff at low density:
        // This relies on primitive flooring procedure
        if ((opt_rho_cut > 0) &&  // ensure we have a non-trivial cut
            (rho < opt_rho_cut))
        {
          phydro->w(IDN,k,j,i) = 0;
        }

      }
    }
  }

  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);

  bool id_floor_primitives = pin->GetOrAddBoolean(
    "problem", "id_floor_primitives", true);

  if (id_floor_primitives)
  {
    for (int k = 0; k < ncells3; ++k)
    for (int j = 0; j < ncells2; ++j)
    for (int i = 0; i < ncells1; ++i)
    {
#if USETM
      peos->ApplyPrimitiveFloors(phydro->w, pscalars->r, k, j, i);
#else
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
#endif
    }
  }

  // here we wave geom & primitive hydro
  // do magnetic and radiation fields, as required:

#if MAGNETIC_FIELDS_ENABLED
  InitMagneticFields(this, pin);
#endif

#if M1_ENABLED
  pm1->UpdateGeometry(pm1->geom, pm1->scratch);
  pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
  pm1->CalcFiducialVelocity();

  /*
  pm1->CalcClosure(pm1->storage.u);
  pm1->CalcFiducialFrame(pm1->storage.u);
  pm1->CalcOpacity(pmy_mesh->dt, pm1->storage.u);

  M1::M1::vars_Lab U_C { {pm1->N_GRPS,pm1->N_SPCS},
                         {pm1->N_GRPS,pm1->N_SPCS},
                         {pm1->N_GRPS,pm1->N_SPCS} };

  pm1->SetVarAliasesLab(pm1->storage.u, U_C);

  M1::M1::vars_Source U_S { {pm1->N_GRPS,pm1->N_SPCS},
                            {pm1->N_GRPS,pm1->N_SPCS},
                            {pm1->N_GRPS,pm1->N_SPCS} };

  pm1->SetVarAliasesSource(pm1->storage.u_sources, U_S);


  M1_ILOOP3(k, j, i)
  {
    M1::Equilibrium::SetEquilibrium(*pm1, U_C, U_S, k, j, i);
  }
  */


#endif  // M1_ENABLED

  // Finalize: init geom and (M)HD evolution variables

  peos->PrimitiveToConserved(phydro->w,
                             pscalars->r,
                             pfield->bcc,
                             phydro->u,
                             pscalars->s,
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);

  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  /*
  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);
  */

  // Cleanup
  delete pstar;
}

void Mesh::UserWorkBeforeLoop(ParameterInput *pin)
{
  if (!pin->GetOrAddBoolean("problem", "detect_bounce", false))
  {
    return;
  }

  // monitor how extrema is changing during evolution
  typedef gra::hydro::rescaling::variety_cs variety_cs;
  variety_cs v_cs = variety_cs::conserved_hydro;
  // don't bother with densitized, it is just to detect extrema
  const Real D_max = presc->GlobalMaximum(v_cs, IDN, true, false);

  if (D_max > D_max_last)
  {
    D_max_steps_inc++;
  }
  else
  {
    D_max_steps_dec++;
  }
  D_max_last = D_max;

  if (Globals::my_rank==0)
  {
    std::printf("D_max_steps_inc %d\n", D_max_steps_inc);
    std::printf("D_max_steps_dec %d\n", D_max_steps_dec);
    std::printf("D_max %.3e\n", D_max);
  }

  const int par_D_max_steps_dec = pin->GetOrAddInteger(
    "problem", "D_max_steps_dec", 3
  );

  if (D_max_steps_dec >= par_D_max_steps_dec)
  {
    // Enforce M1 equilibrium globally
#if M1_ENABLED
    // ...
#endif

    // Enable M1 evolution
    pin->SetBoolean("problem", "M1_enabled", true);

    // Disable deleptonization
    opt_dlp_mtd_ = opt_deleptonization_method::None;
    pin->SetString("problem", "deleptonization_method", "None");

    // Flag post-bounce phase
    pin->SetBoolean("problem", "post_bounce", true);

    // Flag post-bounce short-circuit (unflagged in main for restart)
    pin->SetBoolean("problem", "post_bounce_short_circuit", true);
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Application of deleptonization scheme
//========================================================================================
void Mesh::UserWorkInLoop()
{
  using namespace LinearAlgebra;

  // Triggered during: GRMHD_Z4c::UserWork;

  MeshBlock * pmb = pblock;
  Coordinates * pco;
  EquationOfState * peos = pblock->peos;
  Hydro *ph;
  PassiveScalars *ps;
  Field *pf;
  Z4c * pz4c;

  const Real mb = peos->GetEOS().GetBaryonMass();
  const Real E_nu_avg = opt_E_nu_avg;
  const Real rho_trap = opt_rho_trap;

  auto method_Liebendoerfer = [&]()
  {
    while (pmb != nullptr)
    {
      peos = pmb->peos;
      ph = pmb->phydro;
      ps = pmb->pscalars;
      pf = pmb->pfield;
      pco = pmb->pcoord;

      CC_GLOOP3(k, j, i)
      {
        const Real rho = ph->w(IDN,k,j,i);
        Real & Y_e = ps->r(IYE,k,j,i);
        Real Y[MAX_SPECIES]  { 0 };
        Y[IYE] = Y_e;

        // Electron fraction update -------------------------------------------
        const Real Y_e_bar = pdelept->Ye_of_rho(rho);

        const Real delta_Y_e = std::min(
          0.0,
          Y_e_bar - Y_e
        );

        if (delta_Y_e < 0.0)
        {
          Y_e += delta_Y_e;  // N.B: Update directly to state-vector

          // next need entropy update -----------------------------------------
          const Real n = rho / mb;
          const Real p = ph->w(IPR,k,j,i);
          const Real T = peos->GetEOS().GetTemperatureFromP(n, p, Y);
          Real s = peos->GetEOS().GetEntropyPerBaryon(n, T, Y);

          // Prepare chemical potentials (BD: TODO- check)
          // Should read mu_nu := mu_e - mu_n + mu_p
          const Real mu_nu = (
            peos->GetEOS().GetElectronLeptonChemicalPotential(n, T, Y) // -
            // peos->GetEOS().GetBaryonChemicalPotential(n, T, Y) +
            // peos->GetEOS().GetChargeChemicalPotential(n, T, Y)
          );

          if ((mu_nu > E_nu_avg) &&  // MeV units
              (rho < rho_trap))      // code units
          {
            s -= delta_Y_e * (mu_nu - E_nu_avg) / T;
          }

          // Y_e has been updated
          Y[IYE] = Y_e;
          const Real Tnew = peos->GetEOS().GetTemperatureFromEntropy(n, s, Y);
          // const Real e = peos->GetEOS().GetEnergy(n, Tnew, Y);
          const Real pnew = peos->GetEOS().GetPressure(n, Tnew, Y);

          // push back to prim state vector:
          ph->w(IPR,k,j,i) = pnew;
          ph->temperature(k,j,i) = Tnew;

          peos->PrimitiveToConserved(
            ph->w,
            ps->r,
            pf->bcc,
            ph->u,
            ps->s,
            pco,
            i, i,
            j, j,
            k, k
          );
        }
      }

      pmb = pmb->next;
    }
  };

  auto method_Simple = [&]()
  {
    // prepare scratches
    AT_N_sca alpha_(pblock->ncells1);
    AT_N_sym gamma_dd_(pblock->ncells1);
    AT_N_sym gamma_uu_(pblock->ncells1);
    AT_N_sca sqrt_detgamma_(pblock->ncells1);
    AT_N_vec w_util_u_(     pblock->ncells1);
    AT_C_sca W_(pblock->ncells1);

    Primitive::UnitSystem *us_gs = &Primitive::GeometricSolar;
    Primitive::UnitSystem *us_nu = &Primitive::Nuclear;

    const Real us_fac_MeV2Msun = (
      us_nu->EnergyConversion(*us_gs) // MeV->Msun
    );

    while (pmb != nullptr)
    {
      peos = pmb->peos;
      ph = pmb->phydro;
      ps = pmb->pscalars;
      pz4c = pmb->pz4c;

      AT_N_sym sl_adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
      AT_N_sca sl_adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);


      CC_GLOOP2(k, j)
      {
        // prepare geometric & derived ----------------------------------------
        for (int a=0; a<N; ++a)
        {
          CC_GLOOP1(i)
          {
            w_util_u_(a,i) = ph->w(IVX+a,k,j,i);
          }

          for (int b=a; b<N; ++b)
          CC_GLOOP1(i)
          {
            gamma_dd_(a,b,i) = sl_adm_gamma_dd(a,b,k,j,i);
          }
        }
        CC_GLOOP1(i)
        {
          alpha_(i) = sl_adm_alpha(k,j,i);
        }

        // Prepare determinant-like
        CC_GLOOP1(i)
        {
          const Real detgamma__ = Det3Metric(gamma_dd_, i);
          sqrt_detgamma_(i) = std::sqrt(detgamma__);

          const Real norm2_utilde__ = InnerProductSlicedVec3Metric(
            w_util_u_, gamma_dd_, i
          );
          W_(i) = std::sqrt(1. + norm2_utilde__);
        }

        // update matter fields here ------------------------------------------
        CC_GLOOP1(i)
        {
          const Real rho = ph->w(IDN,k,j,i);
          const Real n = rho / mb;
          Real & tau = ph->u(IEN,k,j,i);
          const Real Y_e = ps->r(IYE,k,j,i);

          // Electron fraction update -------------------------------------------
          const Real Y_e_bar = pdelept->Ye_of_rho(rho);

          const Real delta_Y_e = std::min(
            0.0,
            Y_e_bar - Y_e
          );

          // reset electron fraction & update tau variable ----------------------
          ps->r(IYE,k,j,i) = Y_e_bar;

          // E_nu_avg [MeV] -> [Msun]
          tau -= (alpha_(i) * sqrt_detgamma_(i) * W_(i) * n *
                  delta_Y_e * (us_fac_MeV2Msun * E_nu_avg));

          // double check (simple-simple)
          if ((delta_Y_e < 0) && (E_nu_avg > 0))
          {
            static const int coarse_flag = 0;
            peos->ConservedToPrimitive(
              ph->u,
              ph->w1,
              ph->w,
              ps->s,
              ps->r,
              pf->bcc,
              pco,
              i, i,
              j, j,
              k, k,
              coarse_flag
            );
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
      // update Y_e, entropy call, PrimToCon
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

}

//========================================================================================
namespace {
//========================================================================================

void skipline(FILE *fptr) {
  char c;
  do {
    c = fgetc(fptr);
  } while (c != '\n');
}

StellarProfile::StellarProfile(ParameterInput *pin) : siz(0) {
  string fname = pin->GetString("problem", "progenitor");
  // count number of lines in file --------------------------------------------
  std::ifstream in;
  in.open(fname.c_str());
  std::string line;
  if (!in.is_open())
  {
    stringstream msg;
    msg << "### FATAL ERROR problem/progenitor: " << string(fname) << " "
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

      for (int ix=mass; ix<SP_NVARS; ++ix)
      {
        pvars[ix][el] = std::stod(vs[ix-mass+1]);
      }
      el++;
    }
  }
  in.close();

  // compute derived quantities -----------------------------------------------

  // Compute the metric (Newtonian limit)

  // mass at face 1 (=0 at face 0)
  Real massi = 4.0/3.0 * PI * pow(pr[1],3) * pvars[rho][0];
  Phi = new Real[siz];
  Phi[0] = 0.0;
  Phi[1] = massi/pr[1];
  for (int i = 2; i < siz; ++i) {
    Real dr = pr[i] - pr[i-1];
    massi += ( 4.0 * PI * pow(pr[i-1], 2) * pvars[rho][i-1] )*dr;
    Phi[i] = Phi[i-1] + ( massi /( pow(pr[i], 2) ) ) * dr;
  }
  Real const dPhi = Phi[siz-1] - massi/pr[siz-1];
  for (int i = 0; i < siz; ++i) {
    Phi[i] -= dPhi; 
    pvars[alp][i] = sqrt(1. + 2. * Phi[i]);
    pvars[gxx][i] = 1. - 2. * Phi[i];
   }
}

StellarProfile::~StellarProfile() {
  delete[] pr;
  for (int vi = 0; vi < SP_NVARS; ++vi) {
    delete[] pvars[vi];
  }
  delete[] Phi;
}

Real StellarProfile::Eval(int vi, Real rad) const {
  assert(vi >= 0 && vi < SP_NVARS);
  if (rad <= pr[0]) {
    return pvars[vi][0];
  }
  if (rad >= pr[siz - 1]) {
    return pvars[vi][siz - 1];
  }
  int offset = lower_bound(pr, pr + siz, rad) - pr - 1;
  Real lam = (rad - pr[offset]) / (pr[offset + 1] - pr[offset]);
  return pvars[vi][offset] * (1 - lam) + pvars[vi][offset + 1] * lam;
}

Deleptonization::Deleptonization(ParameterInput *pin) {
  // Default values are SFHo from 1701.02752
  log10_rho1 = pin->GetOrAddReal("deleptonization", "log10_rho1", 7.795);
  log10_rho2 = pin->GetOrAddReal("deleptonization", "log10_rho2", 12.816);
  Ye_2 = pin->GetOrAddReal("deleptonization", "Ye_2", 0.308);
  Ye_c = pin->GetOrAddReal("deleptonization", "Ye_c", 0.0412);
  Ye_H = pin->GetOrAddReal("deleptonization", "Ye_H", 0.257);
}

Real Deleptonization::Ye_of_rho(Real rho) const {
  Real const Ye_1 = 0.5;
  Real const log10_rhoH = 15;

  Real const log10_rho = log10(rho * UDENS);
  Real const x = max(-1.0, min(1.0, (2 * log10_rho - log10_rho2 - log10_rho1) /
                                        (log10_rho2 - log10_rho1)));
  Real const m = (Ye_H - Ye_2) / (log10_rhoH - log10_rho2);

  if (log10_rho > log10_rho2) {
    return Ye_2 + m * (log10_rho - log10_rho2);
  } else {
    return 0.5 * (Ye_2 + Ye_1) + 0.5 * x * (Ye_2 - Ye_1) +
           Ye_c * (1 - abs(x) + 4 * abs(x) * (abs(x) - 0.5) * (abs(x) - 1));
  }
}

int RefinementCondition(MeshBlock *pmb)
{
  Hydro *ph = pmb->phydro;
  Coordinates *pco = pmb->pcoord;

  switch (opt_refm_)
  {
    case opt_refinement_method::MassPerMeshBlock:
    {
      // accumulate on all physical cells
      Real M_loc = 0;
      CC_ILOOP3(k,j,i)
      {
        const Real vol = pco->GetCellVolume(k,j,i);
        M_loc += ph->u(IDN,k,j,i) * vol;
      }

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
      Real max_mass = -std::numeric_limits<Real>::infinity();

      CC_GLOOP3(k, j, i)
      {
        const Real cell_vol  = pco->GetCellVolume(k,j,i);
        const Real cell_mass = ph->u(IDN,k,j,i) * cell_vol;

        max_mass = std::max(max_mass, cell_mass);
      }

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

Real MaxRho(MeshBlock *pmb, int iout)
{
  Real max_rho = 0.0;
  AthenaArray<Real> &w = pmb->phydro->w;

  CC_ILOOP3(k, j, i)
  {
    max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
  }

  return max_rho;
}

Real MaxTemp(MeshBlock *pmb, int iout)
{
  Real max_T = -std::numeric_limits<Real>::infinity();
  CC_ILOOP3(k, j, i)
  {
    max_T = std::max(max_T, pmb->phydro->temperature(k,j,i));
  }
  return max_T;
}

Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A)
{
  return Omega_0/(1.0 + SQR(rad/Omega_A));
}

// Rotation law motivated by GRB progenitor
// https://arxiv.org/abs/1012.1853
// https://arxiv.org/abs/astro-ph/0508175
// This implementation is taken from Zelmani:
// https://bitbucket.org/zelmani/zelmani/
//   src/master/ZelmaniStarMapper/src/StarMapper_Map1D3D.F90
Real OmegaGRB_lam(Real rad, Real drtrans, Real rfe)
{
  return 0.5*(1.0 + tanh((rad - rfe)/drtrans));
}

Real OmegaGRB(Real rad, Real Omega_0, Real Omega_A,
	      Real rfe,Real drtrans, Real dropfac)
{ 
  Real const lam = OmegaGRB_lam(rad, drtrans, rfe);
  Real const Omega_r = OmegaLaw(rad, Omega_0, Omega_A);
  Real const Omega_rfe = OmegaLaw(rfe, Omega_0, Omega_A);
  Real const fac = dropfac/(1 + pow(abs(rad-rfe)/Omega_A, 1.0/3.0));  
  return ( (1.0-lam) * Omega_r + lam* Omega_rfe/fac );
}


#if MAGNETIC_FIELDS_ENABLED

void InitMagneticFields(MeshBlock *pmb, ParameterInput *pin)
{
  GRDynamical * pcoord { static_cast<GRDynamical*>(pmb->pcoord) };
  Field * pfield { pmb->pfield };

  // Initialize magnetic field
  // No metric weighting here
  pfield->b.x1f.ZeroClear();
  pfield->b.x2f.ZeroClear();
  pfield->b.x3f.ZeroClear();
  pfield->bcc.ZeroClear();

  AthenaArray<Real> Acc(NFIELD,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  // Initialize cell centred potential
  for (int k=0; k<pmb->ncells3; k++)
  for (int j=0; j<pmb->ncells2; j++)
  for (int i=0; i<pmb->ncells1; i++) {
    const Real zp = pcoord->x3v(k);
    const Real yp = pcoord->x2v(j);
    const Real xp = pcoord->x1v(i);

    const Real rad_cyl_sqr = SQR(xp) + SQR(yp);
    const Real rad_cyl = sqrt(rad_cyl_sqr);
    const Real oo_rad_cyl = 1.0/rad_cyl;

    const Real rad = sqrt( rad_cyl_sqr + SQR(zp) );
    const Real oo_rad = 1.0/rad;

    // Sph2Cart Jacobian
    const Real drdx = xp * oo_rad;
    const Real drdy = yp * oo_rad;
    const Real drdz = zp * oo_rad;

    const Real dthdx = (xp * zp) * SQR(oo_rad) * oo_rad_cyl;
    const Real dthdy = (yp * zp) * SQR(oo_rad) * oo_rad_cyl;
    const Real dthdz = - rad_cyl * SQR(oo_rad);

    const Real dphdx = - yp * SQR(oo_rad_cyl);
    const Real dphdy = xp * SQR(oo_rad_cyl);
    const Real dphdz = 0.0;

    // A_i model of e.g.
    // https://arxiv.org/abs/1004.2896
    // https://arxiv.org/abs/1403.1230
    Real Ar = 0.0;
    Real Ath = 0.0;
    Real Aph = 0.0;
    if (opt_B0_amp > 0.0 && opt_B0_rad > 0.0) {
      Aph = opt_B0_amp * rad_cyl /( pow(rad,3) + pow(opt_B0_rad,3) );
    }

    Acc(0,k,j,i) = drdx * Ar + dthdx * Ath + dphdx * Aph;
    Acc(1,k,j,i) = drdy * Ar + dthdy * Ath + dphdy * Aph;
    Acc(2,k,j,i) = drdz * Ar + dthdz * Ath + dphdz * Aph;

  }

  // Construct cell centred B field from cell centred potential
  for(int k=pmb->ks-1; k<=pmb->ke+1; k++)
  for(int j=pmb->js-1; j<=pmb->je+1; j++)
  for(int i=pmb->is-1; i<=pmb->ie+1; i++)
  {
    const Real dx1 = pcoord->dx1v(i);
    const Real dx2 = pcoord->dx2v(j);
    const Real dx3 = pcoord->dx3v(k);

    /*
    pfield->bcc(0,k,j,i) = -((Acc(1,k+1,j,i) - Acc(1,k-1,j,i))/(2.0*dx3));
    pfield->bcc(1,k,j,i) =  ((Acc(0,k+1,j,i) - Acc(0,k-1,j,i))/(2.0*dx3));
    pfield->bcc(2,k,j,i) =  ((Acc(1,k,j,i+1) - Acc(1,k,j,i-1))/(2.0*dx1) -
                             (Acc(0,k,j+1,i) - Acc(0,k,j-1,i))/(2.0*dx2));
    */

    const Real F0 = 1.0 / (2.0 * dx1);
    const Real F1 = 1.0 / (2.0 * dx2);
    const Real F2 = 1.0 / (2.0 * dx3);

    const Real d1A0 = F0 * (Acc(0,k,j+1,i)-Acc(0,k,j-1,i));
    const Real d2A0 = F2 * (Acc(0,k+1,j,i)-Acc(0,k-1,j,i));

    const Real d0A1 = F0 * (Acc(1,k,j,i+1)-Acc(1,k,j,i-1));
    const Real d2A1 = F2 * (Acc(1,k+1,j,i)-Acc(1,k-1,j,i));

    const Real d0A2 = F0 * (Acc(2,k,j,i+1)-Acc(2,k,j,i-1));
    const Real d1A2 = F1 * (Acc(2,k,j+1,i)-Acc(2,k,j-1,i));

    pfield->bcc(0,k,j,i) = d1A2-d2A1;
    pfield->bcc(1,k,j,i) = d2A0-d0A2;
    pfield->bcc(2,k,j,i) = d0A1-d1A0;
  }

  // Initialise face centred field by averaging cc field
  for(int k=pmb->ks; k<=pmb->ke;   k++)
  for(int j=pmb->js; j<=pmb->je;   j++)
  for(int i=pmb->is; i<=pmb->ie+1; i++)
  {
    pfield->b.x1f(k,j,i) = 0.5*(pfield->bcc(0,k,j,i-1) +
                                pfield->bcc(0,k,j,i));
  }

  for(int k=pmb->ks; k<=pmb->ke;   k++)
  for(int j=pmb->js; j<=pmb->je+1; j++)
  for(int i=pmb->is; i<=pmb->ie;   i++)
  {
    pfield->b.x2f(k,j,i) = 0.5*(pfield->bcc(1,k,j-1,i) +
                                pfield->bcc(1,k,j,i));
  }

  for(int k=pmb->ks; k<=pmb->ke+1; k++)
  for(int j=pmb->js; j<=pmb->je;   j++)
  for(int i=pmb->is; i<=pmb->ie;   i++)
  {
    pfield->b.x3f(k,j,i) = 0.5*(pfield->bcc(2,k-1,j,i) +
                                pfield->bcc(2,k,j,i));
  }
}


Real DivBface(MeshBlock *pmb, int iout) {
  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += ((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/ dx +
                 (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/ dy +
                 (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/ dz) * vol;
      }
    }
  }
  return divB;
}

#endif

}  // namespace
