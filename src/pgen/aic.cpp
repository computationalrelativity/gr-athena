//========================================================================================
// Problem : Accretion-Induced Collapse (AIC)
//========================================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include <vector>
#include <fstream>
#include <algorithm>

// Athena++ core headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/seed_magnetic_field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

// General Relativity & Neutrino Transport headers
#include "../z4c/ahf.hpp"
#include "../z4c/z4c.hpp"
#if M1_ENABLED
#include "../m1/m1.hpp"
#include "../m1/m1_set_equilibrium.hpp"
#endif

// RNS library for rotating initial data
#include "RNS.h"  

#if not FLUID_ENABLED
#error "This problem generator requires fluid (configure with -f)."
#endif

// Base unit for Density (CGS to Code Units conversion from ccsn.cpp)
#define UDENS (6.1762691458861632e+17)

using namespace std;
using namespace gra::aliases;

//========================================================================================
// Global Variables, Classes, and Function Declarations
//========================================================================================
namespace {

  // --- 1. RNS Initial Data Variables ---
  static ini_data* rns_data;
  Primitive::ColdEOS<Primitive::COLDEOS_POLICY>* ceos = NULL;
  Real mb_rnsc = 931.191715903434;  // RNSC uses this mass factor

  // --- 2. Deleptonization (Electron Capture) Variables ---
  static const int IYE = 0;  // Index in passive scalars array for Electron Fraction (Y_e)

  class Deleptonization {
    public:
      Deleptonization(ParameterInput* pin);
      Real Ye_of_rho(Real rho) const;
    private:
      std::vector<Real> rho_table_;
      std::vector<Real> Ye_table_;

      Real log10_rho1, log10_rho2, Ye_2, Ye_c, Ye_H;
  };

  Deleptonization* pdelept = nullptr;

  enum class opt_deleptonization_method { Liebendoerfer, Simple, None };
  opt_deleptonization_method opt_dlp_mtd_;
  
  bool use_ye_of_rho_table = true;
  bool opt_update_conserved = false;
  bool opt_update_entropy   = true;

  Real opt_E_nu_avg;
  Real opt_rho_trap;
  Real opt_rho_cut;

  Real fac_MeVfm3_code;          // n [fm^-3] * E [MeV] -> code energy density

  // --- 3. Magnetic Field Variables ---
  Real opt_B0_amp;
  Real opt_B0_rad;

  // --- 4. Adaptive Mesh Refinement (AMR) Variables ---
  Real opt_delta_min_m;
  Real opt_delta_max_m;
  
  enum class opt_refinement_method { none, MassPerMeshBlock, MaxMassInCell, MaxMassInCellTracker };
  opt_refinement_method opt_refm_;

  // --- 5. Bounce Detection Variables ---
  Real D_max_last = -std::numeric_limits<Real>::infinity();
  int D_max_steps_inc = 0;
  int D_max_steps_dec = 0;

  // --- Function Declarations (To be defined later) ---
  int RefinementCondition(MeshBlock* pmb);
  bool BounceShortCircuit(Mesh* pm, ParameterInput* pin);
  Real MaxMassInCell(MeshBlock* pmb, int iout);
  Real MassPerMeshBlock(MeshBlock* pmb, int iout);
  Real MaxLevel(MeshBlock* pmb, int iout);
  void SeedMagneticFields(MeshBlock* pmb, ParameterInput* pin);
  void SeedSuwaVarmaMagneticFields(MeshBlock* pmb, ParameterInput* pin);
  void Equilibriate_M1(Mesh* pm, ParameterInput* pin);

  // Field data dumped (for user output / visualization)
  struct user_dumps {
    enum { RefinementCondition, EntropyPerbaryon, N };
  };

} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Initializes problem-specific data (RNS, AMR, Deleptonization) globally.
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput* pin)
{
  // 1. Enroll standard Athena++ Physics
  EnrollUserStandardHydro(pin);
  EnrollUserStandardField(pin);
  EnrollUserStandardZ4c(pin);
  EnrollUserStandardM1(pin);

  // 2. Initialize RNS Solver for White Dwarf (from gr_rns.cpp)
  if (!resume_flag) { // Only run if starting from t=0 (not a restart)
    string set_name = "problem";
    RNS_params_set_default();
    // Read the parameter file (e.g., tovgamma2.par)
    string inputfile = pin->GetOrAddString("problem", "filename", "wd_rns.par");
    RNS_params_set_inputfile((char*)inputfile.c_str());
    
    // Command RNS library to calculate the 2D rotating equilibrium
    rns_data = RNS_make_initial_data();
    
    // Initialize Cold EOS for mapping density to pressure
    ceos = new Primitive::ColdEOS<Primitive::COLDEOS_POLICY>;
    InitColdEOS(ceos, pin);
  }

  // 3. Initialize Adaptive Mesh Refinement (AMR)
  if (adaptive == true) {
    EnrollUserRefinementCondition(RefinementCondition);
  }

  // Map the refinement method string from .par file to the enum
  std::ostringstream msg;
  static const std::map<std::string, opt_refinement_method> opt_ref{
    { "none", opt_refinement_method::none },
    { "MassPerMeshBlock", opt_refinement_method::MassPerMeshBlock },
    { "MaxMassInCell", opt_refinement_method::MaxMassInCell },
    { "MaxMassInCellTracker", opt_refinement_method::MaxMassInCellTracker }
  };
  
  auto itr_ref = opt_ref.find(pin->GetOrAddString("problem", "refinement_method", "none"));
  if (itr_ref != opt_ref.end()) {
    opt_refm_ = itr_ref->second;
  } else {
    msg << "problem/refinement_method unknown" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Read AMR mass thresholds
  const Real INF = std::numeric_limits<Real>::infinity();
  opt_delta_min_m = pin->GetOrAddReal("problem", "delta_min_m", -INF);
  opt_delta_max_m = pin->GetOrAddReal("problem", "delta_max_m", INF);

  // 4. Initialize Deleptonization Parameters
  static const std::map<std::string, opt_deleptonization_method> opt_lep{
    { "Liebendoerfer", opt_deleptonization_method::Liebendoerfer },
    { "Simple", opt_deleptonization_method::Simple },
    { "None", opt_deleptonization_method::None }
  };

  auto itr_lep = opt_lep.find(pin->GetOrAddString("problem", "deleptonization_method", "Simple"));
  if (itr_lep != opt_lep.end()) {
    opt_dlp_mtd_ = itr_lep->second;
  } else {
    msg << "problem/deleptonization_method unknown" << std::endl;
    ATHENA_ERROR(msg);
  }

  opt_update_conserved = pin->GetOrAddBoolean("problem", "update_conserved", false);
  opt_update_entropy   = pin->GetOrAddBoolean("problem", "update_entropy", true);
  opt_E_nu_avg         = pin->GetOrAddReal("problem", "E_nu_avg", 10.0);
  opt_rho_trap         = pin->GetOrAddReal("problem", "rho_trap", 1e12) / UDENS;
  opt_rho_cut          = pin->GetOrAddReal("problem", "rho_cut", -INF) / UDENS;
  use_ye_of_rho_table = pin->GetOrAddBoolean("problem", "use_ye_of_rho_table", true);
  pdelept = new Deleptonization(pin);

  fac_MeVfm3_code = Primitive::Nuclear.PressureConversion(Primitive::GeometricSolar);

  // 5. Initialize Magnetic Field Parameters
  opt_B0_amp = pin->GetOrAddReal("problem", "B0_amp", 0.0);
  opt_B0_rad = pin->GetOrAddReal("problem", "B0_rad", 0.0);

  // 6. Enroll Bounce Detection and Output Trackers
  EnrollUserMainLoopBreak(BounceShortCircuit);
  EnrollUserHistoryOutput(MaxMassInCell, "max_MassInCell", UserHistoryOperation::max);
  EnrollUserHistoryOutput(MassPerMeshBlock, "max_MassPerMB", UserHistoryOperation::max);
  EnrollUserHistoryOutput(MaxLevel, "max_level", UserHistoryOperation::max);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Maps RNS initial data to the 3D grid and sets up Ye and Magnetic Fields.
// Took mostly from gr_rns.cpp
// Additions: Ye injection (initial_Ye) declaring the WD as a C/O WD, and magnetic field from both rns and ccsn as an option
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput* pin)
{
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", false);
  MB_info* mbi = &(pz4c->mbi);

  //---------------------------------------------------------------------------
  // 1. Interpolate ADM Metric from RNS
  //---------------------------------------------------------------------------
  if (verbose && Globals::my_rank == 0)
    std::cout << "Interpolating ADM metric on current MeshBlock." << std::endl;

  int imin[3] = { 0, 0, 0 };
  int n[3]    = { mbi->nn1, mbi->nn2, mbi->nn3 };
  int sz      = n[0] * n[1] * n[2];

  Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
  Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];
  Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
  Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];
  Real *alp = new Real[sz], *betax = new Real[sz], *betay = new Real[sz], *betaz = new Real[sz];
  Real *x = new Real[n[0]], *y = new Real[n[1]], *z = new Real[n[2]];

  // Populate node coordinates for metric
  for (int i = 0; i < n[0]; ++i) x[i] = mbi->x1(i);
  for (int i = 0; i < n[1]; ++i) y[i] = mbi->x2(i);
  for (int i = 0; i < n[2]; ++i) z[i] = mbi->x3(i);

  // Call RNS library to fill the metric arrays
  RNS_Cartesian_interpolation(
    rns_data, imin, n, n, x, y, z, alp, betax, betay, betaz,
    gxx, gxy, gxz, gyy, gyz, gzz, Kxx, Kxy, Kxz, Kyy, Kyz, Kzz,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
  );

  // Map to Athena++ 3D grid
  for (int k = 0; k < mbi->nn3; ++k) {
    for (int j = 0; j < mbi->nn2; ++j) {
      for (int i = 0; i < mbi->nn1; ++i) {
        int flat_ix = i + n[0] * (j + n[1] * k);

        pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i) = gxx[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i) = gxy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i) = gxz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i) = gyy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i) = gyz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i) = gzz[flat_ix];

        pz4c->storage.adm(Z4c::I_ADM_Kxx, k, j, i) = Kxx[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kxy, k, j, i) = Kxy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kxz, k, j, i) = Kxz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kyy, k, j, i) = Kyy[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kyz, k, j, i) = Kyz[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_Kzz, k, j, i) = Kzz[flat_ix];

        pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i) = alp[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_betax, k, j, i) = betax[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_betay, k, j, i) = betay[flat_ix];
        pz4c->storage.adm(Z4c::I_ADM_betaz, k, j, i) = betaz[flat_ix];
      }
    }
  }

  // Free metric arrays
  delete[] gxx; delete[] gxy; delete[] gxz; delete[] gyy; delete[] gyz; delete[] gzz;
  delete[] Kxx; delete[] Kxy; delete[] Kxz; delete[] Kyy; delete[] Kyz; delete[] Kzz;
  delete[] alp; delete[] betax; delete[] betay; delete[] betaz;
  delete[] x; delete[] y; delete[] z;

  // Convert ADM metric to Z4c conformal variables
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  //---------------------------------------------------------------------------
  // 2. Interpolate Fluid Primitives from RNS
  //---------------------------------------------------------------------------
  if (verbose && Globals::my_rank == 0)
    std::cout << "Interpolating primitives on current MeshBlock." << std::endl;

  n[0] = ncells1; n[1] = ncells2; n[2] = ncells3;
  sz   = n[0] * n[1] * n[2];

  Real *rho = new Real[sz], *pres = new Real[sz];
  Real *ux = new Real[sz], *uy = new Real[sz], *uz = new Real[sz];
  x = new Real[n[0]]; y = new Real[n[1]]; z = new Real[n[2]];

  // Populate cell-center coordinates for fluid
  for (int i = 0; i < n[0]; ++i) x[i] = pcoord->x1v(i);
  for (int i = 0; i < n[1]; ++i) y[i] = pcoord->x2v(i);
  for (int i = 0; i < n[2]; ++i) z[i] = pcoord->x3v(i);

  RNS_Cartesian_interpolation(
    rns_data, imin, n, n, x, y, z, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    rho, NULL, NULL, NULL, NULL, ux, uy, uz, pres
  );


//   Real rho_min = pin->GetReal("hydro", "dfloor");
//   Real initial_Ye = pin->GetOrAddReal("problem", "initial_Ye", 0.5); // Standard C/O WD Ye

//   for (int k = 0; k < ncells3; ++k) {
//     for (int j = 0; j < ncells2; ++j) {
//       for (int i = 0; i < ncells1; ++i) {
//         int flat_ix = i + n[0] * (j + n[1] * k);

// #if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
//         rho[flat_ix] *= ceos->mb / mb_rnsc;  // adjust for rns baryon mass
// #endif
//         // commenting out to rewrite the RNS ID        
//         // if (rho[flat_ix] > rho_min) {
//         //   pres[flat_ix] = ceos->GetPressure(rho[flat_ix]);
//         // }

//         // Apply density cutoff (from ccsn.cpp logic)
//         if (opt_rho_cut > 0 && rho[flat_ix] < opt_rho_cut) {
//             rho[flat_ix] = rho_min;
//         }

//         phydro->w(IDN, k, j, i) = rho[flat_ix];
//         phydro->w(IPR, k, j, i) = pres[flat_ix];
//         phydro->w(IVX, k, j, i) = ux[flat_ix];
//         phydro->w(IVY, k, j, i) = uy[flat_ix];
//         phydro->w(IVZ, k, j, i) = uz[flat_ix];

//         // --- The AIC Twist: Explicitly set Electron Fraction ---
//         // pscalars->r(IYE, k, j, i) = initial_Ye;
//         pscalars->r(IYE, k, j, i) = pdelept->Ye_of_rho(rho[flat_ix]);
//       }
//     }
//   }

  Real rho_min = pin->GetReal("hydro", "dfloor");
  Real initial_Ye = pin->GetOrAddReal("problem", "initial_Ye", 0.5);

  const Real mb = ceos->GetBaryonMass();
  auto& reos = peos->GetEOS();

  for (int k = 0; k < ncells3; ++k) {
    for (int j = 0; j < ncells2; ++j) {
      for (int i = 0; i < ncells1; ++i) {

        int flat_ix = i + n[0] * (j + n[1] * k);

		if (rho[flat_ix] > rho_min) {
			const Real P_ID   = pres[flat_ix];
			        
	        Real rho_ID = ceos->GetDensityFromPressure(P_ID);
	        
	        if (!std::isfinite(rho_ID)) {
	          std::cout << "BAD rho_ID: "
	                    << "i=" << i << " j=" << j << " k=" << k
	                    << " P_ID=" << P_ID
	                    << " rho_ID=" << rho_ID
	                    << std::endl;
	        }
	        
	        // Apply density cutoff
	        if (opt_rho_cut > 0 && rho_ID < opt_rho_cut) {
	          rho_ID = rho_min;
	        }
	
	        const Real n_b = rho_ID / mb;
	
	        Real Y_old[MAX_SPECIES]{0.0};
	        Y_old[IYE] = initial_Ye;
	
	        const Real T = reos.GetTemperatureFromP(
	            n_b, P_ID, Y_old);
	
			if (!std::isfinite(T)) {
				std::cout << "BAD T: "
	            << "i=" << i << " j=" << j << " k=" << k
	            << " P_ID=" << P_ID
	            << " rho_ID=" << rho_ID
	            << " n_b=" << n_b
	            << " Ye_old=" << initial_Ye
	            << " T=" << T
	            << std::endl;
			}
	
	        Real Ye_new = pdelept->Ye_of_rho(rho_ID);
	
	        Real Y_new[MAX_SPECIES]{0.0};
	        Y_new[IYE] = Ye_new;
	
	        const Real P_new = reos.GetPressure(
	            n_b, T, Y_new);
			if (!std::isfinite(P_new)) {
			  std::cout << "BAD P_new: "
			            << "i=" << i << " j=" << j << " k=" << k
			            << " P_ID=" << P_ID
			            << " rho_ID=" << rho_ID
			            << " T=" << T
			            << " Ye_new=" << Ye_new
			            << " P_new=" << P_new
			            << std::endl;
			}
			
	        phydro->w(IDN, k, j, i) = rho_ID;
	        phydro->w(IPR, k, j, i) = P_new;
	
	        phydro->w(IVX, k, j, i) = ux[flat_ix];
	        phydro->w(IVY, k, j, i) = uy[flat_ix];
	        phydro->w(IVZ, k, j, i) = uz[flat_ix];
	
	        pscalars->r(IYE, k, j, i) = Ye_new;
		}
      }
    }
  }

  delete[] rho; delete[] pres; delete[] ux; delete[] uy; delete[] uz;
  delete[] x; delete[] y; delete[] z;

  // {
  //   Real local_max = 0.0;

  //   for (int k = 0; k < ncells3; ++k)
  //     for (int j = 0; j < ncells2; ++j)
  //       for (int i = 0; i < ncells1; ++i)
  //         local_max = std::max(local_max, phydro->w(IDN,k,j,i));

  //   Real global_max = local_max;

  // #ifdef MPI_PARALLEL
  //   MPI_Allreduce(&local_max, &global_max, 1,
  //                 MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  // #endif

  //   if (Globals::my_rank == 0)
  //     std::cout << "rho max AFTER copy = "
  //               << std::scientific << global_max << std::endl;
  // }

  //---------------------------------------------------------------------------
  // 3. Finalize: Primitives -> Conserved & Floors
  //---------------------------------------------------------------------------
  int il = is - NGHOST, iu = ie + NGHOST;
  int jl = (block_size.nx2 > 1) ? js - NGHOST : js, ju = (block_size.nx2 > 1) ? je + NGHOST : je;
  int kl = (block_size.nx3 > 1) ? ks - NGHOST : ks, ku = (block_size.nx3 > 1) ? ke + NGHOST : ke;

  peos->PrimitiveToConserved(phydro->w, pscalars->r, pfield->bcc, phydro->u, pscalars->s, 
                             pcoord, il, iu, jl, ju, kl, ku);
  // {
  //   Real local_max = 0.0;

  //   for (int k = 0; k < ncells3; ++k)
  //     for (int j = 0; j < ncells2; ++j)
  //       for (int i = 0; i < ncells1; ++i)
  //         local_max = std::max(local_max, phydro->w(IDN,k,j,i));

  //   Real global_max = local_max;

  // #ifdef MPI_PARALLEL
  //   MPI_Allreduce(&local_max, &global_max, 1,
  //                 MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  // #endif

  //   if (Globals::my_rank == 0)
  //     std::cout << "rho max AFTER PrimitiveToConserved = "
  //               << std::scientific << global_max << std::endl;
  // }
  bool id_floor_primitives = pin->GetOrAddBoolean("problem", "id_floor_primitives", true);
  if (id_floor_primitives) {
    for (int k = 0; k < ncells3; ++k)
      for (int j = 0; j < ncells2; ++j)
        for (int i = 0; i < ncells1; ++i) {
          PrimHelper::ApplyPrimitiveFloors(peos->GetEOS(), phydro->w, pscalars->r, k, j, i);
        }
  }
  // {
  //   Real local_max = 0.0;

  //   for (int k = 0; k < ncells3; ++k)
  //     for (int j = 0; j < ncells2; ++j)
  //       for (int i = 0; i < ncells1; ++i)
  //         local_max = std::max(local_max, phydro->w(IDN,k,j,i));

  //   Real global_max = local_max;

  // #ifdef MPI_PARALLEL
  //   MPI_Allreduce(&local_max, &global_max, 1,
  //                 MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  // #endif

  //   if (Globals::my_rank == 0)
  //     std::cout << "rho max AFTER primitive floors = "
  //               << std::scientific << global_max << std::endl;
  // }
  //---------------------------------------------------------------------------
  // 4. Seed Dipole Magnetic Field (from ccsn.cpp)
  //---------------------------------------------------------------------------
#if MAGNETIC_FIELDS_ENABLED
  std::string bfield_type = pin->GetOrAddString("problem", "bfield_type", "dipole");

  if (bfield_type == "dipole") {
    // Exactly as in ccsn.cpp (no guards)
    if (opt_B0_amp > 0.0 && opt_B0_rad > 0.0) {
      const Real B0_amp_local = opt_B0_amp;
      const Real B0_rad_local = opt_B0_rad;

      SeedFaceBFromEdgePotential(this,
        [=](Real x, Real y, Real z, Real /*p*/, Real /*rho*/, Real& Ax, Real& Ay, Real& Az) {
          const Real rad_cyl_sqr = SQR(x) + SQR(y);
          const Real rad_cyl     = std::sqrt(rad_cyl_sqr);
          const Real oo_rad_cyl  = 1.0 / rad_cyl;

          const Real rad    = std::sqrt(rad_cyl_sqr + SQR(z));
          const Real oo_rad = 1.0 / rad;

          const Real dphdx = -y * SQR(oo_rad_cyl);
          const Real dphdy = x * SQR(oo_rad_cyl);
          const Real dphdz = 0.0;

          const Real Aph = B0_amp_local * rad_cyl / (std::pow(rad, 3) + std::pow(B0_rad_local, 3));

          Ax = dphdx * Aph;
          Ay = dphdy * Aph;
          Az = dphdz * Aph;
        });
    }
  } else if (bfield_type == "rns") {
    // Exactly as in gr_rns.cpp
    SeedMagneticFields(this, pin);

  } else if (bfield_type == "suwa_varma") {

    SeedSuwaVarmaMagneticFields(this, pin);

  } else {
    std::stringstream msg;
    msg << "problem/bfield_type unknown" << std::endl;
    ATHENA_ERROR(msg);
  }
#endif

  return;
}

//========================================================================================
// Deleptonization Class Implementation
//========================================================================================
namespace {

Deleptonization::Deleptonization(ParameterInput* pin) {
  // Default values are SFHo fits from 1701.02752 (Liebendoerfer parameterization)
  log10_rho1 = pin->GetOrAddReal("deleptonization", "log10_rho1", 7.795);
  log10_rho2 = pin->GetOrAddReal("deleptonization", "log10_rho2", 12.816);
  Ye_2       = pin->GetOrAddReal("deleptonization", "Ye_2", 0.308);
  Ye_c       = pin->GetOrAddReal("deleptonization", "Ye_c", 0.0412);
  Ye_H       = pin->GetOrAddReal("deleptonization", "Ye_H", 0.257);
  use_ye_of_rho_table = pin->GetOrAddBoolean("problem", "use_ye_of_rho_table", true);

  if (use_ye_of_rho_table) {
    std::string filename =
        pin->GetOrAddString("problem", "Ye_rho_table", "");

    if (filename.empty()) {
      throw std::runtime_error("Ye_rho_table was not specified");
    }

    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error(
          "Could not open Ye-rho table: " + filename);
    }

    Real Ye, rho;

    // rho from the table is in log10
    while (file >> rho >> Ye) {
      Ye_table_.push_back(Ye);
      rho_table_.push_back(rho);
    }

    file.close();

    if (rho_table_.size() < 2) {
      throw std::runtime_error(
          "Ye-rho table must contain at least two points");
    }
    //if (Globals::my_rank == 0) {
    //	std::cout << "Loaded Ye-rho table with "
    //          	<< rho_table_.size()
    //          	<< " points." << std::endl;
    //}          	
  }
}

Real Deleptonization::Ye_of_rho(Real rho) const {

  if (!use_ye_of_rho_table) {
    Real const Ye_1       = 0.5;
    Real const log10_rhoH = 15;

    Real const log10_rho = log10(rho * UDENS);
    Real const x = std::max(-1.0, std::min(1.0, 
                  (2 * log10_rho - log10_rho2 - log10_rho1) / (log10_rho2 - log10_rho1)));
    Real const m = (Ye_H - Ye_2) / (log10_rhoH - log10_rho2);

    if (log10_rho > log10_rho2) {
      return Ye_2 + m * (log10_rho - log10_rho2);
    } else {
      return 0.5 * (Ye_2 + Ye_1) + 0.5 * x * (Ye_2 - Ye_1) +
            Ye_c * (1 - std::abs(x) + 4 * std::abs(x) * (std::abs(x) - 0.5) * (std::abs(x) - 1));
    }
  } else {
    // Convert code density to cgs density
    Real log10_rho_cgs = log10(rho * UDENS);

    // ------------------------------------------------------------
    // Below the first table point
    // ------------------------------------------------------------
    if (log10_rho_cgs <= rho_table_.front()) {
      return Ye_table_.front();
    }

    // ------------------------------------------------------------
    // Above the last table point
    // ------------------------------------------------------------
    if (log10_rho_cgs >= rho_table_.back()) {
      return Ye_table_.back();
    }

    // ------------------------------------------------------------
    // Find interval containing rho
    // ------------------------------------------------------------
    auto it = std::lower_bound(
        rho_table_.begin(),
        rho_table_.end(),
        log10_rho_cgs);

    int i = std::distance(rho_table_.begin(), it);

    // ------------------------------------------------------------
    // Linear interpolation
    // ------------------------------------------------------------
    Real rho1 = rho_table_[i - 1];
    Real rho2 = rho_table_[i];

    Real Ye1 = Ye_table_[i - 1];
    Real Ye2 = Ye_table_[i];

    Real f = (log10_rho_cgs - rho1) / (rho2 - rho1);

    return Ye1 + f * (Ye2 - Ye1);
  }
}

} // namespace

//========================================================================================
//! \fn void Mesh::UserWorkInLoop(ParameterInput *pin)
//  \brief Application of deleptonization scheme to drive the collapse.
//  completely picked from ccsn
//========================================================================================
void Mesh::UserWorkInLoop(ParameterInput* pin)
{
  MeshBlock* pmb = pblock;

  const Real E_nu_avg = opt_E_nu_avg;
  const Real rho_trap = opt_rho_trap;

  // -------------------------------------------------------------------------
  // Method 1: Liebendoerfer (Complex Entropy Update)
  // -------------------------------------------------------------------------
  auto method_Liebendoerfer = [&]() {
    while (pmb != nullptr) {
      EquationOfState* peos = pmb->peos;
      Hydro* ph = pmb->phydro;
      PassiveScalars* ps = pmb->pscalars;
      Field* pf = pmb->pfield;
      Coordinates* pco = pmb->pcoord;
      Z4c* pz4c = pmb->pz4c;

      AA aux_s; aux_s.InitWithShallowSlice(ph->derived_ms, IX_SPB, 1);
      AA aux_T; aux_T.InitWithShallowSlice(ph->derived_ms, IX_T, 1);
      AA aux_h; aux_h.InitWithShallowSlice(ph->derived_ms, IX_ETH, 1);
      AA aux_e; aux_e.InitWithShallowSlice(ph->derived_ms, IX_SEN, 1);

      auto& reos = peos->GetEOS();
      const Real mb_eos = reos.GetBaryonMass();
      AT_N_sca sqrt_detgamma(pz4c->storage.aux_extended, Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma);

      CC_GLOOP3(k, j, i) {
        const Real rho = ph->w(IDN, k, j, i);
        Real& Y_e = ps->r(IYE, k, j, i); 
        Real Y_old[MAX_SPECIES]{ 0 };
        Real Y_new[MAX_SPECIES]{ 0 };
        Y_old[IYE] = Y_e;

        // Calculate Target Ye based on density
        const Real Y_e_bar   = pdelept->Ye_of_rho(rho);
        const Real delta_Y_e = std::min(0.0, Y_e_bar - Y_e);

        if (delta_Y_e < 0.0) { // Electron capture triggered
          Y_e += delta_Y_e;
          Y_new[IYE] = Y_e;
          const Real n = rho / mb_eos;

          if (opt_update_entropy) {
            const Real mu_nu = reos.GetElectronLeptonChemicalPotential(n, aux_T(k, j, i), Y_old);
            aux_s(k, j, i) = reos.GetEntropyPerBaryon(n, aux_T(k, j, i), Y_old);
            // Only drain entropy if neutrinos can escape (rho < rho_trap)
            if ((mu_nu > E_nu_avg) && (rho < rho_trap)) {
              aux_s(k, j, i) -= delta_Y_e * (mu_nu - E_nu_avg) / aux_T(k, j, i);
            }
          } else {
            aux_s(k, j, i) = reos.GetEntropyPerBaryon(n, aux_T(k, j, i), Y_new);
          }

          // update derived hydro quantities
          aux_T(k, j, i) = reos.GetTemperatureFromEntropy(n, aux_s(k, j, i), Y_new);
          aux_h(k, j, i) = reos.GetEnthalpy(n, aux_T(k, j, i), Y_new);
          aux_e(k, j, i) = reos.GetSpecificInternalEnergy(n, aux_T(k, j, i), Y_new);

          if (opt_update_conserved) {
            // extract total energy density
            Real E = reos.GetEnergy(n, aux_T(k, j, i), Y_new);

            // Have new state (s, T, h, E) adjust conserved variables:
            ph->u(IEN, k, j, i) = sqrt_detgamma(k, j, i) * E - ph->u(IDN, k, j, i);
            ps->s(IYE, k, j, i) = ph->u(IDN, k, j, i) * Y_e;

            // update complementary primitive variables
            static const int coarse_flag = 0;
            peos->ConservedToPrimitive(ph->u, ph->w1, ph->w, ps->s, ps->r, pf->bcc, 
                                       pco, i, i, j, j, k, k, coarse_flag);
          } else {
            // remaining primitive quantities updated:
            ph->w(IPR, k, j, i) = reos.GetPressure(n, aux_T(k, j, i), Y_new);

            // new conserved variables
            peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s, 
                                       pco, i, i, j, j, k, k);
          }
        }
      }
      pmb = pmb->next;
    }
  };

  // -------------------------------------------------------------------------
  // Method 2: Simple (Direct Energy Subtraction)
  // -------------------------------------------------------------------------
  auto method_Simple = [&]() {
    while (pmb != nullptr) {
      EquationOfState* peos = pmb->peos;
      Hydro* ph = pmb->phydro;
      PassiveScalars* ps = pmb->pscalars;
      Field* pf = pmb->pfield;
      Coordinates* pco = pmb->pcoord;
      Z4c* pz4c = pmb->pz4c;

      AT_N_sca alpha(pz4c->storage.adm, Z4c::I_ADM_alpha);
      AT_N_sca sqrt_detgamma(pz4c->storage.aux_extended, Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma);
      AA aux_W; aux_W.InitWithShallowSlice(ph->derived_ms, IX_LOR, 1);

      auto& reos = peos->GetEOS();
      const Real mb_eos = reos.GetBaryonMass();

      CC_GLOOP2(k, j) {
        CC_GLOOP1(i) {
          const Real rho = ph->w(IDN, k, j, i);
          const Real n   = rho / mb_eos;
          Real& tau      = ph->u(IEN, k, j, i);
          const Real Y_e = ps->r(IYE, k, j, i);

          const Real Y_e_bar = pdelept->Ye_of_rho(rho);
          const Real delta_Y_e = std::min(0.0, Y_e_bar - Y_e);

          if (E_nu_avg > 0 && delta_Y_e < 0) {  // only actual captures, never raise Ye
            // reset electron fraction & update tau variable ------------------
            ps->r(IYE, k, j, i) = Y_e_bar;
            ps->s(IYE, k, j, i) = ph->u(IDN, k, j, i) * Y_e_bar;

            // Neutrino energy loss: delta_Y_e < 0 from electron capture,
            // so this term is negative, reducing tau. n dYe E_nu is an energy
            // density, hence PressureConversion.
            tau += (alpha(k, j, i) * sqrt_detgamma(k, j, i) * aux_W(k, j, i) * n *
                    delta_Y_e * E_nu_avg * fac_MeVfm3_code);

            static const int coarse_flag = 0;
            peos->ConservedToPrimitive(ph->u, ph->w1, ph->w, ps->s, ps->r, pf->bcc,
                                       pco, i, i, j, j, k, k, coarse_flag);
          }
        }
      }
      pmb = pmb->next;
    }
  };

  // -------------------------------------------------------------------------
  // Execute Chosen Method and Update Gravity
  // -------------------------------------------------------------------------
  switch (opt_dlp_mtd_) {
    case opt_deleptonization_method::Liebendoerfer:
      method_Liebendoerfer();
      break;
    case opt_deleptonization_method::Simple:
      method_Simple();
      break;
    case opt_deleptonization_method::None:
      break;
    default:
      assert(false);
  }

  // Tell the Z4c gravity solver that the fluid's stress-energy tensor has changed
  const auto& pmb_array = GetMeshBlocksCached();
  FinalizeZ4cADM_Matter(pmb_array);
}

void Mesh::UserWorkAfterLoop(ParameterInput* pin) {
  if (pdelept != nullptr) {
    delete pdelept;
    pdelept = nullptr;
  }
}

//========================================================================================
// Adaptive Mesh Refinement (AMR) Trackers
//========================================================================================
namespace {

Real MassPerMeshBlock(MeshBlock* pmb, int iout) {
  Hydro* ph = pmb->phydro;
  Coordinates* pco = pmb->pcoord;
  Real M_loc = 0;
  CC_NS_ILOOP3(k, j, i) {
    const Real vol = pco->GetCellVolume(k, j, i);
    M_loc += ph->u(IDN, k, j, i) * vol;
  }
  return M_loc;
}

Real MaxMassInCell(MeshBlock* pmb, int iout) {
  Hydro* ph = pmb->phydro;
  Coordinates* pco = pmb->pcoord;
  Real max_mass = -std::numeric_limits<Real>::infinity();
  CC_NS_GLOOP3(k, j, i) {
    const Real cell_vol  = pco->GetCellVolume(k, j, i);
    const Real cell_mass = ph->u(IDN, k, j, i) * cell_vol;
    max_mass = std::max(max_mass, cell_mass);
  }
  return max_mass;
}

Real MaxLevel(MeshBlock* pmb, int iout) {
  return pmb->pmy_mesh->M_info.max_level;
}

int RefinementCondition(MeshBlock* pmb) {
  switch (opt_refm_) {
    case opt_refinement_method::MassPerMeshBlock: {
      Real M_loc = MassPerMeshBlock(pmb, 0);
      if (M_loc > opt_delta_max_m) return 1;
      if (M_loc < opt_delta_min_m) return -1;
      break;
    }
    case opt_refinement_method::MaxMassInCell: {
      Real max_mass = MaxMassInCell(pmb, 0);
      if (max_mass > opt_delta_max_m) return 1;
      if (max_mass < opt_delta_min_m) return -1;
      break;
    }
    case opt_refinement_method::MaxMassInCellTracker: {
      // Allow standard ExtremaTracker to increase refinement level first
      int ref_tr = Mesh::StandardRefinementCondition(pmb);
      if (ref_tr == 1) return 1;

      // Fallback to mass threshold if tracker didn't trigger. Derefine only
      // where the tracker does not pin the level (ref_tr == -1), otherwise its
      // whole region flips down and up again every deref_count cycles.
      Real max_mass = MaxMassInCell(pmb, 0);
      if (max_mass > opt_delta_max_m) return 1;
      if (max_mass < opt_delta_min_m && ref_tr == -1) return -1;
      break;
    }
    case opt_refinement_method::none:
      break;
    default:
      assert(false);
  }
  return 0;
}

//========================================================================================
// Bounce Detection MPI Helpers
//========================================================================================

Real GlobalMaxConservedDensity(Mesh* pm) {
  Real local_max = -std::numeric_limits<Real>::infinity();
  MeshBlock* pmb = pm->pblock;
  while (pmb != nullptr) {
    Hydro* ph = pmb->phydro;
    CC_NS_ILOOP3(k, j, i) {
      local_max = std::max(local_max, ph->u(IDN, k, j, i));
    }
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  Real global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return global_max;
#else
  return local_max;
#endif
}

bool BounceShortCircuit(Mesh* pm, ParameterInput* pin) {
  if (!pin->GetOrAddBoolean("problem", "detect_bounce", false)) return false;

  if (pin->GetOrAddBoolean("problem", "post_bounce_short_circuit", false)) {
    pin->SetBoolean("problem", "post_bounce_short_circuit", false); // Reset for restart
    pin->SetBoolean("problem", "detect_bounce", false); // Disable future detection
    
    if (Globals::my_rank == 0)
      std::cout << "Writing post_bounce restart..." << std::endl;
      
    pin->SetString("problem", "restart_tag", "post_bounce");
    return true; // Break the main simulation loop
  }
  return false;
}

void Equilibriate_M1(Mesh* pm, ParameterInput* pin)
{
#if M1_ENABLED
  if (Globals::my_rank == 0) {
    std::printf("Imposing M1 equilibrium...\n");
  }
  int nthreads = pm->GetNumMeshThreads();
  const auto& pmb_array = pm->GetMeshBlocksCached();
  const int nmb         = pmb_array.size();

#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nmb; ++i) {
    MeshBlock* pmb = pmb_array[i];
    M1::M1* pm1    = pmb->pm1;

    pm1->UpdateGeometry(pm1->geom, pm1->scratch);
    pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
    pm1->CalcFiducialVelocity();

    M1::M1::vars_Lab U_C{ { pm1->N_GRPS, pm1->N_SPCS }, { pm1->N_GRPS, pm1->N_SPCS }, { pm1->N_GRPS, pm1->N_SPCS } };
    pm1->SetVarAliasesLab(pm1->storage.u, U_C);
    M1::M1::vars_Source U_S{ { pm1->N_GRPS, pm1->N_SPCS }, { pm1->N_GRPS, pm1->N_SPCS }, { pm1->N_GRPS, pm1->N_SPCS } };
    pm1->SetVarAliasesSource(pm1->storage.u_sources, U_S);

    M1_ILOOP3(k, j, i) {
      M1::Equilibrium::SetEquilibrium(*pm1, U_C, U_S, k, j, i);
    }
  }
#endif
}

void SeedMagneticFields(MeshBlock* pmb, ParameterInput* pin)
{
  int imin[3] = { 0, 0, 0 };
  int imax[3] = { 1, 1, 1 };
  double x[1] = { 0.0 };
  Real rhomax;
  Real prnsmax;

  RNS_Cartesian_interpolation(
    rns_data, imin, imax, imax, x, x, x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, &rhomax, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &prnsmax
  );

  Real pgasmax = ceos->GetPressure(rhomax);
  if (Globals::my_rank == 0) printf("rhomax=%.5e prnsmax=%.5e pmax=%.5e\n", rhomax, prnsmax, pgasmax);

  Real pcut    = pin->GetReal("problem", "pcut") * pgasmax;
  int magindex = pin->GetInteger("problem", "magindex");
  Real b_amp = pin->GetReal("problem", "b_amp") * 0.5 / (pgasmax - pcut) / 8.351416e19;

  SeedFaceBFromEdgePotential(pmb,
    [=](Real x, Real y, Real /*z*/, Real p, Real rho, Real& Ax, Real& Ay, Real& Az) {
      Real amp = b_amp * std::max(p - pcut, 0.0) * std::pow(1.0 - rho / rhomax, magindex);
      Ax = -y * amp;
      Ay = x * amp;
      Az = 0.0;
    });
}

void SeedSuwaVarmaMagneticFields(
    MeshBlock* pmb,
    ParameterInput* pin)
{
  // Parameters are read in code units.
  const Real r0 =
      pin->GetReal("problem", "r0");

  const Real Bpol =
      pin->GetReal("problem", "Bpol");

  const Real Btor =
      pin->GetReal("problem", "Btor");

  if (r0 <= 0.0) {
    std::stringstream msg;
    msg << "problem/r0 must be positive." << std::endl;
    ATHENA_ERROR(msg);
  }

  SeedFaceBFromEdgePotential(
      pmb,
      [=](Real x, Real y, Real z,
          Real /*p*/, Real /*rho*/,
          Real& Ax, Real& Ay, Real& Az) {

        const Real r2 = x*x + y*y + z*z;
        const Real r  = std::sqrt(r2);

        // Avoid division by zero at the coordinate origin.
        if (r <= TINY_NUMBER) {
          Ax = 0.0;
          Ay = 0.0;
          Az = 0.0;
          return;
        }

        const Real r3  = r*r*r;
        const Real r03 = r0*r0*r0;

        // f(r) = r0^3 * r / [2 * (r^3 + r0^3)]
        const Real f =
            r03 * r /
            (2.0 * (r3 + r03));

        const Real inv_r  = 1.0 / r;
        const Real inv_r2 = 1.0 / r2;

        // Toroidal-component contribution.
        const Real A_tor_x =
            f * Btor * z * x * inv_r2;

        const Real A_tor_y =
            f * Btor * z * y * inv_r2;

        const Real A_tor_z =
            f * Btor * z * z * inv_r2;

        // Poloidal-component contribution.
        const Real A_pol_x =
            -f * Bpol * y * inv_r;

        const Real A_pol_y =
             f * Bpol * x * inv_r;

        const Real A_pol_z = 0.0;

        Ax = A_tor_x + A_pol_x;
        Ay = A_tor_y + A_pol_y;
        Az = A_tor_z + A_pol_z;
      });
}

} // namespace

//========================================================================================
//! \fn void Mesh::UserWorkBeforeLoop(ParameterInput *pin)
//========================================================================================
void Mesh::UserWorkBeforeLoop(ParameterInput* pin)
{
//   // --- START CUSTOM DIAGNOSTICS ---
//   Real local_max = 0.0;
//   MeshBlock* pmb_diag = pblock;

//   while (pmb_diag != nullptr) {
//     Hydro* ph = pmb_diag->phydro;
    
//     // Loop over all active interior cells in this specific MeshBlock
//     for (int k = pmb_diag->ks; k <= pmb_diag->ke; ++k) {
//       for (int j = pmb_diag->js; j <= pmb_diag->je; ++j) {
//         for (int i = pmb_diag->is; i <= pmb_diag->ie; ++i) {
//           local_max = std::max(local_max, ph->w(IDN, k, j, i));
//         }
//       }
//     }
//     pmb_diag = pmb_diag->next;
//   }

//   Real global_max = local_max;
  
// #ifdef MPI_PARALLEL
//   MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
// #endif

//   {
//     std::cout << "rho max AT START OF UserWorkBeforeLoop = " 
//               << std::scientific << global_max << std::endl;
//     std::cout << "rho max [CGS] = " 
//               << global_max * UDENS << std::endl;
//   }
//   // --- END CUSTOM DIAGNOSTICS ---

  if (pin->GetOrAddBoolean("problem", "inject_equilibrium", false)) {
    Equilibriate_M1(this, pin);
    pin->SetBoolean("problem", "inject_equilibrium", false);
  }

  if (!pin->GetOrAddBoolean("problem", "detect_bounce", false)) {
    return;
  }

  enum class opt_bounce_detection_method { local_maximum, maximum, entropy, None };
  opt_bounce_detection_method opt_bdm;
  std::ostringstream msg;

  static const std::map<std::string, opt_bounce_detection_method> opt_bdm_{
    { "local_maximum", opt_bounce_detection_method::local_maximum },
    { "maximum", opt_bounce_detection_method::maximum },
    { "entropy", opt_bounce_detection_method::entropy },
    { "None", opt_bounce_detection_method::None }
  };

  auto itr = opt_bdm_.find(pin->GetOrAddString("problem", "bounce_detection_method", "None"));
  if (itr != opt_bdm_.end()) {
    opt_bdm = itr->second;
  } else {
    msg << "problem/bounce_detection_method unknown" << std::endl;
    ATHENA_ERROR(msg);
  }

  Real opt_bdm_rho_min     = 0;
  Real opt_bdm_rho_max     = 0;
  Real opt_bdm_spb_max     = 0;
  Real opt_bdm_r_max       = 0;
  int par_D_max_steps_dec  = 0;
  Real rat_D_max_threshold = 0;

  switch (opt_bdm) {
    case opt_bounce_detection_method::local_maximum:
      par_D_max_steps_dec = pin->GetOrAddInteger("problem", "D_max_steps_dec", 3);
      rat_D_max_threshold = pin->GetOrAddReal("problem", "rat_D_max_threshold", 0.001);
      break;
    case opt_bounce_detection_method::maximum:
      opt_bdm_rho_max = pin->GetOrAddReal("problem", "bdm_rho_max", 2.0e12);
      break;
    case opt_bounce_detection_method::entropy:
      opt_bdm_rho_min = pin->GetOrAddReal("problem", "bdm_rho_min", 1.0e10);
      opt_bdm_spb_max = pin->GetOrAddReal("problem", "bdm_spb_max", 3);
      opt_bdm_r_max   = pin->GetOrAddReal("problem", "bdm_r_max", 30.0);
      break;
    case opt_bounce_detection_method::None:
      break;
    default:
      assert(false);
  }

  bool at_bounce = false;

  switch (opt_bdm) {
    case opt_bounce_detection_method::local_maximum: {
      const Real D_max = GlobalMaxConservedDensity(this);
      const bool do_check = std::abs(1 - D_max_last / D_max) > rat_D_max_threshold;
      if (do_check) {
        if (D_max > D_max_last) {
          D_max_steps_inc++;
          D_max_steps_dec = 0;
        } else {
          D_max_steps_inc = 0;
          D_max_steps_dec++;
        }
        D_max_last = D_max;
      }
      at_bounce = D_max_steps_dec >= par_D_max_steps_dec;
      if (Globals::my_rank == 0) {
        std::printf("do_check %d; ", do_check);
        std::printf("D_max_steps_inc %d; ", D_max_steps_inc);
        std::printf("D_max_steps_dec %d; ", D_max_steps_dec);
        std::printf("D_max %.17e\n", D_max);
      }
      break;
    }
    case opt_bounce_detection_method::maximum: {
      MeshBlock* pmb = pblock;
      Real rho_max   = 0;
      while (pmb != nullptr) {
        Hydro* ph = pmb->phydro;
        CC_NS_ILOOP3(k, j, i) {
          Real rho_cell = ph->w(IDN, k, j, i) * UDENS;
          rho_max       = std::max(rho_max, rho_cell);
          at_bounce     = at_bounce or (rho_cell > opt_bdm_rho_max);
        }
        pmb = pmb->next;
      }
      if (at_bounce && Globals::my_rank == 0) {
        std::printf("rho_max [CGS] %.3e\n", rho_max);
      }
      break;
    }
    case opt_bounce_detection_method::entropy: {
      MeshBlock* pmb   = pblock;
      Real rho_trigger = 0;
      Real spb_trigger = 0;
      while (pmb != nullptr) {
        Hydro* ph = pmb->phydro;
        CC_NS_ILOOP3(k, j, i) {
          Real rho_cur = ph->w(IDN, k, j, i) * UDENS;
          Real spb_cur = ph->derived_ms(IX_SPB, k, j, i);
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);
          Real r  = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
          bool triggered = (rho_cur > opt_bdm_rho_min) && (spb_cur > opt_bdm_spb_max) && (r < opt_bdm_r_max);
          if (triggered && !at_bounce) {
            rho_trigger = rho_cur;
            spb_trigger = spb_cur;
          }
          at_bounce = at_bounce or triggered;
        }
        pmb = pmb->next;
      }
      if (at_bounce && Globals::my_rank == 0) {
        std::printf("rho [CGS] %.3e; spb %.3e\n", rho_trigger, spb_trigger);
      }
      break;
    }
    case opt_bounce_detection_method::None:
      break;
    default:
      assert(false);
  }

#ifdef MPI_PARALLEL
  {
    int local_bounce  = at_bounce ? 1 : 0;
    int global_bounce = 0;
    MPI_Allreduce(&local_bounce, &global_bounce, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    at_bounce = (global_bounce != 0);
  }
#endif

  if (at_bounce) {
    if (Globals::my_rank == 0) {
      std::printf("Bounce detected... @ %.13e\n", time);
    }
    pin->SetReal("problem", "t_bounce", time);
    const bool equilibriate_post_bounce = pin->GetOrAddBoolean("problem", "equilibriate_post_bounce", true);
    if (equilibriate_post_bounce) {
      Equilibriate_M1(this, pin);
    }
    pin->SetBoolean("problem", "M1_enabled", true);
    opt_dlp_mtd_ = opt_deleptonization_method::None;
    pin->SetString("problem", "deleptonization_method", "None");
    if (Globals::my_rank == 0) {
      std::printf("Deleptonization disabled\n");
    }
    pin->SetBoolean("problem", "post_bounce", true);
    pin->SetBoolean("problem", "post_bounce_short_circuit", true);
  }
}
