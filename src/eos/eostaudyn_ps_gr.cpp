//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file eostaudyn_ps_gr.cpp
//  \brief Implements functions for going between primitive and conserved
//  variables in general-relativistic (magneto)hydrodynamics, as well as for
//  computing wavespeeds. Unified hydro/MHD file - magnetic field handling
//  gated by MAGNETIC_FIELDS_ENABLED.

// includes -------------------------------------------------------------------

// C++ headers
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/linear_algebra.hpp"  // Det. & friends
#include "../z4c/z4c.hpp"
#include "eos.hpp"

// #ifdef Z4C_AHF
#include "../z4c/ahf.hpp"
// #endif

// BD: TODO - a lot of the det / inv calculations could be stream-lined
//            and refactored for speed...

// ----------------------------------------------------------------------------

namespace
{

// Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> eos;
// Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>
// ps{&eos};

typedef Primitive::PrimitiveSolver<Primitive::EOS_POLICY,
                                   Primitive::ERROR_POLICY>
  PS;

// Declarations
static void PrimitiveToConservedSingle(AA& prim,
                                       AA& prim_scalar,
#if MAGNETIC_FIELDS_ENABLED
                                       AA& bb_cc,
#endif
                                       AA& cons,
                                       AA& cons_scalar,
                                       AA& derived_ms,
                                       EquationOfState::geom_sliced_cc& gsc,
                                       int k,
                                       int j,
                                       int i,
                                       PS& ps);

}  // namespace

//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock* pmb, ParameterInput* pin)
    : ps{ &eos }
{
  pmy_block_ = pmb;
  Real density_floor =
    pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024 * (FLT_MIN)));
  Real temperature_floor =
    pin->GetOrAddReal("hydro", "tfloor", std::sqrt(1024 * (FLT_MIN)));

#if MAGNETIC_FIELDS_ENABLED
  Real bsq_max = pin->GetOrAddReal("hydro", "bsq_max", 1e6);
#endif
  verbose = pin->GetOrAddBoolean("hydro", "verbose", true);

  // BD: TODO - dead parameter
  restrict_cs2 = pin->GetOrAddBoolean("hydro", "restrict_cs2", false);

  max_cs_W = pin->GetOrAddReal("hydro", "max_cs_W", 10.0);
  max_cs2  = 1.0 - SQR(1.0 / max_cs_W);

  warn_unrestricted_cs2 =
    pin->GetOrAddBoolean("hydro", "warn_unrestricted_cs2", false);
  recompute_temperature =
    pin->GetOrAddBoolean("hydro", "recompute_temperature", true);
  smooth_temperature =
    pin->GetOrAddBoolean("hydro", "smooth_temperature", false);
  recompute_enthalpy =
    pin->GetOrAddBoolean("hydro", "recompute_enthalpy", false);

  // control PrimitiveSolver tolerances / iterates
  ps.SetRootfinderTol(pin->GetOrAddReal("hydro", "c2p_acc", 1e-15));
  ps.SetRootfinderMaxIter(pin->GetOrAddInteger("hydro", "max_iter", 30));
  ps.SetTightenBracket(pin->GetOrAddBoolean("hydro", "tighten_bracket", true));
  ps.SetValidateDensity(
    pin->GetOrAddBoolean("hydro", "c2p_validate_density", true));
  ps.SetToms748(pin->GetOrAddBoolean("hydro", "use_toms_748", false));

  ps.SetMaxVelocityLorentz(pin->GetOrAddReal("hydro", "max_W", 1000.0));

  // Set up the EOS
#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
  std::string table = pin->GetString("hydro", "table");
  eos.SetCodeUnitSystem(&Primitive::GeometricSolar);
  eos.ReadTableFromFile(table);
  Real mb           = eos.GetBaryonMass();
  Real n_max_factor = pin->GetOrAddReal("hydro", "n_max_factor", 1.0);
  eos.SetMaximumDensity(eos.GetMaximumDensity() * n_max_factor);
  Real T_max_factor = pin->GetOrAddReal("hydro", "T_max_factor", 1.0);
  eos.SetMaximumTemperature(eos.GetMaximumTemperature() * T_max_factor);

#elif defined(USE_IDEAL_GAS)
  // Baryon mass
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos.SetBaryonMass(mb);

#elif defined(USE_PIECEWISE_POLY)
  int n = pin->GetInteger("hydro", "n");
  Real gammas[n];
  for (int i = 0; i < n; i++)
    gammas[i] = pin->GetReal("hydro", "gamma" + std::to_string(i));
  Real rhos[n];
  rhos[0] = 0;  // needed in initialization but ignored in the function
  for (int i = 1; i < n; i++)
    rhos[i] = pin->GetReal("hydro", "rho" + std::to_string(i));
  Real P0 = pin->GetReal("hydro", "P0");
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos.InitializeFromData(rhos, gammas, P0, mb, n);
  Real gamma_th = pin->GetOrAddReal("hydro", "gamma_th", 5.0 / 3.0);
  eos.SetThermalGamma(gamma_th);
#endif

  // Set the number density floor.
  eos.SetDensityFloor(density_floor / mb);
  Real threshold = pin->GetOrAddReal("hydro", "dthreshold", 1.0);
  eos.SetThreshold(threshold);
  // Set the temperature floor.
  eos.SetTemperatureFloor(temperature_floor);
  for (int i = 0; i < eos.GetNSpecies(); i++)
  {
    std::stringstream ss;
    ss << "y" << i << "_atmosphere";
    Real atmosphere = pin->GetOrAddReal("hydro", ss.str(), 0.5);
    eos.SetSpeciesAtmosphere(atmosphere, i);
  }

#if MAGNETIC_FIELDS_ENABLED
  eos.SetMaximumMagnetization(bsq_max);
#endif

  // Enable or disable momentum limiting in the C2P solver.
  eos.SetLimitMomenta(pin->GetOrAddBoolean("hydro", "limit_momenta", false));

  // If we're working with an ideal gas, we need to fix the adiabatic constant.
#ifdef USE_IDEAL_GAS
  eos.SetGamma(pin->GetOrAddReal("hydro", "gamma", 2.0));
#elif defined(USE_HYBRID_EOS)
  eos.SetThermalGamma(pin->GetOrAddReal("hydro", "gamma_th", 2.0));
#endif
}

// Initialize cold eos object based on parameters
void InitColdEOS(Primitive::ColdEOS<Primitive::COLDEOS_POLICY>* eos,
                 ParameterInput* pin)
{
#if defined(USE_COMPOSE_EOS) || defined(USE_HYBRID_EOS)
  std::string table = pin->GetString("hydro", "table");

  // read in species names
  std::string species_names[NSCALARS];
  for (int i = 0; i < NSCALARS; i++)
  {
    species_names[i] =
      pin->GetOrAddString("hydro", "species" + std::to_string(i + 1), "e");
  }

  eos->ReadColdSliceFromFile(table, species_names);
  eos->SetCodeUnitSystem(&Primitive::GeometricSolar);

#elif defined(USE_IDEAL_GAS)
  Real k_adi = pin->GetReal("hydro", "k_adi");
  eos->SetK(k_adi);
  Real gamma_adi = pin->GetReal("hydro", "gamma");
  eos->SetGamma(gamma_adi);

#elif defined(USE_PIECEWISE_POLY)
  int n = pin->GetInteger("hydro", "n");
  Real gammas[n];
  for (int i = 0; i < n; i++)
    gammas[i] = pin->GetReal("hydro", "gamma" + std::to_string(i));
  Real rhos[n];
  rhos[0] = 0;  // needed in initialization but ignored in the function
  for (int i = 1; i < n; i++)
    rhos[i] = pin->GetReal("hydro", "rho" + std::to_string(i));
  Real P0 = pin->GetReal("hydro", "P0");
  Real mb = pin->GetOrAddReal("hydro", "bmass", 1.0);
  eos->InitializeFromData(rhos, gammas, P0, mb, n);
  Real gamma_th = pin->GetOrAddReal("hydro", "gamma_th", 5.0 / 3.0);
  eos->SetThermalGamma(gamma_th);

  Real temperature_floor_ =
    pin->GetOrAddReal("hydro", "tfloor", std::sqrt(1024 * (FLT_MIN)));
  eos->SetTemperature(temperature_floor_);
#endif

  Real density_floor =
    pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024 * (FLT_MIN)));
  eos->SetDensityFloor(density_floor);
}

//----------------------------------------------------------------------------------------
// Variable inverter
// Inputs:
//   cons: conserved quantities
//   prim_old: primitive quantities from previous half timestep
//   bb: face-centered magnetic field
//   pco: pointer to Coordinates
//   il,iu,jl,ju,kl,ku: index bounds of region to be updated
// Outputs:
//   prim: primitives
//   bb_cc: cell-centered magnetic field
// Notes:
//   follows Noble et al. 2006, ApJ 641 626 (N)
//       writing wgas_rel for W = \gamma^2 w
//       writing d for D
//       writing q for Q
//       writing qq for \tilde{Q}
//       writing uu for \tilde{u}
//       writing vv for v
//   implements formulas assuming no magnetic field

void EquationOfState::ConservedToPrimitive(AA& cons,
                                           const AA& prim_old,
                                           AA& prim,
                                           AA& cons_scalar,
                                           AA& prim_scalar,
                                           AA& bb_cc,
                                           Coordinates* pco,
                                           int il,
                                           int iu,
                                           int jl,
                                           int ju,
                                           int kl,
                                           int ku,
                                           int coarse_flag,
                                           bool skip_physical)
{
  MeshBlock* pmb = pmy_block_;
  Mesh* pm       = pmb->pmy_mesh;
  Hydro* ph      = pmb->phydro;
  Field* pf      = pmb->pfield;

  geom_sliced_cc gsc;

  AT_N_sca& alpha_          = gsc.alpha_;
  AT_N_sym& gamma_dd_       = gsc.gamma_dd_;
  AT_N_sym& gamma_uu_       = gsc.gamma_uu_;
  AT_N_sca& sqrt_det_gamma_ = gsc.sqrt_det_gamma_;

  AA c2p_status;
  c2p_status.InitWithShallowSlice(ph->derived_ms, IX_C2P, 1);

  AA temperature;
  temperature.InitWithShallowSlice(ph->derived_ms, IX_T, 1);

  const Real mb    = ps.GetEOS()->GetBaryonMass();
  const Real max_v = ps.GetEOS()->GetMaxVelocity();
  const Real max_W = OO(std::sqrt(1 - SQR(max_v)));

  // sanitize loop limits (coarse / fine auto-switched)
  int IL = il;
  int IU = iu;
  int JL = jl;
  int JU = ju;
  int KL = kl;
  int KU = ku;
  SanitizeLoopLimits(IL, IU, JL, JU, KL, KU, coarse_flag, pco);

  for (int k = KL; k <= KU; ++k)
    for (int j = JL; j <= JU; ++j)
    {
      GeometryToSlicedCC(gsc, k, j, IL, IU, coarse_flag, pco);

// do actual variable conversion ------------------------------------------
#pragma omp simd
      for (int i = IL; i <= IU; ++i)
      {
        if (skip_physical && (pmb->is <= i) && (i <= pmb->ie) &&
            (pmb->js <= j) && (j <= pmb->je) && (pmb->ks <= k) &&
            (k <= pmb->ke))
        {
          continue;
        }

        // Check if the state is admissible; if not we reset to atmo.
        bool is_admissible =
          IsAdmissiblePoint(cons, prim, SQR(sqrt_det_gamma_(i)), k, j, i);

        if (ph->opt_excision.excise_c2p)
        {
          if (ph->opt_excision.use_taper)
          {
            bool can_excise = CanExcisePoint(
              true, alpha_, pco->x1v, pco->x2v, pco->x3v, i, j, k);
            if (can_excise)
            {
              for (int n = 0; n < NHYDRO; ++n)
              {
                prim(n, k, j, i) *= ph->excision_mask(k, j, i);
              }
            }
          }
          else
          {
            bool can_excise = CanExcisePoint(
              true, alpha_, pco->x1v, pco->x2v, pco->x3v, i, j, k);
            is_admissible = is_admissible && !can_excise;
          }
        }

        if (!is_admissible)
        {
          PrimHelper::SetPrimAtmo(
            eos, prim, prim_scalar, k, j, i, &temperature);
          // SetEuclideanCC(gsc, i);
          PrimitiveToConservedSingle(prim,
                                     prim_scalar,
#if MAGNETIC_FIELDS_ENABLED
                                     bb_cc,
#endif
                                     cons,
                                     cons_scalar,
                                     pmy_block_->phydro->derived_ms,
                                     gsc,
                                     k,
                                     j,
                                     i,
                                     ps);

          if (c2p_status(k, j, i) == 0)
            c2p_status(k, j, i) = static_cast<int>(Primitive::Error::EXCISED);
        }

        // Deal with PrimitiveSolver interface
        Real g3d[NSPMETRIC] = { gamma_dd_(0, 0, i), gamma_dd_(0, 1, i),
                                gamma_dd_(0, 2, i), gamma_dd_(1, 1, i),
                                gamma_dd_(1, 2, i), gamma_dd_(2, 2, i) };

        const Real sqrt_detgamma    = sqrt_det_gamma_(i);
        const Real detgamma         = SQR(sqrt_detgamma);
        const Real oo_sqrt_detgamma = OO(sqrt_detgamma);

        Real g3u[NSPMETRIC] = { gamma_uu_(0, 0, i), gamma_uu_(0, 1, i),
                                gamma_uu_(0, 2, i), gamma_uu_(1, 1, i),
                                gamma_uu_(1, 2, i), gamma_uu_(2, 2, i) };

        // Extract and undensitize the conserved variables.
        Real cons_pt[NCONS]     = { 0.0 };
        Real cons_old_pt[NCONS] = {
          0.0
        };  // Redundancy in case things go bad.
        cons_pt[IDN] = cons_old_pt[IDN] =
          cons(IDN, k, j, i) * oo_sqrt_detgamma;
        cons_pt[IM1] = cons_old_pt[IM1] =
          cons(IM1, k, j, i) * oo_sqrt_detgamma;
        cons_pt[IM2] = cons_old_pt[IM2] =
          cons(IM2, k, j, i) * oo_sqrt_detgamma;
        cons_pt[IM3] = cons_old_pt[IM3] =
          cons(IM3, k, j, i) * oo_sqrt_detgamma;
        cons_pt[IEN] = cons_old_pt[IEN] =
          cons(IEN, k, j, i) * oo_sqrt_detgamma;

        // Extract the scalars
        for (int n = 0; n < NSCALARS; n++)
        {
          cons_pt[IYD + n] = cons_old_pt[IYD + n] =
            cons_scalar(n, k, j, i) * oo_sqrt_detgamma;
        }

        // Find the primitive variables.
        Real prim_pt[NPRIM] = { 0.0 };

#if MAGNETIC_FIELDS_ENABLED
        // Extract the magnetic field.
        Real b3u[NMAG] = { bb_cc(IB1, k, j, i) * oo_sqrt_detgamma,
                           bb_cc(IB2, k, j, i) * oo_sqrt_detgamma,
                           bb_cc(IB3, k, j, i) * oo_sqrt_detgamma };
#else
        Real b3u[NMAG] = { 0.0 };  // Assume no magnetic field.
#endif

        if (is_admissible)
        {
          Primitive::SolverResult result =
            ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);

          // retain result of c2p
          if (c2p_status(k, j, i) == 0)
          {
            c2p_status(k, j, i) = static_cast<int>(result.error);
          }

          // If the lapse or metric determinant fall below zero, we're probably
          // in an unphysical regime for a fluid, like a black hole or
          // something. Primitive failure is expected and will just result in a
          // floor being applied.
          if (verbose &&
              (result.error != Primitive::Error::SUCCESS && (detgamma > 0)))
          {
            std::cerr << "There was an error during the primitive solve!\n";
            std::cerr << "  Iteration: " << pmy_block_->pmy_mesh->ncycle
                      << "\n";
            std::cerr << "  Error: "
                      << Primitive::ErrorString[(int)result.error] << "\n";
            std::cerr << "  i=" << i << ", j=" << j << ", k=" << k << "\n";
            const Real x1 = pmy_block_->pcoord->x1v(i);
            const Real x2 = pmy_block_->pcoord->x2v(j);
            const Real x3 = pmy_block_->pcoord->x3v(k);
            std::cerr << "  (x1,x2,x3) " << x1 << "," << x2 << "," << x3
                      << "\n";
            std::cerr << "  g3d = [" << g3d[S11] << ", " << g3d[S12] << ", "
                      << g3d[S13] << ", " << g3d[S22] << ", " << g3d[S23]
                      << ", " << g3d[S33] << "]\n";
            std::cerr << "  g3u = [" << g3u[S11] << ", " << g3u[S12] << ", "
                      << g3u[S13] << ", " << g3u[S22] << ", " << g3u[S23]
                      << ", " << g3u[S33] << "]\n";
            std::cerr << "  detgamma  = " << detgamma << "\n";
            std::cerr << "  sqrt_detgamma = " << sqrt_detgamma << "\n";
            std::cerr << "  D = " << cons_old_pt[IDN] << "\n";
            std::cerr << "  S_1 = " << cons_old_pt[IM1] << "\n";
            std::cerr << "  S_2 = " << cons_old_pt[IM2] << "\n";
            std::cerr << "  S_3 = " << cons_old_pt[IM3] << "\n";
            std::cerr << "  tau = " << cons_old_pt[IEN] << "\n";
#if MAGNETIC_FIELDS_ENABLED
            std::cerr << "  b_u = [" << bb_cc(IB1, k, j, i) << ", "
                      << bb_cc(IB2, k, j, i) << ", " << bb_cc(IB3, k, j, i)
                      << "]\n";
#endif
          }

          // Update the primitive variables.
          prim(IDN, k, j, i) = prim_pt[IDN] * mb;
          prim(IVX, k, j, i) = prim_pt[IVX];
          prim(IVY, k, j, i) = prim_pt[IVY];
          prim(IVZ, k, j, i) = prim_pt[IVZ];
          prim(IPR, k, j, i) = prim_pt[IPR];

          for (int n = 0; n < NSCALARS; n++)
          {
            prim_scalar(n, k, j, i) = prim_pt[IYF + n];
          }

          // Write back conserved variables only if they were modified.
          if (result.cons_adjusted)
          {
            cons(IDN, k, j, i) = cons_pt[IDN] * sqrt_detgamma;
            cons(IM1, k, j, i) = cons_pt[IM1] * sqrt_detgamma;
            cons(IM2, k, j, i) = cons_pt[IM2] * sqrt_detgamma;
            cons(IM3, k, j, i) = cons_pt[IM3] * sqrt_detgamma;
            cons(IEN, k, j, i) = cons_pt[IEN] * sqrt_detgamma;
          }
          if (result.cons_adjusted || result.scalars_adjusted)
          {
            for (int n = 0; n < NSCALARS; n++)
            {
              cons_scalar(n, k, j, i) = cons_pt[IYD + n] * sqrt_detgamma;
            }
          }
        }
        else
        {
          // didn't need to run, but do need to extract variables
          // (see derived below)

          prim_pt[IDN] = prim(IDN, k, j, i) / mb;
          prim_pt[IVX] = prim(IVX, k, j, i);
          prim_pt[IVY] = prim(IVY, k, j, i);
          prim_pt[IVZ] = prim(IVZ, k, j, i);
          prim_pt[IPR] = prim(IPR, k, j, i);
          prim_pt[ITM] = temperature(k, j, i);

          for (int n = 0; n < NSCALARS; n++)
          {
            prim_pt[IYF + n] = prim_scalar(n, k, j, i);
          }
        }

        // BD: not clear why these behave differently (limiting)?
        if (recompute_temperature)
        {
          ph->derived_ms(IX_T, k, j, i) = ps.GetEOS()->GetTemperatureFromP(
            prim_pt[IDN], prim_pt[IPR], &prim_pt[IYF]);
        }
        else
        {
          ph->derived_ms(IX_T, k, j, i) = prim_pt[ITM];
        }

        // as in `PrimitiveSolver`
        ph->derived_ms(IX_LOR, k, j, i) =
          std::max(1.0, cons_pt[IDN] / (prim_pt[IDN] * mb));

        ph->derived_ms(IX_LOR, k, j, i) =
          std::min(max_W, ph->derived_ms(IX_LOR, k, j, i));

        // enthalpy update required at all substeps
        ph->derived_ms(IX_ETH, k, j, i) = GetEOS().GetEnthalpy(
          prim_pt[IDN], ph->derived_ms(IX_T, k, j, i), &prim_pt[IYF]);

        // required [CT specifically] on all substeps
        const Real oo_W  = OO(ph->derived_ms(IX_LOR, k, j, i));
        const Real alpha = gsc.alpha_(i);

        for (int a = 0; a < N; ++a)
        {
          ph->derived_int(IX_TR_V1 + a, k, j, i) =
            (alpha * oo_W * prim(IVX + a, k, j, i) - gsc.beta_u_(a, i));
        }
      }

      // derived quantities
      // ----------------------------------------------------- BD: TODO - some
      // optional reconstructed quantities should perhaps go here
    }
}

//----------------------------------------------------------------------------------------
// Function for converting all primitives to conserved variables
// Inputs:
//   prim: primitives
//   bb_cc: cell-centered magnetic field
//   pco: pointer to Coordinates
//   il,iu,jl,ju,kl,ku: index bounds of region to be updated
// Outputs:
//   cons: conserved variables
// Notes:
//   single-cell function exists for other purposes; call made to that function
//   rather
//       than having duplicate code

void EquationOfState::PrimitiveToConserved(AA& prim,
                                           AA& prim_scalar,
                                           AA& bb_cc,
                                           AA& cons,
                                           AA& cons_scalar,
                                           Coordinates* pco,
                                           int il,
                                           int iu,
                                           int jl,
                                           int ju,
                                           int kl,
                                           int ku)
{
  geom_sliced_cc gsc;

  // sanitize loop limits (coarse / fine auto-switched)
  const bool coarse_flag = false;
  int IL                 = il;
  int IU                 = iu;
  int JL                 = jl;
  int JU                 = ju;
  int KL                 = kl;
  int KU                 = ku;
  SanitizeLoopLimits(IL, IU, JL, JU, KL, KU, coarse_flag, pco);

  for (int k = KL; k <= KU; ++k)
    for (int j = JL; j <= JU; ++j)
    {
      GeometryToSlicedCC(gsc, k, j, IL, IU, coarse_flag, pco);
      // Calculate the conserved variables at every point.
      for (int i = IL; i <= IU; ++i)
      {
        PrimitiveToConservedSingle(prim,
                                   prim_scalar,
#if MAGNETIC_FIELDS_ENABLED
                                   bb_cc,
#endif
                                   cons,
                                   cons_scalar,
                                   pmy_block_->phydro->derived_ms,
                                   gsc,
                                   k,
                                   j,
                                   i,
                                   ps);
      }
    }
}

//----------------------------------------------------------------------------------------
namespace
{

static void PrimitiveToConservedSingle(AA& prim,
                                       AA& prim_scalar,
#if MAGNETIC_FIELDS_ENABLED
                                       AA& bb_cc,
#endif
                                       AA& cons,
                                       AA& cons_scalar,
                                       AA& derived_ms,
                                       EquationOfState::geom_sliced_cc& gsc,
                                       int k,
                                       int j,
                                       int i,
                                       PS& ps)
{
  AT_N_sym& gamma_dd_       = gsc.gamma_dd_;
  AT_N_sca& sqrt_det_gamma_ = gsc.sqrt_det_gamma_;

  // Extract the primitive variables
  Real prim_pt[NPRIM] = { 0.0 };
  Real Y[MAX_SPECIES] = { 0.0 };

  Real mb      = ps.GetEOS()->GetBaryonMass();
  prim_pt[IDN] = prim(IDN, k, j, i) / mb;
  prim_pt[IVX] = prim(IVX, k, j, i);
  prim_pt[IVY] = prim(IVY, k, j, i);
  prim_pt[IVZ] = prim(IVZ, k, j, i);
  prim_pt[IPR] = prim(IPR, k, j, i);

  for (int n = 0; n < NSCALARS; n++)
  {
    Y[n] = prim_scalar(n, k, j, i);
  }

  // Get temperature and apply floor
  ps.GetEOS()->ApplyDensityLimits(prim_pt[IDN]);
  ps.GetEOS()->ApplySpeciesLimits(Y);
  prim_pt[ITM] =
    ps.GetEOS()->GetTemperatureFromP(prim_pt[IDN], prim_pt[IPR], Y);
  bool result = ps.GetEOS()->ApplyPrimitiveFloor(
    prim_pt[IDN], &prim_pt[IVX], prim_pt[IPR], prim_pt[ITM], Y);

  for (int n = 0; n < NSCALARS; n++)
  {
    prim_pt[IYF + n] = Y[n];
  }

  // Extract the metric and calculate the determinant..
  Real g3d[NSPMETRIC] = { gamma_dd_(0, 0, i), gamma_dd_(0, 1, i),
                          gamma_dd_(0, 2, i), gamma_dd_(1, 1, i),
                          gamma_dd_(1, 2, i), gamma_dd_(2, 2, i) };
  Real sdetg          = sqrt_det_gamma_(i);
  Real detg           = SQR(sdetg);

#if MAGNETIC_FIELDS_ENABLED
  // Extract and undensitize the magnetic field.
  Real bu[NMAG] = { bb_cc(IB1, k, j, i) / sdetg,
                    bb_cc(IB2, k, j, i) / sdetg,
                    bb_cc(IB3, k, j, i) / sdetg };
#else
  Real bu[NMAG] = { 0.0 };
#endif

  // Perform the primitive solve.
  Real cons_pt[NCONS];

  ps.PrimToCon(prim_pt, cons_pt, bu, g3d);

  // DEBUG ONLY
  if (!std::isfinite(cons_pt[IEN]))
  {
    std::cerr << "Tau is not finite!\n";
    std::cerr << "  Error occurred at (" << i << ", " << j << ", " << k
              << ")\n";
    std::cerr << "  Primitive variables:\n";
    std::cerr << "    rho = " << prim(IDN, k, j, i) << "\n";
    std::cerr << "    ux  = " << prim(IVX, k, j, i) << "\n";
    std::cerr << "    uy  = " << prim(IVY, k, j, i) << "\n";
    std::cerr << "    uz  = " << prim(IVZ, k, j, i) << "\n";
    std::cerr << "    P   = " << prim(IPR, k, j, i) << "\n";
    std::cerr << "  Conserved variables:\n";
    std::cerr << "    D   = " << cons_pt[IDN] << "\n";
    std::cerr << "    Sx  = " << cons_pt[IM1] << "\n";
    std::cerr << "    Sy  = " << cons_pt[IM2] << "\n";
    std::cerr << "    Sz  = " << cons_pt[IM3] << "\n";
    std::cerr << "    tau = " << cons_pt[IEN] << "\n";
    std::cerr << "  Metric:\n";
    std::cerr << "    g3d = {" << g3d[S11] << ", " << g3d[S12] << ", "
              << g3d[S13] << ", " << g3d[S22] << ", " << g3d[S23] << ", "
              << g3d[S33] << "}\n";
    std::cerr << "    detg  = " << detg << "\n";
    std::cerr << "    sdetg = " << sdetg << "\n";
  }

  // Push the densitized conserved variables to Athena.
  cons(IDN, k, j, i) = cons_pt[IDN] * sdetg;
  cons(IM1, k, j, i) = cons_pt[IM1] * sdetg;
  cons(IM2, k, j, i) = cons_pt[IM2] * sdetg;
  cons(IM3, k, j, i) = cons_pt[IM3] * sdetg;
  cons(IEN, k, j, i) = cons_pt[IEN] * sdetg;
  for (int n = 0; n < NSCALARS; n++)
  {
    cons_scalar(n, k, j, i) = cons_pt[IYD + n] * sdetg;
  }

  // If we floored things, we'll need to readjust the primitives.
  if (result)
  {
    prim(IDN, k, j, i) = prim_pt[IDN] * mb;
    prim(IVX, k, j, i) = prim_pt[IVX];
    prim(IVY, k, j, i) = prim_pt[IVY];
    prim(IVZ, k, j, i) = prim_pt[IVZ];
    prim(IPR, k, j, i) = prim_pt[IPR];
    for (int n = 0; n < NSCALARS; n++)
    {
      prim_scalar(n, k, j, i) = prim_pt[IYF + n];
    }

    derived_ms(IX_LOR, k, j, i) = 1;
  }

  derived_ms(IX_T, k, j, i) = prim_pt[ITM];
}

}  // namespace
