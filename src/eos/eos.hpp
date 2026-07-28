#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"                   // Real
#include "../athena_arrays.hpp"            // AthenaArray
#include "../coordinates/coordinates.hpp"  // Coordinates

#if FLUID_ENABLED
// PrimitiveSolver headers
#include "../z4c/primitive/coldeos.hpp"
#include "../z4c/primitive/eos.hpp"
#include "../z4c/primitive/primitive_solver.hpp"
#include "include_eos.hpp"
#include "primitive_solver_helper.hpp"  // PrimHelper
#endif

// Declarations
class Hydro;
class ParameterInput;
struct FaceField;

//! \class EquationOfState
//  \brief data and functions that implement EoS

class EquationOfState
{
  friend class Hydro;

  public:
  EquationOfState(MeshBlock* pmb, ParameterInput* pin);

  bool verbose               = true;
  bool restrict_cs2          = false;
  Real max_cs_W              = 10;  // 0.99c
  Real max_cs2               = 1.0 - SQR(1.0 / max_cs_W);
  bool recompute_temperature = true;
  bool smooth_temperature    = false;
  bool recompute_enthalpy    = false;

  // BD: Avoid messy macro pollution with some polymorphism & interfaces ------

#if FLUID_ENABLED
  void ConservedToPrimitive(AthenaArray<Real>& cons,
                            const AthenaArray<Real>& prim_old,
                            AthenaArray<Real>& prim,
                            AthenaArray<Real>& cons_scalar,
                            AthenaArray<Real>& prim_scalar,
                            AthenaArray<Real>& bcc,
                            Coordinates* pco,
                            int il,
                            int iu,
                            int jl,
                            int ju,
                            int kl,
                            int ku,
                            int coarseflag,
                            bool skip_physical);

  void ConservedToPrimitive(AthenaArray<Real>& cons,
                            const AthenaArray<Real>& prim_old,
                            AthenaArray<Real>& prim,
                            AthenaArray<Real>& cons_scalar,
                            AthenaArray<Real>& prim_scalar,
                            AthenaArray<Real>& bcc,
                            Coordinates* pco,
                            int il,
                            int iu,
                            int jl,
                            int ju,
                            int kl,
                            int ku,
                            int coarseflag)
  {
    ConservedToPrimitive(cons,
                         prim_old,
                         prim,
                         cons_scalar,
                         prim_scalar,
                         bcc,
                         pco,
                         il,
                         iu,
                         jl,
                         ju,
                         kl,
                         ku,
                         coarseflag,
                         false);
  }

  // Similarly for PrimitiveToConserved ---------------------------------------

  void PrimitiveToConserved(AthenaArray<Real>& prim,
                            AthenaArray<Real>& prim_scalar,
                            AthenaArray<Real>& bc,
                            AthenaArray<Real>& cons,
                            AthenaArray<Real>& cons_scalar,
                            Coordinates* pco,
                            int il,
                            int iu,
                            int jl,
                            int ju,
                            int kl,
                            int ku);
#endif  // FLUID_ENABLED

  // --------------------------------------------------------------------------
  // Check state vector at a point makes sense & we are not
  bool IsAdmissiblePoint(const AA& cons,
                         const AA& prim,
                         const Real adm_detgamma,
                         const int k,
                         const int j,
                         const int i);

  bool CanExcisePoint(const bool is_slice,
                      AT_N_sca& alpha,
                      AA& x1,
                      AA& x2,
                      AA& x3,
                      const int i,
                      const int j,
                      const int k);

  bool CanExcisePoint(Real& excision_factor,
                      const bool is_slice,
                      AT_N_sca& alpha,
                      AA& x1,
                      AA& x2,
                      AA& x3,
                      const int i,
                      const int j,
                      const int k);

  void SanitizeLoopLimits(int& il,
                          int& iu,
                          int& jl,
                          int& ju,
                          int& kl,
                          int& ku,
                          const bool coarse_flag,
                          Coordinates* pco);

#if FLUID_ENABLED
  // Check if conserved density is under a floor cutoff factor.
  // Returns true when every cell in the given range satisfies the threshold.
  bool ConservedDensityWithinFloorThreshold(AA& u,
                                            AA& sqrt_detgamma,
                                            const Real undensitized_dfloor_fac,
                                            int il,
                                            int iu,
                                            int jl,
                                            int ju,
                                            int kl,
                                            int ku);
#endif

  // Use the same logic for slicing geometric entities to CC.
  struct geom_sliced_cc
  {
    // sliced
    AT_N_sym sl_adm_gamma_dd;
    AT_N_sca sl_alpha;
    AT_N_sca sl_chi;
    AT_N_sca sl_adm_sqrt_detgamma;
    AT_N_vec sl_beta_u;
    // interpolated to CC
    AT_N_sca alpha_;
    AT_N_sca rchi_;
    AT_N_vec beta_u_;
    AT_N_sym gamma_dd_;
    // derived on CC
    AT_N_sym gamma_uu_;
    AT_N_sca sqrt_det_gamma_;
    // start false to get first alloc. then it prevents later realloc
    bool is_scratch_allocated = false;
  };

  void StatePrintPoint(const std::string& tag,
                       MeshBlock* pmb,
                       geom_sliced_cc& gsc,
                       const int k,
                       const int j,
                       const int i,
                       const bool terminate);

  void GeometryToSlicedCC(geom_sliced_cc& gsc,
                          const int k,
                          const int j,
                          const int il,
                          const int iu,
                          const bool coarse_flag,
                          Coordinates* pco);

  // Various derived quantities -----------------------------------------------
  void DerivedQuantities(AA& hyd_der_ms,
                         AA& hyd_der_int,
                         AA& fld_der_ms,
                         AA& cons,
                         AA& cons_scalar,
                         AA& prim,
                         AA& prim_scalar,
                         AA& bcc,
                         geom_sliced_cc& gsc,
                         Coordinates* pco,
                         int k,
                         int j,
                         int il,
                         int iu,
                         int coarseflag,
                         bool skip_physical);

  void DerivedQuantities(AA& hyd_der_ms,
                         AA& hyd_der_int,
                         AA& fld_der_ms,
                         AA& cons,
                         AA& cons_scalar,
                         AA& prim,
                         AA& prim_scalar,
                         AA& bcc,
                         geom_sliced_cc& gsc,
                         Coordinates* pco,
                         int k,
                         int j,
                         int il,
                         int iu,
                         int coarseflag)
  {
    DerivedQuantities(hyd_der_ms,
                      hyd_der_int,
                      fld_der_ms,
                      cons,
                      cons_scalar,
                      prim,
                      prim_scalar,
                      bcc,
                      gsc,
                      pco,
                      k,
                      j,
                      il,
                      iu,
                      coarseflag,
                      false);
  }

  bool NeighborsEncloseValue(const AA& src,
                             const int n,
                             const int k,
                             const int j,
                             const int i,
                             const AA_B& mask,
                             const int num_neighbors,
                             const bool exclude_first_extrema,
                             const Real fac_min = 1.0,
                             const Real fac_max = 1.0);

  void NearestNeighborSmooth(AA& tar,
                             const AA& src,
                             const int kl,
                             const int ku,
                             const int jl,
                             const int ju,
                             const int il,
                             const int iu,
                             bool exclude_first_extrema);

  // Smooth derived_ms(IX_T,:) by nearest-neighbour averaging and refresh
  // derived enthalpy / cs2 / entropy-per-baryon as appropriate.
  // No-op when smooth_temperature is false.
  // w1(0,:) is used as scratch and left in an unspecified state on return
  // (callers should invoke RetainState(w1, w, ...) afterwards).
  void SmoothTemperatureAndRecompute(AA& w,
                                     AA& w1,
                                     AA& derived_ms,
                                     const AA& r,
                                     int il,
                                     int iu,
                                     int jl,
                                     int ju,
                                     int kl,
                                     int ku,
                                     bool recompute_cs2,
                                     bool recompute_entropy);

  Real NearestNeighborSmooth(const AA& src,
                             const int n,
                             const int k,
                             const int j,
                             const int i,
                             const AA_B& mask,
                             const int num_neighbors,
                             const bool keep_base_point,
                             const bool exclude_first_extrema,
                             const bool use_hybrid_mean_median,
                             const Real sigma_frac = 0.0);

  Real NearestNeighborSmoothWeighted(const AA& src,
                                     const int n,
                                     const int k,
                                     const int j,
                                     const int i,
                                     const AA_B& mask,
                                     const int num_neighbors,
                                     const bool keep_base_point,
                                     const bool exclude_first_extrema,
                                     const bool use_hybrid_mean_median,
                                     const Real sigma_frac = 0.0,
                                     const Real alpha      = 0.5);

  Real NearestNeighborSmooth(
    const AA& src,
    const int n,
    const int k,
    const int j,
    const int i,
    const AA_B& mask,
    const int num_neighbors,
    const bool keep_base_point,
    const bool exclude_first_extrema,
    const bool use_robust_weights,  // toggle robust weighting
    const Real alpha,               // blend factor [0,1]
    const Real sigma_frac,    // fraction of base value for robust weighting
    const Real max_dev_frac,  // new: max deviation allowed (e.g. 0.1 = 10%)
    const Real sigma_s_frac =
      0.5  // fraction of num_neighbors for spatial weight
  );

#if FLUID_ENABLED
#if defined(USE_TRANSITION_EOS)
  // Reference baryon mass [MeV]: m_u minus the largest binding energy per
  // baryon the RHINE mass model can produce (model_1 output range minimum,
  // -0.939160168170929 MeV/baryon). SCEB = 0 then corresponds exactly to
  // the network's most bound state, so network evolution keeps SCEB (and
  // the specific internal energy) non-negative.
  static constexpr Real transition_baryon_mass_MeV =
    931.4939509082333 - 0.939160168170929;  // = 930.5547907400624

  // One-time setup of the RHINE nuclear-network emulator (model files,
  // pmode, unit conversions). Called from every EquationOfState
  // constructor; only the first call loads the models. An empty
  // rhine_models_path disables RHINE (rates stay zero).
  static void InitTransitionNetwork(ParameterInput* pin);

  // Transition physics on one (k, j) row of PHYSICAL cells: NSE
  // composition/binding reset (w == 1) and RHINE rate evaluation (w < 1),
  // writing IX_TRANS / IX_HEAT / IX_FNU. With dt_apply_code <= 0 the rates
  // are diagnostic only (pass 1); with dt_apply_code > 0 (the RK substep
  // beta*dt) they are additionally applied as explicit sources to
  // cons_scalar and the tau neutrino sink, with the '0' reference
  // composition frozen at full-step start (pscalars->r0).
  void TransitionNetworkStep(AA& prim,
                             AA& prim_scalar,
                             AA& cons,
                             AA& cons_scalar,
                             AA& hyd_der_ms,
                             AA& hyd_der_int,
                             geom_sliced_cc& gsc,
                             Coordinates* pco,
                             int k,
                             int j,
                             const Real dt_apply_code = 0.0);

  // Per-RK-substep RHINE source application over this EOS's MeshBlock
  // (pass 2). No-op unless hydro/rhine_apply = true and models are loaded.
  // At stage 1 snapshots pscalars->r into pscalars->r0.
  void TransitionNetworkApply(const Real dt_scaled, const int stage);

  // Re-synchronize the advected composition of NSE cells (w == 1) with
  // the table. Called once per time step, AFTER the final RK stage's C2P:
  // resyncing mid-step would overwrite registers between stage
  // combinations and invalidate the frozen r0 reference the network's
  // endpoint positivity is certified against (the Xp < 0 events at the
  // NSE-surface strip). Post-step, the next step's r0 snapshot picks up
  // the resynced state -- reference consistent by construction.
  void TransitionNSEResync();

  // Output-cadence diagnostics only (IX_TRANS, IX_XERR); the physics
  // lives in TransitionNetworkStep.
  void TransitionDiagnostics(AA& prim,
                             AA& prim_scalar,
                             AA& hyd_der_ms,
                             int k,
                             int j,
                             int i);
#endif
  inline Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>&
  GetEOS()
  {
    return eos;
  }
#endif

  private:
  MeshBlock* pmy_block_;  // ptr to MeshBlock containing this EOS
#if FLUID_ENABLED
  // If we're using the PrimitiveSolver framework, we need to declare the
  // EOS and PrimitiveSolver objects.
  Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> eos;
  Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>
    ps;
#endif
};

#if FLUID_ENABLED
void InitColdEOS(Primitive::ColdEOS<Primitive::COLDEOS_POLICY>* eos,
                 ParameterInput* pin);
#endif

#endif  // EOS_EOS_HPP_
