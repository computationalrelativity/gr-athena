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
