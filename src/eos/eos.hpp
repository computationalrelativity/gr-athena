#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"         // Real
#include "../athena_arrays.hpp"  // AthenaArray
#include "../coordinates/coordinates.hpp" // Coordinates

#if FLUID_ENABLED
// PrimitiveSolver headers
#include "../z4c/primitive/eos.hpp"
#include "../z4c/primitive/coldeos.hpp"
#include "../z4c/primitive/primitive_solver.hpp"
#include "include_eos.hpp"
#endif

// Declarations
class Hydro;
class ParameterInput;
struct FaceField;

//! \class EquationOfState
//  \brief data and functions that implement EoS

class EquationOfState {
  friend class Hydro;
 public:
  EquationOfState(MeshBlock *pmb, ParameterInput *pin);

  bool verbose = true;
  bool restrict_cs2 = false;
  Real max_cs_W = 10;  // 0.99c
  Real max_cs2  = 1.0 - SQR(1.0 / max_cs_W);
  bool warn_unrestricted_cs2 = false;
  bool recompute_temperature = true;
  bool smooth_temperature = false;
  bool recompute_enthalpy = false;

  // BD: Avoid messy macro pollution with some polymorphism & interfaces ------

#if FLUID_ENABLED
  void ConservedToPrimitive(AthenaArray<Real> &cons,
                            const AthenaArray<Real> &prim_old,
                            AthenaArray<Real> &prim,
                            AthenaArray<Real> &cons_scalar,
                            AthenaArray<Real> &prim_scalar,
                            AthenaArray<Real> &bcc, Coordinates *pco,
                            int il, int iu,
                            int jl, int ju,
                            int kl, int ku,
                            int coarseflag,
                            bool skip_physical);

  void ConservedToPrimitive(AthenaArray<Real> &cons,
                            const AthenaArray<Real> &prim_old,
                            AthenaArray<Real> &prim,
                            AthenaArray<Real> &cons_scalar,
                            AthenaArray<Real> &prim_scalar,
                            AthenaArray<Real> &bcc, Coordinates *pco,
                            int il, int iu,
                            int jl, int ju,
                            int kl, int ku,
                            int coarseflag)
  {
    ConservedToPrimitive(cons, prim_old, prim, cons_scalar, prim_scalar,
                         bcc, pco,
                         il, iu, jl, ju, kl, ku, coarseflag, false);
  }

  // Similarly for PrimitiveToConserved ---------------------------------------

  void PrimitiveToConserved(AthenaArray<Real> &prim,
                            AthenaArray<Real> &prim_scalar,
                            AthenaArray<Real> &bc,
                            AthenaArray<Real> &cons,
                            AthenaArray<Real> &cons_scalar,
                            Coordinates *pco,
                            int il, int iu, int jl, int ju, int kl, int ku);
#endif // FLUID_ENABLED

  // --------------------------------------------------------------------------
  // Check state vector at a point makes sense & we are not
  bool IsAdmissiblePoint(
    const AA & cons,
    const AA & prim,
    const AT_N_sca & adm_detgamma_,
    const int k, const int j, const int i);

  bool CanExcisePoint(
    const bool is_slice,
    AT_N_sca & alpha,
    AA & x1,
    AA & x2,
    AA & x3,
    const int i, const int j, const int k);

  bool CanExcisePoint(
    Real & excision_factor,
    const bool is_slice,
    AT_N_sca & alpha,
    AA & x1,
    AA & x2,
    AA & x3,
    const int i, const int j, const int k);

  void SanitizeLoopLimits(
    int & il, int & iu,
    int & jl, int & ju,
    int & kl, int & ku,
    const bool coarse_flag,
    Coordinates *pco);

  // Use the same logic for slicing geometric entities to CC.
  struct geom_sliced_cc {
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
    AT_N_sca det_gamma_;
    AT_N_sca oo_det_gamma_;
    // start false to get first alloc. then it prevents later realloc
    bool is_scratch_allocated = false;
  };

  void StatePrintPoint(
    const std::string & tag,
    MeshBlock *pmb,
    geom_sliced_cc & gsc,
    const int k, const int j, const int i,
    const bool terminate);


  void GeometryToSlicedCC(
    geom_sliced_cc & gsc,
    const int k, const int j, const int il, const int iu,
    const bool coarse_flag,
    Coordinates *pco
  );

  // Various derived quantities -----------------------------------------------
  void DerivedQuantities(
    AA &hyd_der_ms,
    AA &hyd_der_int,
    AA &fld_der_ms,
    AA &cons, AA &cons_scalar,
    AA &prim, AA &prim_scalar,
    AA &bcc, geom_sliced_cc & gsc,
    Coordinates *pco,
    int k,
    int j,
    int il, int iu,
    int coarseflag,
    bool skip_physical);

  void DerivedQuantities(
    AA &hyd_der_ms,
    AA &hyd_der_int,
    AA &fld_der_ms,
    AA &cons, AA &cons_scalar,
    AA &prim, AA &prim_scalar,
    AA &bcc, geom_sliced_cc & gsc,
    Coordinates *pco,
    int k,
    int j,
    int il, int iu,
    int coarseflag)
  {
    DerivedQuantities(hyd_der_ms,
                      hyd_der_int,
                      fld_der_ms,
                      cons, cons_scalar,
                      prim, prim_scalar,
                      bcc, gsc, pco,
                      k, j, il, iu, coarseflag, false);
  }

  bool NeighborsEncloseValue(
    const AA &src,
    const int n,
    const int k,
    const int j,
    const int i,
    const AA_B &mask,
    const int num_neighbors,
    const bool exclude_first_extrema,
    const Real fac_min=1.0,
    const Real fac_max=1.0
  );

  void NearestNeighborSmooth(
    AA &tar,
    const AA &src,
    const int kl, const int ku,
    const int jl, const int ju,
    const int il, const int iu,
    bool exclude_first_extrema);

  Real NearestNeighborSmooth(
    const AA &src,
    const int n,
    const int k,
    const int j,
    const int i,
    const AA_B &mask,
    const int num_neighbors,
    const bool keep_base_point,
    const bool exclude_first_extrema,
    const bool use_hybrid_mean_median,
    const Real sigma_frac=0.0
  );

  Real NearestNeighborSmoothWeighted(
    const AA &src,
    const int n,
    const int k,
    const int j,
    const int i,
    const AA_B &mask,
    const int num_neighbors,
    const bool keep_base_point,
    const bool exclude_first_extrema,
    const bool use_hybrid_mean_median,
    const Real sigma_frac=0.0,
    const Real alpha=0.5
  );

  Real NearestNeighborSmooth(
    const AA &src,
    const int n,
    const int k,
    const int j,
    const int i,
    const AA_B &mask,
    const int num_neighbors,
    const bool keep_base_point,
    const bool exclude_first_extrema,
    const bool use_robust_weights,   // toggle robust weighting
    const Real alpha,                // blend factor [0,1]
    const Real sigma_frac,           // fraction of base value for robust weighting
    const Real max_dev_frac,         // new: max deviation allowed (e.g. 0.1 = 10%)
    const Real sigma_s_frac = 0.5    // fraction of num_neighbors for spatial weight
  );

  // BD: TODO - clean up this mess ---v

  // pass k, j, i to following 2x functions even though x1-sliced input array is expected
  // in order to accomodate position-dependent floors
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this,prim,k,j) linear(i)
#if FLUID_ENABLED
  void ApplyPrimitiveFloors(AthenaArray<Real> &prim, AthenaArray<Real> &prim_scalar, int k, int j, int i);
#else
  void ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i);
  void ApplyPrimitiveFloors(const int dir,
                            AthenaArray<Real> &prim_l,
                            AthenaArray<Real> &prim_r, int i);
#endif

#if FLUID_ENABLED
  void SetPrimAtmo(
    AA &prim,
    AA &prim_scalar,
    const int k, const int j, const int i)
  {
    Real prim_pt[NPRIM] = {0.0};
    ps.GetEOS()->DoFailureResponse(prim_pt);

    // Update the primitive variables.
    prim(IDN, k, j, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
    prim(IVX, k, j, i) = prim_pt[IVX];
    prim(IVY, k, j, i) = prim_pt[IVY];
    prim(IVZ, k, j, i) = prim_pt[IVZ];
    prim(IPR, k, j, i) = prim_pt[IPR];
    for(int n=0; n<NSCALARS; n++){
      prim_scalar(n, k, j, i) = prim_pt[IYF + n];
    }
  }

  void SetPrimAtmo(
    AA &temperature,
    AA &prim,
    AA &prim_scalar,
    const int k, const int j, const int i)
  {
    Real prim_pt[NPRIM] = {0.0};
    ps.GetEOS()->DoFailureResponse(prim_pt);

    // Update the primitive variables.
    prim(IDN, k, j, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
    prim(IVX, k, j, i) = prim_pt[IVX];
    prim(IVY, k, j, i) = prim_pt[IVY];
    prim(IVZ, k, j, i) = prim_pt[IVZ];
    prim(IPR, k, j, i) = prim_pt[IPR];
    temperature(k,j,i) = prim_pt[ITM];
    for(int n=0; n<NSCALARS; n++){
      prim_scalar(n, k, j, i) = prim_pt[IYF + n];
    }
  }

  void SetPrimAtmo(
    AA &prim_,
    AA &prim_scalar_,
    const int i)
  {
    Real prim_pt[NPRIM] = {0.0};
    ps.GetEOS()->DoFailureResponse(prim_pt);

    // Update the primitive variables.
    prim_(IDN, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
    prim_(IVX, i) = prim_pt[IVX];
    prim_(IVY, i) = prim_pt[IVY];
    prim_(IVZ, i) = prim_pt[IVZ];
    prim_(IPR, i) = prim_pt[IPR];
    for(int n=0; n<NSCALARS; n++){
      prim_scalar_(n, i) = prim_pt[IYF + n];
    }
  }

  void SetPrimAtmo(
    AA &temperature_,
    AA &prim_,
    AA &prim_scalar_,
    const int i)
  {
    Real prim_pt[NPRIM] = {0.0};
    ps.GetEOS()->DoFailureResponse(prim_pt);

    // Update the primitive variables.
    prim_(IDN, i) = prim_pt[IDN]*ps.GetEOS()->GetBaryonMass();
    prim_(IVX, i) = prim_pt[IVX];
    prim_(IVY, i) = prim_pt[IVY];
    prim_(IVZ, i) = prim_pt[IVZ];
    prim_(IPR, i) = prim_pt[IPR];
    temperature_(i) = prim_pt[ITM];
    for(int n=0; n<NSCALARS; n++){
      prim_scalar_(n, i) = prim_pt[IYF + n];
    }
  }
#else
  void SetPrimAtmo(
    AA &temperature,
    AA &prim,
    AA &prim_scalar,
    const int k, const int j, const int i)
  {
    assert(false);
  }

  void SetPrimAtmo(
    AA &prim,
    AA &prim_scalar,
    const int k, const int j, const int i)
  {
    assert(false);
  }

  void SetPrimAtmo(
    AA &temperature,
    AA &prim,
    AA &prim_scalar,
    const int i)
  {
    assert(false);
  }

  void SetPrimAtmo(
    AA &prim,
    AA &prim_scalar,
    const int i)
  {
    assert(false);
  }
#endif // FLUID_ENABLED


// BD: TODO - many functions don't do what their names suggest.
//            Sound speed != eigenvalue..
// Should clean this up at some point, for now, just add actual sound speed

  // Sound speed functions in different regimes
#if !GENERAL_RELATIVITY  // Newtonian: GR defined as no-op
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  Real SoundSpeed(const Real prim[(NHYDRO)]);
#if !MAGNETIC_FIELDS_ENABLED  // Newtonian hydro: Newtonian MHD defined as no-op
  Real FastMagnetosonicSpeed(const Real[], const Real) {return 0.0;}
#else  // Newtonian MHD
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  Real FastMagnetosonicSpeed(const Real prim[(NWAVE)], const Real bx);
#endif  // !MAGNETIC_FIELDS_ENABLED
  void SoundSpeedsSR(Real, Real, Real, Real, Real *, Real *) {return;}
  void FastMagnetosonicSpeedsSR(
      const AthenaArray<Real> &, const AthenaArray<Real> &,
      int, int, int, int, int, AthenaArray<Real> &,
      AthenaArray<Real> &) {return;}
  void SoundSpeedsGR(Real, Real, Real, Real, Real, Real, Real, Real *, Real *)
  {return;}
  void FastMagnetosonicSpeedsGR(Real, Real, Real, Real, Real, Real, Real, Real, Real *,
                                Real *) {return;}
#else  // GR: Newtonian defined as no-op
  Real SoundSpeed(const Real[]) {return 0.0;}
  Real FastMagnetosonicSpeed(const Real[], const Real) {return 0.0;}
#if !MAGNETIC_FIELDS_ENABLED  // GR hydro: GR+SR MHD defined as no-op
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)

#if FLUID_ENABLED
  void SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
                     Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
#else
  void SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
                     Real *plambda_plus, Real *plambda_minus);
#endif // FLUID_ENABLED
  void FastMagnetosonicSpeedsSR(
      const AthenaArray<Real> &, const AthenaArray<Real> &,
      int, int, int, int, int, AthenaArray<Real> &,
      AthenaArray<Real> &) {return;}
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#if FLUID_ENABLED
  void SoundSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1,
                     Real g00, Real g01, Real g11,
                     Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);

  void SoundSpeedsGR(Real cs_2, Real rho_h, Real pgas, Real u0, Real u1,
                     Real g00, Real g01, Real g11,
                     Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
#else
  void SoundSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1,
                     Real g00, Real g01, Real g11,
                     Real *plambda_plus, Real *plambda_minus);
#endif // FLUID_ENABLED
  void FastMagnetosonicSpeedsGR(Real, Real, Real, Real, Real, Real, Real, Real,
                                Real *, Real *) {return;}
#else  // GR MHD: GR+SR hydro defined as no-op
  void SoundSpeedsSR(Real, Real, Real, Real, Real *, Real *) {return;}
  void FastMagnetosonicSpeedsSR(
      const AthenaArray<Real> &prim, const AthenaArray<Real> &bbx_vals,
      int k, int j, int il, int iu, int ivx,
      AthenaArray<Real> &lambdas_p, AthenaArray<Real> &lambdas_m);
  void SoundSpeedsGR(Real, Real, Real, Real, Real, Real, Real, Real *, Real *)
  {return;}
#if FLUID_ENABLED
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  void FastMagnetosonicSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1, Real b_sq,
                                Real g00, Real g01, Real g11,
                                Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
  void FastMagnetosonicSpeedsGR(Real cs_2, Real rho_h, Real pgas, Real u0, Real u1, Real b_sq,
                                Real g00, Real g01, Real g11,
                                Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
#else
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  void FastMagnetosonicSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1, Real b_sq,
                                Real g00, Real g01, Real g11,
                                Real *plambda_plus, Real *plambda_minus);
#endif  // FLUID_ENABLED
#endif  // !MAGNETIC_FIELDS_ENABLED (GR)
#endif  // !GENERAL_RELATIVITY

  Real GetDensityFloor() const {return density_floor_;}
#if FLUID_ENABLED
  inline Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& GetEOS() {
    return eos;
  }
  Real GetTemperatureFloor() const {return temperature_floor_;}
#endif
  Real GetGamma() const {return gamma_;}

 private:
  MeshBlock *pmy_block_;                 // ptr to MeshBlock containing this EOS
  Real gamma_;                            // ratio of specific heats
  Real density_floor_;                   // density floor
  Real scalar_floor_; // dimensionless concentration floor
#if FLUID_ENABLED
  // If we're using the PrimitiveSolver framework, we need to declare the
  // EOS and PrimitiveSolver objects.
  Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> eos;
  Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> ps;
  Real temperature_floor_;
#endif
};

#if FLUID_ENABLED
void InitColdEOS(Primitive::ColdEOS<Primitive::COLDEOS_POLICY> *eos,
                 ParameterInput *pin);
#endif

#endif // EOS_EOS_HPP_
