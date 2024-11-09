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
#include <limits>     // std::numeric_limits<float>

// Athena++ headers
#include "../athena.hpp"         // Real
#include "../athena_arrays.hpp"  // AthenaArray
#include "../coordinates/coordinates.hpp" // Coordinates
#include "../utils/interp_table.hpp"

#if USETM
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

  // BD: Avoid messy macro pollution with some polymorphism & interfaces ------
  void PassiveScalarConservedToPrimitive(AthenaArray<Real> &s,
                                         const AthenaArray<Real> &w,
                                         const AthenaArray<Real> &r_old,
                                         AthenaArray<Real> &r,
                                         Coordinates *pco,
                                         int il, int iu,
                                         int jl, int ju,
                                         int kl, int ku);

#if USETM
  void ConservedToPrimitive(AthenaArray<Real> &cons,
                            const AthenaArray<Real> &prim_old,
                            AthenaArray<Real> &prim,
                            AthenaArray<Real> &cons_scalar,
                            AthenaArray<Real> &prim_scalar,
                            AthenaArray<Real> &bcc, Coordinates *pco,
                            int il, int iu,
                            int jl, int ju,
                            int kl, int ku,
                            int coarseflag);
#else
  void ConservedToPrimitive(AthenaArray<Real> &cons,
                            const AthenaArray<Real> &prim_old,
                            AthenaArray<Real> &prim,
                            AthenaArray<Real> &bcc,
                            Coordinates *pco,
                            int il, int iu,
                            int jl, int ju,
                            int kl, int ku,
                            int coarseflag);

  inline void ConservedToPrimitive(AthenaArray<Real> &cons,
                                   const AthenaArray<Real> &prim_old,
                                   AthenaArray<Real> &prim,
                                   AthenaArray<Real> &cons_scalar,
                                   AthenaArray<Real> &prim_scalar,
                                   AthenaArray<Real> &bcc,
                                   Coordinates *pco,
                                   int il, int iu,
                                   int jl, int ju,
                                   int kl, int ku,
                                   int coarseflag)
  {
    ConservedToPrimitive(cons, prim_old, prim, bcc, pco,
                         il, iu, jl, ju, kl, ku, coarseflag);

    if (NSCALARS > 0)
    {
      PassiveScalarConservedToPrimitive(cons_scalar,
                                        prim,
                                        prim_scalar, // old and new distinction
                                        prim_scalar, // is not used for anything
                                        pco,
                                        il, iu, jl, ju, kl, ku);
    }
  }
#endif // USETM

  // Similarly for PrimitiveToConserved ---------------------------------------

  void PassiveScalarPrimitiveToConserved(const AthenaArray<Real> &r,
                                         const AthenaArray<Real> &w,
                                         AthenaArray<Real> &s,
                                         Coordinates *pco,
                                         int il, int iu,
                                         int jl, int ju,
                                         int kl, int ku);

#if USETM
  void PrimitiveToConserved(AthenaArray<Real> &prim,
                            AthenaArray<Real> &prim_scalar,
                            AthenaArray<Real> &bc,
                            AthenaArray<Real> &cons,
                            AthenaArray<Real> &cons_scalar,
                            Coordinates *pco,
                            int il, int iu, int jl, int ju, int kl, int ku);
#else
  // Define prototype without scalars
  void PrimitiveToConserved(AthenaArray<Real> &prim,
                            AthenaArray<Real> &bc,
                            AthenaArray<Real> &cons,
                            Coordinates *pco,
                            int il, int iu, int jl, int ju, int kl, int ku);

  // Handle split-passive scalar reconstruction
  inline void PrimitiveToConserved(AthenaArray<Real> &prim,
                                   AthenaArray<Real> &prim_scalar,
                                   AthenaArray<Real> &bc,
                                   AthenaArray<Real> &cons,
                                   AthenaArray<Real> &cons_scalar,
                                   Coordinates *pco,
                                   int il, int iu,
                                   int jl, int ju,
                                   int kl, int ku)
  {
#if FLUID_ENABLED
    PrimitiveToConserved(prim, bc, cons, pco, il, iu, jl, ju, kl, ku);

    if (NSCALARS > 0)
    {
      PassiveScalarPrimitiveToConserved(prim_scalar, prim, cons_scalar, pco,
                                        il, iu, jl, ju, kl, ku);
    }
#endif
  }

#endif

  // --------------------------------------------------------------------------
  // Check state vector at a point makes sense & we are not
  bool IsAdmissiblePoint(
    const AA & cons,
    const AA & prim,
    const AT_N_sca & adm_detgamma_,
    const int k, const int j, const int i);

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
    // interpolated to CC
    AT_N_sca alpha_;
    AT_N_sca rchi_;
    AT_N_sym gamma_dd_;
    // derived on CC
    AT_N_sym gamma_uu_;
    AT_N_sca det_gamma_;
    AT_N_sca oo_det_gamma_;
    // start false to get first alloc. then it prevents later realloc
    bool is_scratch_allocated = false;
  };

  void GeometryToSlicedCC(
    geom_sliced_cc & gsc,
    const int k, const int j, const int il, const int iu,
    const bool coarse_flag,
    Coordinates *pco
  );

  // BD: TODO - clean up this mess ---v

  // pass k, j, i to following 2x functions even though x1-sliced input array is expected
  // in order to accomodate position-dependent floors
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this,prim,k,j) linear(i)
#if USETM
  void ApplyPrimitiveFloors(AthenaArray<Real> &prim, AthenaArray<Real> &prim_scalar, int k, int j, int i);
#else
  void ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i);
  void ApplyPrimitiveFloors(const int dir,
                            AthenaArray<Real> &prim_l,
                            AthenaArray<Real> &prim_r, int i);
#endif

#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this,s,n,k,j) linear(i)
  void ApplyPassiveScalarFloors(AthenaArray<Real> &s, int n, int k, int j, int i);

#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this,s,w,r,n,k,j) linear(i)
  void ApplyPassiveScalarPrimitiveConservedFloors(
    AthenaArray<Real> &s, const AthenaArray<Real> &w, AthenaArray<Real> &r,
    int n, int k, int j, int i);

#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this,prim,k,j) linear(i)
  bool CheckPrimitivePhysical(const AthenaArray<Real> &prim, int k, int j, int i)
  {
    bool is_physical = true;

    Real p = NAN;
    Real rho = NAN;

    if(prim.GetDim4()==1)
    {
      p   = prim(IPR,i);
      rho = prim(IDN,i);
    }
    else if(prim.GetDim4()==5)
    {
      p   = prim(IPR,k,j,i);
      rho = prim(IDN,k,j,i);
    }

    // N.B.!
    // to look like standard conditions need to put rho_ = rho h

#if USETM
    Real mb = GetEOS().GetBaryonMass();
    Real n = rho / mb;
    // FIXME: Generalize to work with EOSes accepting particle fractions.
    Real Y[MAX_SPECIES] = {0.0};
#if NSCALARS>0
    for (int l=0; l<NSCALARS; l++)
    {
      // Y[l] = w_r(l,i);
      Y[l] = pmy_block_->pscalars->r(l,k,j,i);
    }
#endif

    Real T = GetEOS().GetTemperatureFromP(n, p, Y);
    Real rho_ = rho*GetEOS().GetEnthalpy(n, T, Y);
#else
    Real gamma_adi = GetGamma();
    Real rho_ = rho + gamma_adi/(gamma_adi-1.0) * p;  // EOS dep.
#endif

    // +ve density
    is_physical = is_physical && ((rho_ > 0));

    // null energy condition (EC)
    is_physical = is_physical && (rho_ + p >= 0);
    // weak EC
    is_physical = is_physical && ((rho_ >= 0) && (rho_ + p >= 0));
    // dominant EC
    is_physical = is_physical && (rho_ >= std::abs(p));
    // strong EC
    is_physical = is_physical && ((rho_ + p >= 0) && (rho_ + 3.0 * p >= 0));

    return is_physical;
  }

// BD: TODO - many functions don't do what their names suggest.
//            Sound speed != eigenvalue..
// Should clean this up at some point, for now, just add actual sound speed
#if GENERAL_RELATIVITY && Z4C_ENABLED && FLUID_ENABLED


  #if USETM
    // ...
  #else
    #if MAGNETIC_FIELDS_ENABLED
      // ...
    #else
      Real GRHD_SoundSpeed(const Real w_rho, const Real w_p);
      // \Kappa := PD[p, epsilon]
      Real GRHD_Kappa(const Real w_rho);
      Real GRHD_Enthalpy(const Real w_rho, const Real w_p);
      // void GRHD_Eigenvalues();
    #endif  // MAGNETIC_FIELDS_ENABLED
  #endif

#endif

  // Sound speed functions in different regimes
#if !RELATIVISTIC_DYNAMICS  // Newtonian: SR, GR defined as no-op
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  Real SoundSpeed(const Real prim[(NHYDRO)]);
  // Define flooring function for fourth-order EOS as no-op for SR, GR regimes
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this,prim,cons,bcc,k,j) linear(i)
  void ApplyPrimitiveConservedFloors(
      AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
      int k, int j, int i);
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
#elif !GENERAL_RELATIVITY  // SR: Newtonian, GR defined as no-op
  Real SoundSpeed(const Real[]) {return 0.0;}
  Real FastMagnetosonicSpeed(const Real[], const Real) {return 0.0;}
  void ApplyPrimitiveConservedFloors(
      AthenaArray<Real> &, AthenaArray<Real> &, AthenaArray<Real> &,
      int, int, int) {return;}
#if !MAGNETIC_FIELDS_ENABLED  // SR hydro: SR MHD defined as no-op
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  void SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
                     Real *plambda_plus, Real *plambda_minus);
  void FastMagnetosonicSpeedsSR(
      const AthenaArray<Real> &, const AthenaArray<Real> &,
      int, int, int, int, int, AthenaArray<Real> &,
      AthenaArray<Real> &) {return;}
#else  // SR MHD: SR hydro defined as no-op
  void SoundSpeedsSR(Real, Real, Real, Real, Real *, Real *) {return;}
  void FastMagnetosonicSpeedsSR(
      const AthenaArray<Real> &prim, const AthenaArray<Real> &bbx_vals,
      int k, int j, int il, int iu, int ivx,
      AthenaArray<Real> &lambdas_p, AthenaArray<Real> &lambdas_m);
#endif  // !MAGNETIC_FIELDS_ENABLED
  void SoundSpeedsGR(Real, Real, Real, Real, Real, Real, Real, Real *, Real *)
  {return;}
  void FastMagnetosonicSpeedsGR(Real, Real, Real, Real, Real, Real, Real, Real,
                                Real *, Real *) {return;}
#else  // GR: Newtonian defined as no-op
  Real SoundSpeed(const Real[]) {return 0.0;}
  Real FastMagnetosonicSpeed(const Real[], const Real) {return 0.0;}
  void ApplyPrimitiveConservedFloors(
      AthenaArray<Real> &, AthenaArray<Real> &, AthenaArray<Real> &,
      int, int, int) {return;}
#if !MAGNETIC_FIELDS_ENABLED  // GR hydro: GR+SR MHD defined as no-op
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)

#if USETM
  void SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
                     Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
#else
  void SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
                     Real *plambda_plus, Real *plambda_minus);
#endif // USETM
  void FastMagnetosonicSpeedsSR(
      const AthenaArray<Real> &, const AthenaArray<Real> &,
      int, int, int, int, int, AthenaArray<Real> &,
      AthenaArray<Real> &) {return;}
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
#if USETM
  void SoundSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1,
                     Real g00, Real g01, Real g11,
                     Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
#else
  void SoundSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1,
                     Real g00, Real g01, Real g11,
                     Real *plambda_plus, Real *plambda_minus);
#endif // USETM
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
#if USETM
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  void FastMagnetosonicSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1, Real b_sq,
                                Real g00, Real g01, Real g11,
                                Real *plambda_plus, Real *plambda_minus, Real prim_scalar[NSCALARS]);
#else
#pragma omp declare simd simdlen(SIMD_WIDTH) uniform(this)
  void FastMagnetosonicSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1, Real b_sq,
                                Real g00, Real g01, Real g11,
                                Real *plambda_plus, Real *plambda_minus);
#endif  // USETM
#endif  // !MAGNETIC_FIELDS_ENABLED (GR)
#endif  // #else (#if !RELATIVISTIC_DYNAMICS, #elif !GENERAL_RELATIVITY)

  Real PresFromRhoEg(Real rho, Real egas);
  Real EgasFromRhoP(Real rho, Real pres);
  Real AsqFromRhoP(Real rho, Real pres);
  Real GetIsoSoundSpeed() const {return iso_sound_speed_;}
  Real GetDensityFloor() const {return density_floor_;}
  Real GetPressureFloor() const {return pressure_floor_;}
  EosTable* ptable; // pointer to EOS table data
#if USETM
  inline Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>& GetEOS() {
    return eos;
  }
  Real GetTemperatureFloor() const {return temperature_floor_;}
#endif
#if GENERAL_EOS
  Real GetGamma();
#else // not GENERAL_EOS
  Real GetGamma() const {return gamma_;}
#endif

 private:
  // (C++11) in-class Default Member Initializer (fallback option):
  const Real float_min{std::numeric_limits<float>::min()};
  MeshBlock *pmy_block_;                 // ptr to MeshBlock containing this EOS
  Real iso_sound_speed_, gamma_;         // isothermal Cs, ratio of specific heats
  Real density_floor_, pressure_floor_;  // density and pressure floors
  Real energy_floor_;                    // energy floor
  Real scalar_floor_; // dimensionless concentration floor
  Real sigma_max_, beta_min_;            // limits on ratios of gas quantities to pmag
  Real gamma_max_;                       // maximum Lorentz factor
  Real rho_min_, rho_pow_;               // variables to control power-law denity floor
  Real pgas_min_, pgas_pow_;             // variables to control power-law pressure floor
  Real rho_unit_, inv_rho_unit_;         // physical unit/sim unit for mass density
  Real egas_unit_, inv_egas_unit_;       // physical unit/sim unit for energy density
  Real vsqr_unit_, inv_vsqr_unit_;       // physical unit/sim unit for speed^2
  AthenaArray<Real> g_, g_inv_;          // metric and its inverse, used in GR
  AthenaArray<Real> fixed_;              // cells with problems, used in GR hydro
  AthenaArray<Real> normal_dd_;          // normal-frame densities, used in GR MHD
  AthenaArray<Real> normal_ee_;          // normal-frame energies, used in GR MHD
  AthenaArray<Real> normal_mm_;          // normal-frame momenta, used in GR MHD
  AthenaArray<Real> normal_bb_;          // normal-frame fields, used in GR MHD
  AthenaArray<Real> normal_tt_;          // normal-frame M.B, used in GR MHD
  void InitEosConstants(ParameterInput *pin);
#if USETM
  // If we're using the PrimitiveSolver framework, we need to declare the
  // EOS and PrimitiveSolver objects.
  Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> eos;
  Primitive::PrimitiveSolver<Primitive::EOS_POLICY, Primitive::ERROR_POLICY> ps;
  Real temperature_floor_;

  // Debugging parameters
  Real dbg_err_tol_abs;
  Real dbg_err_tol_rel;
  bool dbg_report_all;
#endif
};

#if USETM
void InitColdEOS(Primitive::ColdEOS<Primitive::COLDEOS_POLICY> *eos,
                 ParameterInput *pin);
#endif

#endif // EOS_EOS_HPP_
