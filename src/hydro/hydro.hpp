#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../bvals/cc/hydro/bvals_hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "srcterms/hydro_srcterms.hpp"

class MeshBlock;
class ParameterInput;

using namespace gra::aliases;

// TODO(felker): consider adding a struct FaceFlux w/ overloaded ctor in athena.hpp, or:
// using FaceFlux = AA[3];

//! \class Hydro
//  \brief hydro data and functions

class Hydro {
  friend class Field;
  friend class EquationOfState;
 public:
  Hydro(MeshBlock *pmb, ParameterInput *pin);

  // data
  // TODO(KGF): make this private, if possible
  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Hydro

  // conserved and primitive variables
  AA u, w; // time-integrator memory register #1
  AA u1, w1;       // time-integrator memory register #2
  AA u2;           // time-integrator memory register #3
  // (no more than MAX_NREGISTER allowed)

  // storage for derived quantities (HydroDerivedIndex); matter-sampling
  AA derived_ms;
  // storage for derived quantities for internal usage (HydroInternalDerivedIndex);
  AA derived_int;

  AA flux[3];  // face-averaged flux vector
  AA lo_flux[3];

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AA coarse_cons_, coarse_prim_;
  int refinement_idx{-1};

  // for reconstruction failure, should both states be floored?
  bool floor_both_states = false;
  bool flux_reconstruction = false;
  bool split_lr_fallback = false;
  bool flux_table_limiter = false;

  HydroBoundaryVariable hbvar;
  HydroSourceTerms hsrc;
  HydroDiffusion hdif;

  struct {
    Real alpha_threshold;           // excise hydro if alpha < alpha_excision
    bool horizon_based;             // use horizon for excise
    Real horizon_factor;            // factor to multiply horizon radius
    bool hybrid_hydro;              // control whether to use ahf+hydro excision
    Real hybrid_fac_min_alpha;      // cut values below this * min_alpha
    Real use_taper;                 // taper instead of hard-cut?
    bool excise_hydro_freeze_evo;   // use with taper
    bool excise_hydro_taper;        // taper (cons) state-vector
    Real taper_pow;                 // taper(x) ^ taper_pow
    bool excise_hydro_damping;      // replace hydro evo with exponential decay
    bool excise_c2p;
    bool excise_flux;
    Real hydro_damping_factor;
  } opt_excision;

  AA excision_mask;
  AA fallback_mask;

  // --------------------------------------------------------------------------
  struct ixn_cons
  {
    enum
    {
      D,     // matches IDN, IM1, IM2, IM3, IEN
      S_d_1,
      S_d_2,
      S_d_3,
      tau,
      N
    };

    static constexpr char const * const names[] = {
      "hydro.cons.D",
      "hydro.cons.S_d_1",
      "hydro.cons.S_d_2",
      "hydro.cons.S_d_3",
      "hydro.cons.tau",
    };
  };

  struct ixn_prim
  {
    enum
    {
      rho,   // matches IDN, IVX, IVY, IVZ, IPR
      util_u_1,
      util_u_2,
      util_u_3,
      p,
      N
    };

    static constexpr char const * const names[] = {
      "hydro.prim.rho",
      "hydro.prim.util_u_1",
      "hydro.prim.util_u_2",
      "hydro.prim.util_u_3",
      "hydro.prim.p",
    };
  };

  struct ixn_derived_ms
  {
    // Uses "HydroDerivedIndex"
    static constexpr char const * const names[] = {
      "hydro.aux.c2p_status",
      "hydro.aux.W",
      "hydro.aux.T",
      "hydro.aux.h",
      "hydro.aux.s",
      "hydro.aux.e",
      "hydro.aux.u_t",
      "hydro.aux.hu_t",
      "hydro.aux.cs2",
      "hydro.aux.Omega",
    };
  };

  struct ixn_derived_int
  {
    // Uses "HydroDerivedIndex"
    static constexpr char const * const names[] = {
      "hydro.aux.V_u_x",
      "hydro.aux.V_u_y",
      "hydro.aux.V_u_z",
    };
  };

  // scratches ----------------------------------------------------------------
public:
  AT_N_sca sqrt_detgamma_;
  AT_N_sca detgamma_;     // spatial met det
  AT_N_sca oo_detgamma_;  // 1 / spatial met det

  AT_N_sca alpha_;
  AT_N_sca oo_alpha_;     // 1 / alpha
  AT_N_vec beta_u_;
  AT_N_sym gamma_dd_;
  AT_N_sym gamma_uu_;

  AT_N_sca chi_;

  AT_N_vec w_v_u_l_;
  AT_N_vec w_v_u_r_;

  AT_C_sca w_norm2_v_l_;
  AT_C_sca w_norm2_v_r_;

  AT_C_sca lambda_p_l;
  AT_C_sca lambda_m_l;
  AT_C_sca lambda_p_r;
  AT_C_sca lambda_m_r;
  AT_C_sca lambda;

  // primitive vel. (covar.)
  AT_N_vec w_util_d_l_;
  AT_N_vec w_util_d_r_;

  // Lorentz factor
  AT_C_sca W_l_;
  AT_C_sca W_r_;

  // h * rho
  AT_C_sca w_hrho_l_;
  AT_C_sca w_hrho_r_;

  // prim / cons shaped scratches
  AT_H_vec cons_l_;
  AT_H_vec cons_r_;

  AT_H_vec flux_l_;
  AT_H_vec flux_r_;

  // Particular to magnetic fields --------------------------------------------
  AT_N_sca oo_sqrt_detgamma_;

  AT_C_sca oo_W_l_;
  AT_C_sca oo_W_r_;

  AT_N_vec w_v_d_l_;
  AT_N_vec w_v_d_r_;

  AT_N_vec alpha_w_vtil_u_l_;
  AT_N_vec alpha_w_vtil_u_r_;

  AT_N_vec beta_d_;

  AT_N_vec q_scB_u_l_;  // \mathcal{B}^k:= B^k / \sqrt{\gamma}
  AT_N_vec q_scB_u_r_;

  AT_C_sca b0_l_;
  AT_C_sca b0_r_;

  AT_C_sca b2_l_;
  AT_C_sca b2_r_;

  AT_N_vec bi_u_l_;
  AT_N_vec bi_u_r_;

  AT_N_vec bi_d_l_;
  AT_N_vec bi_d_r_;

public:
  // functions ----------------------------------------------------------------
  void NewBlockTimeStep();    // computes new timestep on a MeshBlock
  void AddFluxDivergence(const Real wght, AA &u_out);
  void CalculateFluxes(AA &w,
                       AA &r,
                       FaceField &b,
                       AA &bcc,
                       AA(& hflux)[3],
                       AA(& sflux)[3],
                       Reconstruction::ReconstructionVariant rv,
                       const int num_enlarge_layer=0);

  void CalculateFluxesCombined(AA &w,
                               AA &r,
                               FaceField &b,
                               AA &bcc,
                               AA(& hflux)[3],
                               AA(& sflux)[3],
                               Reconstruction::ReconstructionVariant rv,
                               const int num_enlarge_layer=0);

  void CheckStateWithFluxDivergence(
    const Real wght,
    AA &u,
    AA &s,
    AA(& hflux)[3],
    AA(& sflux)[3],
    bool &all_valid,
    AA_B &mask,
    const int num_enlarge_layer
  );

  void CheckStateWithFluxDivergenceDMP(
    const Real wght,
    AA &u,
    AA &u_old,
    AA &s,
    AA &s_old,
    AA(& hflux)[3],
    AA(& sflux)[3],
    bool &all_valid,
    AA_B &mask,
    const int num_enlarge_layer
  );

  void HybridizeFluxes(
    AA (&hflux)[3], AA (&sflux)[3],
    AA (&lo_hflux)[3], AA (&lo_sflux)[3],
    const AA_B &mask);

  void LimitMaskFluxDivergence(
    const Real wght,
    AA &u,
    AA &s,
    AA(& hflux)[3],
    AA(& sflux)[3],
    AA &mask,
    const int num_enlarge_layer
  );

  void LimitFluxes(
    AA & mask_theta,
    AA(& hflux)[3],
    AA(& sflux)[3]
  );

  void EnforceFloorsLimits(
    AA &u,
    AA &s,
    const int num_enlarge_layer
  );

  bool ConservedDensityWithinFloorThreshold(
    AA &u,
    const Real undensitized_dfloor_fac,
    const int num_enlarge_layer
  );

    // BD: TODO- To remove
  void CalculateFluxes_FluxReconstruction(
    AA &w, FaceField &b,
    AA &bcc, const int order);

  void CalculateFluxes_STS();

#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
  void RiemannSolver(
    const int ivx,
    const int k, const int j,
    const int il, const int iu,
    AA &prim_l_,
    AA &prim_r_,
    AA &pscalars_l_,
    AA &pscalars_r_,
    AA &aux_l_,
    AA &aux_r_,
    AT_N_sca & alpha_,
    AT_N_sca & oo_alpha_,
    AT_N_vec & beta_u_,
    AT_N_sym & gamma_dd_,
    AT_N_sca & detgamma_,
    AT_N_sca & oo_detgamma_,
    AT_N_sca & sqrt_detgamma_,
    AA &flux,
    AA &s_flux,
    const AA &dxw_,
    const Real lambda_rescaling);
#else  // MHD:
  void RiemannSolver(
    const int ivx,
    const int k, const int j,
    const int il, const int iu,
    const AA &B,
    AA &prim_l_,
    AA &prim_r_,
    AA &pscalars_l_,
    AA &pscalars_r_,
    AA &aux_l_,
    AA &aux_r_,
    AT_N_sca & alpha_,
    AT_N_sca & oo_alpha_,
    AT_N_vec & beta_u_,
    AT_N_sym & gamma_dd_,
    AT_N_sca & detgamma_,
    AT_N_sca & oo_detgamma_,
    AT_N_sca & sqrt_detgamma_,
    AA &flux,
    AA &s_flux,
    AA &ey,
    AA &ez,
    AA &wct,
    const AA &dxw_,
    const Real lambda_rescaling);
#endif

  inline void RetainState(
    AA & w1,
    const AA & w,
    const int il, const int iu,
    const int jl, const int ju,
    const int kl, const int ku
  )
  {
    for (int n=0; n<NHYDRO; ++n)
    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    for (int i=il; i<=iu; ++i)
    {
      w1(n,k,j,i) = w(n,k,j,i);
    }
  }

 private:
  AA dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep
  // scratch space used to compute fluxes
  AA dxw_;
  // 2D
  AA wl_, wr_, wlb_;
  AA rl_, rr_, rlb_;

  // reconstruction for auxiliaries
  AA al_, ar_, alb_;

  AA dflx_;

  TimeStepFunc UserTimeStep_;

  void AddDiffusionFluxes();
  Real GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt);
};

namespace fluxes {

// Split flux based on local eigenvalues
//
// Takes max over all lambda cpts & directional (ivx-aligned) faces
void SplitFluxLLFMax(MeshBlock * pmb,
                     const int k, const int j,
                     const int il, const int iu,
                     const int ivx,
                     AA &u,
                     AA &flux,
                     AA &lambda,
                     AA &flux_m,
                     AA &flux_p);

}  // namespace fluxes

namespace fluxes::grhd {

// Dense assembly of fluxes
void AssembleFluxes(MeshBlock * pmb,
                    const int k, const int j,
                    const int il, const int iu,
                    const int ivx,
                    AA &f,
                    AA &w,
                    AA &u);

}  // namespace fluxes::grhd

namespace characteristic::grhd {

// Dense assembly of lambda
void AssembleEigenvalues(MeshBlock * pmb,
                         const int k, const int j,
                         const int il, const int iu,
                         const int ivx,
                         AA &lambda,
                         AA &w,
                         AA &u);

}  // namespace characteristic::grhd


#endif // HYDRO_HYDRO_HPP_
