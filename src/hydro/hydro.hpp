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

// TODO(felker): consider adding a struct FaceFlux w/ overloaded ctor in athena.hpp, or:
// using FaceFlux = AthenaArray<Real>[3];

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
  AthenaArray<Real> u, w; // time-integrator memory register #1
  AthenaArray<Real> u1, w1;       // time-integrator memory register #2
  AthenaArray<Real> u2;           // time-integrator memory register #3
  // (no more than MAX_NREGISTER allowed)

#if USETM
// Storage for temperature output
  AthenaArray<Real> temperature;
#endif

  AthenaArray<Real> flux[3];  // face-averaged flux vector

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AthenaArray<Real> coarse_cons_, coarse_prim_;
  int refinement_idx{-1};

  // BD: TODO - make the mask more useful
  // prim: w, cons: q
  // q<->w can fail; in this situation values need to be reset
  // It is helpful to make a mask to this end
  AthenaArray<bool> q_reset_mask;
  AthenaArray<Real> c2p_status;


  // for reconstruction failure, should both states be floored?
  bool floor_both_states = false;
  bool flux_reconstruction = false;

  HydroBoundaryVariable hbvar;
  HydroSourceTerms hsrc;
  HydroDiffusion hdif;

  // scratches ----------------------------------------------------------------
public:
  AT_N_sca sqrt_detgamma_;
  AT_N_sca detgamma_;     // spatial met det
  AT_N_sca oo_detgamma_;  // 1 / spatial met det

  AT_N_sca alpha_;
  AT_N_vec beta_u_;
  AT_N_sym gamma_dd_;
  AT_N_sym gamma_uu_;

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
  void AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out);
  void CalculateFluxes(AthenaArray<Real> &w, FaceField &b,
                       AthenaArray<Real> &bcc, const int order);

  void CalculateFluxes_FluxReconstruction(
    AthenaArray<Real> &w, FaceField &b,
    AthenaArray<Real> &bcc, const int order);


  // debug join hydro+passive scalar recon.
  void CalculateFluxesCombined(AthenaArray<Real> &w, FaceField &b,
                               AthenaArray<Real> &bcc, const int order);

  void CalculateFluxes_STS();

#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
  void RiemannSolver(
      const int k, const int j, const int il, const int iu,
      const int ivx,
      AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx,
      const AthenaArray<Real> &dxw);
#else  // MHD:
  void RiemannSolver(
      const int k, const int j, const int il, const int iu,
      const int ivx, const AthenaArray<Real> &bx,
      AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx,
      AthenaArray<Real> &ey, AthenaArray<Real> &ez,
      AthenaArray<Real> &wct, const AthenaArray<Real> &dxw);
#endif

  inline void FallbackInadmissiblePrimitiveX1_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
  {
    // Either 0 or 1, depending on l/r convention
    static const int I = DGB_RECON_X1_OFFSET;

    #pragma omp simd
    for (int i=il-1; i<=iu; ++i)
    {
      if ((zl_(IDN,i+I) < 0) || (zr_(IDN,i) < 0))
      {
        for (int n=0; n<NWAVE; ++n)
        {
          zl_(n,i+I) = f_zl_(n,i+I);
          zr_(n,i  ) = f_zr_(n,i  );
        }
      }
    }

    if (pmy_block->precon->xorder_use_fb_unphysical)
    {
      EquationOfState *peos = pmy_block->peos;
      #pragma omp simd
      for (int i=il-1; i<=iu; ++i)
      {
        const bool pl_ = peos->CheckPrimitivePhysical(zl_, -1, -1, i+I);
        const bool pr_ = peos->CheckPrimitivePhysical(zr_, -1, -1, i);
        if (!pl_ || !pr_)
        {
          for (int n=0; n<NWAVE; ++n)
          {
            zl_(n,i+I) = f_zl_(n,i+I);
            zr_(n,i  ) = f_zr_(n,i  );
          }
        }
      }
    }

  }

  inline bool CheckInadmissiblePrimitiveX1_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int il, const int iu)
  {
    // Either 0 or 1, depending on l/r convention
    static const int I = DGB_RECON_X1_OFFSET;

    for (int i=il; i<=iu; ++i)
    {
      if ((zl_(IDN,i+I) < 0) || (zr_(IDN,i) < 0))
      {
        return true;
      }

      const bool pl_ = pmy_block->peos->CheckPrimitivePhysical(zl_, -1, -1, i+I);
      const bool pr_ = pmy_block->peos->CheckPrimitivePhysical(zr_, -1, -1, i);

      if (!pl_ || !pr_)
      {
        return true;
      }
    }
    return false;
  }

  inline void FallbackInadmissiblePrimitiveX2_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if ((zl_(IDN,i) < 0) || (zr_(IDN,i) < 0))
      {
        for (int n=0; n<NWAVE; ++n)
        {
          zl_(n,i) = f_zl_(n,i);
          zr_(n,i) = f_zr_(n,i);
        }
      }
    }

    if (pmy_block->precon->xorder_use_fb_unphysical)
    {
      EquationOfState *peos = pmy_block->peos;
      #pragma omp simd
      for (int i=il; i<=iu; ++i)
      {
        const bool pl_ = peos->CheckPrimitivePhysical(zl_, -1, -1, i);
        const bool pr_ = peos->CheckPrimitivePhysical(zr_, -1, -1, i);
        if (!pl_ || !pr_)
        {
          for (int n=0; n<NWAVE; ++n)
          {
            zl_(n,i) = f_zl_(n,i);
            zr_(n,i) = f_zr_(n,i);
          }
        }
      }
    }
  }

  inline void FallbackInadmissiblePrimitiveX3_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if ((zl_(IDN,i) < 0) || (zr_(IDN,i) < 0))
      {
        for (int n=0; n<NWAVE; ++n)
        {
          zl_(n,i) = f_zl_(n,i);
          zr_(n,i) = f_zr_(n,i);
        }
      }
    }

    if (pmy_block->precon->xorder_use_fb_unphysical)
    {
      EquationOfState *peos = pmy_block->peos;
      #pragma omp simd
      for (int i=il; i<=iu; ++i)
      {
        const bool pl_ = peos->CheckPrimitivePhysical(zl_, -1, -1, i);
        const bool pr_ = peos->CheckPrimitivePhysical(zr_, -1, -1, i);
        if (!pl_ || !pr_)
        {
          for (int n=0; n<NWAVE; ++n)
          {
            zl_(n,i) = f_zl_(n,i);
            zr_(n,i) = f_zr_(n,i);
          }
        }
      }
    }
  }

#if USETM
  inline void FloorPrimitiveX1_(
    AthenaArray<Real> & wl_,
    AthenaArray<Real> & wr_,
    AthenaArray<Real> & rl_,
    AthenaArray<Real> & rr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    // Either 0 or 1, depending on l/r convention
    static const int I = DGB_RECON_X1_OFFSET;

    EquationOfState *peos = pmy_block->peos;

    // N.B. indicial range is bumped left to account for reconstruction
    // convention in X1
    #pragma omp simd
    for (int i=il-1; i<=iu; ++i)
    {
      peos->ApplyPrimitiveFloors(wl_,rl_,k,j,i+I);
      peos->ApplyPrimitiveFloors(wr_,rr_,k,j,i);
    }
  }
#else
  inline void FloorPrimitiveX1_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    // Either 0 or 1, depending on l/r convention
    static const int I = DGB_RECON_X1_OFFSET;

    EquationOfState * peos = pmy_block->peos;

    // N.B. indicial range is bumped left to account for reconstruction
    // convention in X1
    #pragma omp simd
    for (int i=il-1; i<=iu; ++i)
    {
      if (floor_both_states)
      {
        const int dir = 1;
        peos->ApplyPrimitiveFloors(dir, zl_, zr_, i);
      }
      else
      {
        peos->ApplyPrimitiveFloors(zl_,k,j,i+I);
        peos->ApplyPrimitiveFloors(zr_,k,j,i);
      }
    }
  }
#endif

#if USETM
  inline void FloorPrimitiveX2_(
    AthenaArray<Real> & wl_,
    AthenaArray<Real> & wr_,
    AthenaArray<Real> & rl_,
    AthenaArray<Real> & rr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      peos->ApplyPrimitiveFloors(wl_,rl_,k,j,i);
      peos->ApplyPrimitiveFloors(wr_,rr_,k,j,i);
    }
  }
#else
  inline void FloorPrimitiveX2_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (floor_both_states)
      {
        const int dir = 2;
        peos->ApplyPrimitiveFloors(dir, zl_, zr_, i);
      }
      else
      {
        peos->ApplyPrimitiveFloors(zl_,k,j,i);
        peos->ApplyPrimitiveFloors(zr_,k,j,i);
      }
    }
  }
#endif



#if USETM
  inline void FloorPrimitiveX3_(
    AthenaArray<Real> & wl_,
    AthenaArray<Real> & wr_,
    AthenaArray<Real> & rl_,
    AthenaArray<Real> & rr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      peos->ApplyPrimitiveFloors(wl_,rl_,k,j,i);
      peos->ApplyPrimitiveFloors(wr_,rr_,k,j,i);
    }
  }
#else
  inline void FloorPrimitiveX3_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (floor_both_states)
      {
        const int dir = 3;
        peos->ApplyPrimitiveFloors(dir, zl_, zr_, i);
      }
      else
      {
        peos->ApplyPrimitiveFloors(zl_,k,j,i);
        peos->ApplyPrimitiveFloors(zr_,k,j,i);
      }
    }
  }
#endif

 private:
  AthenaArray<Real> dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep
  // scratch space used to compute fluxes
  AthenaArray<Real> dxw_;
  // 2D
  AthenaArray<Real> wl_, wr_, wlb_;
  AthenaArray<Real> r_wl_, r_wr_, r_wlb_;
  AthenaArray<Real> rl_, rr_, rlb_;

  AthenaArray<Real> dflx_;

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
                     AthenaArray<Real> &u,
                     AthenaArray<Real> &flux,
                     AthenaArray<Real> &lambda,
                     AthenaArray<Real> &flux_m,
                     AthenaArray<Real> &flux_p);

}  // namespace fluxes

namespace fluxes::grhd {

// Dense assembly of fluxes
void AssembleFluxes(MeshBlock * pmb,
                    const int k, const int j,
                    const int il, const int iu,
                    const int ivx,
                    AthenaArray<Real> &f,
                    AthenaArray<Real> &w,
                    AthenaArray<Real> &u);

}  // namespace fluxes::grhd

namespace characteristic::grhd {

// Dense assembly of lambda
void AssembleEigenvalues(MeshBlock * pmb,
                         const int k, const int j,
                         const int il, const int iu,
                         const int ivx,
                         AthenaArray<Real> &lambda,
                         AthenaArray<Real> &w,
                         AthenaArray<Real> &u);

}  // namespace characteristic::grhd


#endif // HYDRO_HYDRO_HPP_
