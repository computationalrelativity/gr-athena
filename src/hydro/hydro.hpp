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
#include "../athena_arrays.hpp"
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
  AthenaArray<Real> u, w, w_init; // time-integrator memory register #1
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

  // prim: w, cons: q
  // q<->w can fail; in this situation values need to be reset
  // It is helpful to make a mask to this end
  AthenaArray<bool> q_reset_mask;

  // for reconstruction failure, should both states be floored?
  bool floor_both_states = false;

  // fourth-order intermediate quantities
  AthenaArray<Real> u_cc, w_cc;      // cell-centered approximations

  HydroBoundaryVariable hbvar;
  HydroSourceTerms hsrc;
  HydroDiffusion hdif;

  // functions
  void NewBlockTimeStep();    // computes new timestep on a MeshBlock
  void AddFluxDivergence(const Real wght, AthenaArray<Real> &u_out);
  void CalculateFluxes(AthenaArray<Real> &w, FaceField &b,
                       AthenaArray<Real> &bcc, const int order);

  void CalculateFluxesRef(AthenaArray<Real> &w, FaceField &b,
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

  void AddGravityFlux();
  void AddGravityFluxWithGflx();
  void CalculateGravityFlux(AthenaArray<Real> &phi_in);

  inline void FallbackInadmissiblePrimitiveX1_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if ((zl_(IDN,i+1) < 0) || (zr_(IDN,i) < 0))
      {
        for (int n=0; n<NWAVE; ++n)
        {
          zl_(n,i+1) = f_zl_(n,i+1);
          zr_(n,i  ) = f_zr_(n,i  );
        }
      }
    }

    if (pmy_block->precon->xorder_fallback_unphysical)
    {
      EquationOfState *peos = pmy_block->peos;
      #pragma omp simd
      for (int i=il; i<=iu; ++i)
      {
        const bool pl_ = peos->CheckPrimitivePhysical(zl_, -1, -1, i+1);
        const bool pr_ = peos->CheckPrimitivePhysical(zr_, -1, -1, i);
        if (!pl_ || !pr_)
        {
          for (int n=0; n<NWAVE; ++n)
          {
            zl_(n,i+1) = f_zl_(n,i+1);
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
    for (int i=il; i<=iu; ++i)
    {
      if ((zl_(IDN,i+1) < 0) || (zr_(IDN,i) < 0))
      {
        return true;
      }

      const bool pl_ = pmy_block->peos->CheckPrimitivePhysical(zl_, -1, -1, i+1);
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

    if (pmy_block->precon->xorder_fallback_unphysical)
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

    if (pmy_block->precon->xorder_fallback_unphysical)
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
    AthenaArray<Real> & sl_,
    AthenaArray<Real> & sr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState *peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      peos->ApplyPrimitiveFloors(wl_,sl_,k,j,i+1);
      peos->ApplyPrimitiveFloors(wr_,sr_,k,j,i);
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
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (floor_both_states)
      {
        const int dir = 1;
        peos->ApplyPrimitiveFloors(dir, zl_, zr_, i);
      }
      else
      {
        peos->ApplyPrimitiveFloors(zl_,k,j,i+1);
        peos->ApplyPrimitiveFloors(zr_,k,j,i);
      }
    }
  }
#endif

#if USETM
  inline void FloorPrimitiveX2_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & sl_,
    AthenaArray<Real> & sr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      peos->ApplyPrimitiveFloors(wl_,sl_,k,j,i);
      peos->ApplyPrimitiveFloors(wr_,sr_,k,j,i);
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
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & sl_,
    AthenaArray<Real> & sr_,
    const int k,
    const int j,
    const int il, const int iu)
  {
    EquationOfState * peos = pmy_block->peos;

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      peos->ApplyPrimitiveFloors(wl_,sl_,k,j,i);
      peos->ApplyPrimitiveFloors(wr_,sr_,k,j,i);
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


  // Debug
  void HydroRHS(   AthenaArray<Real> & u_cons, AthenaArray<Real> & u_rhs);
  void AddHydroRHS(AthenaArray<Real> & rhs,
                   Real const wght,
                   AthenaArray<Real> &u_out);

  void Hydro_IdealEoS_Prim2Cons(
    const Real Gamma,
    AthenaArray<Real> & prim,
    AthenaArray<Real> & cons,
    const int il, const int iu,
    const int jl, const int ju,
    const int kl, const int ku);

  void Hydro_IdealEoS_Cons2Prim(
    const Real Gamma,
    AthenaArray<Real> & cons,
    AthenaArray<Real> & prim,
    const int il, const int iu,
    const int jl, const int ju,
    const int kl, const int ku);

  void Hydro_IdealEoS_Cons2Prim(
    const Real Gamma,
    AthenaArray<Real> & cons,
    AthenaArray<Real> & prim,
    AthenaArray<Real> & prim_old,
    const int il, const int iu,
    const int jl, const int ju,
    const int kl, const int ku);


 private:

  int fix_fluxes;
  int zero_div;
  AthenaArray<Real> dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep
  // scratch space used to compute fluxes
  AthenaArray<Real> dxw_;
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_;
  // 2D
  AthenaArray<Real> wl_, wr_, wlb_;
  AthenaArray<Real> r_wl_, r_wr_, r_wlb_;
  AthenaArray<Real> rl_, rr_, rlb_;

  AthenaArray<Real> dflx_;
  AthenaArray<Real> bb_normal_;    // normal magnetic field, for (SR/GR)MHD
  AthenaArray<Real> lambdas_p_l_;  // most positive wavespeeds in left state
  AthenaArray<Real> lambdas_m_l_;  // most negative wavespeeds in left state
  AthenaArray<Real> lambdas_p_r_;  // most positive wavespeeds in right state
  AthenaArray<Real> lambdas_m_r_;  // most negative wavespeeds in right state
  // 2D GR
  AthenaArray<Real> g_, gi_;       // metric and inverse, for some GR Riemann solvers
  AthenaArray<Real> cons_;         // conserved state, for some GR Riemann solvers

  // self-gravity
  AthenaArray<Real> gflx[3], gflx_old[3]; // gravity tensor (old Athena style)

  // fourth-order hydro
  // 4D scratch arrays
  AthenaArray<Real> scr1_nkji_, scr2_nkji_;
  AthenaArray<Real> wl3d_, wr3d_;
  // 1D scratch arrays
  AthenaArray<Real> laplacian_l_fc_, laplacian_r_fc_;

  TimeStepFunc UserTimeStep_;

  void AddDiffusionFluxes();
  Real GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt);
};
#endif // HYDRO_HYDRO_HPP_
