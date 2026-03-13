#ifndef RECONSTRUCT_RECONSTRUCTION_HPP_
#define RECONSTRUCT_RECONSTRUCTION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file reconstruction.hpp
//  \brief defines class Reconstruction, data and functions for spatial reconstruction

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"

// Forward declarations
class MeshBlock;
class ParameterInput;

//! \class Reconstruction
//  \brief member functions implement various spatial reconstruction algorithms

class Reconstruction {
 public:
  Reconstruction(MeshBlock *pmb, ParameterInput *pin);

  // data
  // reconstruction style variant selection
  enum class ReconstructionVariant {
    none,donate,lin_vl,lin_mc2,ppm,ceno3,mp3,mp5,mp7,mp5_R,
    weno5,weno5z,weno5d_si
  };
  ReconstructionVariant xorder_style;
  ReconstructionVariant xorder_style_fb;

  ReconstructionVariant xorder_style_p;
  ReconstructionVariant xorder_style_p_fb;

  Real xorder_eps;                        // epsilon control parameters
  bool xorder_use_fb;                     // try order reduction
  bool xorder_floor_primitives = false;   // apply floors to reconstructed states?
  bool xorder_limit_species = false;      // limit reconstructed species?
  Real xorder_fb_dfloor_fac = 1;          // multiply dfloor by this to consider fb
  Real xorder_fb_Y_min_fac;               // fiddle factors
  Real xorder_fb_Y_max_fac;

  bool xorder_use_hlle = false;           // promote LLF solver?
  bool xorder_upwind_scalars = true;      // should passive scalars be upwinded?

  bool xorder_use_dmp = false;            // approximate DMP
  bool xorder_use_dmp_scalars;

  bool xorder_min_tau_zero = false;

  Real xorder_dmp_min;                    // fiddle factors controlling decay / growth
  Real xorder_dmp_max;

  const bool xorder_use_auxiliaries = true;  // reconstruct derived quantities?
  bool xorder_use_aux_T;                     // reconstruct temperature?
  bool xorder_use_aux_h;                     // reconstruct enthalpy?
  bool xorder_use_aux_W;                     // reconstruct lorentz?
  bool xorder_use_aux_cs2;                   // reconstruct cs^2?

  bool xorder_limit_fluxes = false;
  bool enforce_limits_integration = false;
  bool enforce_limits_flux_div = false;

  bool uniform[3];

  // x1-sliced arrays of interpolation coefficients and limiter parameters:
  AthenaArray<Real> c1i, c2i, c3i, c4i, c5i, c6i;  // coefficients for PPM in x1
  AthenaArray<Real> c1j, c2j, c3j, c4j, c5j, c6j;  // coefficients for PPM in x2
  AthenaArray<Real> c1k, c2k, c3k, c4k, c5k, c6k;  // coefficients for PPM in x3

  // Refactored interface -----------------------------------------------------
  // More general, cleaner, switches variant case for a slice, not point-wise

  // Convenience function to fix ranges for indices during reconstruction
  inline void SetIndicialLimitsCalculateFluxes(
    const int dir,
    int & il, int & iu,
    int & jl, int & ju,
    int & kl, int & ku,
    const int num_enlarge_layer
  )
  {
    MeshBlock * pmb = pmy_block_;
    Mesh * pm = pmb->pmy_mesh;

    const int is = pmb->is-num_enlarge_layer;
    const int ie = pmb->ie+num_enlarge_layer;

    const int js = pmb->js-num_enlarge_layer;
    const int je = pmb->je+num_enlarge_layer;

    const int ks = pmb->ks-num_enlarge_layer;
    const int ke = pmb->ke+num_enlarge_layer;

    switch (dir)
    {
      case 3:
      {
        il = is-1;
        iu = ie+1;

        jl = js-1;
        ju = je+1;

        kl = ks;
        ku = ke+1;

        break;
      }
      case 2:
      {
        il = is-1;
        iu = ie+1;

        jl = js;
        ju = je+1;

        kl = ks-(pm->f3); // if 3d
        ku = ke+(pm->f3);

        break;
      }
      case 1:
      {
        il = is;
        iu = ie+1;

        jl = js-(pm->f2 || pm->f3);  // 2d or 3d
        ju = je+(pm->f2 || pm->f3);

        kl = ks-(pm->f3); // if 3d
        ku = ke+(pm->f3);

        break;
      }
      default:
      {
        assert(false);
      }
    }
  }

  // Convention (map to Shu):
  // zl_(i+1) <- z_{i+1/2}; zl_(i) = z_{i+1/2}^-
  // zr_(i  ) <- z_{i-1/2}; zr_(i) = z_{i+1/2}^+
  // N.B. for this direction and il=iu=i, zl_(i+1) and zr_(i) are populated.
  void ReconstructFieldX1(const ReconstructionVariant rv,
                          AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  // Convention (map to Shu):
  // zl_(i) <- z_{j+1/2,i} = z_{j+1/2,i}^-
  // zr_(i) <- z_{j-1/2,i} = z_{j-1/2,i}^+
  void ReconstructFieldX2(const ReconstructionVariant rv,
                          AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  // Convention (map to Shu):
  // zl_(i) <- z_{k+1/2,j,i} = z_{k+1/2,j,i}^-
  // zr_(i) <- z_{k-1/2,j,i} = z_{k-1/2,j,i}^+
  void ReconstructFieldX3(const ReconstructionVariant rv,
                          AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  // Delegate to the above based on ivx = {1, 2, 3}
  void ReconstructFieldXd(const ReconstructionVariant rv,
                          AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int ivx,  // ivx = d = 1, 2, 3
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);
  // --------------------------------------------------------------------------

private:
  // Available methods
  void ReconstructDonateX1(AthenaArray<Real> &z,
                           AthenaArray<Real> &zl_,
                           AthenaArray<Real> &zr_,
                           const int n_tar,
                           const int n_src,
                           const int k,
                           const int j,
                           const int il, const int iu);

  void ReconstructDonateX2(AthenaArray<Real> &z,
                           AthenaArray<Real> &zl_,
                           AthenaArray<Real> &zr_,
                           const int n_tar,
                           const int n_src,
                           const int k,
                           const int j,
                           const int il, const int iu);

  void ReconstructDonateX3(AthenaArray<Real> &z,
                           AthenaArray<Real> &zl_,
                           AthenaArray<Real> &zr_,
                           const int n_tar,
                           const int n_src,
                           const int k,
                           const int j,
                           const int il, const int iu);

  void ReconstructLinearVLX1(AthenaArray<Real> &z,
                             AthenaArray<Real> &zl_,
                             AthenaArray<Real> &zr_,
                             const int n_tar,
                             const int n_src,
                             const int k,
                             const int j,
                             const int il, const int iu);

  void ReconstructLinearVLX2(AthenaArray<Real> &z,
                             AthenaArray<Real> &zl_,
                             AthenaArray<Real> &zr_,
                             const int n_tar,
                             const int n_src,
                             const int k,
                             const int j,
                             const int il, const int iu);

  void ReconstructLinearVLX3(AthenaArray<Real> &z,
                             AthenaArray<Real> &zl_,
                             AthenaArray<Real> &zr_,
                             const int n_tar,
                             const int n_src,
                             const int k,
                             const int j,
                             const int il, const int iu);

  void ReconstructLinearMC2X1(AthenaArray<Real> &z,
                              AthenaArray<Real> &zl_,
                              AthenaArray<Real> &zr_,
                              const int n_tar,
                              const int n_src,
                              const int k,
                              const int j,
                              const int il, const int iu);

  void ReconstructLinearMC2X2(AthenaArray<Real> &z,
                              AthenaArray<Real> &zl_,
                              AthenaArray<Real> &zr_,
                              const int n_tar,
                              const int n_src,
                              const int k,
                              const int j,
                              const int il, const int iu);

  void ReconstructLinearMC2X3(AthenaArray<Real> &z,
                              AthenaArray<Real> &zl_,
                              AthenaArray<Real> &zr_,
                              const int n_tar,
                              const int n_src,
                              const int k,
                              const int j,
                              const int il, const int iu);

  void ReconstructPPMX1(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructPPMX2(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructPPMX3(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructCeno3X1(AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  void ReconstructCeno3X2(AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  void ReconstructCeno3X3(AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  void ReconstructWeno5ZX1(AthenaArray<Real> &z,
                           AthenaArray<Real> &zl_,
                           AthenaArray<Real> &zr_,
                           const int n_tar,
                           const int n_src,
                           const int k,
                           const int j,
                           const int il, const int iu);

  void ReconstructWeno5ZX2(AthenaArray<Real> &z,
                           AthenaArray<Real> &zl_,
                           AthenaArray<Real> &zr_,
                           const int n_tar,
                           const int n_src,
                           const int k,
                           const int j,
                           const int il, const int iu);

  void ReconstructWeno5ZX3(AthenaArray<Real> &z,
                           AthenaArray<Real> &zl_,
                           AthenaArray<Real> &zr_,
                           const int n_tar,
                           const int n_src,
                           const int k,
                           const int j,
                           const int il, const int iu);

  void ReconstructWeno5X1(AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  void ReconstructWeno5X2(AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  void ReconstructWeno5X3(AthenaArray<Real> &z,
                          AthenaArray<Real> &zl_,
                          AthenaArray<Real> &zr_,
                          const int n_tar,
                          const int n_src,
                          const int k,
                          const int j,
                          const int il, const int iu);

  void ReconstructWeno5dsiX1(AthenaArray<Real> &z,
                             AthenaArray<Real> &zl_,
                             AthenaArray<Real> &zr_,
                             const int n_tar,
                             const int n_src,
                             const int k,
                             const int j,
                             const int il, const int iu);

  void ReconstructWeno5dsiX2(AthenaArray<Real> &z,
                             AthenaArray<Real> &zl_,
                             AthenaArray<Real> &zr_,
                             const int n_tar,
                             const int n_src,
                             const int k,
                             const int j,
                             const int il, const int iu);

  void ReconstructWeno5dsiX3(AthenaArray<Real> &z,
                             AthenaArray<Real> &zl_,
                             AthenaArray<Real> &zr_,
                             const int n_tar,
                             const int n_src,
                             const int k,
                             const int j,
                             const int il, const int iu);

  void ReconstructMP3X1(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP3X2(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP3X3(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP5X1(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP5X2(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP5X3(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP7X1(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP7X2(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP7X3(AthenaArray<Real> &z,
                        AthenaArray<Real> &zl_,
                        AthenaArray<Real> &zr_,
                        const int n_tar,
                        const int n_src,
                        const int k,
                        const int j,
                        const int il, const int iu);

  void ReconstructMP5RX1(AthenaArray<Real> &z,
                         AthenaArray<Real> &zl_,
                         AthenaArray<Real> &zr_,
                         const int n_tar,
                         const int n_src,
                         const int k,
                         const int j,
                         const int il, const int iu);

  void ReconstructMP5RX2(AthenaArray<Real> &z,
                         AthenaArray<Real> &zl_,
                         AthenaArray<Real> &zr_,
                         const int n_tar,
                         const int n_src,
                         const int k,
                         const int j,
                         const int il, const int iu);

  void ReconstructMP5RX3(AthenaArray<Real> &z,
                         AthenaArray<Real> &zl_,
                         AthenaArray<Real> &zr_,
                         const int n_tar,
                         const int n_src,
                         const int k,
                         const int j,
                         const int il, const int iu);

private:
  MeshBlock* pmy_block_;  // ptr to MeshBlock containing this Reconstruction

  // scratch arrays used in reconstruction functions
  AthenaArray<Real> scr02_i_, scr03_i_, scr04_i_, scr05_i_;
  AthenaArray<Real> scr06_i_, scr07_i_, scr08_i_, scr09_i_, scr10_i_;
  AthenaArray<Real> scr11_i_, scr12_i_, scr13_i_, scr14_i_;
  AthenaArray<Real> scr2_ni_, scr3_ni_, scr4_ni_, scr5_ni_;
  AthenaArray<Real> scr6_ni_;


};
#endif // RECONSTRUCT_RECONSTRUCTION_HPP_
