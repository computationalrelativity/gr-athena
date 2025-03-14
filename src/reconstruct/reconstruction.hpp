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
  // switches for reconstruction method variants:
  int xorder;   // roughly the formal order of accuracy of overall reconstruction method

  // for xorder == 5 we can switch between reconstruction styles
  enum class ReconstructionVariant {
    none,donate,lin_vl,lin_mc2,ppm,ceno3,mp3,mp5,mp7,mp5_R,
    weno5,weno5z,weno5d_si
  };
  ReconstructionVariant xorder_style;
  ReconstructionVariant xorder_style_fb;

  Real xorder_eps;                        // epsilon control parameters
  bool xorder_use_fb;                     // try order reduction
  bool xorder_use_fb_unphysical = false;  // try energy conditions

  bool xorder_use_auxiliaries;            // reconstruct derived quantities?
  bool xorder_use_aux_T;                  // reconstruct temperature?
  bool xorder_use_aux_h;                  // reconstruct enthalpy?

  bool characteristic_projection; // reconstruct on characteristic or primitive hydro vars
  bool uniform[3], curvilinear[2];
  // (Cartesian reconstruction formulas are used for x3 azimuthal coordinate in both
  // cylindrical and spherical-polar coordinates)

  // x1-sliced arrays of interpolation coefficients and limiter parameters:
  AthenaArray<Real> c1i, c2i, c3i, c4i, c5i, c6i;  // coefficients for PPM in x1
  AthenaArray<Real> hplus_ratio_i, hminus_ratio_i; // for curvilinear PPMx1
  AthenaArray<Real> c1j, c2j, c3j, c4j, c5j, c6j;  // coefficients for PPM in x2
  AthenaArray<Real> hplus_ratio_j, hminus_ratio_j; // for curvilinear PPMx2
  AthenaArray<Real> c1k, c2k, c3k, c4k, c5k, c6k;  // coefficients for PPM in x3
  AthenaArray<Real> hplus_ratio_k, hminus_ratio_k; // for curvilinear PPMx3

  // Refactored interface -----------------------------------------------------
  // More general, cleaner, switches variant case for a slice, not point-wise

  // Convenience function to fix ranges for indices during reconstruction
  inline void SetIndicialLimitsCalculateFluxes(
    const int dir,
    int & il, int & iu,
    int & jl, int & ju,
    int & kl, int & ku
  )
  {
    MeshBlock * pmb = pmy_block_;
    Mesh * pm = pmb->pmy_mesh;

    const int is = pmb->is;
    const int ie = pmb->ie;

    const int js = pmb->js;
    const int je = pmb->je;

    const int ks = pmb->ks;
    const int ke = pmb->ke;

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
  // --------------------------------------------------------------------------

  // logic for some variable collections --------------------------------------
  inline void ReconstructPrimitivesX1_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NHYDRO; ++n)
    {
      // wl_ populated at i+1 on Recon. call
      ReconstructFieldX1(rv, z, zl_, zr_, n, n, k, j, il-1, iu);
    }
  }

  inline void ReconstructMagneticFieldX1_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    // wl_ populated at i+1 on Recon. call
#if MAGNETIC_FIELDS_ENABLED
    ReconstructFieldX1(rv, z, zl_, zr_, IBY, IB2, k, j, il-1, iu);
    ReconstructFieldX1(rv, z, zl_, zr_, IBZ, IB3, k, j, il-1, iu);
#endif
  }

  inline void ReconstructPassiveScalarsX1_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    // wl_ populated at i+1 on Recon. call
    for (int n=0; n<NSCALARS; ++n)
    {
      ReconstructFieldX1(rv, z, zl_, zr_, n, n, k, j, il-1, iu);
    }
  }

  inline void ReconstructHydroAuxiliariesX1_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    // wl_ populated at i+1 on Recon. call
    for (int n=0; n<NDRV_HYDRO; ++n)
    {
      if (((n == IX_T) && xorder_use_aux_T) ||
           (n == IX_ETH && xorder_use_aux_h))
        ReconstructFieldX1(rv, z, zl_, zr_, n, n, k, j, il-1, iu);
    }
  }

  inline void ReconstructPrimitivesX2_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NHYDRO; ++n)
    {
      ReconstructFieldX2(rv, z, zl_, zr_, n, n, k, j, il, iu);
    }
  }

  inline void ReconstructMagneticFieldX2_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
#if MAGNETIC_FIELDS_ENABLED
    ReconstructFieldX2(rv, z, zl_, zr_, IBY, IB3, k, j, il, iu);
    ReconstructFieldX2(rv, z, zl_, zr_, IBZ, IB1, k, j, il, iu);
#endif
  }

  inline void ReconstructPassiveScalarsX2_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      ReconstructFieldX2(rv, z, zl_, zr_, n, n, k, j, il, iu);
    }
  }

  inline void ReconstructHydroAuxiliariesX2_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NDRV_HYDRO; ++n)
    {
      if (((n == IX_T) && xorder_use_aux_T) ||
           (n == IX_ETH && xorder_use_aux_h))
        ReconstructFieldX2(rv, z, zl_, zr_, n, n, k, j, il, iu);
    }
  }

  inline void ReconstructPrimitivesX3_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NHYDRO; ++n)
    {
      ReconstructFieldX3(rv, z, zl_, zr_, n, n, k, j, il, iu);
    }
  }

  inline void ReconstructMagneticFieldX3_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
#if MAGNETIC_FIELDS_ENABLED
    ReconstructFieldX3(rv, z, zl_, zr_, IBY, IB1, k, j, il, iu);
    ReconstructFieldX3(rv, z, zl_, zr_, IBZ, IB2, k, j, il, iu);
#endif
  }

  inline void ReconstructPassiveScalarsX3_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      ReconstructFieldX3(rv, z, zl_, zr_, n, n, k, j, il, iu);
    }
  }

  inline void ReconstructHydroAuxiliariesX3_(
    ReconstructionVariant rv,
    AthenaArray<Real> & z,
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    const int k, const int j, const int il, const int iu)
  {
    for (int n=0; n<NDRV_HYDRO; ++n)
    {
      if (((n == IX_T) && xorder_use_aux_T) ||
           (n == IX_ETH && xorder_use_aux_h))
        ReconstructFieldX3(rv, z, zl_, zr_, n, n, k, j, il, iu);
    }
  }

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

  // scratch arrays used in PLM and PPM reconstruction functions
  AthenaArray<Real> scr01_i_, scr02_i_, scr03_i_, scr04_i_, scr05_i_;
  AthenaArray<Real> scr06_i_, scr07_i_, scr08_i_, scr09_i_, scr10_i_;
  AthenaArray<Real> scr11_i_, scr12_i_, scr13_i_, scr14_i_;
  AthenaArray<Real> scr1_ni_, scr2_ni_, scr3_ni_, scr4_ni_, scr5_ni_;
  AthenaArray<Real> scr6_ni_, scr7_ni_, scr8_ni_;

private:
  // refactored (dead code)
  /*
  // functions
  // linear transformations of vectors between primitive and characteristic variables
  void LeftEigenmatrixDotVector(
      const int ivx, const int il, const int iu,
      const AthenaArray<Real> &b1, const AthenaArray<Real> &w, AthenaArray<Real> &vect);
  void RightEigenmatrixDotVector(
      const int ivx, const int il, const int iu,
      const AthenaArray<Real> &b1, const AthenaArray<Real> &w, AthenaArray<Real> &vect);

  // reconstruction functions of various orders in each dimension
  void DonorCellX1(const int k, const int j, const int il, const int iu,
                   const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                   AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void DonorCellX2(const int k, const int j, const int il, const int iu,
                   const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                   AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void DonorCellX3(const int k, const int j, const int il, const int iu,
                   const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                   AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void PiecewiseLinearX1(const int k, const int j, const int il, const int iu,
                         const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                         AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void PiecewiseLinearX2(const int k, const int j, const int il, const int iu,
                         const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                         AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void PiecewiseLinearX3(const int k, const int j, const int il, const int iu,
                         const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                         AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void PiecewiseParabolicX1(const int k, const int j, const int il, const int iu,
                            const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                            AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void PiecewiseParabolicX2(const int k, const int j, const int il, const int iu,
                            const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                            AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  void PiecewiseParabolicX3(const int k, const int j, const int il, const int iu,
                            const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
                            AthenaArray<Real> &wl, AthenaArray<Real> &wr);

  // overloads for non-fluid (cell-centered Hydro prim. and magnetic field) reconstruction
  void DonorCellX1(const int k, const int j, const int il, const int iu,
                   const AthenaArray<Real> &q,
                   AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void DonorCellX2(const int k, const int j, const int il, const int iu,
                   const AthenaArray<Real> &q,
                   AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void DonorCellX3(const int k, const int j, const int il, const int iu,
                   const AthenaArray<Real> &q,
                   AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void PiecewiseLinearX1(const int k, const int j, const int il, const int iu,
                         const AthenaArray<Real> &q,
                         AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void PiecewiseLinearX2(const int k, const int j, const int il, const int iu,
                         const AthenaArray<Real> &q,
                         AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void PiecewiseLinearX3(const int k, const int j, const int il, const int iu,
                         const AthenaArray<Real> &q,
                         AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void PiecewiseParabolicX1(const int k, const int j, const int il, const int iu,
                            const AthenaArray<Real> &q,
                            AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void PiecewiseParabolicX2(const int k, const int j, const int il, const int iu,
                            const AthenaArray<Real> &q,
                            AthenaArray<Real> &ql, AthenaArray<Real> &qr);

  void PiecewiseParabolicX3(const int k, const int j, const int il, const int iu,
                            const AthenaArray<Real> &q,
                            AthenaArray<Real> &ql, AthenaArray<Real> &qr);
  */


};
#endif // RECONSTRUCT_RECONSTRUCTION_HPP_
