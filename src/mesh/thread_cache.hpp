#ifndef MESH_THREAD_CACHE_HPP_
#define MESH_THREAD_CACHE_HPP_
//========================================================================================
//! \file thread_cache.hpp
//  \brief Per-thread scratch data for eliminating redundant computation across passes.
//
//  Currently holds:
//    1. Face-centered (FC) geometry fields cached during the high-order flux pass
//       so that the low-order fallback pass (hybridization) can reuse them without
//       re-interpolating the spacetime metric to face centers.
//    2. Low-order fallback flux scratch arrays (lo_hflux, lo_sflux) used during
//       hybridization - moved here from per-MeshBlock storage to save memory
//       (~2 MB per MeshBlock * nMeshBlocks -> ~2 MB * nthreads).
//
//  Uses a staged allocation API: each subsystem's scratch is allocated by a
//  separate method (AllocateFCGeom, AllocateLOFlux, ...).  The caller in
//  mesh.cpp decides which to call based on compile-time and runtime flags.
//========================================================================================

// C++ headers
#include <algorithm>  // std::max

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_aliases.hpp"
#include "../utils/floating_point.hpp"

using namespace gra::aliases;

//----------------------------------------------------------------------------------------
//! \struct ThreadCache
//  \brief Generic per-thread scratch container.
//
//  FC geometry cache layout:
//    fc_geom(dir * N_FC_GEOM_FIELDS + field, k, j, i)
//  where dir in {0,1,2} for X1,X2,X3 face directions,
//  field indexes the enum below, and (k,j,i) use nverts-sized dimensions.
//  All 3 directions are stored simultaneously so the HO pass can fill
//  X1->X2->X3 sequentially and the subsequent LO pass can read any direction.

struct ThreadCache {

  // Enumerate the FC geometry fields we cache per direction
  enum FCGeomField {
    FC_ALPHA = 0,
    FC_BETA_X,
    FC_BETA_Y,
    FC_BETA_Z,
    FC_GXX,
    FC_GXY,
    FC_GXZ,
    FC_GYY,
    FC_GYZ,
    FC_GZZ,
    FC_SQRT_DETGAMMA,
    FC_GAMMA_UU_DIAG,   // only the diagonal component gamma^{dd} for flux direction d
    N_FC_GEOM_FIELDS    // = 12
  };

  // 4D array: (3 * N_FC_GEOM_FIELDS, nverts3, nverts2, nverts1)
  AthenaArray<Real> fc_geom;

  // Per-thread scratch for low-order fallback fluxes (hybridization).
  // Face-centered arrays, one per spatial direction.  Left default-constructed
  // (empty) unless AllocateLOFlux() is called.
  AthenaArray<Real> lo_hflux[3];   // (NHYDRO,  ncells3, ncells2, ncells1+1) etc.
  AthenaArray<Real> lo_sflux[3];   // (NSCALARS, ncells3, ncells2, ncells1+1) etc.

  //--------------------------------------------------------------------------------------
  //! \fn void AllocateFCGeom(int ncells1, int ncells2, int ncells3)
  //  \brief Allocate the FC geometry cache.  ncells include ghost cells.
  void AllocateFCGeom(int ncells1, int ncells2, int ncells3) {
    int nv1 = ncells1 + 1;
    int nv2 = std::max(ncells2, 1) + (ncells2 > 1 ? 1 : 0);
    int nv3 = std::max(ncells3, 1) + (ncells3 > 1 ? 1 : 0);
    fc_geom.NewAthenaArray(3 * N_FC_GEOM_FIELDS, nv3, nv2, nv1);
  }

  //--------------------------------------------------------------------------------------
  //! \fn void AllocateLOFlux(...)
  //  \brief Allocate hydro + scalar low-order fallback flux scratch.
  //
  //  ncells1/2/3 include ghost cells.
  //  f2, f3: multi-dimensional flags from Mesh.
  //
  //  NHYDRO > 0 and NSCALARS > 0 guards are compile-time constants;
  //  the compiler eliminates dead branches entirely.
  void AllocateLOFlux(int ncells1, int ncells2, int ncells3,
                      bool f2, bool f3) {
    using DS = AthenaArray<Real>::DataStatus;

    if (NHYDRO > 0) {
      lo_hflux[0].NewAthenaArray(NHYDRO, ncells3, ncells2, ncells1 + 1);
      lo_hflux[1] = AthenaArray<Real>(NHYDRO, ncells3, ncells2 + 1, ncells1,
                                      (f2 ? DS::allocated : DS::empty));
      lo_hflux[2] = AthenaArray<Real>(NHYDRO, ncells3 + 1, ncells2, ncells1,
                                      (f3 ? DS::allocated : DS::empty));
    }

    if (NSCALARS > 0) {
      lo_sflux[0].NewAthenaArray(NSCALARS, ncells3, ncells2, ncells1 + 1);
      lo_sflux[1] = AthenaArray<Real>(NSCALARS, ncells3, ncells2 + 1, ncells1,
                                      (f2 ? DS::allocated : DS::empty));
      lo_sflux[2] = AthenaArray<Real>(NSCALARS, ncells3 + 1, ncells2, ncells1,
                                      (f3 ? DS::allocated : DS::empty));
    }
  }

  //--------------------------------------------------------------------------------------
  //! \fn void StoreFCGeometry(...)
  //  \brief Write the 12 FC geometry fields for a single pencil (k,j) in direction dir.
  //
  //  dir: 0=X1, 1=X2, 2=X3 (i.e. ivx-1)
  //  ivx: 1=IVX, 2=IVY, 3=IVZ - used only to select which gamma^{dd} diagonal to store

  inline void StoreFCGeometry(
    int dir, int k, int j, int il, int iu,
    const AT_N_sca & alpha,
    const AT_N_vec & beta_u,
    const AT_N_sym & gamma_dd,
    const AT_N_sca & sqrt_detgamma,
    const AT_N_sym & gamma_uu,
    int ivx)
  {
    const int base = dir * N_FC_GEOM_FIELDS;
    const int d = ivx - 1;  // diagonal index for gamma_uu

    for (int i = il; i <= iu; ++i) {
      fc_geom(base + FC_ALPHA,          k, j, i) = alpha(i);
      fc_geom(base + FC_BETA_X,         k, j, i) = beta_u(0, i);
      fc_geom(base + FC_BETA_Y,         k, j, i) = beta_u(1, i);
      fc_geom(base + FC_BETA_Z,         k, j, i) = beta_u(2, i);
      fc_geom(base + FC_GXX,            k, j, i) = gamma_dd(0, 0, i);
      fc_geom(base + FC_GXY,            k, j, i) = gamma_dd(0, 1, i);
      fc_geom(base + FC_GXZ,            k, j, i) = gamma_dd(0, 2, i);
      fc_geom(base + FC_GYY,            k, j, i) = gamma_dd(1, 1, i);
      fc_geom(base + FC_GYZ,            k, j, i) = gamma_dd(1, 2, i);
      fc_geom(base + FC_GZZ,            k, j, i) = gamma_dd(2, 2, i);
      fc_geom(base + FC_SQRT_DETGAMMA,  k, j, i) = sqrt_detgamma(i);
      fc_geom(base + FC_GAMMA_UU_DIAG,  k, j, i) = gamma_uu(d, d, i);
    }
  }

  //--------------------------------------------------------------------------------------
  //! \fn void LoadFCGeometry(...)
  //  \brief Read 12 primary fields from cache and recompute 4 cheap derived quantities.
  //
  //  Derived:
  //    oo_alpha       = 1 / regularize_near_zero(alpha, eps_alpha)
  //    detgamma       = sqrt_detgamma^2
  //    oo_detgamma    = 1 / detgamma
  //    oo_sqrt_detgamma = 1 / sqrt_detgamma

  inline void LoadFCGeometry(
    int dir, int k, int j, int il, int iu,
    AT_N_sca & alpha,
    AT_N_sca & oo_alpha,
    AT_N_vec & beta_u,
    AT_N_sym & gamma_dd,
    AT_N_sca & sqrt_detgamma,
    AT_N_sca & oo_sqrt_detgamma,
    AT_N_sca & detgamma,
    AT_N_sca & oo_detgamma,
    AT_N_sym & gamma_uu,
    int ivx,
    Real eps_alpha)
  {
    using namespace FloatingPoint;

    const int base = dir * N_FC_GEOM_FIELDS;
    const int d = ivx - 1;

    for (int i = il; i <= iu; ++i) {
      // Load primary cached fields
      alpha(i)          = fc_geom(base + FC_ALPHA,          k, j, i);
      beta_u(0, i)      = fc_geom(base + FC_BETA_X,         k, j, i);
      beta_u(1, i)      = fc_geom(base + FC_BETA_Y,         k, j, i);
      beta_u(2, i)      = fc_geom(base + FC_BETA_Z,         k, j, i);
      gamma_dd(0, 0, i) = fc_geom(base + FC_GXX,            k, j, i);
      gamma_dd(0, 1, i) = fc_geom(base + FC_GXY,            k, j, i);
      gamma_dd(0, 2, i) = fc_geom(base + FC_GXZ,            k, j, i);
      gamma_dd(1, 1, i) = fc_geom(base + FC_GYY,            k, j, i);
      gamma_dd(1, 2, i) = fc_geom(base + FC_GYZ,            k, j, i);
      gamma_dd(2, 2, i) = fc_geom(base + FC_GZZ,            k, j, i);
      sqrt_detgamma(i)  = fc_geom(base + FC_SQRT_DETGAMMA,  k, j, i);
      gamma_uu(d, d, i) = fc_geom(base + FC_GAMMA_UU_DIAG,  k, j, i);

      // Recompute cheap derived quantities
      Real alpha_reg     = regularize_near_zero(alpha(i), eps_alpha);
      oo_alpha(i)        = OO(alpha_reg);
      detgamma(i)        = SQR(sqrt_detgamma(i));
      oo_detgamma(i)     = OO(detgamma(i));
      oo_sqrt_detgamma(i) = OO(sqrt_detgamma(i));
    }
  }

};

#endif // MESH_THREAD_CACHE_HPP_
