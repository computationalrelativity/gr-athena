//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file reconstruction.cpp
//  \brief

// C headers

// C++ headers
#include <cmath>      // abs()
#include <sstream>
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "reconstruction.hpp"

namespace {

typedef Reconstruction::ReconstructionVariant ReconVar;

void GetVariant(
  MeshBlock *pmb, ParameterInput *pin,
  const std::string &sxorder,
  const std::string &sxorder_eps,
  ReconVar &xorder_style,
  Real &xorder_eps)
{

  xorder_eps = pin->GetOrAddReal("time", sxorder_eps, 1e-40);
  std::string str_xorder_style = pin->GetOrAddString(
    "time", sxorder, "lin_vl");

  if (str_xorder_style == "donate")
  {
    xorder_style = ReconVar::donate;
  }
  else if (str_xorder_style == "lin_vl")
  {
    xorder_style = ReconVar::lin_vl;
  }
  else if (str_xorder_style == "lin_mc2")
  {
    xorder_style = ReconVar::lin_mc2;
  }
  else if (str_xorder_style == "ppm")
  {
    xorder_style = ReconVar::ppm;
  }
  else if (str_xorder_style == "ceno3")
  {
    xorder_style = ReconVar::ceno3;
  }
  else if (str_xorder_style == "mp3")
  {
    xorder_style = ReconVar::mp3;
  }
  else if (str_xorder_style == "mp5")
  {
    xorder_style = ReconVar::mp5;
  }
  else if (str_xorder_style == "mp7")
  {
    xorder_style = ReconVar::mp7;
  }
  else if (str_xorder_style == "mp5_R")
  {
    xorder_style = ReconVar::mp5_R;
  }
  else if (str_xorder_style == "weno5")
  {
    xorder_style = ReconVar::weno5;
  }
  else if (str_xorder_style == "weno5z")
  {
    xorder_style = ReconVar::weno5z;
  }
  else if (str_xorder_style == "weno5d_si")
  {
    xorder_style = ReconVar::weno5d_si;
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
        << "xorder=" << str_xorder_style
        << " not valid choice for reconstruction"<< std::endl;
    ATHENA_ERROR(msg);
  }

}

}


Reconstruction::Reconstruction(MeshBlock *pmb, ParameterInput *pin)
  : uniform{true, true, true},
    pmy_block_{pmb}
{

  // Read and set type of spatial reconstruction
  GetVariant(pmb, pin, "xorder", "xorder_eps", xorder_style, xorder_eps);

  xorder_use_fb = pin->GetOrAddBoolean("time", "xorder_use_fb", false);

  if (xorder_use_fb)
  {
    GetVariant(pmb, pin,
               "xorder_fb",
               "xorder_eps",
               xorder_style_fb,
               xorder_eps);
  }

  xorder_floor_primitives = pin->GetOrAddBoolean(
    "time", "xorder_floor_primitives", true
  );

  xorder_limit_species = pin->GetOrAddBoolean(
    "time", "xorder_limit_species", true
  );

  xorder_fb_dfloor_fac = pin->GetOrAddReal(
    "time", "xorder_fb_dfloor_fac", 1.0
  );

  xorder_fb_Y_min_fac = pin->GetOrAddReal(
    "time", "xorder_fb_Y_min_fac", 1.0
  );

  xorder_fb_Y_max_fac = pin->GetOrAddReal(
    "time", "xorder_fb_Y_max_fac", 1.0
  );

  xorder_upwind_scalars = pin->GetOrAddBoolean(
    "time", "xorder_upwind_scalars", true
  );

  xorder_use_dmp = pin->GetOrAddBoolean(
    "time", "xorder_use_dmp", false
  );

  xorder_use_dmp_scalars = pin->GetOrAddBoolean(
    "time", "xorder_use_dmp_scalars",
    (xorder_use_dmp) ? true : false
  );

  xorder_dmp_min = pin->GetOrAddReal(
    "time", "xorder_dmp_min", 0.9
  );

  xorder_dmp_max = pin->GetOrAddReal(
    "time", "xorder_dmp_max", 1.1
  );

  if (xorder_use_dmp && !xorder_use_fb)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in Reconstruction constructor" << std::endl
        << "xorder_use_dmp requires xorder_use_fb."<< std::endl;
    ATHENA_ERROR(msg);
  }

  xorder_min_tau_zero = pin->GetOrAddBoolean(
    "time", "xorder_min_tau_zero", false
  );

  xorder_use_aux_T = pin->GetOrAddBoolean(
    "time", "xorder_use_aux_T", false
  );

  xorder_use_aux_h = pin->GetOrAddBoolean(
    "time", "xorder_use_aux_h", false
  );

  xorder_use_aux_W = pin->GetOrAddBoolean(
    "time", "xorder_use_aux_W", false
  );

  xorder_use_aux_cs2 = pin->GetOrAddBoolean(
    "time", "xorder_use_aux_cs2", false
  );

  xorder_limit_fluxes = pin->GetOrAddBoolean(
    "time", "xorder_limit_fluxes", false
  );

  enforce_limits_integration = pin->GetOrAddBoolean(
    "time", "enforce_limits_integration", false
  );

  enforce_limits_flux_div = pin->GetOrAddBoolean(
    "time", "enforce_limits_flux_div", false
  );

  // Check for incompatible choices with broader solver configuration
  // --------------------------------

  // Detect nonuniform meshes
  if (pmb->block_size.x1rat != 1.0)
    uniform[X1DIR] = false;
  if (pmb->block_size.x2rat != 1.0)
    uniform[X2DIR] = false;
  if (pmb->block_size.x3rat != 1.0)
    uniform[X3DIR] = false;

  // Uniform mesh with --coord=cartesian or GR: Minkowski, Schwarzschild, Kerr-Schild,
  // GR-User will use the uniform Cartesian limiter and reconstruction weights

  // Allocate memory for scratch arrays used in reconstruction
  int nc1 = pmb->ncells1;
  scr02_i_.NewAthenaArray(nc1);

  scr2_ni_.NewAthenaArray(NWAVE, nc1);
  scr3_ni_.NewAthenaArray(NWAVE, nc1);
  scr4_ni_.NewAthenaArray(NWAVE, nc1);

  {
    Coordinates *pco = pmb->pcoord;
    scr03_i_.NewAthenaArray(nc1);
    scr04_i_.NewAthenaArray(nc1);
    scr05_i_.NewAthenaArray(nc1);
    scr06_i_.NewAthenaArray(nc1);
    scr07_i_.NewAthenaArray(nc1);
    scr08_i_.NewAthenaArray(nc1);
    scr09_i_.NewAthenaArray(nc1);
    scr10_i_.NewAthenaArray(nc1);
    scr11_i_.NewAthenaArray(nc1);
    scr12_i_.NewAthenaArray(nc1);
    scr13_i_.NewAthenaArray(nc1);
    scr14_i_.NewAthenaArray(nc1);

    scr5_ni_.NewAthenaArray(NWAVE, nc1);
    scr6_ni_.NewAthenaArray(NWAVE, nc1);

    // Precompute PPM coefficients in x1-direction ---------------------------------------
    c1i.NewAthenaArray(nc1);
    c2i.NewAthenaArray(nc1);
    c3i.NewAthenaArray(nc1);
    c4i.NewAthenaArray(nc1);
    c5i.NewAthenaArray(nc1);
    c6i.NewAthenaArray(nc1);

    // Cartesian-like x1 coordinate
    // 4th order reconstruction weights along Cartesian-like x1 w/ uniform spacing
    if (uniform[X1DIR]) {
#pragma omp simd
      for (int i=(pmb->is)-NGHOST; i<=(pmb->ie)+NGHOST; ++i) {
        // reducing general formula corresponds to Mignone eq B.4 weights:
        // (-1/12, 7/12, 7/12, -1/12)
        c1i(i) = 0.5;
        c2i(i) = 0.5;
        c3i(i) = 0.5;
        c4i(i) = 0.5;
        c5i(i) = 1.0/6.0;
        c6i(i) = -1.0/6.0;
      }
    } else { // coefficients along Cartesian-like x1 with nonuniform mesh spacing
#pragma omp simd
      for (int i=(pmb->is)-NGHOST+1; i<=(pmb->ie)+NGHOST-1; ++i) {
        Real& dx_im1 = pco->dx1f(i-1);
        Real& dx_i   = pco->dx1f(i  );
        Real& dx_ip1 = pco->dx1f(i+1);
        Real qe = dx_i/(dx_im1 + dx_i + dx_ip1);       // Outermost coeff in CW eq 1.7
        c1i(i) = qe*(2.0*dx_im1+dx_i)/(dx_ip1 + dx_i); // First term in CW eq 1.7
        c2i(i) = qe*(2.0*dx_ip1+dx_i)/(dx_im1 + dx_i); // Second term in CW eq 1.7
        if (i > (pmb->is)-NGHOST+1) {  // c3-c6 are not computed in first iteration
          Real& dx_im2 = pco->dx1f(i-2);
          Real qa = dx_im2 + dx_im1 + dx_i + dx_ip1;
          Real qb = dx_im1/(dx_im1 + dx_i);
          Real qc = (dx_im2 + dx_im1)/(2.0*dx_im1 + dx_i);
          Real qd = (dx_ip1 + dx_i)/(2.0*dx_i + dx_im1);
          qb = qb + 2.0*dx_i*qb/qa*(qc-qd);
          c3i(i) = 1.0 - qb;
          c4i(i) = qb;
          c5i(i) = dx_i/qa*qd;
          c6i(i) = -dx_im1/qa*qc;
        }
      }
    }

    // Precompute PPM coefficients in x2-direction ---------------------------------------
    if (pmb->block_size.nx2 > 1) {
      int nc2 = pmb->ncells2;
      c1j.NewAthenaArray(nc2);
      c2j.NewAthenaArray(nc2);
      c3j.NewAthenaArray(nc2);
      c4j.NewAthenaArray(nc2);
      c5j.NewAthenaArray(nc2);
      c6j.NewAthenaArray(nc2);

      // Cartesian-like x2 coordinate
      // 4th order reconstruction weights along Cartesian-like x2 w/ uniform spacing
      if (uniform[X2DIR]) {
#pragma omp simd
        for (int j=(pmb->js)-NGHOST; j<=(pmb->je)+NGHOST; ++j) {
          c1j(j) = 0.5;
          c2j(j) = 0.5;
          c3j(j) = 0.5;
          c4j(j) = 0.5;
          c5j(j) = 1.0/6.0;
          c6j(j) = -1.0/6.0;
        }
      } else { // coefficients along Cartesian-like x2 with nonuniform mesh spacing
#pragma omp simd
        for (int j=(pmb->js)-NGHOST+2; j<=(pmb->je)+NGHOST-1; ++j) {
          Real& dx_jm1 = pco->dx2f(j-1);
          Real& dx_j   = pco->dx2f(j  );
          Real& dx_jp1 = pco->dx2f(j+1);
          Real qe = dx_j/(dx_jm1 + dx_j + dx_jp1);       // Outermost coeff in CW eq 1.7
          c1j(j) = qe*(2.0*dx_jm1 + dx_j)/(dx_jp1 + dx_j); // First term in CW eq 1.7
          c2j(j) = qe*(2.0*dx_jp1 + dx_j)/(dx_jm1 + dx_j); // Second term in CW eq 1.7

          if (j > (pmb->js)-NGHOST+1) {  // c3-c6 are not computed in first iteration
            Real& dx_jm2 = pco->dx2f(j-2);
            Real qa = dx_jm2 + dx_jm1 + dx_j + dx_jp1;
            Real qb = dx_jm1/(dx_jm1 + dx_j);
            Real qc = (dx_jm2 + dx_jm1)/(2.0*dx_jm1 + dx_j);
            Real qd = (dx_jp1 + dx_j)/(2.0*dx_j + dx_jm1);
            qb = qb + 2.0*dx_j*qb/qa*(qc-qd);
            c3j(j) = 1.0 - qb;
            c4j(j) = qb;
            c5j(j) = dx_j/qa*qd;
            c6j(j) = -dx_jm1/qa*qc;
          }
        }
      } // end nonuniform Cartesian-like
    } // end 2D or 3D

    // Precompute PPM coefficients in x3-direction
    if (pmb->block_size.nx3 > 1) {
      int nc3 = pmb->ncells3;
      c1k.NewAthenaArray(nc3);
      c2k.NewAthenaArray(nc3);
      c3k.NewAthenaArray(nc3);
      c4k.NewAthenaArray(nc3);
      c5k.NewAthenaArray(nc3);
      c6k.NewAthenaArray(nc3);

      // reconstruction coefficients in x3, Cartesian-like coordinate:
      if (uniform[X3DIR]) { // uniform spacing
#pragma omp simd
        for (int k=(pmb->ks)-NGHOST; k<=(pmb->ke)+NGHOST; ++k) {
          c1k(k) = 0.5;
          c2k(k) = 0.5;
          c3k(k) = 0.5;
          c4k(k) = 0.5;
          c5k(k) = 1.0/6.0;
          c6k(k) = -1.0/6.0;
        }

      } else { // nonuniform spacing
#pragma omp simd
        for (int k=(pmb->ks)-NGHOST+2; k<=(pmb->ke)+NGHOST-1; ++k) {
          Real& dx_km1 = pco->dx3f(k-1);
          Real& dx_k   = pco->dx3f(k  );
          Real& dx_kp1 = pco->dx3f(k+1);
          Real qe = dx_k/(dx_km1 + dx_k + dx_kp1);       // Outermost coeff in CW eq 1.7
          c1k(k) = qe*(2.0*dx_km1+dx_k)/(dx_kp1 + dx_k); // First term in CW eq 1.7
          c2k(k) = qe*(2.0*dx_kp1+dx_k)/(dx_km1 + dx_k); // Second term in CW eq 1.7

          if (k > (pmb->ks)-NGHOST+1) {  // c3-c6 are not computed in first iteration
            Real& dx_km2 = pco->dx3f(k-2);
            Real qa = dx_km2 + dx_km1 + dx_k + dx_kp1;
            Real qb = dx_km1/(dx_km1 + dx_k);
            Real qc = (dx_km2 + dx_km1)/(2.0*dx_km1 + dx_k);
            Real qd = (dx_kp1 + dx_k)/(2.0*dx_k + dx_km1);
            qb = qb + 2.0*dx_k*qb/qa*(qc-qd);
            c3k(k) = 1.0 - qb;
            c4k(k) = qb;
            c5k(k) = dx_k/qa*qd;
            c6k(k) = -dx_km1/qa*qc;
          }
        }
      }
    }
  } // end scratch allocation and coefficient precomputation
}

// Refactored interface -------------------------------------------------------
void Reconstruction::ReconstructFieldX1(
  const ReconstructionVariant rv,
  AthenaArray<Real> &z,
  AthenaArray<Real> &zl_,
  AthenaArray<Real> &zr_,
  const int n_tar,
  const int n_src,
  const int k,
  const int j,
  const int il, const int iu)
{
  typedef Reconstruction::ReconstructionVariant ReconVar;

  switch (rv)
  {
    case (ReconVar::donate):
    {
      ReconstructDonateX1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::lin_vl):
    {
      ReconstructLinearVLX1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::lin_mc2):
    {
      ReconstructLinearMC2X1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::ppm):
    {
      ReconstructPPMX1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::ceno3):
    {
      ReconstructCeno3X1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5):
    {
      ReconstructWeno5X1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5z):
    {
      ReconstructWeno5ZX1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5d_si):
    {
      ReconstructWeno5dsiX1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp3):
    {
      ReconstructMP3X1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp5):
    {
      ReconstructMP5X1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp7):
    {
      ReconstructMP7X1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp5_R):
    {
      ReconstructMP5RX1(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void Reconstruction::ReconstructFieldX2(
  const ReconstructionVariant rv,
  AthenaArray<Real> &z,
  AthenaArray<Real> &zl_,
  AthenaArray<Real> &zr_,
  const int n_tar,
  const int n_src,
  const int k,
  const int j,
  const int il, const int iu)
{
  typedef Reconstruction::ReconstructionVariant ReconVar;

  switch (rv)
  {
    case (ReconVar::donate):
    {
      ReconstructDonateX2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::lin_vl):
    {
      ReconstructLinearVLX2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::lin_mc2):
    {
      ReconstructLinearMC2X2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::ppm):
    {
      ReconstructPPMX2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::ceno3):
    {
      ReconstructCeno3X2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5):
    {
      ReconstructWeno5X2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5z):
    {
      ReconstructWeno5ZX2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5d_si):
    {
      ReconstructWeno5dsiX2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp3):
    {
      ReconstructMP3X2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp5):
    {
      ReconstructMP5X2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp7):
    {
      ReconstructMP7X2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp5_R):
    {
      ReconstructMP5RX2(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void Reconstruction::ReconstructFieldX3(
  const ReconstructionVariant rv,
  AthenaArray<Real> &z,
  AthenaArray<Real> &zl_,
  AthenaArray<Real> &zr_,
  const int n_tar,
  const int n_src,
  const int k,
  const int j,
  const int il, const int iu)
{
  typedef Reconstruction::ReconstructionVariant ReconVar;

  switch (rv)
  {
    case (ReconVar::donate):
    {
      ReconstructDonateX3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::lin_vl):
    {
      ReconstructLinearVLX3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::lin_mc2):
    {
      ReconstructLinearMC2X3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::ppm):
    {
      ReconstructPPMX3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::ceno3):
    {
      ReconstructCeno3X3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5):
    {
      ReconstructWeno5X3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5z):
    {
      ReconstructWeno5ZX3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::weno5d_si):
    {
      ReconstructWeno5dsiX3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp3):
    {
      ReconstructMP3X3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp5):
    {
      ReconstructMP5X3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp7):
    {
      ReconstructMP7X3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    case (ReconVar::mp5_R):
    {
      ReconstructMP5RX3(z, zl_, zr_, n_tar, n_src, k, j, il, iu);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void Reconstruction::ReconstructFieldXd(
  const ReconstructionVariant rv,
  AthenaArray<Real> &z,
  AthenaArray<Real> &zl_,
  AthenaArray<Real> &zr_,
  const int ivx,
  const int n_tar,
  const int n_src,
  const int k,
  const int j,
  const int il, const int iu)
{
  switch (ivx)
  {
    case 1:
    {
      ReconstructFieldX1(
        rv, z, zl_, zr_,
        n_tar, n_src,
        k, j, il, iu
      );
      break;
    }
    case 2:
    {
      ReconstructFieldX2(
        rv, z, zl_, zr_,
        n_tar, n_src,
        k, j, il, iu
      );
      break;
    }
    case 3:
    {
      ReconstructFieldX3(
        rv, z, zl_, zr_,
        n_tar, n_src,
        k, j, il, iu
      );
      break;
    }
  }
}
// ----------------------------------------------------------------------------
