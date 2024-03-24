//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_dynamical.cpp
//  \brief Used for arbitrary dynamially evolving coordinates in general relativity
//  Original implementation by CJ White.

// C headers

// C++ headers
#include <cmath>  // sqrt()

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../parameter_input.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "coordinates.hpp"
#include "../utils/linear_algebra.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

//SB TODO this needs a cleanup
#define CLOOP1(i)				\
  _Pragma("omp simd")				\
  for (int i=is; i<=ie; ++i)

//----------------------------------------------------------------------------------------
// GRDynamical Constructor
// Inputs:
//   pmb: pointer to MeshBlock containing this grid
//   pin: pointer to runtime inputs
//   flag: true if object is for coarse grid only in an AMR calculation

GRDynamical::GRDynamical(MeshBlock *pmb, ParameterInput *pin, bool flag)
  : Coordinates(pmb, pin, flag) {

  // Set object names
  RegionSize& block_size = pmy_block->block_size;
  fix_sources = pin->GetOrAddInteger("hydro","fix_sources",0);
  zero_sources = pin->GetOrAddInteger("hydro","zero_sources",0);
  // set more indices
  int ill = il - ng;
  int iuu = iu + ng;
  int jll, juu;
  if (block_size.nx2 > 1) {
    jll = jl - ng;
    juu = ju + ng;
  } else {
    jll = jl;
    juu = ju;
  }
  int kll, kuu;
  if (block_size.nx3 > 1) {
    kll = kl - ng;
    kuu = ku + ng;
  } else {
    kll = kl;
    kuu = ku;
  }
  // needed for coarse representation
  chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);

  // Initialize volume-averaged coordinates and spacings: r-direction
  for (int i=ill; i<=iuu; ++i) {
    Real r_m = x1f(i);
    Real r_p = x1f(i+1);
    x1v(i) = 0.5 * (r_m + r_p);  // at least 2nd-order accurate
  }
  for (int i=ill; i<=iuu-1; ++i) {
    dx1v(i) = x1v(i+1) - x1v(i);
  }

  // Initialize volume-averaged coordinates and spacings: theta-direction
  if (block_size.nx2 == 1) {
    Real theta_m = x2f(jl);
    Real theta_p = x2f(jl+1);
    x2v(jl) = 0.5 * (theta_m + theta_p);  // at least 2nd-order accurate
    dx2v(jl) = dx2f(jl);
  } else {
    for (int j=jll; j<=juu; ++j) {
      Real theta_m = x2f(j);
      Real theta_p = x2f(j+1);
      x2v(j) = 0.5 * (theta_m + theta_p);  // at least 2nd-order accurate
    }
    for (int j=jll; j<=juu-1; ++j) {
      dx2v(j) = x2v(j+1) - x2v(j);
    }
  }

  // Initialize volume-averaged coordinates and spacings: phi-direction
  if (block_size.nx3 == 1) {
    Real phi_m = x3f(kl);
    Real phi_p = x3f(kl+1);
    x3v(kl) = 0.5 * (phi_m + phi_p);  // at least 2nd-order accurate
    dx3v(kl) = dx3f(kl);
  } else {
    for (int k=kll; k<=kuu; ++k) {
      Real phi_m = x3f(k);
      Real phi_p = x3f(k+1);
      x3v(k) = 0.5 * (phi_m + phi_p);  // at least 2nd-order accurate
    }
    for (int k=kll; k<=kuu-1; ++k) {
      dx3v(k) = x3v(k+1) - x3v(k);
    }
  }

  // Initialize area-averaged coordinates used with MHD AMR
  if (pm->multilevel && MAGNETIC_FIELDS_ENABLED) {
    for (int i=ill; i<=iuu; ++i) {
      x1s2(i) = x1s3(i) = x1v(i);
    }
    if (block_size.nx2 == 1) {
      x2s1(jl) = x2s3(jl) = x2v(jl);
    } else {
      for (int j=jll; j<=juu; ++j) {
        x2s1(j) = x2s3(j) = x2v(j);
      }
    }
    if (block_size.nx3 == 1) {
      x3s1(kl) = x3s2(kl) = x3v(kl);
    } else {
      for (int k=kll; k<=kuu; ++k) {
        x3s1(k) = x3s2(k) = x3v(k);
      }
    }
  }

  // Allocate arrays for geometric quantities
  if (!coarse_flag) {
    g_.NewAthenaArray(NMETRIC, nc1+1);
    gi_.NewAthenaArray(NMETRIC, nc1+1);
  }

  // Allocate scratch arrays
  AthenaArray<Real> g, g_inv, dg_dx1, dg_dx2, dg_dx3, transformation;
  g.NewAthenaArray(NMETRIC);
  g_inv.NewAthenaArray(NMETRIC);
  dg_dx1.NewAthenaArray(NMETRIC);
  dg_dx2.NewAthenaArray(NMETRIC);
  dg_dx3.NewAthenaArray(NMETRIC);
  if (!coarse_flag) {
    transformation.NewAthenaArray(2, NTRIANGULAR);
  }

  // Set up finite differencing -----------------------------------------------
  fd_is_defined = true;
  fd_cc = new FiniteDifference::Uniform(
    nc1, nc2, nc3,
    dx1v(0), dx2v(0), dx3v(0)
  );

  fd_cx = new FiniteDifference::Uniform(
    cx_nc1, cx_nc2, cx_nc3,
    dx1v(0), dx2v(0), dx3v(0)
  );

  fd_vc = new FiniteDifference::Uniform(
    nv1, nv2, nv3,
    dx1f(0), dx2f(0), dx3f(0)
  );

  // intergrid interpolators --------------------------------------------------
  // for metric <-> matter sampling conversion
  //
  // if metric_vc then need {vc->fc, vc->cc}
  // if metric_cx then only need cx->fc
  //
  // this motivates the choices of ghosts below

  const int ng_c = (coarse_flag)? NCGHOST_CX : NGHOST;
  const int ng_v = (coarse_flag)? NCGHOST    : NGHOST;

  int N[] = {
    nc1 - 2 * ng_c,
    nc2 - 2 * ng_c,
    nc3 - 2 * ng_c
  };
  Real rdx[] = {
    1./(x1f(1)-x1f(0)),
    1./(x2f(1)-x2f(0)),
    1./(x3f(1)-x3f(0))
  };

  const int dim = (pmb->pmy_mesh->f3) ? 3 : (
    (pmb->pmy_mesh->f2) ? 2 : 1
  );

  ig_is_defined = true;

  ig_1N = new IIG_1N(dim, &N[0], &rdx[0], ng_c, ng_v);
  ig_2N = new IIG_2N(dim, &N[0], &rdx[0], ng_c, ng_v);
  ig_NN = new IIG_NN(dim, &N[0], &rdx[0], ng_c, ng_v);

  // Metric quantities not initialised in constructor, need to wait for UpdateMetric()
  // to be called once VC metric is initialised in pgen
}

//----------------------------------------------------------------------------------------
// EdgeXLength functions: compute physical length at cell edge-X as vector
// Edge1(i,j,k) located at (i,j-1/2,k-1/2), i.e. (x1v(i), x2f(j), x3f(k))
// Edge2(i,j,k) located at (i-1/2,j,k-1/2), i.e. (x1f(i), x2v(j), x3f(k))
// Edge3(i,j,k) located at (i-1/2,j-1/2,k), i.e. (x1f(i), x2f(j), x3v(k))

void GRDynamical::Edge1Length(const int k, const int j, const int il, const int iu,
			      AthenaArray<Real> &lengths) {
  // \Delta L \approx \sqrt{-g} \Delta x^1
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // lengths(i) = coord_len1_kji_(k,j,i);
    //WGC for fluxcorrection_fc want just dx1
    lengths(i) = dx1f(i);
  }
  return;
}

void GRDynamical::Edge2Length(const int k, const int j, const int il, const int iu,
                         AthenaArray<Real> &lengths) {
  // \Delta L \approx \sqrt{-g} \Delta x^2
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // lengths(i) = coord_len2_kji_(k,j,i);
    //WGC for fluxcorrection_fc want just dx2
    lengths(i) = dx2f(j);
  }
  return;
}

void GRDynamical::Edge3Length(const int k, const int j, const int il, const int iu,
			      AthenaArray<Real> &lengths) {
  // \Delta L \approx \sqrt{-g} \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // lengths(i) = coord_len3_kji_(k,j,i);
    //WGC for fluxcorrection_fc want just dx3
    lengths(i) = dx3f(k);
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetEdgeXLength functions: return length of edge-X at (i,j,k)

Real GRDynamical::GetEdge1Length(const int k, const int j, const int i) {
  // \Delta L \approx \sqrt{-g} \Delta x^1
  // return coord_len1_kji_(k,j,i);
  //WGC for ProlongateInternalField make dx1
  return dx1f(i);
}

Real GRDynamical::GetEdge2Length(const int k, const int j, const int i) {
  // \Delta L \approx \sqrt{-g} \Delta x^2
  // return coord_len2_kji_(k,j,i);
  //WGC for ProlongateInternalField make dx2
  return dx2f(j);
}

Real GRDynamical::GetEdge3Length(const int k, const int j, const int i) {
  // \Delta L \approx \sqrt{-g} \Delta x^3
  // return coord_len3_kji_(k,j,i);
  //WGC for ProlongateInternalField make dx3
  return dx3f(k);
}

//----------------------------------------------------------------------------------------
// CenterWidthX functions: return physical width in X-dir at (i,j,k) cell-center

void GRDynamical::CenterWidth1(const int k, const int j, const int il, const int iu,
			       AthenaArray<Real> &dx1) {
  // \Delta W \approx \sqrt{g_{11}} \Delta x^1
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // dx1(i) = coord_width1_kji_(k,j,i);
    //WGC replace for CT weight without vol weighting
    dx1(i) = dx1v(i);
  }
  return;
}

void GRDynamical::CenterWidth2(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &dx2) {
  // \Delta W \approx \sqrt{g_{22}} \Delta x^2
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // dx2(i) = coord_width2_kji_(k,j,i);
    dx2(i) = dx2v(j);
  }
  return;
}

void GRDynamical::CenterWidth3(const int k, const int j, const int il, const int iu,
                          AthenaArray<Real> &dx3) {
  // \Delta W \approx \sqrt{g_{33}} \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // dx3(i) = coord_width3_kji_(k,j,i);
    dx3(i) = dx3v(k);
  }
  return;
}

//----------------------------------------------------------------------------------------
// FaceXArea functions: compute area of face with normal in X-dir as vector
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
// Outputs:
//   areas: 1D array of interface areas orthogonal to X-face

void GRDynamical::Face1Area(const int k, const int j, const int il, const int iu,
                       AthenaArray<Real> &areas) {
  // \Delta A \approx \sqrt{-g} \Delta x^2 \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // areas(i) = coord_area1_kji_(k,j,i);
    //WC: for ProlongateInternalField make dx2*dx3
    areas(i) = dx2f(j)*dx3f(k);
  }
  return;
}

void GRDynamical::Face2Area(const int k, const int j, const int il, const int iu,
                       AthenaArray<Real> &areas) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    // areas(i) = coord_area2_kji_(k,j,i);
    //WC: for ProlongateInternalField make dx1*dx3
    areas(i) = dx1f(i)*dx3f(k);
  }
  return;
}

void GRDynamical::Face3Area(const int k, const int j, const int il, const int iu,
			    AthenaArray<Real> &areas) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^2
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    //    areas(i) = coord_area3_kji_(k,j,i);
    //WC: for ProlongateInternalField make dx1*dx2
    areas(i) = dx1f(i)*dx2f(j);
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetFaceXArea functions: return area of face with normal in X-dir at (i,j,k)
// Inputs:
//   k,j,i: x3-, x2-, and x1-indices
// return:
//   interface area orthogonal to X-face

Real GRDynamical::GetFace1Area(const int k, const int j, const int i) {
  // \Delta A \approx \sqrt{-g} \Delta x^2 \Delta x^3
  //removed coord_area?_kji_
  //  return coord_area1_kji_(k,j,i);
  return dx2f(j)*dx3f(k);
}

Real GRDynamical::GetFace2Area(const int k, const int j, const int i) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^3
  //  return coord_area2_kji_(k,j,i);
  return dx1f(i)*dx3f(k);
}

Real GRDynamical::GetFace3Area(const int k, const int j, const int i) {
  // \Delta A \approx \sqrt{-g} \Delta x^1 \Delta x^2
  //  return coord_area3_kji_(k,j,i);
  return dx1f(i)*dx2f(j);
}

//----------------------------------------------------------------------------------------
// Cell Volume function: compute volume of cell as vector
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
// Outputs:
//   volumes: 1D array of cell volumes

void GRDynamical::CellVolume(const int k, const int j, const int il, const int iu,
			     AthenaArray<Real> &volumes) {
  // \Delta V \approx \sqrt{-g} \Delta x^1 \Delta x^2 \Delta x^3
#pragma omp simd
  for (int i=il; i<=iu; ++i) {
    //  WC edit: make not volume weighted for restrictCC in mesh_refinement
    //    volumes(i) = coord_vol_kji_(k,j,i);
    volumes(i) = dx1v(i)*dx2v(j)*dx3v(k);
  }
  return;
}

//----------------------------------------------------------------------------------------
// GetCellVolume: returns cell volume at (i,j,k)
// Inputs:
//   k,j,i: phi-, theta-, and r-indices
// Outputs:
//   returned value: cell volume

Real GRDynamical::GetCellVolume(const int k, const int j, const int i) {
  // \Delta V \approx \sqrt{-g} \Delta x^1 \Delta x^2 \Delta x^3
  //  return coord_vol_kji_(k,j,i);
  //  WC edit: make not volume weighted for restrictCC in mesh_refinement
  return dx1v(i)*dx2v(j)*dx3v(k);
}

//----------------------------------------------------------------------------------------
// Coordinate (geometric) source term function
// Inputs:
//   dt: size of timestep
//   flux: 3D array of fluxes
//   prim: 3D array of primitive values at beginning of half timestep
//   bb_cc: 3D array of cell-centered magnetic fields
// Outputs:
//   cons: source terms added to 3D array of conserved variables
#ifdef DBG_MA_SOURCES
void GRDynamical::AddCoordTermsDivergence(
  const Real dt,
  const AthenaArray<Real> *flux,
  const AthenaArray<Real> &prim,
#if USETM
  const AthenaArray<Real> &prim_scalar,
#endif
  const AthenaArray<Real> &bb_cc,
  AthenaArray<Real> &cons)
#else
void GRDynamical::_AddCoordTermsDivergence(
  const Real dt,
  const AthenaArray<Real> *flux,
  const AthenaArray<Real> &prim,
#if USETM
  const AthenaArray<Real> &prim_scalar,
#endif
  const AthenaArray<Real> &bb_cc,
  AthenaArray<Real> &cons)
#endif // DBG_MA_SOURCES
{

  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;

  const int il = pmb->is, iu = pmb->ie;
  const int jl = pmb->js, ju = pmb->je;
  const int kl = pmb->ks, ku = pmb->ke;

  // this is for matter
  int ms_nn1 = pmb->ncells1;

  // --------------------------------------------------------------------------
  // Perform variable resampling when required
  Z4c * pz4c = pmb->pz4c;

  // Slice z4c metric quantities
  AT_N_sym adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym adm_K_dd(    pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec adm_beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

  // Slice matter
  AT_N_sca sl_w_rho(   const_cast<AthenaArray<Real>&>(prim), IDN);
  AT_N_sca sl_w_p(     const_cast<AthenaArray<Real>&>(prim), IPR);
  AT_N_vec sl_w_util_u(const_cast<AthenaArray<Real>&>(prim), IVX);
#if NSCALARS > 0
<<<<<<< HEAD
  AT_N_vec sl_scalars_r;
  // AT_N_vec sl_scalars_r(const_cast<AthenaArray<Real>&>(prim_scalar), 0);
=======
  AT_N_vec sl_scalars_r(const_cast<AthenaArray<Real>&>(prim_scalar), 0);
>>>>>>> 74019e3a (Changes for Tabulated EoSs)
#endif

  // Scratch for matter sampling
  AT_N_sca ms_detg_(     ms_nn1); // det metric
  AT_N_sca ms_sqrt_detg_(ms_nn1); // sqrt det metric
  AT_N_sca ms_oo_detg_(  ms_nn1); // 1 / det metric
  AT_N_sca ms_alpha_(    ms_nn1); // lapse
  AT_N_sca ms_w_hrho_(   ms_nn1); // h rho (EoS dep.)
  AT_N_sca ms_W_(        ms_nn1); // Lorentz factor
  AT_N_vec ms_beta_u_(   ms_nn1); // beta^i
  AT_N_vec ms_w_util_u_( ms_nn1); // fluid velocity
  AT_N_vec ms_w_util_d_( ms_nn1); // fluid velocity

  AT_N_sca ms_K_(        ms_nn1); // trace extrinsic

  // AT_N_sca ms_src_wei_( ms_nn1);  // alpha * sqrt(det)

  AT_N_sym ms_adm_gamma_dd_(ms_nn1); // gamma_{ij}
  AT_N_sym ms_adm_gamma_uu_(ms_nn1); // gamma^{ij}
  AT_N_sym ms_adm_K_dd_(    ms_nn1); // K_{ij}

  AT_N_VS2 ms_Gamma_ddd_(ms_nn1);  // Christoffel symbols of 1st kind
  AT_N_VS2 ms_Gamma_udd_(ms_nn1);  // Christoffel symbols of 2nd kind
  AT_N_vec ms_Gamma_d_(  ms_nn1);    // contracted Christoffel


  // Derivative scratch
  AT_N_vec ms_dalpha_d_(      ms_nn1); // pd_i alpha
  AT_N_T2  ms_dbeta_du_(      ms_nn1); // pd_i beta^j
  AT_N_VS2 ms_adm_dgamma_ddd_(ms_nn1); // pd_i gamma_{jk}

  // Source terms
  AT_N_sca ms_S_tau_(ms_nn1);
  AT_N_vec ms_S_S_d_(ms_nn1);

  // --------------------------------------------------------------------------

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    // Note: internally maps geometric sampling to matter sampling
    GetGeometricFieldCC(ms_adm_gamma_dd_, adm_gamma_dd, k, j);
    GetGeometricFieldCC(ms_adm_K_dd_,     adm_K_dd,     k, j);
    GetGeometricFieldCC(ms_alpha_,        adm_alpha,    k, j);
    GetGeometricFieldCC(ms_beta_u_,       adm_beta_u,   k, j);

#if !defined(DBG_FD_CX_COORDDIV) || !defined(Z4C_CX_ENABLED)
    for(int a=0; a<NDIM; ++a)
    {
      GetGeometricFieldDerCC(ms_adm_dgamma_ddd_, adm_gamma_dd, a, k, j);
      GetGeometricFieldDerCC(ms_dalpha_d_,       adm_alpha,    a, k, j);
      GetGeometricFieldDerCC(ms_dbeta_du_,       adm_beta_u,   a, k, j);
    }
#else
    for (int a=0; a<NDIM; ++a)
    ILOOP1(i)
    {
      ms_dalpha_d_(a,i) = fd_cx->Dx(a, adm_alpha(k,j,i));
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    ILOOP1(i)
    {
      ms_dbeta_du_(b,a,i) = fd_cx->Dx(b, adm_beta_u(a,k,j,i));
    }

    // Tensors
    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    for (int c=0; c<NDIM; ++c)
    ILOOP1(i)
    {
      ms_adm_dgamma_ddd_(c,a,b,i) = fd_cx->Dx(c, adm_gamma_dd(a,b,k,j,i));
    }
#endif // DBG_FD_CX_COORDDIV

    // prepare g^{ij}
    for (int i=il; i<=iu; ++i)
    {
      ms_detg_(i) = Det3Metric(ms_adm_gamma_dd_, i);
      ms_sqrt_detg_(i) = std::sqrt(ms_detg_(i));
      ms_oo_detg_(i) = 1. / ms_detg_(i);
      Inv3Metric(ms_oo_detg_, ms_adm_gamma_dd_, ms_adm_gamma_uu_, i);
    }

    // prepare K
    ms_K_.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    for (int b = 0; b < NDIM; ++b)
    for (int i=il; i<=iu; ++i)
    {
      ms_K_(i) += ms_adm_gamma_uu_(a,b,i) * ms_adm_K_dd_(a,b,i);
    }

    // Prepare Christoffels (Gamma_{ijk})
    for (int c=0; c<NDIM; ++c)
    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    for (int i=il; i<=iu; ++i)
    {
      ms_Gamma_ddd_(c,a,b,i) = 0.5*(ms_adm_dgamma_ddd_(a,b,c,i) +
                                    ms_adm_dgamma_ddd_(b,a,c,i) -
                                    ms_adm_dgamma_ddd_(c,a,b,i));
    }

    // Prepare Christoffels (Gamma^i_{jk})
    ms_Gamma_udd_.ZeroClear();
    for (int c = 0; c < NDIM; ++c)
    for (int a = 0; a < NDIM; ++a)
    for (int b = a; b < NDIM; ++b)
    for (int d = 0; d < NDIM; ++d)
    for (int i=il; i<=iu; ++i)
    {
      ms_Gamma_udd_(c,a,b,i) += ms_adm_gamma_uu_(c,d,i) * ms_Gamma_ddd_(d,a,b,i);
    }

    // Prepare Christoffels (Gamma^k_{ik})
    ms_Gamma_d_.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for (int i=il; i<=iu; ++i)
    {
      ms_Gamma_d_(a,i) += ms_Gamma_udd_(b,a,b,i);
    }


    // prepare h*rho
    for (int i=il; i<=iu; ++i)
    {
#if USETM
      Real n = sl_w_rho(k,j,i)/pmy_block->peos->GetEOS().GetBaryonMass();
      // FIXME: Generalize to work with EOSes accepting particle fractions.
      Real Y[MAX_SPECIES] = {0.0};
<<<<<<< HEAD
#if NSCALARS > 0
      for (int l=0; l<NSCALARS; l++){
        Y[l] = sl_scalars_r(l,k,j,i);
      }
#endif
=======
      for (int l=0; l<NSCALARS; l++){
        Y[l] = sl_scalars_r(l,k,j,i);
      }
>>>>>>> 74019e3a (Changes for Tabulated EoSs)
      Real T = pmb->peos->GetEOS().GetTemperatureFromP(n,  sl_w_p(k,j,i), Y);
      ms_w_hrho_(i) = sl_w_rho(k,j,i)*pmb->peos->GetEOS().GetEnthalpy(n, T, Y);
#else
      Real gamma_adi = pmy_block->peos->GetGamma();
      const Real Eos_Gamma_ratio = gamma_adi / (gamma_adi - 1.0);

      ms_w_hrho_(i) = sl_w_rho(k,j,i) + Eos_Gamma_ratio * sl_w_p(k,j,i);
#endif
    }

    // prepare util^i (this slicing is needed for IP impl. below)
    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      ms_w_util_u_(a,i) = sl_w_util_u(a,k,j,i);
    }

    // prepare util_i
    SlicedVecMet3Contraction(ms_w_util_d_, ms_w_util_u_, ms_adm_gamma_dd_,
                             il, iu);

    // Lorentz factor
    for (int i=il; i<=iu; ++i)
    {
      Real const norm2_utilde = InnerProductSlicedVec3Metric(
        ms_w_util_u_, ms_adm_gamma_dd_, i
      );

      ms_W_(i) = std::sqrt(1. + norm2_utilde);
    }


    // Prepare updates terms for sources

    // S_tau ------------------------------------------------------------------
    ms_S_tau_.ZeroClear();

    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    for (int i=il; i<=iu; ++i)
    {
      ms_S_tau_(i) += ms_alpha_(i) * ms_w_hrho_(i) * (
        ms_w_util_u_(a,i) * ms_w_util_u_(b,i) * ms_adm_K_dd_(a,b,i)
      );
    }

    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      ms_S_tau_(i) -= ms_w_hrho_(i) * ms_W_(i) * (
        ms_w_util_u_(a,i) * ms_dalpha_d_(a,i)
      );
    }

    for (int i=il; i<=iu; ++i)
    {
      ms_S_tau_(i) += ms_alpha_(i) * sl_w_p(k,j,i) * ms_K_(i);
    }


    // S_d --------------------------------------------------------------------
    ms_S_S_d_.ZeroClear();

    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      ms_S_S_d_(a,i) += sl_w_p(k,j,i) * (
        ms_dalpha_d_(a,i) + ms_alpha_(i) * ms_Gamma_d_(a,i)
      );

      ms_S_S_d_(a,i) -= SQR(ms_W_(i)) * ms_w_hrho_(i) * (
        ms_dalpha_d_(a,i)
      );
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    for (int i=il; i<=iu; ++i)
    {
      ms_S_S_d_(a,i) += ms_W_(i) * ms_w_hrho_(i) * (
        ms_w_util_d_(b,i) * ms_dbeta_du_(a,b,i)
      );
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    for (int c=0; c<NDIM; ++c)
    for (int i=il; i<=iu; ++i)
    {
      ms_S_S_d_(a,i) += ms_w_hrho_(i) * ms_alpha_(i) * (
        ms_w_util_u_(b,i) * ms_w_util_d_(c,i) * ms_Gamma_udd_(c,a,b,i)
      );
    }

    // Add sources
    for (int i=il; i<=iu; ++i)
    {
      cons(IEN,k,j,i) += dt * ms_S_tau_(i) * ms_sqrt_detg_(i);
    }

    for (int c=0; c<NDIM; ++c)
    for (int i=il; i<=iu; ++i)
    {
      cons(IM1+c,k,j,i) += dt * ms_S_S_d_(c,i) * ms_sqrt_detg_(i);
    }

  }

}

#ifdef DBG_MA_SOURCES
void GRDynamical::_AddCoordTermsDivergence(
  const Real dt,
  const AthenaArray<Real> *flux,
  const AthenaArray<Real> &prim,
#if USETM
  const AthenaArray<Real> &prim_scalar,
#endif
  const AthenaArray<Real> &bb_cc,
  AthenaArray<Real> &cons)
#else
void GRDynamical::AddCoordTermsDivergence(
  const Real dt,
  const AthenaArray<Real> *flux,
  const AthenaArray<Real> &prim,
#if USETM
  const AthenaArray<Real> &prim_scalar,
#endif
  const AthenaArray<Real> &bb_cc,
  AthenaArray<Real> &cons)
#endif // DBG_MA_SOURCES
{
  // BD: TODO - cleanup (see DBG_MA_SOURCES approach)

  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;

  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;
  const int nn1 = pmb->ncells1;

  AT_N_sca alpha(nn1);              // lapse
  AT_N_sca detgamma(nn1);           // 3-metric determinant
  AT_N_sca sqrt_detgamma(nn1);
  AT_N_sca Wlor(nn1);               // lorentz factor
  AT_N_sca wtot(nn1);               // rho*h
  AT_N_sca pgas(nn1);               // pressure
  AT_N_sca rho(nn1);                // density
  AT_N_sca Stau(nn1);               // source term tau eq

  AT_N_vec dalpha_d(nn1);  // pd_i alpha
  AT_N_vec beta_u(nn1);
  AT_N_vec beta_d(nn1);
  AT_N_vec utilde_u(nn1);  // primitive gamma^i_a u^a
  AT_N_vec v_u(nn1);       // 3 velocity v^i
  AT_N_vec v_d(nn1);       // 3 vel v_i
  AT_N_vec SS_d(nn1);      // Source term for S_i eq

  AT_N_sym gamma_dd(nn1);
  AT_N_sym gamma_uu(nn1);
  AT_N_sym K_dd(nn1);

  AT_N_T2  dbeta_du(nn1);    // pd_i beta^j

  AT_N_VS2 dgamma_ddd(nn1);  // pd_i gamma_{jk}

  // ----------------------------------------------------------------------------
  AT_N_sca b0_u, bsq, u0, T00;
  AT_N_vec bb_u, bi_u, bi_d, T0i_u, T0i_d;
  AT_N_sym Tij_uu;
  // ----------------------------------------------------------------------------

  // perform variable resampling when required
  Z4c * pz4c = pmy_block->pz4c;

  AT_N_sca adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec adm_beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_sym adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym adm_K_dd(    pz4c->storage.adm, Z4c::I_ADM_Kxx);

  AA pgas_init(nn1), rho_init(nn1), w_init(nn1);

  // ----------------------------------------------------------------------------
  if (MAGNETIC_FIELDS_ENABLED)
  {
    b0_u.NewAthenaTensor(nn1);
    bb_u.NewAthenaTensor(nn1);
    bi_u.NewAthenaTensor(nn1);
    bi_d.NewAthenaTensor(nn1);
    bsq.NewAthenaTensor(nn1);
    u0.NewAthenaTensor(nn1);
    T00.NewAthenaTensor(nn1);
    T0i_u.NewAthenaTensor(nn1);
    T0i_d.NewAthenaTensor(nn1);
    Tij_uu.NewAthenaTensor(nn1);
  }
  // ----------------------------------------------------------------------------

  for (int k=ks; k<=ke; ++k)
  for (int j=js; j<=je; ++j)
  {
    GetGeometricFieldCC(gamma_dd, adm_gamma_dd, k, j);
    GetGeometricFieldCC(K_dd,     adm_K_dd,     k, j);
    GetGeometricFieldCC(alpha,    adm_alpha,    k, j);
    GetGeometricFieldCC(beta_u,   adm_beta_u,   k, j);

#if !defined(DBG_FD_CX_COORDDIV) || !defined(Z4C_CX_ENABLED)
    for (int a=0; a<NDIM; ++a)
    {
      GetGeometricFieldDerCC(dgamma_ddd, adm_gamma_dd, a, k, j);
      GetGeometricFieldDerCC(dalpha_d,   adm_alpha,    a, k, j);
      GetGeometricFieldDerCC(dbeta_du,   adm_beta_u,   a, k, j);
    }
#else
    for (int a=0; a<NDIM; ++a)
    ILOOP1(i)
    {
      dalpha_d(a,i) = fd_cx->Dx(a, adm_alpha(k,j,i));
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    ILOOP1(i)
    {
      dbeta_du(b,a,i) = fd_cx->Dx(b, adm_beta_u(a,k,j,i));
    }

    // Tensors
    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    for (int c=0; c<NDIM; ++c)
    ILOOP1(i)
    {
      dgamma_ddd(c,a,b,i) = fd_cx->Dx(c, adm_gamma_dd(a,b,k,j,i));
    }
#endif // DBG_FD_CX_COORDDIV


  	CLOOP1(i)
    {
      detgamma(i) = Det3Metric(gamma_dd, i);
      sqrt_detgamma(i) = std::sqrt(detgamma(i));

      Inv3Metric(
        1.0/detgamma(i),
        gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
        gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i),
        &gamma_uu(0,0,i), &gamma_uu(0,1,i), &gamma_uu(0,2,i),
        &gamma_uu(1,1,i), &gamma_uu(1,2,i), &gamma_uu(2,2,i));
    }

    // Read in primitives
    for (int a=0;a<NDIM;++a)
  	CLOOP1(i)
    {
	    utilde_u(a,i) = prim(a+IVX,k,j,i);
  	}

    CLOOP1(i)
    {
      pgas(i) = prim(IPR,k,j,i);
      rho(i)  = prim(IDN,k,j,i);
    }

    // Calculate enthalpy (rho*h) NB EOS specific!
    CLOOP1(i)
    {
#if USETM
      Real n = rho(i)/pmy_block->peos->GetEOS().GetBaryonMass();
      Real Y[MAX_SPECIES] = {0.0};
<<<<<<< HEAD
#if NSCALARS > 0
      for (int l=0; l<NSCALARS; ++l)
      {
        Y[l] = prim_scalar(l,k,j,i);
      }
#endif
=======
      for(int l=0; l<NSCALARS; ++l) {
        Y[l] = prim_scalar(l,k,j,i);
      }
>>>>>>> 74019e3a (Changes for Tabulated EoSs)
      Real T = pmy_block->peos->GetEOS().GetTemperatureFromP(n, pgas(i), Y);
      wtot(i) = rho(i)*pmy_block->peos->GetEOS().GetEnthalpy(n, T, Y);
#else
      const Real gamma_adi = pmy_block->peos->GetGamma();
    	wtot(i) = rho(i) + gamma_adi/(gamma_adi-1.0) * pgas(i);
#endif
    }

    // Calculate Lorentz factor
    Wlor.ZeroClear();

    for (int a=0; a<NDIM;++a)
    for (int b=0; b<NDIM;++b)
	  CLOOP1(i)
    {
	    Wlor(i) += utilde_u(a,i)*utilde_u(b,i)*gamma_dd(a,b,i);
	  }

    CLOOP1(i)
    {
    	Wlor(i) = std::sqrt(1.0+Wlor(i));
    }

    // Calculate shift beta_i
    beta_d.ZeroClear();
    for (int a=0; a<NDIM; ++a)
  	for (int b=0; b<NDIM; ++b)
    CLOOP1(i)
    {
	    beta_d(a,i) += gamma_dd(a,b,i)*beta_u(b,i);
    }

    // Calculate 3 velocity v^i
    for (int a=0; a<NDIM; ++a)
  	CLOOP1(i)
    {
	    v_u(a,i) = utilde_u(a,i)/Wlor(i);
  	}

    // Calculate 3 velocity index down v_i
    v_d.ZeroClear();
    for (int a=0; a<NDIM; ++a)
  	for (int b=0; b<NDIM; ++b)
	  CLOOP1(i)
    {
	    v_d(a,i) += v_u(b,i)*gamma_dd(a,b,i);
	  }

    if(MAGNETIC_FIELDS_ENABLED)
    {
      CLOOP1(i)
      {
        bb_u(0,i) = bb_cc(IB1,k,j,i) / sqrt_detgamma(i);
        bb_u(1,i) = bb_cc(IB2,k,j,i) / sqrt_detgamma(i);
        bb_u(2,i) = bb_cc(IB3,k,j,i) / sqrt_detgamma(i);
      }

      b0_u.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          b0_u(i) += Wlor(i)*bb_u(a,i)*v_d(a,i)/alpha(i);
        }
      }

      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          bi_u(a,i) = (bb_u(a,i) + alpha(i)*b0_u(i)*Wlor(i)*
                       (v_u(a,i) - beta_u(a,i)/alpha(i))) / Wlor(i);
        }
      }

      //  bi_d.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {

        CLOOP1(i)
        {
          bi_d(a,i) = beta_d(a,i) * b0_u(i);
          //bi_d(a,i) += bi_u(b,i)*gamma_dd(a,b,i);
        }

        for (int b=0; b<NDIM; ++b)
        {
          CLOOP1(i)
          {
            bi_d(a,i) += gamma_dd(a,b,i)*bi_u(b,i);
          }
        }
    	}

      CLOOP1(i)
      {
        bsq(i) = alpha(i)*alpha(i)*b0_u(i)*b0_u(i)/(Wlor(i)*Wlor(i));
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CLOOP1(i)
        {
          bsq(i) += bb_u(a,i)*bb_u(b,i)*gamma_dd(a,b,i)/(Wlor(i)*Wlor(i));
        }
      }

      CLOOP1(i)
      {
        u0(i) = Wlor(i)/alpha(i);
      }

      // Tab components
      CLOOP1(i)
      {
        T00(i) = ((wtot(i)+bsq(i))*u0(i)*u0(i) +
                  (pgas(i)+bsq(i)/2.0)*(-1.0/(alpha(i)*alpha(i))) -
                  b0_u(i)*b0_u(i));
      }

      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          T0i_u(a,i) = ((wtot(i)+bsq(i))*u0(i)*Wlor(i) *
                        (v_u(a,i) - beta_u(a,i)/alpha(i)) +
                        (pgas(i)+bsq(i)/2.0) *
                        (beta_u(a,i)/(alpha(i)*alpha(i))) - b0_u(i)*bi_u(a,i));
        }
      }

      T0i_d.ZeroClear();

      // BD: TODO - why is this loop written like this?
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CLOOP1(i)
        {
          T0i_d(a,i) =(((wtot(i) + bsq(i))*Wlor(i)*Wlor(i)*v_d(a,i))/alpha(i) -
                       b0_u(i)*bi_d(a,i));
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CLOOP1(i)
        {
          Tij_uu(a,b,i) = ((wtot(i)+bsq(i))*Wlor(i) *
                           (v_u(a,i) - beta_u(a,i)/alpha(i)) *
                           Wlor(i)*(v_u(b,i) - beta_u(b,i)/alpha(i)) +
                           (pgas(i)+bsq(i)/2.0) *
                           (gamma_uu(a,b,i) - beta_u(a,i)*beta_u(b,i) /
                                              (alpha(i)*alpha(i))) -
                           bi_u(a,i)*bi_u(b,i));
        }
      }

      // momentum source term
      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          SS_d(a,i) = T00(i)*(- alpha(i)*dalpha_d(a,i));
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CLOOP1(i)
        {
          SS_d(a,i) += T0i_d(b,i)*dbeta_du(a,b,i);
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      {
        CLOOP1(i)
        {
          SS_d(a,i) += (0.5*T00(i) *
                        (beta_u(b,i)*beta_u(c,i)*dgamma_ddd(a,b,c,i)) +
                        T0i_u(b,i)*beta_u(c,i)*dgamma_ddd(a,b,c,i) +
                        0.5*Tij_uu(b,c,i)*dgamma_ddd(a,b,c,i));
        }
      }

      // tau-source term
      Stau.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          Stau(i) += (T00(i)*(- beta_u(a,i)*dalpha_d(a,i)) +
                      T0i_u(a,i)*(- dalpha_d(a,i)));
        }
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      {
        CLOOP1(i)
        {
          Stau(i) += (T00(i)*(beta_u(a,i)*beta_u(b,i)*K_dd(a,b,i)) +
                      T0i_u(a,i)*(2.0*beta_u(b,i)*K_dd(a,b,i) ) +
                      Tij_uu(a,b,i)*K_dd(a,b,i));
        }
      }

    }
    else  // GRHD
    {
      // tau source term of hydro sources
      Stau.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          Stau(i) -= wtot(i)*SQR(Wlor(i))*( v_u(a,i)*dalpha_d(a,i))/alpha(i);
        }

        for (int b=0; b<NDIM; ++b)
        {
          CLOOP1(i)
          {
            Stau(i) += (wtot(i)*SQR(Wlor(i)) * v_u(a,i)*v_u(b,i) +
                        pgas(i)*gamma_uu(a,b,i))*K_dd(a,b,i);
          }
        }
      }

      // momentum source term of hydro sources
      SS_d.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          SS_d(a,i) = -(wtot(i)*SQR(Wlor(i)) - pgas(i)) * dalpha_d(a,i)/alpha(i);
        }

        for (int b=0; b<NDIM; ++b)
        {
          CLOOP1(i)
          {
            SS_d(a,i) += wtot(i) *SQR(Wlor(i))*v_d(b,i)*dbeta_du(a,b,i)/alpha(i);
          }

          for (int c=0; c<NDIM; ++c)
          {
            CLOOP1(i)
            {
              SS_d(a,i) += (0.5*(wtot(i)*SQR(Wlor(i))*v_u(b,i)*v_u(c,i) +
                                pgas(i)*gamma_uu(b,c,i))*dgamma_ddd(a,b,c,i));
            }
          }
        }
      }
    } // MAGNETIC_FIELDS_ENABLED

    if(fix_sources==1)
    {
      CLOOP1(i)
      {
        pgas_init(i) = pmy_block->phydro->w_init(IPR,k,j,i);
        rho_init(i) = pmy_block->phydro->w_init(IDN,k,j,i);
#if USETM
        Real n = rho_init(i)/pmy_block->peos->GetEOS().GetBaryonMass();
        // FIXME: Generalize to work with EOSes accepting particle fractions.
        Real Y[MAX_SPECIES] = {0.0};
<<<<<<< HEAD
        for (int l=0; l<NSCALARS; ++l)
        {
=======
        for(int l=0; l<NSCALARS; ++l) {
>>>>>>> 74019e3a (Changes for Tabulated EoSs)
          Y[l] = prim_scalar(l,k,j,i);
        }
        Real T = pmy_block->peos->GetEOS().GetTemperatureFromP(n, pgas_init(i), Y);
        w_init(i) = rho_init(i)*pmy_block->peos->GetEOS().GetEnthalpy(n, T, Y);
#else
        Real gamma_adi = pmy_block->peos->GetGamma();
    	  w_init(i) = rho_init(i) + gamma_adi/(gamma_adi-1.0) * pgas_init(i);
#endif
	      Stau(i) = 0.0;
      }

      for (int a=0; a<NDIM; ++a)
      {
        CLOOP1(i)
        {
          SS_d(a,i) = - (w_init(i) - pgas_init(i))*dalpha_d(a,i)/alpha(i);
        }
        for (int b=0; b<NDIM; ++b)
        for (int c=0; c<NDIM; ++c)
        {
          CLOOP1(i)
          {
            SS_d(a,i) += 0.5*pgas_init(i)*gamma_uu(b,c,i)*dgamma_ddd(a,b,c,i);
          }
        }

      }
    } // fix_sources

    if(zero_sources == 1)
    {
      for (int a=0; a<NDIM;++a)
      {
        CLOOP1(i)
        {
          SS_d(a,i) = 0.0 ;
        }
      }
      CLOOP1(i)
      {
        Stau(i) = 0.0;
      }
    } // zero_sources

    // Add sources
    CLOOP1(i)
    {
      const Real w_vol = dt * alpha(i) * sqrt_detgamma(i);
      cons(IEN,k,j,i) += w_vol * Stau(i);
      cons(IM1,k,j,i) += w_vol * SS_d(0,i);
      cons(IM2,k,j,i) += w_vol * SS_d(1,i);
      cons(IM3,k,j,i) += w_vol * SS_d(2,i);
    }
  } // j, k

}

//----------------------------------------------------------------------------------------
// Function for transforming primitives to locally flat frame: x1-interface
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
//   bb1: 1D array of normal components B^1 of magnetic field, in global coordinates
//   prim_l: 1D array of left primitives, using global coordinates
//   prim_r: 1D array of right primitives, using global coordinates
// Outputs:
//   prim_l: values overwritten in local coordinates
//   prim_r: values overwritten in local coordinates
//   bbx: 1D array of normal magnetic fields, in local coordinates
// Notes:
//   expects \tilde{u}^1/\tilde{u}^2/\tilde{u}^3 in IVX/IVY/IVZ slots
//   expects B^1 in bb1
//   expects B^2/B^3 in IBY/IBZ slots
//   puts \tilde{u}^x/\tilde{u}^y/\tilde{u}^z in IVX/IVY/IVZ slots
//   puts B^x in bbx
//   puts B^y/B^z in IBY/IBZ slots
//   u^\hat{i} = M^\hat{i}_j \tilde{u}^j

void GRDynamical::PrimToLocal1(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &bb1, AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
    AthenaArray<Real> &bbx) {
  // Possibly useful in the future, See GRUser for code.
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming primitives to locally flat frame: x2-interface
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
//   bb2: 1D array of normal components B^2 of magnetic field, in global coordinates
//   prim_l: 1D array of left primitives, using global coordinates
//   prim_r: 1D array of right primitives, using global coordinates
// Outputs:
//   prim_l: values overwritten in local coordinates
//   prim_r: values overwritten in local coordinates
//   bbx: 1D array of normal magnetic fields, in local coordinates
// Notes:
//   expects \tilde{u}^1/\tilde{u}^2/\tilde{u}^3 in IVX/IVY/IVZ slots
//   expects B^2 in bb2
//   expects B^3/B^1 in IBY/IBZ slots
//   puts \tilde{u}^x/\tilde{u}^y/\tilde{u}^z in IVY/IVZ/IVX slots
//   puts B^x in bbx
//   puts B^y/B^z in IBY/IBZ slots
//   u^\hat{i} = M^\hat{i}_j \tilde{u}^j

void GRDynamical::PrimToLocal2(
    const int k, const int j, const int il, const int iu,
    const AthenaArray<Real> &bb2, AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
    AthenaArray<Real> &bbx) {
  // Possibly useful in the future, See GRUser for code.
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming primitives to locally flat frame: x3-interface
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
//   bb3: 1D array of normal components B^3 of magnetic field, in global coordinates
//   prim_l: 1D array of left primitives, using global coordinates
//   prim_r: 1D array of right primitives, using global coordinates
// Outputs:
//   prim_l: values overwritten in local coordinates
//   prim_r: values overwritten in local coordinates
//   bbx: 1D array of normal magnetic fields, in local coordinates
// Notes:
//   expects \tilde{u}^1/\tilde{u}^2/\tilde{u}^3 in IVX/IVY/IVZ slots
//   expects B^3 in bb3
//   expects B^1/B^2 in IBY/IBZ slots
//   puts \tilde{u}^x/\tilde{u}^y/\tilde{u}^z in IVZ/IVX/IVY slots
//   puts B^x in bbx
//   puts B^y/B^z in IBY/IBZ slots
//   u^\hat{i} = M^\hat{i}_j \tilde{u}^j

void GRDynamical::PrimToLocal3(
			       const int k, const int j, const int il, const int iu,
			       const AthenaArray<Real> &bb3, AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r,
			       AthenaArray<Real> &bbx) {
  // Possibly useful in the future, See GRUser for code.
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming fluxes to global frame: x1-interface
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
//   cons: 1D array of conserved quantities, using local coordinates
//   bbx: 1D array of longitudinal magnetic fields, in local coordinates
//   flux: 3D array of hydrodynamical fluxes, using local coordinates
//   ey,ez: 3D arrays of magnetic fluxes (electric fields), using local coordinates
// Outputs:
//   flux: values overwritten in global coordinates
//   ey,ez: values overwritten in global coordinates
// Notes:
//   expects values and x-fluxes of Mx/My/Mz in IM1/IM2/IM3 slots
//   expects values and x-fluxes of By/Bz in IBY/IBZ slots and ey/ez
//   puts x1-fluxes of M1/M2/M3 in IM1/IM2/IM3 slots
//   puts x1-fluxes of B2/B3 in ey/ez

void GRDynamical::FluxToGlobal1(
				const int k, const int j, const int il, const int iu,
				const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx, AthenaArray<Real> &flux,
				AthenaArray<Real> &ey, AthenaArray<Real> &ez) {
  // Possibly useful in the future, See GRUser for code.
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming fluxes to global frame: x2-interface
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
//   cons: 1D array of conserved quantities, using local coordinates
//   bbx: 1D array of longitudinal magnetic fields, in local coordinates
//   flux: 3D array of hydrodynamical fluxes, using local coordinates
//   ey,ez: 3D arrays of magnetic fluxes (electric fields), using local coordinates
// Outputs:
//   flux: values overwritten in global coordinates
//   ey,ez: values overwritten in global coordinates
// Notes:
//   expects values and x-fluxes of Mx/My/Mz in IM2/IM3/IM1 slots
//   expects values and x-fluxes of By/Bz in IBY/IBZ slots and ey/ez
//   puts x2-fluxes of M1/M2/M3 in IM1/IM2/IM3 slots
//   puts x2-fluxes of B3/B1 in ey/ez

void GRDynamical::FluxToGlobal2(
				const int k, const int j, const int il, const int iu,
				const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx, AthenaArray<Real> &flux,
				AthenaArray<Real> &ey, AthenaArray<Real> &ez) {
  // Possibly useful in the future, See GRUser for code.
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming fluxes to global frame: x3-interface
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
//   cons: 1D array of conserved quantities, using local coordinates
//   bbx: 1D array of longitudinal magnetic fields, in local coordinates
//   flux: 3D array of hydrodynamical fluxes, using local coordinates
//   ey,ez: 3D arrays of magnetic fluxes (electric fields), using local coordinates
// Outputs:
//   flux: values overwritten in global coordinates
//   ey,ez: values overwritten in global coordinates
// Notes:
//   expects values and x-fluxes of Mx/My/Mz in IM3/IM1/IM2 slots
//   expects values and x-fluxes of By/Bz in IBY/IBZ slots and ey/ez
//   puts x3-fluxes of M1/M2/M3 in IM1/IM2/IM3 slots
//   puts x3-fluxes of B1/B2 in ey/ez

void GRDynamical::FluxToGlobal3(
				const int k, const int j, const int il, const int iu,
				const AthenaArray<Real> &cons, const AthenaArray<Real> &bbx, AthenaArray<Real> &flux,
				AthenaArray<Real> &ey, AthenaArray<Real> &ez) {
  // Possibly useful in the future, See GRUser for code.
  return;
}