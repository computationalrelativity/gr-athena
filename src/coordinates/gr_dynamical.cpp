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
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../parameter_input.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "coordinates.hpp"

//SB TODO this needs a cleanup
#define CLOOP1(i)				\
  _Pragma("omp simd")				\
  for (int i=is; i<=ie; ++i)

namespace {
  // Declarations
  Real Determinant(const AthenaArray<Real> &g);
  Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
		   Real a31, Real a32, Real a33);
  Real Determinant(Real a11, Real a12, Real a21, Real a22);
  void Invert4Metric(AthenaArray<Real> &ginv, AthenaArray<Real> &g);
  void CalculateTransformation(
			       const AthenaArray<Real> &g,
			       const AthenaArray<Real> &g_inv, int face, AthenaArray<Real> &transformation);
  Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i);
  void Inverse3Metric(Real const detginv,
		      Real const gxx, Real const gxy, Real const gxz,
		      Real const gyy, Real const gyz, Real const gzz,
		      Real * uxx, Real * uxy, Real * uxz,
		      Real * uyy, Real * uyz, Real * uzz);
} // namespace

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

void GRDynamical::AddCoordTermsDivergence(const Real dt, const AthenaArray<Real> *flux,
					  const AthenaArray<Real> &prim, const AthenaArray<Real> &bb_cc,
					  AthenaArray<Real> &cons) {
  
  // Extract indices
  int is = pmy_block->is;
  int ie = pmy_block->ie;
  int js = pmy_block->js;
  int je = pmy_block->je;
  int ks = pmy_block->ks;
  int ke = pmy_block->ke;
  int nn1 = pmy_block->ncells1;
  int a,b,c,d,e;
  const Real idx[3] = {
     1.0/pmy_block->pcoord->dx1v(0),
     1.0/pmy_block->pcoord->dx2v(0),
     1.0/pmy_block->pcoord->dx3v(0),
  };

  // Extract ratio of specific heats
  Real gamma_adi = pmy_block->peos->GetGamma();

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha; //lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> detg; //lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Wlor;  // lorentz factor
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> wtot;  // rho*h
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> pgas;  // pressure
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rho;   // density
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Stau;  // source term tau eq
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dalpha_d;  // pd_i alpha
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;  // beta^i
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_d;  // beta_j
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> utilde_u;  // primitive gamma^i_a u^a
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_u;  // 3 velocity v^i
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_d;  // 3 vel v_i
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> SS_d;  // Source term for S_i eq
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dbeta_du;  // pd_i beta^j
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;  //gamma_{ij}
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_uu;  // gamma^{ij}
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dgamma_ddd;  // pd_i gamma_{jk}
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;  // K_{ij}

  // bfield
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> b0_u, bsq, u0, T00 ; //lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> bb_u, bi_u, bi_d, T0i_u, T0i_d;  // 3 vel v_i
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Tij_uu;  // K_{ij}
  
  //SB TODO these need cleanup
  AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
  AthenaArray<Real> vcgamma_yz,vcgamma_zz,vcbeta_x,vcbeta_y;
  AthenaArray<Real> vcbeta_z, vcalpha;
  AthenaArray<Real> vcK_xx,vcK_xy,vcK_xz,vcK_yy;
  AthenaArray<Real> vcK_yz,vcK_zz;
  
  AthenaArray<Real> pgas_init, rho_init, w_init;
    
  pgas_init.NewAthenaArray(nn1);
  rho_init.NewAthenaArray(nn1);
  w_init.NewAthenaArray(nn1);
  alpha.NewAthenaTensor(nn1);
  detg.NewAthenaTensor(nn1);
  Wlor.NewAthenaTensor(nn1);
  wtot.NewAthenaTensor(nn1);
  pgas.NewAthenaTensor(nn1);
  rho.NewAthenaTensor(nn1);
  Stau.NewAthenaTensor(nn1);
  dalpha_d.NewAthenaTensor(nn1);
  beta_u.NewAthenaTensor(nn1);
  beta_d.NewAthenaTensor(nn1);
  utilde_u.NewAthenaTensor(nn1);
  v_u.NewAthenaTensor(nn1);
  v_d.NewAthenaTensor(nn1);
  SS_d.NewAthenaTensor(nn1);
  dbeta_du.NewAthenaTensor(nn1);
  gamma_dd.NewAthenaTensor(nn1);
  gamma_uu.NewAthenaTensor(nn1);
  dgamma_ddd.NewAthenaTensor(nn1);
  K_dd.NewAthenaTensor(nn1);

  //bfield
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
    
  vcgamma_xx.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
  vcgamma_xy.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
  vcgamma_xz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
  vcgamma_yy.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
  vcgamma_yz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
  vcgamma_zz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  vcK_xx.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_Kxx,1);
  vcK_xy.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_Kxy,1);
  vcK_xz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_Kxz,1);
  vcK_yy.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_Kyy,1);
  vcK_yz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_Kyz,1);
  vcK_zz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_Kzz,1);
  vcbeta_x.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_betax,1);
  vcbeta_y.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_betay,1);
  vcbeta_z.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_betaz,1);
  vcalpha.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_alpha,1);
  
  // Go through cells
  //SB TODO remove CLOOPS when new VC-CC logic is in place
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      
      // populate alpha, beta, gamma, K, derivatives done
#ifdef HYBRID_INTERP
      CLOOP1(i){
	gamma_dd(0,0,i) = VCInterpolation(vcgamma_xx,k,j,i);
	gamma_dd(0,1,i) = VCInterpolation(vcgamma_xy,k,j,i);
	gamma_dd(0,2,i) = VCInterpolation(vcgamma_xz,k,j,i);
	gamma_dd(1,1,i) = VCInterpolation(vcgamma_yy,k,j,i);
	gamma_dd(1,2,i) = VCInterpolation(vcgamma_yz,k,j,i);
	gamma_dd(2,2,i) = VCInterpolation(vcgamma_zz,k,j,i);
	K_dd(0,0,i) = VCInterpolation(vcK_xx,k,j,i);
	K_dd(0,1,i) = VCInterpolation(vcK_xy,k,j,i);
	K_dd(0,2,i) = VCInterpolation(vcK_xz,k,j,i);
	K_dd(1,1,i) = VCInterpolation(vcK_yy,k,j,i);
	K_dd(1,2,i) = VCInterpolation(vcK_yz,k,j,i);
	K_dd(2,2,i) = VCInterpolation(vcK_zz,k,j,i);
	alpha(i) = VCInterpolation(vcalpha,k,j,i);
	beta_u(0,i) = VCInterpolation(vcbeta_x,k,j,i);
	beta_u(1,i) = VCInterpolation(vcbeta_y,k,j,i);
	beta_u(2,i) = VCInterpolation(vcbeta_z,k,j,i);
      }
      for(a=0;a<NDIM;++a){
        CLOOP1(i){
	  dgamma_ddd(a,0,0,i) = idx[a]*VCDiff(a,vcgamma_xx,k,j,i);
	  dgamma_ddd(a,0,1,i) = idx[a]*VCDiff(a,vcgamma_xy,k,j,i);
	  dgamma_ddd(a,0,2,i) = idx[a]*VCDiff(a,vcgamma_xz,k,j,i);
	  dgamma_ddd(a,1,1,i) = idx[a]*VCDiff(a,vcgamma_yy,k,j,i);
	  dgamma_ddd(a,1,2,i) = idx[a]*VCDiff(a,vcgamma_yz,k,j,i);
	  dgamma_ddd(a,2,2,i) = idx[a]*VCDiff(a,vcgamma_zz,k,j,i);
	  dalpha_d(a,i) = idx[a]*VCDiff(a,vcalpha,k,j,i);
	  dbeta_du(a,0,i) = idx[a]*VCDiff(a,vcbeta_x,k,j,i);
	  dbeta_du(a,1,i) = idx[a]*VCDiff(a,vcbeta_y,k,j,i);
	  dbeta_du(a,2,i) = idx[a]*VCDiff(a,vcbeta_z,k,j,i);
        }
      }
      
#else
      CLOOP1(i){
	gamma_dd(0,0,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
	gamma_dd(0,1,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
	gamma_dd(0,2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
	gamma_dd(1,1,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
	gamma_dd(1,2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
	gamma_dd(2,2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));
	K_dd(0,0,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcK_xx(k,j,i));
	K_dd(0,1,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcK_xy(k,j,i));
	K_dd(0,2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcK_xz(k,j,i));
	K_dd(1,1,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcK_yy(k,j,i));
	K_dd(1,2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcK_yz(k,j,i));
	K_dd(2,2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcK_zz(k,j,i));
	alpha(i) = pmy_block->pz4c->ig->map3d_VC2CC(vcalpha(k,j,i));
	beta_u(0,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcbeta_x(k,j,i));
	beta_u(1,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcbeta_y(k,j,i));
	beta_u(2,i) = pmy_block->pz4c->ig->map3d_VC2CC(vcbeta_z(k,j,i));
      } 
      for(a=0;a<NDIM;++a){
        CLOOP1(i){
	  dgamma_ddd(a,0,0,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcgamma_xx(k,j,i));
	  dgamma_ddd(a,0,1,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcgamma_xy(k,j,i));
	  dgamma_ddd(a,0,2,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcgamma_xz(k,j,i));
	  dgamma_ddd(a,1,1,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcgamma_yy(k,j,i));
	  dgamma_ddd(a,1,2,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcgamma_yz(k,j,i));
	  dgamma_ddd(a,2,2,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcgamma_zz(k,j,i));
	  dalpha_d(a,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcalpha(k,j,i));
	  dbeta_du(a,0,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcbeta_x(k,j,i));
	  dbeta_du(a,1,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcbeta_y(k,j,i));
	  dbeta_du(a,2,i) = pmy_block->pz4c->ig->map3d_VC2CC_der(a,vcbeta_z(k,j,i));
        }
      }
#endif
      CLOOP1(i) {
	detg(i) = Det3Metric(gamma_dd, i);
	Inverse3Metric(1.0/detg(i),
		       gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
		       gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i),
		       &gamma_uu(0,0,i), &gamma_uu(0,1,i), &gamma_uu(0,2,i),
		       &gamma_uu(1,1,i), &gamma_uu(1,2,i), &gamma_uu(2,2,i));
      }

      // Read in primitives
      for(a=0;a<NDIM;++a){
	CLOOP1(i){
	  utilde_u(a,i) = prim(a+IVX,k,j,i);
	}
      }
      
      CLOOP1(i){
	pgas(i) = prim(IPR,k,j,i);
	rho(i)  = prim(IDN,k,j,i);
      }
      
      // Calculate enthalpy (rho*h) NB EOS specific!
      CLOOP1(i){
#if USETM
	Real n = rho(i)/pmy_block->peos->GetEOS().GetBaryonMass();
	Real Y[MAX_SPECIES] = {0.0};
	Real T = pmy_block->peos->GetEOS().GetTemperatureFromP(n, pgas(i), Y);
	wtot(i) = n*pmy_block->peos->GetEOS().GetEnthalpy(n, T, Y);
#else
	wtot(i) = rho(i) + gamma_adi/(gamma_adi-1.0) * pgas(i);
#endif
      }

      // Calculate Lorentz factor
      Wlor.ZeroClear();
      for(a=0;a<NDIM;++a){
	for(b=0;b<NDIM;++b){
	  CLOOP1(i){
	    Wlor(i) += utilde_u(a,i)*utilde_u(b,i)*gamma_dd(a,b,i);
	  }
	}
      }
      CLOOP1(i){
	Wlor(i) = std::sqrt(1.0+Wlor(i));
      }

      // Calculate shift beta_i
      beta_d.ZeroClear();
      for(a=0;a<NDIM;++a){
	for(b=0;b<NDIM;++b){
          CLOOP1(i){
	    beta_d(a,i) += gamma_dd(a,b,i)*beta_u(b,i);
          }
	}
      }

      // Calculate 3 velocity v^i
      for(a=0;a<NDIM;++a){
	CLOOP1(i){
	  v_u(a,i) = utilde_u(a,i)/Wlor(i);
	}
      }
      
      // Calculate 3 velocity index down v_i
      v_d.ZeroClear();
      for(a=0;a<NDIM;++a){
	for(b=0;b<NDIM;++b){
	  CLOOP1(i){
	    v_d(a,i) += v_u(b,i)*gamma_dd(a,b,i);
	  }
	}
      }
      
      // tau source term of hydro sources
      Stau.ZeroClear();
      for(a=0;a<NDIM; ++a){ 
	CLOOP1(i){
	  Stau(i) -= wtot(i)*SQR(Wlor(i))*( v_u(a,i)*dalpha_d(a,i))/alpha(i) ;
	}
	for(b = 0; b<NDIM; ++b){ 
	  CLOOP1(i){
	    Stau(i) += (wtot(i)*SQR(Wlor(i)) * v_u(a,i)*v_u(b,i) + pgas(i)*gamma_uu(a,b,i))*K_dd(a,b,i) ;
	  }
	}
      }

      // momentum source term of hydro sources      
      SS_d.ZeroClear();
      for(a = 0; a<NDIM; ++a){  
	CLOOP1(i){
	  SS_d(a,i) = - (wtot(i)*SQR(Wlor(i)) - pgas(i))  * dalpha_d(a,i)/alpha(i);
	}
	for(b = 0; b<NDIM; ++b){  
	  CLOOP1(i){
	    SS_d(a,i) += wtot(i) *SQR(Wlor(i))*v_d(b,i)*dbeta_du(a,b,i)/alpha(i);
	  }
	  for(c = 0; c<NDIM; ++c){  
	    CLOOP1(i){
	      SS_d(a,i) +=  0.5*(wtot(i)*SQR(Wlor(i))*v_u(b,i)*v_u(c,i) + pgas(i)*gamma_uu(b,c,i))*dgamma_ddd(a,b,c,i) ;
	    }
	  }
	}
      }
      
      if(MAGNETIC_FIELDS_ENABLED){
	
	CLOOP1(i){
	  bb_u(0,i) = bb_cc(IB1,k,j,i)/std::sqrt(detg(i)); 
	  bb_u(1,i) = bb_cc(IB2,k,j,i)/std::sqrt(detg(i)); 
	  bb_u(2,i) = bb_cc(IB3,k,j,i)/std::sqrt(detg(i)); 
	}
	
	b0_u.ZeroClear();
	for(a=0;a<NDIM;++a){
	  CLOOP1(i){
	    b0_u(i) += Wlor(i)*bb_u(a,i)*v_d(a,i)/alpha(i);
	  }
	}
	for(a=0;a<NDIM;++a){
	  CLOOP1(i){
	    bi_u(a,i) = (bb_u(a,i) + alpha(i)*b0_u(i)*Wlor(i)*(v_u(a,i) - beta_u(a,i)/alpha(i)))/Wlor(i);
	  }
	}
	
	//  bi_d.ZeroClear();
	for(a=0;a<NDIM;++a){
	  CLOOP1(i){
	    bi_d(a,i) = beta_d(a,i) * b0_u(i);
	    //bi_d(a,i) += bi_u(b,i)*gamma_dd(a,b,i);
	  }
	  for(b=0;b<NDIM;++b){
	    CLOOP1(i){
	      bi_d(a,i) += gamma_dd(a,b,i)*bi_u(b,i);
	    }
	  }
	}
	CLOOP1(i){
	  bsq(i) = alpha(i)*alpha(i)*b0_u(i)*b0_u(i)/(Wlor(i)*Wlor(i));
	}
	
	for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
	    CLOOP1(i){
	      bsq(i) += bb_u(a,i)*bb_u(b,i)*gamma_dd(a,b,i)/(Wlor(i)*Wlor(i));
	    }
	  }
	}
	CLOOP1(i){
	  u0(i) = Wlor(i)/alpha(i);
	}    

	// Tab components
	CLOOP1(i){
	  T00(i) = (wtot(i)+bsq(i))*u0(i)*u0(i) + (pgas(i)+bsq(i)/2.0)*(-1.0/(alpha(i)*alpha(i))) - b0_u(i)*b0_u(i);
	}
	
	for(a = 0; a<NDIM; ++a){  
	  CLOOP1(i){
	    T0i_u(a,i) = (wtot(i)+bsq(i))*u0(i)*Wlor(i)*(v_u(a,i) - beta_u(a,i)/alpha(i)) + (pgas(i)+bsq(i)/2.0)*(beta_u(a,i)/(alpha(i)*alpha(i))) - b0_u(i)*bi_u(a,i);
	  }
	}
	
	T0i_d.ZeroClear();
	for(a = 0; a<NDIM; ++a){  
	  for(b = 0; b<NDIM; ++b){  
	    CLOOP1(i){
	      T0i_d(a,i) =( (wtot(i) + bsq(i))*Wlor(i)*Wlor(i)*v_d(a,i))/alpha(i) - b0_u(i)*bi_d(a,i);
	    }
	  }
	}
	
	for(a = 0; a<NDIM; ++a){  
	  for(b = 0; b<NDIM; ++b){  
	    CLOOP1(i){
	      Tij_uu(a,b,i) = (wtot(i)+bsq(i))*Wlor(i)*(v_u(a,i) - beta_u(a,i)/alpha(i))*Wlor(i)*(v_u(b,i) - beta_u(b,i)/alpha(i)) + (pgas(i)+bsq(i)/2.0)*(gamma_uu(a,b,i) - beta_u(a,i)*beta_u(b,i)/(alpha(i)*alpha(i))) - bi_u(a,i)*bi_u(b,i);
	    }
	  }
	}

	// momentum source term
	for(a = 0; a<NDIM; ++a){  
	  CLOOP1(i){
	    SS_d(a,i) = T00(i)*(- alpha(i)*dalpha_d(a,i));
	  }
	}
	
	for(a = 0; a<NDIM; ++a){  
	  for(b = 0; b<NDIM; ++b){  
	    CLOOP1(i){
	      SS_d(a,i) += T0i_d(b,i)*dbeta_du(a,b,i);
	    }
	  }
	}
	for(a = 0; a<NDIM; ++a){  
	  for(b = 0; b<NDIM; ++b){  
	    for(c = 0; c<NDIM; ++c){  
	      CLOOP1(i){
		SS_d(a,i) += 0.5*T00(i)*(beta_u(b,i)*beta_u(c,i)*dgamma_ddd(a,b,c,i)) + T0i_u(b,i)*beta_u(c,i)*dgamma_ddd(a,b,c,i) + 0.5*Tij_uu(b,c,i)*dgamma_ddd(a,b,c,i);
	      }
	    }
	  }
	}

	// tau-source term
	Stau.ZeroClear();
	for(a = 0; a<NDIM; ++a){  
	  CLOOP1(i){
	    Stau(i) += T00(i)*(- beta_u(a,i)*dalpha_d(a,i)) + T0i_u(a,i)*(- dalpha_d(a,i));
	  }
	}
	
	for(a = 0; a<NDIM; ++a){  
	  for(b = 0; b<NDIM; ++b){  
	    CLOOP1(i){
	      Stau(i) += T00(i)*(beta_u(a,i)*beta_u(b,i)*K_dd(a,b,i))  + T0i_u(a,i)*(2.0*beta_u(b,i)*K_dd(a,b,i) ) + Tij_uu(a,b,i)*K_dd(a,b,i);
	    }
	  }
	}
	
      } // MAGNETIC_FIELDS_ENABLED
      
      if(fix_sources==1){
	CLOOP1(i){
	  pgas_init(i) = pmy_block->phydro->w_init(IPR,k,j,i); 
	  rho_init(i) = pmy_block->phydro->w_init(IDN,k,j,i); 
#if USETM
	  Real n = rho_init(i)/pmy_block->peos->GetEOS().GetBaryonMass();
	  // FIXME: Generalize to work with EOSes accepting particle fractions.
	  Real Y[MAX_SPECIES] = {0.0};
	  Real T = pmy_block->peos->GetEOS().GetTemperatureFromP(n, pgas_init(i), Y);
	  w_init(i) = n*pmy_block->peos->GetEOS().GetEnthalpy(n, T, Y);
#else
	  w_init(i) = rho_init(i) + gamma_adi/(gamma_adi-1.0) * pgas_init(i);
#endif
	  Stau(i) = 0.0;
        }
        for(a=0;a<NDIM;++a){
	  CLOOP1(i){
	    SS_d(a,i) = - (w_init(i) - pgas_init(i))*dalpha_d(a,i)/alpha(i)  ;
	  }
	  for(b=0;b<NDIM;++b){
	    for(c=0;c<NDIM;++c){
	      CLOOP1(i){
		SS_d(a,i) += 0.5*pgas_init(i)*gamma_uu(b,c,i)*dgamma_ddd(a,b,c,i);
	      }
	    }
	  }
        }
      } // fix_sources
      
      if(zero_sources == 1){
	for(a=0;a<NDIM;++a){
	  CLOOP1(i){
	    SS_d(a,i) = 0.0 ;
	  }
	  
	}
	CLOOP1(i){
	  Stau(i) = 0.0;
	}
      } // zero_sources

      // Add sources
      CLOOP1(i){
        cons(IEN,k,j,i) += dt * Stau(i)  *alpha(i)*std::sqrt(detg(i));
        cons(IM1,k,j,i) += dt * SS_d(0,i)*alpha(i)*std::sqrt(detg(i));
        cons(IM2,k,j,i) += dt * SS_d(1,i)*alpha(i)*std::sqrt(detg(i));
        cons(IM3,k,j,i) += dt * SS_d(2,i)*alpha(i)*std::sqrt(detg(i));
      }
      
    } // j
  }// k
  
  // cleanup 1d buffers
  pgas_init.DeleteAthenaArray();
  rho_init.DeleteAthenaArray();
  w_init.DeleteAthenaArray();
  alpha.DeleteAthenaTensor();
  detg.DeleteAthenaTensor();
  Wlor.DeleteAthenaTensor();
  wtot.DeleteAthenaTensor();
  pgas.DeleteAthenaTensor();
  rho.DeleteAthenaTensor();
  Stau.DeleteAthenaTensor();
  dalpha_d.DeleteAthenaTensor();
  beta_u.DeleteAthenaTensor();
  beta_d.DeleteAthenaTensor();
  utilde_u.DeleteAthenaTensor();
  v_u.DeleteAthenaTensor();
  v_d.DeleteAthenaTensor();
  SS_d.DeleteAthenaTensor();
  dbeta_du.DeleteAthenaTensor();
  gamma_dd.DeleteAthenaTensor();
  gamma_uu.DeleteAthenaTensor();
  dgamma_ddd.DeleteAthenaTensor();
  K_dd.DeleteAthenaTensor();

  b0_u.DeleteAthenaTensor();
  bb_u.DeleteAthenaTensor();
  bi_u.DeleteAthenaTensor();
  bi_d.DeleteAthenaTensor();
  bsq.DeleteAthenaTensor();
  u0.DeleteAthenaTensor();
  T00.DeleteAthenaTensor();
  T0i_u.DeleteAthenaTensor();
  T0i_d.DeleteAthenaTensor();
  Tij_uu.DeleteAthenaTensor();

  
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing cell-centered and face-centered metric coefficients
// Inputs:
//   k,j: x3- and x2-indices
//   il,iu: x1-index bounds
// Outputs:
//   g: array of metric components in 1D
//   g_inv: array of inverse metric components in 1D

void GRDynamical::CellMetric(const int k, const int j, const int il, const int iu,
                        AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  //removed metric cell
  /*
    for (int n = 0; n < NMETRIC; ++n) {
    #pragma omp simd
    for (int i=il; i<=iu; ++i) {
    g(n,i) = metric_cell_kji_(0,n,k,j,i);
    g_inv(n,i) = metric_cell_kji_(1,n,k,j,i);
    }
    }
  */
  return;
}

void GRDynamical::Face1Metric(const int k, const int j, const int il, const int iu,
			      AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  //Removed metric_face?_kji_
  /*
    for (int n = 0; n < NMETRIC; ++n) {
    #pragma omp simd
    for (int i=il; i<=iu; ++i) {
    g(n,i) = metric_face1_kji_(0,n,k,j,i);
    g_inv(n,i) = metric_face1_kji_(1,n,k,j,i);
    }
    }
  */
  return;
}

void GRDynamical::Face2Metric(const int k, const int j, const int il, const int iu,
			      AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  /*
    for (int n = 0; n < NMETRIC; ++n) {
    #pragma omp simd
    for (int i=il; i<=iu; ++i) {
    g(n,i) = metric_face2_kji_(0,n,k,j,i);
    g_inv(n,i) = metric_face2_kji_(1,n,k,j,i);
    }
    }
  */
  return;
}

void GRDynamical::Face3Metric(const int k, const int j, const int il, const int iu,
			      AthenaArray<Real> &g, AthenaArray<Real> &g_inv) {
  /*
    for (int n = 0; n < NMETRIC; ++n) {
    #pragma omp simd
    for (int i=il; i<=iu; ++i) {
    g(n,i) = metric_face3_kji_(0,n,k,j,i);
    g_inv(n,i) = metric_face3_kji_(1,n,k,j,i);
    }
    }
  */
  return;
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

//----------------------------------------------------------------------------------------
// Function for raising covariant components of a vector
// Inputs:
//   a_0,a_1,a_2,a_3: covariant components of vector
//   k,j,i: indices of cell in which transformation is desired
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to contravariant 4-vector components

void GRDynamical::RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j, int i,
				  Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  //SB FIXME needed? 
  return;
}

//----------------------------------------------------------------------------------------
// Function for lowering contravariant components of a vector
// Inputs:
//   a0,a1,a2,a3: contravariant components of vector
//   k,j,i: indices of cell in which transformation is desired
// Outputs:
//   pa_0,pa_1,pa_2,pa_3: pointers to covariant 4-vector components

void GRDynamical::LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
				  Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) {
  //SB FIXME needed? 
  return;
}

//----------------------------------------------------------------------------------------
// Function for updating metric values at beginning of each matter timestep
// 

void GRDynamical::UpdateMetric(){

  //SB FIXME needed? (used in pgen)
  
  // Allocate scratch arrays
  AthenaArray<Real> g, g_inv, dg_dx1, dg_dx2, dg_dx3, transformation, K;
  g.NewAthenaArray(NMETRIC);
  K.NewAthenaArray(NSPMETRIC);
  g_inv.NewAthenaArray(NMETRIC);
  dg_dx1.NewAthenaArray(NMETRIC);
  dg_dx2.NewAthenaArray(NMETRIC);
  dg_dx3.NewAthenaArray(NMETRIC);
  if (!coarse_flag) {
    transformation.NewAthenaArray(2, NTRIANGULAR);
  }
  
  RegionSize& block_size = pmy_block->block_size;
  
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
  
  return;
}

//----------------------------------------------------------------------------------------
// 

void GRDynamical::GetDerivs(
  int i, int j, int k,
  AthenaArray<Real>& dg_dx1, AthenaArray<Real>& dg_dx2, AthenaArray<Real>& dg_dx3)
{
  //SB FIXME needed?
}

//----------------------------------------------------------------------------------------
// Function for getting dynamical extrinsic curvature from vertex
// centred ADM variables and interpolating to cell centred ADM variables
void GRDynamical::GetExCurv(int k, int j, int i, AthenaArray<Real>& K){
  //SB FIXME needed?
  //average corner values to get cell centre value
  AthenaArray<Real> src;
  for(int n=Z4c::I_ADM_Kxx; n<Z4c::I_ADM_Kzz+1; ++n){
    src.InitWithShallowSlice(pmy_block->pz4c->storage.adm,n,1);
    K(n-Z4c::I_ADM_Kxx) = 0.125*(src(k,j,i) + src(k,j,i+1) + src(k,j+1,i) + src(k+1,j,i) + src(k,j+1,i+1) + src(k+1,j,i+1) + src(k+1,j+1,i) + src(k+1,j+1,i+1));;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for getting dynamical metric from vertex centred ADM
// variables and interpolating to cell centred ADM variables 
void GRDynamical::GetMetric(int k, int j, int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv){
  Real beta2;
  AthenaArray<Real> src,  chi_coarse, z4cgdd_coarse ;
  // interpolate from regular or coarse representation of z4c vars
  
  for(int n=Z4c::I_Z4c_alpha; n<Z4c::I_Z4c_betaz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.u,n,1);
    }
    else{
      src.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,n,1);
    }
    // lapse, shift from averaging
    g(n-Z4c::I_Z4c_alpha) = 0.125*(src(k,j,i) + src(k,j,i+1) + src(k,j+1,i) + src(k+1,j,i) + src(k,j+1,i+1) + src(k+1,j,i+1) + src(k+1,j+1,i) + src(k+1,j+1,i+1));
  }
  //ADM metric
  for(int n=Z4c::I_ADM_gxx; n<Z4c::I_ADM_gzz+1; ++n){
    // if not coarse use adm variables stored in z4c.storage
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.adm,n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.125*( src(k,j,i) + src(k+1,j,i) + src(k,j+1,i) + src(k,j,i+1) + src(k+1,j+1,i) + src(k+1,j,i+1) + src(k,j+1,i+1) + src(k+1,j+1,i+1));
    }
    else{
      // no adm variables stored in coarse representation - calculate ADM metric from z4c.chi and z4c.gdd in coarse representation 
      chi_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_chi,1);
      z4cgdd_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_gxx-Z4c::I_ADM_gxx+n,1);
      
      g(n-Z4c::I_ADM_gxx+4) = 0.125*(pow(chi_coarse(k,j,i),4./chi_psi_power)*z4cgdd_coarse(k,j,i) + pow(chi_coarse(k,j,i+1),4./chi_psi_power)*z4cgdd_coarse(k,j,i+1) + pow(chi_coarse(k,j+1,i),4./chi_psi_power)*z4cgdd_coarse(k,j+1,i) + pow(chi_coarse(k+1,j,i),4./chi_psi_power)*z4cgdd_coarse(k+1,j,i) + pow(chi_coarse(k,j+1,i+1),4./chi_psi_power)*z4cgdd_coarse(k,j+1,i+1) + pow(chi_coarse(k+1,j,i+1),4./chi_psi_power)*z4cgdd_coarse(k+1,j,i+1) + pow(chi_coarse(k+1,j+1,i),4./chi_psi_power)*z4cgdd_coarse(k+1,j+1,i) + pow(chi_coarse(k+1,j+1,i+1),4./chi_psi_power)*z4cgdd_coarse(k+1,j+1,i+1));
    }
  }
  //Construct inverse metric of cell centred metric
  //g_00 currently contains lapse, update to -alp**2 +beta^i beta_i
  // g(0I) currently contains beta^i. Want beta_i
  // beta_i = 
  Real beta1_d = g(I01)*g(I11) + g(I02)*g(I12) + g(I03)*g(I13);
  Real beta2_d = g(I01)*g(I12) + g(I02)*g(I22) + g(I03)*g(I23);
  Real beta3_d = g(I01)*g(I13) + g(I02)*g(I23) + g(I03)*g(I33);
  
  beta2 = g(I01)*g(I01)*g(I11) + g(I02)*g(I02)*g(I22) + g(I03)*g(I03) * g(I33);
  
  g(I01) = beta1_d;
  g(I02) = beta2_d;
  g(I03) = beta3_d;
  
  g(I00) = -g(I00)*g(I00) + beta2;
  
  //Invert 4 metric
  Invert4Metric(g_inv,g);
  
  return;
}

//----------------------------------------------------------------------------------------
// Same as GetMetric except average 4 corners of a face for face centred metric
// Construct vertex centred 4-metric from lapse, shift and ADM 3-metric
void GRDynamical::GetFace1Metric(int k, int j, int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv){
  Real beta2;
  AthenaArray<Real> src,  chi_coarse, z4cgdd_coarse ;
  for(int n=Z4c::I_Z4c_alpha; n<Z4c::I_Z4c_betaz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.u,n,1);
    }
    else{
      src.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,n,1);
	}
    g(n-Z4c::I_Z4c_alpha) = 0.25*(src(k,j,i) + src(k,j+1,i) + src(k+1,j,i) + src(k+1,j+1,i));
  }
  for(int n=Z4c::I_ADM_gxx; n<Z4c::I_ADM_gzz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.adm,n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.25*( src(k,j,i) + src(k+1,j,i) + src(k,j+1,i) + src(k+1,j+1,i));
    }
    else{
      chi_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_chi,1);
      z4cgdd_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_gxx-Z4c::I_ADM_gxx+n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.25*(pow(chi_coarse(k,j,i),4./chi_psi_power)*z4cgdd_coarse(k,j,i) + pow(chi_coarse(k,j+1,i),4./chi_psi_power)*z4cgdd_coarse(k,j+1,i) + pow(chi_coarse(k+1,j,i),4./chi_psi_power)*z4cgdd_coarse(k+1,j,i) + pow(chi_coarse(k+1,j+1,i),4./chi_psi_power)*z4cgdd_coarse(k+1,j+1,i) );
    }
  }
  // Construct inverse metric of cell centred metric
  // g_00 currently contains lapse, update to -alp**2 +beta^i beta_i
  beta2 = g(I01)*g(I01)*g(I11) + g(I02)*g(I02)*g(I22) + g(I03)*g(I03) * g(I33);
  g(I00) = -g(I00)*g(I00) + beta2;  
  // Invert 4 metric
  Invert4Metric(g_inv,g);  
  return;
}

//----------------------------------------------------------------------------------------
void GRDynamical::GetFace2Metric(int k, int j, int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv){
  Real beta2;
  AthenaArray<Real> src, chi_coarse, z4cgdd_coarse ;
  for(int n=Z4c::I_Z4c_alpha; n<Z4c::I_Z4c_betaz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.u,n,1);
    }
    else{
      src.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,n,1);
    }
    g(n-Z4c::I_Z4c_alpha) = 0.25*(src(k,j,i) + src(k,j,i+1) + src(k+1,j,i) + src(k+1,j,i+1));
  }
  for(int n=Z4c::I_ADM_gxx; n<Z4c::I_ADM_gzz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.adm,n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.25*( src(k,j,i) + src(k+1,j,i) + src(k,j,i+1) + src(k+1,j,i+1));
    }
    else{
      chi_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_chi,1);
      z4cgdd_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_gxx-Z4c::I_ADM_gxx+n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.25*(pow(chi_coarse(k,j,i),4./chi_psi_power)*z4cgdd_coarse(k,j,i) + pow(chi_coarse(k,j,i+1),4./chi_psi_power)*z4cgdd_coarse(k,j,i+1) + pow(chi_coarse(k+1,j,i),4./chi_psi_power)*z4cgdd_coarse(k+1,j,i) + pow(chi_coarse(k+1,j,i+1),4./chi_psi_power)*z4cgdd_coarse(k+1,j,i+1) );
    }
  }
  //Construct inverse metric of cell centred metric
  //g_00 currently contains lapse, update to -alp**2 +beta^i beta_i
  beta2 = g(I01)*g(I01)*g(I11) + g(I02)*g(I02)*g(I22) + g(I03)*g(I03) * g(I33);
  g(I00) = -g(I00)*g(I00) + beta2;

  //Invert 4 metric
  Invert4Metric(g_inv,g);

  return;
}

//----------------------------------------------------------------------------------------
void GRDynamical::GetFace3Metric(int k, int j, int i, AthenaArray<Real>& g, AthenaArray<Real>& g_inv){
  Real beta2;
  AthenaArray<Real> src, chi_coarse, z4cgdd_coarse ;
  for(int n=Z4c::I_Z4c_alpha; n<Z4c::I_Z4c_betaz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.u,n,1);
    }
    else{
      src.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,n,1);
    }
    g(n-Z4c::I_Z4c_alpha) = 0.25*(src(k,j,i) + src(k,j+1,i) + src(k,j,i+1) + src(k,j+1,i+1));
  }
  for(int n=Z4c::I_ADM_gxx; n<Z4c::I_ADM_gzz+1; ++n){
    if(!coarse_flag){
      src.InitWithShallowSlice(pmy_block->pz4c->storage.adm,n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.25*( src(k,j,i) + src(k,j,i+1) + src(k,j+1,i) + src(k,j+1,i+1));
    }
    else{
      chi_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_chi,1);
      z4cgdd_coarse.InitWithShallowSlice(pmy_block->pz4c->coarse_u_,Z4c::I_Z4c_gxx-Z4c::I_ADM_gxx+n,1);
      g(n-Z4c::I_ADM_gxx+4) = 0.25*(pow(chi_coarse(k,j,i),4./chi_psi_power)*z4cgdd_coarse(k,j,i) + pow(chi_coarse(k,j+1,i),4./chi_psi_power)*z4cgdd_coarse(k,j+1,i) + pow(chi_coarse(k,j,i+1),4./chi_psi_power)*z4cgdd_coarse(k,j,i+1) + pow(chi_coarse(k,j+1,i+1),4./chi_psi_power)*z4cgdd_coarse(k,j+1,i+1) );
    }
  }
  //Construct inverse metric of cell centred metric
  //g_00 currently contains lapse, update to -alp**2 +beta^i beta_i
  beta2 = g(I01)*g(I01)*g(I11) + g(I02)*g(I02)*g(I22) + g(I03)*g(I03) * g(I33);
  g(I00) = -g(I00)*g(I00) + beta2;

  //Invert 4 metric
  Invert4Metric(g_inv,g);

  return;
}

namespace {
  //----------------------------------------------------------------------------------------
  // Functions for calculating determinant
  // Inputs:
  //   g: array of covariant metric coefficients
  //   a11,a12,a13,a21,a22,a23,a31,a32,a33: elements of matrix
  //   a11,a12,a21,a22: elements of matrix
  // Outputs:
  //   returned value: determinant

  Real Determinant(const AthenaArray<Real> &g) {
    const Real &a11 = g(I00);
    const Real &a12 = g(I01);
    const Real &a13 = g(I02);
    const Real &a14 = g(I03);
    const Real &a21 = g(I01);
    const Real &a22 = g(I11);
    const Real &a23 = g(I12);
    const Real &a24 = g(I13);
    const Real &a31 = g(I02);
    const Real &a32 = g(I12);
    const Real &a33 = g(I22);
    const Real &a34 = g(I23);
    const Real &a41 = g(I03);
    const Real &a42 = g(I13);
    const Real &a43 = g(I23);
    const Real &a44 = g(I33);
    Real det = a11 * Determinant(a22, a23, a24, a32, a33, a34, a42, a43, a44)
      - a12 * Determinant(a21, a23, a24, a31, a33, a34, a41, a43, a44)
      + a13 * Determinant(a21, a22, a24, a31, a32, a34, a41, a42, a44)
      - a14 * Determinant(a21, a22, a23, a31, a32, a33, a41, a42, a43);
    return det;
  }

  Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
		   Real a31, Real a32, Real a33) {
    Real det = a11 * Determinant(a22, a23, a32, a33)
      - a12 * Determinant(a21, a23, a31, a33)
      + a13 * Determinant(a21, a22, a31, a32);
    return det;
  }

  Real Determinant(Real a11, Real a12, Real a21, Real a22) {
    return a11 * a22 - a12 * a21;
  }

  void Invert4Metric(AthenaArray<Real> &ginv, AthenaArray<Real> &g){
    const Real &a11 = g(I00);
    const Real &a12 = g(I01);
    const Real &a13 = g(I02);
    const Real &a14 = g(I03);
    const Real &a21 = g(I01);
    const Real &a22 = g(I11);
    const Real &a23 = g(I12);
    const Real &a24 = g(I13);
    const Real &a31 = g(I02);
    const Real &a32 = g(I12);
    const Real &a33 = g(I22);
    const Real &a34 = g(I23);
    const Real &a41 = g(I03);
    const Real &a42 = g(I13);
    const Real &a43 = g(I23);
    const Real &a44 = g(I33);

    Real detg = Determinant(g);
    ginv(I00) = Determinant(a22,a23,a24,a32,a33,a34,a42,a43,a44)/detg;
    ginv(I01) = -1.0*Determinant(a12,a13,a14,a32,a33,a34,a42,a43,a44)/detg;
    ginv(I02) = Determinant(a12,a13,a14,a22,a23,a24,a42,a43,a44)/detg;
    ginv(I03) = -1.0*Determinant(a12,a13,a14,a22,a23,a24,a32,a33,a34)/detg;
    ginv(I11) = Determinant(a11,a13,a14,a31,a33,a34,a41,a43,a44)/detg;
    ginv(I12) = -1.0*Determinant(a11,a13,a14,a21,a23,a24,a41,a43,a44)/detg;
    ginv(I13) = Determinant(a11,a13,a14,a21,a23,a24,a31,a33,a34)/detg;
    ginv(I22) = Determinant(a11,a12,a14,a21,a22,a24,a41,a42,a44)/detg;
    ginv(I23) = -1.0*Determinant(a11,a12,a14,a21,a22,a24,a31,a32,a34)/detg;
    ginv(I33) = Determinant(a11,a12,a13,a21,a22,a23,a31,a32,a33)/detg;
  }

  //----------------------------------------------------------------------------------------
  // Function for calculating frame transformation coefficients
  // Inputs:
  //   g,g_inv: arrays of covariant and contravariant metric coefficients
  //   face: 1, 2, or 3 depending on which face is considered
  // Outputs:
  //   transformation: array of transformation coefficients

  void CalculateTransformation(
			       const AthenaArray<Real> &g,
			       const AthenaArray<Real> &g_inv, int face, AthenaArray<Real> &transformation) {
    // Prepare indices
    int index[4][4];
    index[0][0] = I00;
    index[0][1] = I01; index[0][2] = I02; index[0][3] = I03;
    index[1][1] = I11; index[1][2] = I12; index[1][3] = I13;
    index[2][1] = I12; index[2][2] = I22; index[2][3] = I23;
    index[3][1] = I13; index[3][2] = I23; index[3][3] = I33;

    // Shift indices according to face
    int i0 = 0;
    int i1 = face;
    int i2 = 1 + face%3;
    int i3 = 1 + (face+1)%3;

    // Extract metric coefficients
    const Real &g_22 = g(index[i2][i2]);
    const Real &g_23 = g(index[i2][i3]);
    const Real &g_33 = g(index[i3][i3]);
    const Real &g00 = g_inv(index[i0][i0]);
    const Real &g01 = g_inv(index[i0][i1]);
    const Real &g02 = g_inv(index[i0][i2]);
    const Real &g03 = g_inv(index[i0][i3]);
    const Real &g11 = g_inv(index[i1][i1]);
    const Real &g12 = g_inv(index[i1][i2]);
    const Real &g13 = g_inv(index[i1][i3]);

    // Calculate intermediate quantities
    Real aa = -1.0 / std::sqrt(-g00);
    Real bb = 1.0 / std::sqrt(g00 * (g00 * g11 - SQR(g01)));
    Real cc = 1.0 / std::sqrt(g_33);
    Real dd = 1.0 / std::sqrt(g_33 * (g_22 * g_33 - SQR(g_23)));
    Real ee = g01 * g12 - g11 * g02;
    Real ff = g01 * g02 - g00 * g12;
    Real gg = g01 * g13 - g11 * g03;
    Real hh = g01 * g03 - g00 * g13;
    Real ii = SQR(bb)/cc * g00 * (gg + ee * g_23/g_33);
    Real jj = SQR(bb)/cc * g00 * (hh + ff * g_23/g_33);

    // Set local-to-global transformation coefficients
    transformation(0,T00) = aa * g00;
    transformation(0,T10) = aa * g01;
    transformation(0,T20) = aa * g02;
    transformation(0,T30) = aa * g03;
    transformation(0,T11) = bb * (g01 * g01 - g00 * g11);
    transformation(0,T21) = bb * (g01 * g02 - g00 * g12);
    transformation(0,T31) = bb * (g01 * g03 - g00 * g13);
    transformation(0,T22) = dd * g_33;
    transformation(0,T32) = -dd * g_23;
    transformation(0,T33) = cc;

    // Set global-to-local transformation coefficients
    transformation(1,T00) = -aa;
    transformation(1,T10) = bb * g01;
    transformation(1,T11) = -bb * g00;
    transformation(1,T20) = SQR(bb)*ee/dd * g00/g_33;
    transformation(1,T21) = SQR(bb)*ff/dd * g00/g_33;
    transformation(1,T22) = 1.0 / (dd * g_33);
    transformation(1,T30) = ii;
    transformation(1,T31) = jj;
    transformation(1,T32) = 1.0/cc * g_23/g_33;
    transformation(1,T33) = 1.0/cc;
    return;
  }

  Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i)
  {
    return - SQR(gamma(0,2,i))*gamma(1,1,i) + 
      2*gamma(0,1,i)*gamma(0,2,i)*gamma(1,2,i) - 
      gamma(0,0,i)*SQR(gamma(1,2,i)) - SQR(gamma(0,1,i))*gamma(2,2,i) + 
      gamma(0,0,i)*gamma(1,1,i)*gamma(2,2,i);
  }

  void Inverse3Metric(Real const detginv,
		      Real const gxx, Real const gxy, Real const gxz,
		      Real const gyy, Real const gyz, Real const gzz,
		      Real * uxx, Real * uxy, Real * uxz,
		      Real * uyy, Real * uyz, Real * uzz)
  {
    *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
    *uxy = (gxz*gyz  - gxy*gzz)*detginv;
    *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
    *uxz = (-gxz*gyy + gxy*gyz)*detginv;
    *uyz = (gxy*gxz  - gxx*gyz)*detginv;
    *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
    return;
  }

} // namespace
