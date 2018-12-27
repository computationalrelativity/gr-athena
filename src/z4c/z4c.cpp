//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c.cpp
//  \brief implementation of functions in the Z4c class

// Athena++ headers
#include "z4c.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"

#define SQ(X) ((X)*(X))

// constructor, initializes data structures and parameters

Z4c::Z4c(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_block = pmb;

  // Allocate memory for the solution and its time derivative
  int ncells1 = pmy_block->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if(pmy_block->block_size.nx2 > 1) ncells2 = pmy_block->block_size.nx2 + 2*(NGHOST);
  if(pmy_block->block_size.nx3 > 1) ncells3 = pmy_block->block_size.nx3 + 2*(NGHOST);

  storage.u.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  storage.u1.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time","integrator","vl2");
  if (integrator == "ssprk5_4") storage.u2.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  storage.rhs.NewAthenaArray(N_Z4c, ncells3, ncells2, ncells1);
  storage.adm.NewAthenaArray(N_ADM, ncells3, ncells2, ncells1);

  dt1_.NewAthenaArray(ncells1);
  dt2_.NewAthenaArray(ncells1);
  dt3_.NewAthenaArray(ncells1);

  // Parameters
  opts.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opts.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opts.z4c_kappa_damp1 = pin->GetOrAddReal("z4c", "z4c_kappa_damp1", 0.0);
  opts.z4c_kappa_damp2 = pin->GetOrAddReal("z4c", "z4c_kappa_damp2", 0.0);

  // Set aliases
  SetZ4cAliases(storage.rhs, rhs);
  SetADMAliases(storage.adm, adm);

  // Allocate memory for aux 1D vars
  detg.NewAthenaTensor(ncells1);
  g_uu.NewAthenaTensor(ncells1);
  R_dd.NewAthenaTensor(ncells1);
  Kt_dd.NewAthenaTensor(ncells1);

  dalpha_d.NewAthenaTensor(ncells1);
  dchi_d.NewAthenaTensor(ncells1);
  dKhat_d.NewAthenaTensor(ncells1);
  dTheta_d.NewAthenaTensor(ncells1);
  ddalpha_dd.NewAthenaTensor(ncells1);
  dbeta_du.NewAthenaTensor(ncells1);
  ddchi_dd.NewAthenaTensor(ncells1);
  dGam_du.NewAthenaTensor(ncells1);
  dg_ddd.NewAthenaTensor(ncells1);
  dA_ddd.NewAthenaTensor(ncells1);
  ddbeta_ddu.NewAthenaTensor(ncells1);
  ddg_dddd.NewAthenaTensor(ncells1);

  Lchi.NewAthenaTensor(ncells1);
  LKhat.NewAthenaTensor(ncells1);
  LTheta.NewAthenaTensor(ncells1);
  Lalpha.NewAthenaTensor(ncells1);
  LGam_u.NewAthenaTensor(ncells1);
  Lbeta_u.NewAthenaTensor(ncells1);
  Lg_dd.NewAthenaTensor(ncells1);
  LA_dd.NewAthenaTensor(ncells1);
}

// destructor

Z4c::~Z4c()
{
  storage.u.DeleteAthenaArray();
  storage.u1.DeleteAthenaArray();
  storage.u2.DeleteAthenaArray();
  storage.rhs.DeleteAthenaArray();
  storage.adm.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  detg.DeleteAthenaArray();
  g_uu.DeleteAthenaArray();
  R_dd.DeleteAthenaArray();
  Kt_dd.DeleteAthenaArray();

  dalpha_d.DeleteAthenaArray();
  dchi_d.DeleteAthenaArray();
  dKhat_d.DeleteAthenaArray();
  dTheta_d.DeleteAthenaArray();
  ddalpha_dd.DeleteAthenaArray();
  dbeta_du.DeleteAthenaArray();
  ddchi_dd.DeleteAthenaArray();
  dGam_du.DeleteAthenaArray();
  dg_ddd.DeleteAthenaArray();
  dA_ddd.DeleteAthenaArray();
  ddbeta_ddu.DeleteAthenaArray();
  ddg_dddd.DeleteAthenaArray();

  Lchi.DeleteAthenaArray();
  LKhat.DeleteAthenaArray();
  LTheta.DeleteAthenaArray();
  Lalpha.DeleteAthenaArray();
  LGam_u.DeleteAthenaArray();
  Lbeta_u.DeleteAthenaArray();
  Lg_dd.DeleteAthenaArray();
  LA_dd.DeleteAthenaArray();
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetZ4cAliases(AthenaArray<Real> & u, Z4c_vars & z4c)
// \brief Set Z4c aliases

void Z4c::SetZ4cAliases(AthenaArray<Real> & u, Z4c::Z4c_vars & z4c)
{
  z4c.chi.InitWithShallowSlice(u, I_Z4c_chi);
  z4c.Khat.InitWithShallowSlice(u, I_Z4c_Khat);
  z4c.Theta.InitWithShallowSlice(u, I_Z4c_Theta);
  z4c.alpha.InitWithShallowSlice(u, I_Z4c_alpha);
  z4c.Gam_u.InitWithShallowSlice(u, I_Z4c_Gam);
  z4c.beta_u.InitWithShallowSlice(u, I_Z4c_beta);
  z4c.g_dd.InitWithShallowSlice(u, I_Z4c_g);
  z4c.A_dd.InitWithShallowSlice(u, I_Z4c_A);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SetADMAliases(AthenaArray<Real> & u, ADM_vars & z4c)
// \brief Set ADM aliases

void Z4c::SetADMAliases(AthenaArray<Real> & u, Z4c::ADM_vars & adm)
{
  adm.Psi4.InitWithShallowSlice(u, I_Psi4);
  adm.H.InitWithShallowSlice(u, I_ADM_Ham);
  adm.Mom_d.InitWithShallowSlice(u, I_ADM_Mom);
  adm.g_dd.InitWithShallowSlice(u, I_ADM_g);
  adm.K_dd.InitWithShallowSlice(u, I_ADM_K);
}

//----------------------------------------------------------------------------------------
// \!fn Real Z4c::SpatialDet(Real gxx, ... , Real gzz)
// \brief returns determinant of 3-metric

Real Z4c::SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz)
{
  return - SQ(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQ(gyz) - SQ(gxy)*gzz + gxx*gyy*gzz;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::SpatialInv(Real const det,
//           Real const gxx, Real const gxy, Real const gxz,
//           Real const gyy, Real const gyz, Real const gzz,
//           Real * uxx, Real * uxy, Real * uxz,
//           Real * uyy, Real * uyz, Real * uzz)
// \brief returns inverse of 3-metric

void Z4c::SpatialInv(Real const det,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real * uxx, Real * uxy, Real * uxz,
                     Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQ(gyz) + gyy*gzz)/det;
  *uxy = (gxz*gyz  - gxy*gzz)/det;
  *uyy = (-SQ(gxz) + gxx*gzz)/det;
  *uxz = (-gxz*gyy + gxy*gyz)/det;
  *uyz = (gxy*gxz  - gxx*gyz)/det;
  *uzz = (-SQ(gxy) + gxx*gyy)/det;
  return;
}

//----------------------------------------------------------------------------------------
// \!fn Real Z4c::Trace(Real detginv, Real gxx, ... , Real gzz, Real Axx, ..., Real Azz)
// \brief returns Trace of extrinsic curvature

Real Z4c::Trace(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real const Axx, Real const Axy, Real const Axz,
                Real const Ayy, Real const Ayz, Real const Azz)
{
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQ(gxy) - Ayy*SQ(gxz) - Axx*SQ(gyz)
       ));
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::AlgConstr(AthenaArray<Real> & u)
// \brief algebraic constraints projection

void Z4c::AlgConstr(AthenaArray<Real> & u)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  g.array().InitWithShallowSlice(u, gxx_IDX, NCab);
  A.array().InitWithShallowSlice(u, Axx_IDX, Ncab);

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

  const Real oot = 1.0/3.0;
  const Real eps_floor = 1e-12;

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

    //----------------------------------------------------------------------------------------
    // det(g) - 1 = 0

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

  // Determinant of the 3-metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = SpatialDet( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
        g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i) );
        }

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = (detg(i) <=0.) ? (1.0) : (detg(i));
  }

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detginv(i) = 1.0/detg(i);
  }

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    epsg(i) = -1.0 + detg(i);
        }

  // Enforce detg = 1
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    Real aux = (fabs(epsg(i)) < eps_floor) ?  (1.0-oot*epsg(i)) : (pow(detg(i),-oot));
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {
        g(a,b,k,j,i) = aux * g(a,b,k,j,i);
      }
    }
  }

      } // j - loop
    } // k - loop

    //----------------------------------------------------------------------------------------
    // Tr( A ) = 0

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

  // Trace of extrinsic curvature using new metric if rescaled
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = SpatialDet( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
        g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i) );
        }

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    // Note following variable actually contains 1/3 * Tr(A)
    TrA(i) = oot * Trace( 1.0/detg(i),
        g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
        g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i),
        A(0,0,k,j,i), A(0,1,k,j,i), A(0,2,k,j,i),
        A(1,1,k,j,i), A(1,2,k,j,i), A(2,2,k,j,i) );
  }

  // Enforce Tr A = 0
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        A(a,b,k,j,i) = A(a,b,k,j,i) - TrA(i) * g(a,b,k,j,i);
      }
    }
  }

      } // j - loop
    } // k - lopp

  }// parallel block

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & w)
// \brief compute ADM vars from Z4c vars

void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  g.array().InitWithShallowSlice(u, gxx_IDX, NCab);
  A.array().InitWithShallowSlice(u, Axx_IDX, NCab);
  Khat.array().InitWithShallowSlice(u, Khat_IDX, 1);//TODO: check Khat or K?
  chi.array().InitWithShallowSlice(u, chi_IDX, 1);

  ADM_g.array().InitWithShallowSlice(u_adm, ADM_gxx_IDX, NCab);
  ADM_K.array().InitWithShallowSlice(u_adm, ADM_Kxx_IDX, NCab);
  Psi4.array().InitWithShallowSlice(u_adm, ADM_Psi4_IDX, 1);

  const Real oot = 1.0/3.0;

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

    //----------------------------------------------------------------------------------------
    // Psi^4

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    Real psi  = pow(chi(k,j,i),1./chipsipower);
    Psi4(k,j,i) = pow(psi,4.);
  }

      }
    }

    //----------------------------------------------------------------------------------------
    // ADM g_{ij}

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        ADM_g(a,b,k,j,i) = Psi4(k,j,i) * g(a,b,k,j,i);
      }
    }
  }

      }
    }

    //----------------------------------------------------------------------------------------
    // ADM K_{ij}

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        ADM_K(a,b,k,j,i) = Psi4(k,j,i) * A(a,b,k,j,i) +  oot * Khat(k,j,i) * g(a,b,k,j,i);
      }
    }
  }

      }
    }

  } // parallel block

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMToZ4c(AthenaArray<Real> & w, AthenaArray<Real> & u)
// \brief derive Z4c variables from ADM variables
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//

// BAM: Z4c_init()
// https://git.tpi.uni-jena.de/bamdev/z4
// https://git.tpi.uni-jena.de/bamdev/z4/blob/master/z4_init.m

void Z4c::ADMToZ4c(AthenaArray<Real> & u_adm, AthenaArray<Real> & u)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  g.array().InitWithShallowSlice(u, gxx_IDX, NCab);
  chi.array().InitWithShallowSlice(u, chi_IDX, 1);
  A.array().InitWithShallowSlice(u, Axx_IDX, NCab);
  Khat.array().InitWithShallowSlice(u, Khat_IDX, 1); // Set Khat = K ...
  Theta.array().InitWithShallowSlice(u, Theta_IDX, 1); // ... Theta = 0
  Gam().InitWithShallowSlice(u, Gamx_IDX, NCa);

  ADM_g.array().InitWithShallowSlice(u_adm, gxx_IDX, NCab);
  ADM_K.array().InitWithShallowSlice(u_adm, Kxx_IDX, NCab);

  const Real oot = 1.0/3.0;

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

    //----------------------------------------------------------------------------------------
    // Conf. factor, metric, extrisic traceless curvature and trace

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

  // Determinant of the ADM metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = SpatialDet( ADM_g(0,0,k,j,i), ADM_g(0,1,k,j,i), ADM_g(0,2,k,j,i),
        ADM_g(1,1,k,j,i), ADM_g(1,2,k,j,i), ADM_g(2,2,k,j,i) );
        }

  // Inverse of the Conf. factor 1/Psi^4
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    oopsi4(i) = pow(detg(i), - 1.0/3.0);
        }

  // Conf. factor chi
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    chi(k,j,i) = pow(detg(i), 1.0/12.0 * chipsipower);
        }

  // Conf. metric
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        g(a,b,k,j,i) = oopsi4(i) * ADM_g(a,b,k,j,i)
      }
    }
  }

  // Conf. Extr. Curvature
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        Kt(a,b,i) = oopsi4(i) * ADM_K(a,b,k,j,i)
      }
    }
  }

  // Determinant of the Conf. metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = SpatialDet( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
        g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i) );
        }

  // Trace of conf. extr. curvature
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    Khat(k,j,i) = Trace( 1.0/detg(i),
          g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
          g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i),
          Kt(0,0,i), Kt(0,1,i), Kt(0,2,i),
          Kt(1,1,i), Kt(1,2,i), Kt(2,2,i) );
  }

  // Conf. Traceless Extr. Curvature
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        A(a,b,k,j,i) = ( ADM_K(a,b,k,j,i) - oot * Khat(k,j,i) * ADM_g(a,b,k,j,i) ) * oopsi4(i)
      }
    }
  }

      } // j - loop
    } // k - loop

    //----------------------------------------------------------------------------------------
    // Theta

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    Theta(k,j,i) = 0.0;
  }
      }
    }

    //----------------------------------------------------------------------------------------
    // Gamma's

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

  // Determinant of the Conf. metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = SpatialDet( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
        g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i) );
        }

  // Inverse Conf. metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    SpatialInv( detg(i),
          g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
          g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i),
          &ginv(0,0,i), &ginv(0,1,i), & ginv(0,2,i),
          &ginv(1,1,i), &ginv(1,2,i), & ginv(2,2,i) );
        }

  // Derivatives of Inverse Conf. metric // TODO
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    //ddginv(a,b,c, i) = d( ginv(a,b) , c  );
        }
      }
    }
  }

  // Inverse Conf. metric
  for(int a = 0; a < NDIM; ++a) {
    for(int i = is; i <= ie; ++i) {
      Gam(a, k,j,i) = 0.;
#pragma omp simd
        for(int b = a; b < NDIM; ++b) {
    Gam(a, k,j,i) -= dginv(b,a,b, i);
      }
    }
  }

      } // j -loop
    } // k - loop

  } // parallel block

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMConstraints(AthenaArray<Real> & u)
// \brief compute constraints ADM vars

// BAM: adm_constraints_N()
// https://git.tpi.uni-jena.de/bamdev/adm
// https://git.tpi.uni-jena.de/bamdev/adm/blob/master/adm_constraints_N.m

void Z4c::ADMConstraints(AthenaArray<Real> & u)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  ADM_g.array().InitWithShallowSlice(u_adm, ADM_gxx_IDX, NCab);
  ADM_K.array().InitWithShallowSlice(u_adm, ADM_Kxx_IDX, NCab);
  ADM_Ham.array().InitWithShallowSlice(u_adm, ADM_Ham_IDX, 1);
  ADM_Mom.array().InitWithShallowSlice(u_adm, ADM_Momx_IDX, NCa);

  const Real oot = 1./3.;

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

    //----------------------------------------------------------------------------------------

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    // ... dg[a,b,c] ddg[a,b,c,d] dK[a,b,c] ...
  }

  // Determinant
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    detg(i) = SpatialDet( ADM_g(0,0,k,j,i), ADM_g(0,1,k,j,i), ADM_g(0,2,k,j,i),
        ADM_g(1,1,k,j,i), ADM_g(1,2,k,j,i), ADM_g(2,2,k,j,i) );
        }

  // Inverse Conf. metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    SpatialInv( detg(i),
          ADM_g(0,0,k,j,i), ADM_g(0,1,k,j,i), ADM_g(0,2,k,j,i),
          ADM_g(1,1,k,j,i), ADM_g(1,2,k,j,i), ADM_g(2,2,k,j,i),
          &ginv(0,0,i), &ginv(0,1,i), & ginv(0,2,i),
          &ginv(1,1,i), &ginv(1,2,i), & ginv(2,2,i) );
        }

  // Gamma_{abc}
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    gammado(c,a,b, i) = 0.5 * ( dg(a,b,c, i) + dg(b,a,c, i) - dg(c,a,b, i) );
        }
      }
    }
  }

  // Gamma_{ab}^c
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    gamma(c,a,b, i) = 0.0;
    for(int d = 0; d < NDIM; ++d) {
      gamma(c,a,b, i) += ginv(c,d, i) * gammado(d,a,b, i);
    }
        }
      }
    }
  }

  // D_a( K_{bc} )
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    cdK(a,b,c, i) = dK(a,b,c, i);
    for(int d = 0; d < NDIM; ++d) {
      cdK(a,b,c, i) -= ( gamma(d,a,b, i) * K(d,c, k,j,i) - gamma(d,a,c, i) * K(b,d, k,j,i) );
    }
        }
      }
    }
  }

  // D^a K_{bc}
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
    cdKuud(a,b,c, i) = 0.0;
    for(int d = 0; d < NDIM; ++d) {
      cdKuud(a,b,c, i) += ginv(a,d, i) * codelK(d,b,c, i);
    }
        }
      }
    }
  }

  // R_{ab}
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        R(a,b, i) = 0.0;
        for(int c = 0; c < NDIM; ++c) {
    for(int d = 0; d < NDIM; ++d) {
      R(a,b, i) += 0.5 * ginv(c,d, i) * ( - ddg(c,d,a,b, i) - ddg(a,b,c,d, i) + ddg(a,c,b,d, i) + ddg(b,c,a,d, i) );
      for(int e = 0; e < NDIM; ++e) { // check this ...
        R(a,b, i) += 0.5* ginv(c,d, i) * (  gamma(e,a,c, i) * gammado(e,b,d, i) - gamma(e,a,b, i) * gammado(e,c,d, i) ) ;
      }
    }
        }
      }
    }
  }

  // K^a_b
  for(int a = 0; a < NDIM; ++a) {
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        Kud(a,b, i) = 0.0;
        for(int c = 0; c < NDIM; ++c) {
    Kud(a,b, i) += ginv(a,c, i) * K(c,b, k,j,i);
        }
      }
    }
  }

  // Ham
#pragma omp simd
  for(int i = is; i <= ie; ++i)
    Ham(k,j,i) = 0.0; // - 16.0 * PI * ADM_rho(k,j,i);
  for(int a = 0; a < NDIM; ++a)
#pragma omp simd
    for(int i = is; i <= ie; ++i)
      Ham(k,j,i) += Kud(a,a, i) * Kud(a,a, i); // K^2
  for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
#pragma omp simd
        for(int i = is; i <= ie; ++i)
    Ham(k,j,i) += ginv(a,b, i) * R(a,b, i) - Kud(a,b, i) * Kud(b,a, i);  // R + K^2


  // Mom^a
  for(int a = 0; a < NDIM; ++a) {
#pragma omp simd
    for(int i = is; i <= ie; ++i)
      Mom(a, k,j,i) = 0.0; // - 8.0 * PI * ADM_S(a, k,j,i);
    for(int b = 0; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c)
        for(int d = 0; d < NDIM; ++d)
#pragma omp simd
        for(int i = is; i <= ie; ++i)
    Mom(a, k,j,i) += ginv(a,b, i) * cdKudd(c,b,c, i) - ginv(b,c, i) * cdKudd(a,b,c, i);
  }


      } // j - loop
    } // k - loop
  } // parallel block

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMFlat(AthenaArray<Real> & u)
// \brief Initialize ADM vars to Minkowski

void Z4c::ADMFlat(AthenaArray<Real> & u_adm)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  ADM_g.array().InitWithShallowSlice(u_adm, ADM_gxx_IDX, NCab);
  ADM_K.array().InitWithShallowSlice(u_adm, ADM_Kxx_IDX, NCab);
  Psi4.array().InitWithShallowSlice(u_adm, ADM_Psi4_IDX, 1);

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {

#pragma omp simd
        for(int i = is; i <= ie; ++i)
    Psi4(k,j,i) = 1.0;

#pragma omp simd
  for(int i = is; i <= ie; ++i)
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
        ADM_g(a,b,k,j,i) = (a == b) ? (1.0) : (0.0);

#pragma omp simd
  for(int i = is; i <= ie; ++i)
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
        ADM_K(a,b,k,j,i) = 0.0;

      }
    }

  } // parallel block

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::GaugeFlat(AthenaArray<Real> & u)
// \brief Initialize lapse to 1 and shift to 0

void Z4c::GaugeFlat(AthenaArray<Real> & u)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  alpha.array().InitWithShallowSlice(u, alpha_IDX, 1);
  beta.array().InitWithShallowSlice(u, betax_IDX, NCa);

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
#pragma omp simd
        for(int i = is; i <= ie; ++i)
    alpha(k,j,i) = 1.0;
#pragma omp simd
  for(int i = is; i <= ie; ++i)
    for(int a = 0; a < NDIM; ++a)
      beta(a,k,j,i) = 0.0;
      }
    }
  }

  return;
}


