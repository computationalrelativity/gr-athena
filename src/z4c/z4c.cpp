//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file vwave.hpp
//  \brief implementation of functions in the VWave class

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

  u.  NewAthenaArray(NVARS, ncells3, ncells2, ncells1);
  u1. NewAthenaArray(NVARS, ncells3, ncells2, ncells1);
  rhs.NewAthenaArray(NVARS, ncells3, ncells2, ncells1);
  adm.NewAthenaArray(ADMVARS, ncells3, ncells2, ncells1);

  int nthreads = pmy_block->pmy_mesh->GetNumMeshThreads();
  dt1_.NewAthenaArray(nthreads,ncells1);
  dt2_.NewAthenaArray(nthreads,ncells1);
  dt3_.NewAthenaArray(nthreads,ncells1);

  // Parameters
  c = pin->GetOrAddReal("z4c", "c", 1.0);
  chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  z4c_kappa_damp1 = pin->GetOrAddReal("z4c", "z4c_kappa_damp1", 0.0);
  z4c_kappa_damp2 = pin->GetOrAddReal("z4c", "z4c_kappa_damp2", 0.0);
  
  // Allocate memory for aux 1D vars

  ginv.NewAthenaTensor(ncells1); // inverse of conf metric
  detg.NewAthenaTensor(ncells1); // det(g)
  epsg.NewAthenaTensor(ncells1); // 1 - det(g) 
  detginv.NewAthenaTensor(ncells1); // 1 / det(g)
  TrA.NewAthenaTensor(ncells1); // Tr(A)
  oopsi4.NewAthenaTensor(ncells1); // 1 / Psi^4

  // a
  da.NewAthenaTensor(ncells1); // lapse 1st drvts 
  dchi.NewAthenaTensor(ncells1); // conf.factor 1st drvts 
  dTheta.NewAthenaTensor(ncells1); // Theta 1st drvts 
  // a,b
  Kt.NewAthenaTensor(ncells1); // extrisic curvature conformally rescaled 
  dda.NewAthenaTensor(ncells1); // lapse 2nd drvts
  db.NewAthenaTensor(ncells1); // shift 1st drvts
  ddchi.NewAthenaTensor(ncells1); // conf. factor 2nd drvts
  dG.NewAthenaTensor(ncells1); // Gamma 1st drvts 
  R.NewAthenaTensor(ncells1); // conf. Ricci
  // a,b,c 
  ddb.NewAthenaTensor(ncells1); // shift 2nd drvts
  dg.NewAthenaTensor(ncells1); // Metric 1st drvts 
  dKhat.NewAthenaTensor(ncells1); // hat Gamma 1st drvts 
  dA.NewAthenaTensor(ncells1); // Extr. curvature 1st drvts 
  // a,b,c,d
  ddg.NewAthenaTensor(ncells1); // Metric 2nd drvts 
  
  lieKhat.NewAthenaTensor(ncells1); // Lie Khat
  liechi.NewAthenaTensor(ncells1); // Lie chi
  liealpha.NewAthenaTensor(ncells1); // Lie lapse
  // a
  advG.NewAthenaTensor(ncells1); // Lie Gamma
  advb.NewAthenaTensor(ncells1); // shift advective drvts
  // a,b
  lieg.NewAthenaTensor(ncells1); // Lie g 
  lieA.NewAthenaTensor(ncells1); // Lie A
  advTheta.NewAthenaTensor(ncells1); // Theta advective drvts

}

// destructor

Z4c::~Z4c()
{
  u.DeleteAthenaArray();
  u1.DeleteAthenaArray();
  rhs.DeleteAthenaArray();
  adm.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  detg.DeleteAthenaTensor(); // det(g)
  epsg.DeleteAthenaTensor(); // 1 - det(g) 
  detginv.DeleteAthenaTensor(); // 1/det(g)
  TrA.DeleteAthenaTensor(); // Tr(A)

  da.DeleteAthenaTensor(); // lapse 1st drvts 
  dchi.DeleteAthenaTensor(); // conf.factor 1st drvts 
  dTheta.DeleteAthenaTensor(); // Theta 1st drvts 

  dda.DeleteAthenaTensor(); // lapse 2nd drvts
  db.DeleteAthenaTensor(); // shift 1st drvts
  ddchi.DeleteAthenaTensor(); // conf. factor 2nd drvts
  dG.DeleteAthenaTensor(); // Gamma 1st drvts 
  R.DeleteAthenaTensor(); // conf. Ricci

  ddb.DeleteAthenaTensor(); // shift 2nd drvts
  dg.DeleteAthenaTensor(); // Metric 1st drvts 
  dKhat.DeleteAthenaTensor(); // hat Gamma 1st drvts 
  dA.DeleteAthenaTensor(); // Extr. curvature 1st drvts 

  ddg.DeleteAthenaTensor(); // Metric 2nd drvts 
  
  lieKhat.DeleteAthenaTensor(); // Lie Khat
  liechi.DeleteAthenaTensor(); // Lie chi
  liealpha.DeleteAthenaTensor(); // Lie lapse

  advG.DeleteAthenaTensor(); // Lie Gamma
  advb.DeleteAthenaTensor(); // shift advective drvts

  lieg.DeleteAthenaTensor(); // Lie g 
  lieA.DeleteAthenaTensor(); // Lie A
  advTheta.DeleteAthenaTensor(); // Theta advective drvts

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
//		       Real const gxx, Real const gxy, Real const gxz,
//		       Real const gyy, Real const gyz, Real const gzz,
//		       Real * uxx, Real * uxy, Real * uxz,
//		       Real * uyy, Real * uyz, Real * uzz)
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


