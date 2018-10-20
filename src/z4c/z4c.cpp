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

  u.  NewAthenaArray(2, ncells3, ncells2, ncells1);
  u1. NewAthenaArray(2, ncells3, ncells2, ncells1);
  rhs.NewAthenaArray(2, ncells3, ncells2, ncells1);

  int nthreads = pmy_block->pmy_mesh->GetNumMeshThreads();
  dt1_.NewAthenaArray(nthreads,ncells1);
  dt2_.NewAthenaArray(nthreads,ncells1);
  dt3_.NewAthenaArray(nthreads,ncells1);

  // Allocate memory for aux vars
  
  detg.NewAthenaTensor(ncells1); // det(g)
  epsg.NewAthenaTensor(ncells1); // 1 - det(g) 
  detginv.NewAthenaTensor(ncells1); // 1/det(g)
  TrA.NewAthenaTensor(ncells1); // Tr(A)

  // a
  da.NewAthenaTensor(ncells1); // lapse 1st drvts 
  dchi.NewAthenaTensor(ncells1); // conf.factor 1st drvts 
  dTheta.NewAthenaTensor(ncells1); // Theta 1st drvts 
  // a,b
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
// \!fn Real Z4c::Trace(Real gxx, ... , Real gzz, Real Axx, ..., Real Azz)
// \brief returns Trace of extrinsic curvature

Real Z4c::Trace(Real const gxx, Real const gxy, Real const gxz,
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

  g.array().InitWithShallowSlice(u, gab_IDX, 6);
  A.array().InitWithShallowSlice(u, Aab_IDX, 6);

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
    // 

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
	
	// Determinant of the metric
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  detg[i] = SpatialDet( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i), 
				g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i) );            
        }

	// Enforce detg = 1 
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  detg[i] = (detg[i] <=0.) ? (1.0) : (detg[i]);
	}
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  detginv[i] = 1.0/detg[i];
	}
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  epsg[i] = -1.0 + detg[i];  
        }
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  Real aux = (fabs(epsg[i]) < eps_floor) ?  (aux = 1.0-oot*epsg[i]) : (pow(detg[i],-oot));
	  for(int a = 0; a < NDIM; ++a) {
	    for(int b = a; b < NDIM; ++b) {
	      g(a,b,k,j,i) = aux * g(a,b,k,j,i); 
	    }
	  }
	}

	// Trace of extrinsic curvature using new metric if rescaled 
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  detg[i] = SpatialDet( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i), 
				g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i) );            
        }
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  detginv[i] = 1.0/detg[i];
	}
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
	  // Note following variable is actually 1/3 * Tr(A)  
	  TrA[i] = oot * Trace( g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i), 
				g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i),
				A(0,0,k,j,i), A(0,1,k,j,i), A(0,2,k,j,i), 
				A(1,1,k,j,i), A(1,2,k,j,i), A(2,2,k,j,i) );
	}
	
	// Enforce Tr A = 0
	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      A(a,b,k,j,i) = A(a,b,k,j,i) - TrA[i] * g(a,b,k,j,i); 
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

void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & w)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  g.array().InitWithShallowSlice(u, gab_IDX, 6);
  A.array().InitWithShallowSlice(u, Aab_IDX, 6);
  K.array().InitWithShallowSlice(u, K_IDX, 1);
  Chi.array().InitWithShallowSlice(u, Chi_IDX, 1);

  ADM_g.array().InitWithShallowSlice(w, gab_IDX, 6);
  ADM_K.array().InitWithShallowSlice(w, Kab_IDX, 6);

  const Real chipsipower = - 4.0; //Getd("z4_chi_psipower");//TODO get from pars
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
	  Real psi  = pow(chi(k,j,i),1./chipsipower);
	  psi4[i] = pow(psi,4.);
	}

	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      ADM_g(a,b,k,j,i) = psi4[i] * g(a,b,k,j,i);
	    }
	  }
	}

	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      ADM_K(a,b,k,j,i) = psi4[i] * A(a,b,k,j,i) +  oot * K(k,j,i) * g(a,b,k,j,i);
	    }
	  }
	}
	
      } // j - loop
    } // k - loop
  } // parallel block

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMToZ4c(AthenaArray<Real> & w, AthenaArray<Real> & u)
// \brief compute Z4c vars from ADM vars


// BAM: Z4c_init()

/* derive z4 variables from ADM variables
   p  = detgbar^(-1/3) 
   p0 = psi^(-4)

   gtilde_ij = p gbar_ij
   Ktilde_ij = p p0 K_ij

   phi = - log(p) / 4
   K   = gtildeinv^ij Ktilde_ij
   Atilde_ij = Ktilde_ij - gtilde_ij K / 3

   G^i = - del_j gtildeinv^ji
*/

void Z4c::ADMToZ4c(AthenaArray<Real> & w, AthenaArray<Real> & u)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  g.array().InitWithShallowSlice(u, gab_IDX, 6);
  A.array().InitWithShallowSlice(u, Aab_IDX, 6);
  K.array().InitWithShallowSlice(u, K_IDX, 1);
  Chi.array().InitWithShallowSlice(u, Chi_IDX, 1);

  ADM_g.array().InitWithShallowSlice(w, gab_IDX, 6);
  ADM_K.array().InitWithShallowSlice(w, Kab_IDX, 6);

  const Real chipsipower = - 4.0; //Getd("z4_chi_psipower");//TODO get from pars
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
	  // ...
	}

	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      //...
	    }
	  }
	}

	
      } // j - loop
    } // k - loop
  } // parallel block

  return;
}
