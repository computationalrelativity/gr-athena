//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_setup_tests.cpp
//  \brief intial data for essential tests

// C++ standard headers
#include <algorithm> // max
#include <cmath> // sqrt
#include <iomanip>
#include <iostream>
#include <sstream> // stringstream
#include <fstream>

// Athena++ headers
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "m1.hpp"

#define SQ(X) ((X)*(X))

namespace {

  // Estimate the fractional volume of the intersection between a cell and a
  // sphere of radius R with center in the origin
  Real volume(Real R, Real xp, Real yp, Real zp,
	      Real dx, Real dy, Real dz) {
    int const NPOINTS = 10;
    
    int inside = 0;
    int count = 0;
    for (int i = 0; i < NPOINTS; ++i) {
      Real const myx = (xp - dx/2.) + (i + 0.5)*(dx/NPOINTS);
      for (int j = 0; j < NPOINTS; ++j) {
            Real const myy = (yp - dy/2.) + (j + 0.5)*(dy/NPOINTS);
            for (int k = 0; k < NPOINTS; ++k) {
	      Real const myz = (zp - dz/2.) + (k + 0.5)*(dz/NPOINTS);
	      count++;
	      if (SQ(myx) + SQ(myy) + SQ(myz) < SQ(R)) {
		inside++;
	      }
            }
        }
    }
    return static_cast<Real>(inside)/static_cast<Real>(count);
  }
  
} // namespace


//----------------------------------------------------------------------------------------
// \!fn void M1::SetupZeroLabVars(AthenaArray<Real> & u)
// \brief Initialize to zero. This is used for kerrschild, shadow and sphere.
void M1::SetZeroLabVars(AthenaArray<Real> & u)
{
  Lab_vars vec;
  SetLabVarsAliases(u, vec);  

  // Initialize to zero
  vec.E.ZeroClear();
  vec.N.ZeroClear();
  vec.F_d.ZeroClear();
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetupZeroFiduVars(AthenaArray<Real> & u)
// \brief Initialize to zero. This is used for kerrschild, shadow and sphere.
void M1::SetZeroFiduVars(AthenaArray<Real> & u)
{
  Fidu_vars vec;
  SetFiduVarsAliases(u, vec);  

  // Initialize to zero
  vec.vel_u.ZeroClear();
  vec.Wlorentz.Fill(1.0);
}


//----------------------------------------------------------------------------------------
// \!fn void M1::SetupBeamTest(AthenaArray<Real> & u)
// \brief Setup beam on flat spacetime

void M1::SetupBeamTest(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;

  // Initialize to zero
  SetZeroLabVars(u);
  
  Lab_vars vec;
  SetLabVarsAliases(u, vec);

  // Beam direction (normalized)
  Real nx = beam_dir[0];
  Real ny = beam_dir[1];
  Real nz = beam_dir[2];
  Real n2 = SQ(nx) + SQ(ny) + SQ(nz);
  if (n2 > 0) {
    Real nn = std::sqrt(n2);
    nx /= nn;
    ny /= nn;
    nz /= nn;
  }
  else {
    nx = 1.0;
    nz = ny = 0.0;
  }

  GCLOOP3(k,j,i) { 

    Real z = pmb->pcoord->x3v(k);
    Real y = pmb->pcoord->x2v(j);
    Real x = pmb->pcoord->x1v(i);
    Real proj = nx*x + ny*y + nz*z;
    Real offset2 = SQR(x-nx*x) + SQR(y-ny*y) + SQR(z-nz*z);
    //std::cout << "z2 = " << z*z << std::endl;
    //std::cout << "y2 = " << y*y << std::endl;
    //std::cout << "x2 = " << SQR(x-beam_position) << std::endl;
    if (SQR(x-beam_position[0]) + SQR(y-beam_position[1]) + SQR(z-beam_position[2]) < SQR(beam_width)) {
      for (int ig=0; ig<ngroups*nspecies; ++ig) {
        vec.N(ig,k,j,i) = 1.0;
        vec.E(ig,k,j,i) = 1.0;
        vec.F_d(0,ig,k,j,i) = nx;
        vec.F_d(1,ig,k,j,i) = ny;
        vec.F_d(2,ig,k,j,i) = nz;
      }
    } else {
      for (int ig=0; ig<ngroups*nspecies; ++ig) {
        vec.N(ig,k,j,i) = (rad_N_floor>0)? rad_N_floor : 0.0;
        vec.E(ig,k,j,i) = (rad_E_floor>0)? rad_E_floor : 0.0;
        vec.F_d(0,ig,k,j,i) = 0.0;
        vec.F_d(1,ig,k,j,i) = 0.0;
        vec.F_d(2,ig,k,j,i) = 0.0;
      }
    }
    
    // There is no fluid, but set the fiducial velocity to zero
    for(int a = 0; a < NDIM; ++a) {
      fidu.vel_u(a,k,j,i) = 0.0;
    }
    fidu.Wlorentz(k,j,i) = 1.0;
  } // k, j, i 
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetupDiffusionTest(AthenaArray<Real> & u)
// \brief Setup diffusion test on flat spacetime

void M1::SetupDiffusionTest(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;
  
  // Initialize to zero
  SetZeroLabVars(u);

  Lab_vars vec;
  SetLabVarsAliases(u, vec);  

  Real const vel = medium_velocity;
  Real const wlorentz = 1.0/std::sqrt(1.0 - SQR(medium_velocity));
  
  GCLOOP3(k,j,i) {
    Real const x = pmb->pcoord->x1v(i);;
    Real const y = pmb->pcoord->x2v(j);;
    Real const z = pmb->pcoord->x3v(k);;
    
    for (int ig = 0; ig < ngroups*nspecies; ++ig) {
      if (diff_profile == "step") {
        if (x > -0.5 && x < 0.5) {
          vec.E(ig,k,j,i) = 1.0;
        }
      } else if (diff_profile == "gaussian") {
	        vec.E(ig,k,j,i) = std::exp(-SQR(3*x));
      }
      else {
        std::stringstream msg;
        msg << "Unknown diffusion profile " << diff_profile << std::endl;
        ATHENA_ERROR(msg);
      }
      
      vec.N(ig,k,j,i) = vec.E(ig,k,j,i);

      // Set the medium velocity
	    fidu.vel_u(0,k,j,i) = vel;
      fidu.vel_u(1,k,j,i) = 0.0;
      fidu.vel_u(2,k,j,i) = 0.0;
      fidu.Wlorentz(k,j,i) = wlorentz;
      // Use thick closure to compute the fluxes
      Real const W = fidu.Wlorentz(k,j,i);
      Real const Jo3 = vec.E(ig,k,j,i)/(4*SQR(W) - 1);
      for (int a = 0; a < NDIM; ++a) {
	      vec.F_d(a,ig,k,j,i) = 4*SQR(W)*fidu.vel_u(a,k,j,i)*Jo3;
      }
    } // ig loop
  } // CLOOP3
}

void M1::SetupAdvectionJumpTest(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;

  // Initialize to zero
  SetZeroLabVars(u);
  
  Lab_vars vec;
  SetLabVarsAliases(u, vec);

  GCLOOP3(k,j,i) { 
    Real x = pmb->pcoord->x1v(i);
    if (x+0.5 < 0.0) {
      for (int ig=0; ig<ngroups*nspecies; ++ig) {
        vec.N(ig,k,j,i) = 1.0;
        vec.E(ig,k,j,i) = 1.0;
        vec.F_d(0,ig,k,j,i) = 1.0;
        vec.F_d(1,ig,k,j,i) = 0.0;
        vec.F_d(2,ig,k,j,i) = 0.0;
      }
    } else {
      for (int ig=0; ig<ngroups*nspecies; ++ig) {
        vec.N(ig,k,j,i) = (rad_N_floor>0)? rad_N_floor : 0.0;
        vec.E(ig,k,j,i) = (rad_E_floor>0)? rad_E_floor : 0.0;
        vec.F_d(0,ig,k,j,i) = 0.0;
        vec.F_d(1,ig,k,j,i) = 0.0;
        vec.F_d(2,ig,k,j,i) = 0.0;
      }
    }
    
    if (x < 0.) {
      fidu.vel_u(0,k,j,i) = 0.87;
    } else {
      fidu.vel_u(0,k,j,i) = -0.87;
    }
    for(int a = 1; a < NDIM; ++a) {
      fidu.vel_u(a,k,j,i) = 0.0;
    }
    fidu.Wlorentz(k,j,i) = 1.0/std::sqrt(1.0 - SQR(fidu.vel_u(0,k,j,i)));
  } // k, j, i  
}


//----------------------------------------------------------------------------------------
// \!fn void M1::SetupEquilibriumTest(AthenaArray<Real> & u)
// \brief Setup equilibrium test on flat spacetime

void M1::SetupEquilibriumTest(AthenaArray<Real> & u)
{
  //MeshBlock * pmb = pmy_block;
  
  // Initialize to zero
  SetZeroLabVars(u);
  
  Lab_vars vec;
  SetLabVarsAliases(u, vec);  

  CLOOP3(k,j,i) {

    assert(ngroups == 1);
    assert(nspecies == 3);

    Real const W = fidu.Wlorentz(k,j,i);

    for (int is = 0; is < nspecies; ++is) {
      Real const Jnu = equil_nudens_1[is];
      vec.E(is,k,j,i)  = (4.*W*W - 1.)/3.*Jnu;
      vec.N(is,k,j,i)  = equil_nudens_0[is]*W;
      for (int a = 0; a < NDIM; ++a) {
	      vec.F_d(a,is,k,j,i) = 4./3.*SQ(W)*fidu.vel_u(a,k,j,i)*Jnu;
      }
    }
  } // CLOOP3
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetupKerrSchildMask() 
// \brief Setup the mask for Kerrschild test

void M1::SetupKerrSchildMask(AthenaArray<Real> & u) 
{
  MeshBlock * pmb = pmy_block;

  Lab_vars vec;
  SetLabVarsAliases(u, vec); 

  // Init excision mask to zero
  m1_mask.ZeroClear();
  GCLOOP3(k,j,i) {
    Real const x = pmb->pcoord->x1v(i);
    Real const y = pmb->pcoord->x2v(j);
    Real const z = pmb->pcoord->x3v(k);

    if (SQ(x) + SQ(y) + SQ(z) < SQ(kerr_mask_radius)) {
      m1_mask(k,j,i) = 1;
      for (int ig = 0; ig < nspecies*ngroups; ++ig) {
        vec.E(ig,k,j,i) = 0.0;
        vec.N(ig,k,j,i) = 0.0;
        vec.F_d(0,ig,k,j,i) = 0.0;
        vec.F_d(1,ig,k,j,i) = 0.0;
        vec.F_d(2,ig,k,j,i) = 0.0;
      }
    }
  } // GCLOOP3

}


//----------------------------------------------------------------------------------------
// \!fn void M1::SetupKerrBeamTest()
// \brief Setup the hydro variables for shadow and sphere tests
void M1::SetupKerrBeamTest(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;

  // Initialize to zero
  SetZeroLabVars(u);
  Lab_vars vec;
  SetLabVarsAliases(u, vec);

  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> beta_d;  
  TensorPointwise<Real, Symmetries::NONE, MDIM, 0> alpha;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> F_u;

  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  beta_d.NewTensorPointwise();
  alpha.NewTensorPointwise();
  F_u.NewTensorPointwise();
  F_d.NewTensorPointwise();

  Real const eps = 0.01;
  
  GCLOOP3(k,j,i) {

    Real z = pmb->pcoord->x3v(k);
    Real y = pmb->pcoord->x2v(j);
    Real x = pmb->pcoord->x1v(i);

    for (int ig=0; ig<ngroups*nspecies; ++ig) {
      vec.N(ig,k,j,i) = 0.0;
      vec.E(ig,k,j,i) = 0.0;
      vec.F_d(0,ig,k,j,i) = 0.0;
      vec.F_d(1,ig,k,j,i) = 0.0;
      vec.F_d(2,ig,k,j,i) = 0.0;
    }
    
    // There is no fluid, but set the fiducial velocity to zero
    for(int a = 0; a < NDIM; ++a) {
      fidu.vel_u(a,k,j,i) = 0.0;
    }
    fidu.Wlorentz(k,j,i) = 1.0;
    
  } // k, j, i 
  
}

void M1::KerrBeamInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &uu,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec); 

  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> g_dd;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> beta_u;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> beta_d;  
  TensorPointwise<Real, Symmetries::NONE, NDIM, 0> alpha;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> F_u;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> K_dd; 

  g_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  beta_d.NewTensorPointwise();
  alpha.NewTensorPointwise();
  F_u.NewTensorPointwise();
  F_d.NewTensorPointwise();
  K_dd.NewTensorPointwise();

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real z = pmb->pcoord->x3v(k);
        Real y = pmb->pcoord->x2v(j);
        Real x = pmb->pcoord->x1v(il-i);
        if (!m1_mask(k,j,il-i) &&
                        std::abs(y) <= kerr_beam_width &&
                        std::abs(z - beam_position[2]) <= 0.5*kerr_beam_width) {

          pmb->pz4c->ADMKerrSchild(x, y, z, alpha, beta_u, g_dd, K_dd);
          utils::tensor::contract(g_dd, beta_u, beta_d);
          Real const detg = SpatialDet(g_dd);
          Real const volform = std::sqrt(detg);
          Real const beta2 = utils::tensor::dot(beta_u, beta_d);
          Real const a = (-beta_d(0) + sqrt(SQR(beta_d(0)) - beta2 +
                              SQR(alpha())*0.99))/g_dd(0,0);
          for (int ig=0; ig<ngroups*nspecies; ++ig) {

            if (nspecies > 1) {
              vec.N(ig,k,j,il-i) = 1.0;
            }
            vec.E(ig,k,j,il-i) = volform;
            
            F_u(0) = vec.E(ig,k,j,il-i)/alpha() * (a + beta_u(0));
            F_u(1) = vec.E(ig,k,j,il-i)/alpha() * beta_u(1);
            F_u(2) = vec.E(ig,k,j,il-i)/alpha() * beta_u(2);

            utils::tensor::contract(g_dd, F_u, F_d);
            vec.F_d(0,ig,k,j,il-i) = F_d(0);
            vec.F_d(1,ig,k,j,il-i) = F_d(1);
            vec.F_d(2,ig,k,j,il-i) = F_d(2);
          }
        } else {
          for (int ig=0; ig<ngroups*nspecies; ++ig) {
            vec.N(ig,k,j,il-i) = 0.0;
            vec.E(ig,k,j,il-i) = 0.0;
            vec.F_d(0,ig,k,j,il-i) = 0.0;
            vec.F_d(1,ig,k,j,il-i) = 0.0;
            vec.F_d(2,ig,k,j,il-i) = 0.0;
          }
        }
      }
    }
  }
}

void M1::BeamInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &uu,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec); 

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real z = pmb->pcoord->x3v(k);
        Real y = pmb->pcoord->x2v(j);
        Real x = pmb->pcoord->x1v(il-i);
        if (std::abs(y) <= beam_width &&
            std::abs(z - beam_position[2]) <= 0.5*beam_width) {
          for (int ig=0; ig<ngroups*nspecies; ++ig) {

            if (nspecies > 1) {
              vec.N(ig,k,j,il-i) = 1.0;
            }
            vec.E(ig,k,j,il-i) = 1.0;
            vec.F_d(0,ig,k,j,il-i) = 1.0;
            vec.F_d(1,ig,k,j,il-i) = 0.0;
            vec.F_d(2,ig,k,j,il-i) = 0.0;
          }
        } else {
          for (int ig=0; ig<ngroups*nspecies; ++ig) {
            vec.N(ig,k,j,il-i) = 0.0;
            vec.E(ig,k,j,il-i) = 0.0;
            vec.F_d(0,ig,k,j,il-i) = 0.0;
            vec.F_d(1,ig,k,j,il-i) = 0.0;
            vec.F_d(2,ig,k,j,il-i) = 0.0;
          }
        }
      }
    }
  }
}

void M1::OutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,k,j,il-i) = vec.N(ig,k,j,il);
          vec.E(ig,k,j,il-i) = vec.E(ig,k,j,il);
          vec.F_d(0,ig,k,j,il-i) = vec.F_d(0,ig,k,j,il);
          vec.F_d(1,ig,k,j,il-i) = vec.F_d(1,ig,k,j,il);
          vec.F_d(2,ig,k,j,il-i) = vec.F_d(2,ig,k,j,il);
        }
      }
    }
  }
}

void M1::OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,k,j,iu+i) = vec.N(ig,k,j,iu);
          vec.E(ig,k,j,iu+i) = vec.E(ig,k,j,iu);
          vec.F_d(0,ig,k,j,iu+i) = vec.F_d(0,ig,k,j,iu);
          vec.F_d(1,ig,k,j,iu+i) = vec.F_d(1,ig,k,j,iu);
          vec.F_d(2,ig,k,j,iu+i) = vec.F_d(2,ig,k,j,iu);
        }
      }
    }
  }
}

void M1::OutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,k,jl-j,i) = vec.N(ig,k,jl,i);
          vec.E(ig,k,jl-j,i) = vec.E(ig,k,jl,i);
          vec.F_d(0,ig,k,jl-j,i) = vec.F_d(0,ig,k,jl,i);
          vec.F_d(1,ig,k,jl-j,i) = vec.F_d(1,ig,k,jl,i);
          vec.F_d(2,ig,k,jl-j,i) = vec.F_d(2,ig,k,jl,i);
        }
      }
    }
  }
}

void M1::OutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,k,ju+j,i) = vec.N(ig,k,ju,i);
          vec.E(ig,k,ju+j,i) = vec.E(ig,k,ju,i);
          vec.F_d(0,ig,k,ju+j,i) = vec.F_d(0,ig,k,ju,i);
          vec.F_d(1,ig,k,ju+j,i) = vec.F_d(1,ig,k,ju,i);
          vec.F_d(2,ig,k,ju+j,i) = vec.F_d(2,ig,k,ju,i);
        }
      }
    }
  }
}

void M1::OutflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,kl-k,j,i) = vec.N(ig,kl,j,i);
          vec.E(ig,kl-k,j,i) = vec.E(ig,kl,j,i);
          vec.F_d(0,ig,kl-k,j,i) = vec.F_d(0,ig,kl,j,i);
          vec.F_d(1,ig,kl-k,j,i) = vec.F_d(1,ig,kl,j,i);
          vec.F_d(2,ig,kl-k,j,i) = vec.F_d(2,ig,kl,j,i);
        }
      }
    }
  }
}

void M1::OutflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,ku+k,j,i) = vec.N(ig,ku,j,i);
          vec.E(ig,ku+k,j,i) = vec.E(ig,ku,j,i);
          vec.F_d(0,ig,ku+k,j,i) = vec.F_d(0,ig,ku,j,i);
          vec.F_d(1,ig,ku+k,j,i) = vec.F_d(1,ig,ku,j,i);
          vec.F_d(2,ig,ku+k,j,i) = vec.F_d(2,ig,ku,j,i);
        }
      }
    }
  }
}
  
void M1::ReflectInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,k,j,il-i) = vec.N(ig,k,j,il+i-1);
          vec.E(ig,k,j,il-i) = vec.E(ig,k,j,il+i-1);
          vec.F_d(0,ig,k,j,il-i) = vec.F_d(0,ig,k,j,il+i-1);
          vec.F_d(1,ig,k,j,il-i) = vec.F_d(1,ig,k,j,il+i-1);
          vec.F_d(2,ig,k,j,il-i) = vec.F_d(2,ig,k,j,il+i-1);
        }
      }
    }
  }
}

void M1::ReflectInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,k,jl-j,i) = vec.N(ig,k,jl+j-1,i);
          vec.E(ig,k,jl-j,i) = vec.E(ig,k,jl+j-1,i);
          vec.F_d(0,ig,k,jl-j,i) = vec.F_d(0,ig,k,jl+j-1,i);
          vec.F_d(1,ig,k,jl-j,i) = vec.F_d(1,ig,k,jl+j-1,i);
          vec.F_d(2,ig,k,jl-j,i) = vec.F_d(2,ig,k,jl+j-1,i);
        }
      }
    }
  }
}

void M1::ReflectInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Lab_vars vec;
  SetLabVarsAliases(storage.u, vec);

  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        for (int ig=0; ig<ngroups*nspecies; ++ig) {
          vec.N(ig,kl-k,j,i) = vec.N(ig,kl+k-1,j,i);
          vec.E(ig,kl-k,j,i) = vec.E(ig,kl+k-1,j,i);
          vec.F_d(0,ig,kl-k,j,i) = vec.F_d(0,ig,kl+k-1,j,i);
          vec.F_d(1,ig,kl-k,j,i) = vec.F_d(1,ig,kl+k-1,j,i);
          vec.F_d(2,ig,kl-k,j,i) = vec.F_d(2,ig,kl+k-1,j,i);
        }
      }
    }
  }
}