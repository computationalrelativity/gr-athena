//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file puncture_z4c.cpp
//  \brief implementation of functions in the Z4c class for initializing Apples with Apples tests
//  See  https://arxiv.org/abs/gr-qc/0305023
//       https://arxiv.org/abs/1111.2177

// C++ standard headers
#include <cmath> // pow, rand, sin, sqrt

// random number in [-1,1]
#define RAND_MAX (1.0)
#define RANDOMNUMBER (2.0*(std::rand())-1.0) 

// sin wave for various wave tests
#define SINWAVE(a,d,x) { ((a)*std::sin(2*M_PI*(x)/(d))) };
#define DSINWAVE(a,d,x) { (-(a)*2.0*M_PI/(d)*std::cos(2*M_PI*(x)/(d))) };

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMRobustStability(AthenaArray<Real> & u)
// \brief Initialize ADM vars for robust stability test

// Note the amplitude of the noise (~1e10) should be also rescaled by
// the square of the grid spacing

void Z4c::ADMRobustStability(AthenaArray<Real> & u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  
  // Flat spacetime
  ADMMinkowski(u_adm);

  std::srand(std::time(0)); // seed ?
  
  GLOOP2(k,j) {
    // g_ab
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
	GLOOP1(i) {
	  adm.g_dd(a,b,k,j,i) += RANDOMNUMBER*opt.AwA_amplitude;
	}
      }
    // K_ab
    for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
	GLOOP1(i) {
	  adm.K_dd(a,b,k,j,i) += RANDOMNUMBER*opt.AwA_amplitude;
	}
      }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::LinearWave1(AthenaArray<Real> & u)
// \brief Initialize ADM vars for linear wave test in 1d

void Z4c::ADMLinearWave1(AthenaArray<Real> & u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  
  // Flat spacetime
  ADMMinkowski(u_adm);
  
  // Set direction
  const int alldirections[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };
  int *dir = alldirections[opt.AwA_direction];

  // For propagation along x ...
  GLOOP2(k,j) {
    switch (opt.AwA_direction) {
    case 0: GLOOP1(i) { r(i) = pco->x1v(i); } break;
    case 1: GLOOP1(i) { r(i) = pco->x2v(j); } break;
    case 2: GLOOP1(i) { r(i) = pco->x3v(k); } break; 
    }    
    // g_yy
    GLOOP1(i) {
      adm.g_dd(dir[1],dir[1],k,j,i) += SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i)); 
    }
    // g_zz
    GLOOP1(i) {
      adm.g_dd(dir[2],dir[2],k,j,i) -= SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i));
    }
    // K_yy
    GLOOP1(i) {
      adm.K_dd(dir[1],dir[1],k,j,i) += 0.5*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i));;
    }
    // K_zz
    GLOOP1(i) {
      adm.K_dd(dir[2],dir[2],k,j,i) -= 0.5*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i));
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::LinearWave2(AthenaArray<Real> & u)
// \brief Initialize ADM vars for linear wave test in 2d

// Note we use the same macro as for the 1d,
// so there's a factor of sqrt(2) in the normalization different from literature

void Z4c::ADMLinearWave2(AthenaArray<Real> & u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  
  // Flat spacetime
  ADMMinkowski(u_adm);
  
  GLOOP2(k,j) {
    // g_xx, g_xy
    for(int a = 0; a < NDIM-1; ++a)
      GLOOP1(i) {
	adm.g_dd(a,a,k,j,i) += 0.5*SINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
      }
    // g_xy
    GLOOP1(i) {
      adm.g_dd(0,1,k,j,i) = 0.5*SINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
    }
    // g_zz
    GLOOP1(i) {
      adm.g_dd(2,2,k,j,i) -= SINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
    }
    // K_xx, K_xy, K_yy
    for(int a = 0; a < NDIM-1; ++a)
      for(int b = a; b < NDIM-1; ++b) {
	GLOOP1(i) {
	  adm.K_dd(a,b,k,j,i) = 0.25*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
	}
      }
    // K_zz
    GLOOP1(i) {
      adm.K_dd(2,2,k,j,i) = -0.5*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
    }
  }  
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMGaugeWave1(AthenaArray<Real> & u)
// \brief Initialize ADM vars for gauge wave test in 1d

void Z4c::ADMGaugeWave1(AthenaArray<Real> & u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  
  // Flat spacetime
  ADMMinkowski(u_adm);

  // Set direction
  const int alldirections[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };
  int *dir = alldirections[opt.AwA_direction];			     

  // For propagation along x ...
  GLOOP2(k,j) {
    switch (opt.AwA_direction) {
    case 0: GLOOP1(i) { r(i) = pco->x1v(i); } break;
    case 1: GLOOP1(i) { r(i) = pco->x2v(j); } break;
    case 2: GLOOP1(i) { r(i) = pco->x3v(k); } break; 
    }    
    // g_xx
    GLOOP1(i) {
      adm.g_dd(dir[0],dir[0],k,j,i) += SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i)); 
    }
    // K_xx
    GLOOP1(i) {
      adm.K_dd(dir[0],dir[0],k,j,i) =
	0.5*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i))/
	(std::sqrt(1.0 - SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i))));
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::GaugeGaugeWave(AthenaArray<Real> & u)
// \brief Initialize lapse and zero shift for gauge wave test

void Z4c::GaugeGaugeWaveLapse(AthenaArray<Real> & u)
{
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);
  z4c.alpha.Fill(1.);
  z4c.beta_u.Fill(0.);

  // Set direction
  const int alldirections[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };
  int *dir = alldirections[opt.AwA_direction];			     
  
  GLOOP2(k,j) {
    switch (opt.AwA_direction) {
    case 0: GLOOP1(i) { r(i) = pco->x1v(i); } break;
    case 1: GLOOP1(i) { r(i) = pco->x2v(j); } break;
    case 2: GLOOP1(i) { r(i) = pco->x3v(k); } break; 
    }    
    // alpha
    GLOOP1(i) {
      z4c.alpha(k,j,i) = std::sqrt(1.0 - SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i)));
    }  
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::GaugeGaugeWave(AthenaArray<Real> & u)
// \brief Initialize lapse and shift for shifted gauge wave test

void Z4c::GaugeGaugeWaveLapseShift(AthenaArray<Real> & u)
{
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);
  z4c.alpha.Fill(1.);
  z4c.beta_u.Fill(0.);

  // Set direction
  const int alldirections[3][3] = { {0,1,2}, {2,0,1}, {1,2,0} };
  int *dir = alldirections[opt.AwA_direction];			     
  
  GLOOP2(k,j) {
    switch (opt.AwA_direction) {
    case 0: GLOOP1(i) { r(i) = pco->x1v(i); } break;
    case 1: GLOOP1(i) { r(i) = pco->x2v(j); } break;
    case 2: GLOOP1(i) { r(i) = pco->x3v(k); } break; 
    }    
    // alpha 
    GLOOP1(i) {
      z4c.alpha(k,j,i) = 1.0/(std::sqrt(1.0 + SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i))));
    }  
    // shift
    GLOOP1(i) {
      z4c.beta(dir[0],k,j,i) = - SINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i))/(SQ(z4c.alpha(k,j,i)))
    }  
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMGaugeWave2(AthenaArray<Real> & u)
// \brief Initialize ADM vars for gauge wave test in 2d

// Note we use the same macro as for the 1d,
// so there's a factor of sqrt(2) in the normalization different from literature

void Z4c::ADMGaugeWave2(AthenaArray<Real> & u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  MeshBlock * pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  
  // Flat spacetime
  ADMMinkowski(u_adm);

  GLOOP2(k,j) {
    // g_xx, g_xy
    for(int a = 0; a < NDIM-1; ++a)
      GLOOP1(i) {
	adm.g_dd(a,a,k,j,i) -= 0.5*SINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
      }
    // g_xy
    GLOOP1(i) {
      adm.g_dd(0,1,k,j,i) = 0.5*SINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) );
    }
    // K_xx, K_yy
    for(int a = 0; a < NDIM-1; ++a)
      GLOOP1(i) {
	adm.K_dd(a,a,k,j,i) =
	  0.25*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma,r(i))/
	  (std::sqrt(1.0-SINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) )));
      }
    // K_xy
    GLOOP1(i) {
      adm.K_dd(0,1,k,j,i) = - adm.K_dd(0,0,k,j,i);
    }
    // K_zz
    GLOOP1(i) {
      adm.K_dd(2,2,k,j,i) =
	-0.5*DSINWAVE(opt.AwA_amplitude,opt.AwA_sigma, (pco->x1v(i) - pco->x2v(i)) ); 
    }
  }
}
