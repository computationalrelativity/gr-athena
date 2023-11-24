//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_adm_matter.cpp
//  \brief add the radiation contribution to the ADM matter fields

//TODO: fix/check expressions

// C++ standard headers
//#include <cmath> // pow

// Athena++ headers
#include "m1.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../mesh/mesh.hpp"


void M1::AddToADMMatter(AthenaArray<Real> & u)
{
  MeshBlock * pmb = pmy_block;

  Lab_vars vec;
  SetLabVarsAliases(u, vec);

  // ADM matter vars VC
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> vc_mat_rho;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> vc_mat_S_d;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> vc_mat_S_dd;
  vc_mat_rho.InitWithShallowSlice(pmb->pz4c->storage.mat, Z4c::I_MAT_rho);
  vc_mat_S_d.InitWithShallowSlice(pmb->pz4c->storage.mat, Z4c::I_MAT_Sx);
  vc_mat_S_dd.InitWithShallowSlice(pmb->pz4c->storage.mat, Z4c::I_MAT_Sxx);
  
  // Go through vertexes and add the radiation to ADM matter var
  // TODO: Check species with the new logic
  for (int ig=0; ig<ngroups*nspecies; ++ig) {
    const int shift = ig*mbi.nn1*mbi.nn2*mbi.nn3;
    ILOOP2(k,j) {
      ILOOP1(i) {
        vc_mat_rho(k,j,i) += CCInterpolation(*(&vec.E()+shift),k,j,i);
      }
    
      ILOOP1(i) {      
        for(int a = 0; a < NDIM; ++a) {
          vc_mat_S_d(a,k,j,i) += CCInterpolation(*(&vec.F_d(a)+shift),k,j,i);
        }
      }

      ILOOP1(i) {      
        for(int a = 0; a < NDIM; ++a) {
          for(int b = a; b < NDIM; ++b) {
            vc_mat_S_dd(a,b,k,j,i) += CCInterpolation(*(&rad.P_dd(a,b)+shift),k,j,i);
          }
        }
      }
    } // ILOOP2
  } // ig
}
