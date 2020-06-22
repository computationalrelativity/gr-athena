//  \brief source terms due to transformation to co-expanding grid

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../gravity/gravity.hpp"
#include "../../mesh/mesh.hpp"
#include "../hydro.hpp"
#include "hydro_srcterms.hpp"

#include "../../mesh/mesh.hpp"
#include "../conformal.hpp"

//----------------------------------------------------------------------------------------
//! \fn void HydroSourceTerms::ConformalSourceTerm
//  \brief Adds source terms for expanding grid and time-dependent ejecta from the hydro simulations. 

void HydroSourceTerms::ConformalSourceTerm(const Real t, const Real dt,const AthenaArray<Real> *flux,
                                   const AthenaArray<Real> &prim,
                                   AthenaArray<Real> &cons) {
  //Hydro *pmy_hydro = 
  MeshBlock *pmb = pmy_hydro_->pmy_block;
  Mesh *pmesh=pmb->pmy_mesh;
  
 if (CONFORMAL_SCALING) {
 
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real dx1 = pmb->pcoord->dx1v(i);
          //Real dx2 = pmb->pcoord->dx2v(j);
          //Real dx3 = pmb->pcoord->dx3v(k);
          Real dtodx1 = dt/dx1;

        //Real den = prim(IDN,k,j,i);
        if (NON_BAROTROPIC_EOS) {

          // Update momenta and energy with d/dx1 terms
          cons(IDN,k,j,i) += -dt*3*cons(IDN,k,j,i) * pmy_hydro_->my_conformal.expansionVelocity(pmesh->time);
          cons(IM1,k,j,i) += -dt*3*cons(IM1,k,j,i) * pmy_hydro_->my_conformal.expansionVelocity(pmesh->time);
          cons(IM2,k,j,i) += -dt*3*cons(IM2,k,j,i) * pmy_hydro_->my_conformal.expansionVelocity(pmesh->time);
          cons(IM3,k,j,i) += -dt*3*cons(IM3,k,j,i) * pmy_hydro_->my_conformal.expansionVelocity(pmesh->time);
          cons(IEN,k,j,i) += -dt*3*cons(IEN,k,j,i) * pmy_hydro_->my_conformal.expansionVelocity(pmesh->time);
        }
      }
    }

  }
  return;

                          }
                                   }