#include <cassert> // assert
#include <iostream>
#include <sstream>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "z4c.hpp"
#include "z4c_amr.hpp"

// set some parameters
Z4c_AMR::Z4c_AMR(MeshBlock *pmb)
{
  ParameterInput *const pin = pmb->pmy_in;
  pz4c = pmb->pz4c;
  const Real dmax= std::numeric_limits<Real>::max();
  const Real dmin=-std::numeric_limits<Real>::max();
  ref_method = pin->GetOrAddString("z4c","refinement_method","Linf_box_in_box");
  ref_tol   = pin->GetOrAddReal("z4c","refinement_tol",1e-5);
  dref_tol  = pin->GetOrAddReal("z4c","derefinement_tol",1e-8);
  ref_x1min = pin->GetOrAddReal("z4c","refinement_x1min",dmin);
  ref_x1max = pin->GetOrAddReal("z4c","refinement_x1max",dmax);
  ref_x2min = pin->GetOrAddReal("z4c","refinement_x2min",dmin);
  ref_x2max = pin->GetOrAddReal("z4c","refinement_x2max",dmax);
  ref_x3min = pin->GetOrAddReal("z4c","refinement_x3min",dmin);
  ref_x3max = pin->GetOrAddReal("z4c","refinement_x3max",dmax);
  ref_deriv = pin->GetOrAddReal("z4c","refinement_deriv_order",7);
  ref_pow   = pin->GetOrAddReal("z4c","refinement_deriv_power",1);
  verbose   = pin->GetOrAddBoolean("z4c", "refinement_verbose",false);
}

// using the FD error as an approximation of the error in the meshblock.
// if this error falls below a prescribed value, the meshblock should be refined.
int Z4c_AMR::FDErrorApprox(MeshBlock *pmb)
{
  ParameterInput *const pin = pmb->pmy_in;
  int ret = 0;
  Real err = 0.;
  char region[999] = {0};
  
  if (verbose)
    sprintf(region,"[%0.1f,%0.1f]x[%0.1f,%0.1f]x[%0.1f,%0.1f]",
            pmb->block_size.x1min,pmb->block_size.x1max,
            pmb->block_size.x2min,pmb->block_size.x2max,
            pmb->block_size.x3min,pmb->block_size.x3max);

  // calc. err
  err = amr_err_L2_derive_chi_pow(pmb,ref_deriv,ref_pow);
  
  // check the region of interest for the refinement
  if (pmb->block_size.x1min < ref_x1min || pmb->block_size.x1max > ref_x1max)
  {
    if (verbose) 
      printf("out of bound %s.\n",region);
    ret = 0;
  }
  else if (pmb->block_size.x2min < ref_x2min || pmb->block_size.x2max > ref_x2max)
  {
    if (verbose) 
      printf("out of bound %s.\n",region);
    ret = 0;
  }
  else if (pmb->block_size.x3min < ref_x3min || pmb->block_size.x3max > ref_x3max)
  {
    if (verbose) 
      printf("out of bound %s.\n",region);
    ret = 0;
  }
  
  // compare with the error bounds
  else if (err > ref_tol)
  {
    if (verbose)
      printf("err > ref-tol:   %e > %e  ==> refine %s.\n",err,ref_tol,region);
    ret = 1.;
  }
  else if (err < dref_tol)
  {
    if (verbose)
      printf("err < deref-tol: %e < %e  ==> derefine %s.\n",err,dref_tol,region);
    ret = -1;
  }
  else 
  {
    if (verbose)
      printf("dref-tol <= err <= ref-tol: %e <= %e <= %e ==> nothing %s.\n",
              dref_tol,err,ref_tol,region);
    ret = 0;
  }
  
  fflush(stdout);
  
  return ret;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c:: amr_err_L2_derive_chi_pow(MeshBlock *const pmy_block, const int p)
// \brief returning the L2 norm of error basse one some derivative of chi
//

// NOTE: don't change pmy_block variable name as it's used in macros 
// such as IX_KU,IX_KL, etc.
Real Z4c_AMR::amr_err_L2_derive_chi_pow(MeshBlock *const pmy_block, 
                                        const int deriv_order, const int p)
{
  Z4c::Z4c_vars z4c;
  Real L2_norm = 0.;
  Real derive_aa_ijk[NDIM] = {0};
  Real derive_ijk = 0.;
  const int npts = (IX_KU-IX_KL)*(IX_JU-IX_KL)*(IX_IU-IX_IL);
  Real h1, h2, h3, hmax; // grid space
  // find grid spaces
  h1 = pmy_block->pcoord->x1f(1)-pmy_block->pcoord->x1f(0);
  h2 = pmy_block->pcoord->x2f(1)-pmy_block->pcoord->x2f(0);
  h3 = pmy_block->pcoord->x3f(1)-pmy_block->pcoord->x3f(0);
  hmax = std::max(h1,h2);
  hmax = std::max(hmax,h3);

  z4c.chi.InitWithShallowSlice(pz4c->storage.u, pz4c->I_Z4c_chi);
  
  
  // calc. L2 norm of 7th derivative
  if (deriv_order == 7)
  {
    ILOOP2(k,j) {
      ILOOP1(i) {
        // d^7 chi(ijk)/d(xyz)^7
        for(int a = 0; a < NDIM; ++a) {
          derive_aa_ijk[a] = pz4c->FD.Dx7(a, z4c.chi(k,j,i));
        }
        derive_ijk = 0.;
        // (d^7 chi(ijk)/dx^7)^p + (d^7 chi(ijk)/dy^7)^p + (d^7 chi(ijk)/dz^7)^p
        for(int a = 0; a < NDIM; ++a) {
          derive_ijk += std::pow(derive_aa_ijk[a],p);
        }
        L2_norm += std::pow(derive_ijk,2);
      }
    }
  }
  // calc. L2 norm of 2nd derivative
  else if (deriv_order == 2)
  {
    ILOOP2(k,j) {
      ILOOP1(i) {
        // d^2 chi(ijk)/d(xyz)^2
        for(int a = 0; a < NDIM; ++a) {
          derive_aa_ijk[a] = pz4c->FD.Dxx(a, z4c.chi(k,j,i));
        }
        derive_ijk = 0.;
        // (d^2 chi(ijk)/dx^2)^p + (d^2 chi(ijk)/dy^2)^p + (d^2 chi(ijk)/dz^2)^p
        for(int a = 0; a < NDIM; ++a) {
          derive_ijk += std::pow(derive_aa_ijk[a],p);
        }
        L2_norm += std::pow(derive_ijk,2);
      }
    }
  }
  else
  {
    std::stringstream msg;
    msg << "No such derivative" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  L2_norm /= npts;
  L2_norm *= std::pow(hmax,6);
  L2_norm = std::sqrt(L2_norm);

  return L2_norm;
}
  
Z4c_AMR::~Z4c_AMR()
{
}
