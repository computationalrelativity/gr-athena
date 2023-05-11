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

// print the results
// note: at 'if (Verbose)' when Verbose = 0, the if block is ignored by the compiler
#define Verbose (0)

// set some parameters
Z4c_AMR::Z4c_AMR(MeshBlock *pmb,ParameterInput *pin):
pz4c(pmb->pz4c),
pin(pin)
{
  const Real dmax =  std::numeric_limits<Real>::max();
  const Real dmin = -std::numeric_limits<Real>::max();
  Real h1, h2, h3; // grid space
  
  ref_method = pin->GetOrAddString("z4c","refinement_method","Linf_box_in_box");
  ref_x1min = pin->GetOrAddReal("z4c","refinement_x1min",dmin);
  ref_x1max = pin->GetOrAddReal("z4c","refinement_x1max",dmax);
  ref_x2min = pin->GetOrAddReal("z4c","refinement_x2min",dmin);
  ref_x2max = pin->GetOrAddReal("z4c","refinement_x2max",dmax);
  ref_x3min = pin->GetOrAddReal("z4c","refinement_x3min",dmin);
  ref_x3max = pin->GetOrAddReal("z4c","refinement_x3max",dmax);
  ref_deriv = pin->GetOrAddReal("z4c","refinement_deriv_order",7);
  ref_pow   = pin->GetOrAddReal("z4c","refinement_deriv_power",1);
  ref_gwh   = pin->GetOrAddReal("z4c","refinement_gw_resolution",1);// order of total mass
  ref_gwr   = pin->GetOrAddReal("z4c","refinement_gw_radius",100);// max length of gw extraction radius
  ref_hpow  = pin->GetOrAddReal("z4c","refinement_h_power",4.); // power of grid-space

  ref_FD_r1_inn  = pin->GetOrAddReal("z4c","refinement_FD_radius1_inn",10e10);
  ref_FD_r1_out  = pin->GetOrAddReal("z4c","refinement_FD_radius1_out",10e10);
  ref_FD_r2_inn  = pin->GetOrAddReal("z4c","refinement_FD_radius2_inn",10e10);
  ref_FD_r2_out  = pin->GetOrAddReal("z4c","refinement_FD_radius2_out",10e10);
  ref_PrerefTime = pin->GetOrAddReal("z4c","refinement_preref_time_lt",0.);
  
  ref_IsPreref_Linf = pin->GetOrAddBoolean("z4c","refinement_preref_Linf",0);
  ref_IsPreref_L2   = pin->GetOrAddBoolean("z4c","refinement_preref_L2",0);

  Real hp = pow(ref_hmax,ref_hpow);
  ref_tol1  = pin->GetOrAddReal("z4c","refinement_tol1",10)*hp;
  dref_tol1 = pin->GetOrAddReal("z4c","derefinement_tol1",1)*hp;
  ref_tol2  = pin->GetOrAddReal("z4c","refinement_tol2",10)*hp;
  dref_tol2 = pin->GetOrAddReal("z4c","derefinement_tol2",1)*hp;
   
  // find grid spaces
  assert(NDIM == 3);// the subsequent calculation may get affected if N!=3.
  h1 = pmb->pcoord->x1f(1)-pmb->pcoord->x1f(0);
  h2 = pmb->pcoord->x2f(1)-pmb->pcoord->x2f(0);
  h3 = pmb->pcoord->x3f(1)-pmb->pcoord->x3f(0);
  ref_hmax = std::max(h1,h2);
  ref_hmax = std::max(ref_hmax,h3);
  
  mb_radius = std::sqrt( POW2(pmb->block_size.x3max + pmb->block_size.x3min) + 
                         POW2(pmb->block_size.x2max + pmb->block_size.x2min) + 
                         POW2(pmb->block_size.x1max + pmb->block_size.x1min))/2.;
  
}

// using the FD error as an approximation of the error in the meshblock.
// if this error falls below a prescribed value, the meshblock should be refined.
int Z4c_AMR::FDErrorApprox(MeshBlock *pmb, Real dref_tol, Real ref_tol)
{
  int ret          = 0;
  Real err         = 0.;
  char region[999] = {0};
  
  if (Verbose)
    sprintf(region,"[%0.1f,%0.1f]x[%0.1f,%0.1f]x[%0.1f,%0.1f]",
            pmb->block_size.x1min,pmb->block_size.x1max,
            pmb->block_size.x2min,pmb->block_size.x2max,
            pmb->block_size.x3min,pmb->block_size.x3max);

  
  // check the region of interest for the refinement
  if (pmb->block_size.x1min < ref_x1min || pmb->block_size.x1max > ref_x1max)
  {
    if (Verbose) printf("out of bound %s.\n",region);
    return 0;
  }
  if (pmb->block_size.x2min < ref_x2min || pmb->block_size.x2max > ref_x2max)
  {
    if (Verbose) printf("out of bound %s.\n",region);
    return 0;
  }
  if (pmb->block_size.x3min < ref_x3min || pmb->block_size.x3max > ref_x3max)
  {
    if (Verbose) printf("out of bound %s.\n",region);
    return 0;
  }
  
  // if contains the extraction radius and too coarse refinement for GW
  if (mb_radius < ref_gwr && ref_gwh < ref_hmax)
  {
    if (Verbose)
      printf("box radius < GWr: %e < %e && GWh < h: %e < %e => refine %s.\n",
              mb_radius, ref_gwr, ref_gwh, ref_hmax, region);
    return 1;
  }
  
  
  // calc. err
  // err = amr_err_L2_derive_chi_pow(pmb,ref_deriv,ref_pow);
  err = amr_err_pnt_derive_chi_pow(pmb,ref_deriv,ref_pow);
  
  // compare with the error bounds
  if (err >= ref_tol)
  {
    if (Verbose) printf("err > ref-tol:   %e >= %e  ==> refine %s.\n",err,ref_tol,region);
    ret = 1.;
  }
  else if (err <= dref_tol)
  {
    if (Verbose) printf("err < deref-tol: %e <= %e  ==> derefine %s.\n",err,dref_tol,region);
    ret = -1;
  }
  else 
  {
    if (Verbose) printf("dref-tol < err < ref-tol: %e < %e < %e ==> nothing %s.\n",
                 dref_tol,err,ref_tol,region);
    ret = 0;
  }
  
  if (Verbose) fflush(stdout);
  
  return ret;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c:: amr_err_L2_derive_chi_pow(MeshBlock *const pmy_block, const int p)
// \brief returning the L2 norm of error based on some derivative of chi
//

// NOTE: DON'T change pmy_block variable name as it's used in macros 
// such as IX_KU,IX_KL, etc.
Real Z4c_AMR::amr_err_L2_derive_chi_pow(MeshBlock *const pmy_block, 
                                        const int deriv_order, const int p)
{
  Z4c::Z4c_vars z4c;
  Real L2_norm = 0.;
  Real derive_kji = 0.;
  const int npts = (IX_KU-IX_KL + 1)*(IX_JU-IX_KL + 1)*(IX_IU-IX_IL + 1);

  z4c.chi.InitWithShallowSlice(pz4c->storage.u, pz4c->I_Z4c_chi);
  
  // calc. L2 norm of 7th derivative
  if (deriv_order == 7)
  {
    assert(NGHOST > 3);
    ILOOP2(k,j) {
      ILOOP1(i) {
        derive_kji = 0.;
        // (d^7 chi(kji)/dx^7)^p + (d^7 chi(kji)/dy^7)^p + (d^7 chi(kji)/dz^7)^p
        for(int a = 0; a < NDIM; ++a) {
          derive_kji += std::pow(pz4c->FD.Dx7(a, z4c.chi(k,j,i)),p);
        }
        L2_norm += POW2(derive_kji);
      }
    }
  }
  // calc. L2 norm of 2nd derivative
  else if (deriv_order == 2)
  {
    ILOOP2(k,j) {
      ILOOP1(i) {
        derive_kji = 0.;
        // (d^2 chi(kji)/dx^2)^p + (d^2 chi(kji)/dy^2)^p + (d^2 chi(kji)/dz^2)^p
        for(int a = 0; a < NDIM; ++a) {
          derive_kji += std::pow(pz4c->FD.Dxx(a, z4c.chi(k,j,i)),p);
        }
        L2_norm += POW2(derive_kji);
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
  L2_norm *= std::pow(ref_hmax,6);
  L2_norm = std::sqrt(L2_norm);

  return L2_norm;
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c:: amr_err_pnt_derive_chi_pow(MeshBlock *const pmy_block, const int p)
// \brief returning the point-wise max error of a derivative of chi in a meshblock
//

// NOTE: DON'T change pmy_block variable name as it's used in macros 
// such as IX_KU,IX_KL, etc.
Real Z4c_AMR::amr_err_pnt_derive_chi_pow(MeshBlock *const pmy_block, 
                                        const int deriv_order, const int p)
{
  Z4c::Z4c_vars z4c;
  Real derive_kji = 0.;
  const int npts = (IX_KU-IX_KL + 1)*(IX_JU-IX_KL + 1)*(IX_IU-IX_IL + 1);
  std::vector<Real> err_pnt (npts,0.);
  int kji = 0; // dummy index

  z4c.chi.InitWithShallowSlice(pz4c->storage.u, pz4c->I_Z4c_chi);
  
  // calc. 7th derivative in all dirs
  if (deriv_order == 7)
  {
    assert(NGHOST > 3 && p == 1);// as p = 1 is optimized
    ILOOP2(k,j) {
      ILOOP1(i) {
        derive_kji = 0.;
        // (d^7 chi(kji)/dx^7)^p + (d^7 chi(kji)/dy^7)^p + (d^7 chi(kji)/dz^7)^p
        for(int a = 0; a < NDIM; ++a) {
          //derive_kji += std::pow(pz4c->FD.Dx7(a, z4c.chi(k,j,i)),p); // p != 1
          derive_kji += pz4c->FD.Dx7(a, z4c.chi(k,j,i)); // p = 1 (optimization)
        }
        err_pnt[kji] = std::abs(derive_kji);
        kji++;
      }
    }
  }
  // calc. 2nd derivative in all dirs
  else if (deriv_order == 2)
  {
    ILOOP2(k,j) {
      ILOOP1(i) {
        derive_kji = 0.;
        // (d^2 chi(kji)/dx^2)^p + (d^2 chi(kji)/dy^2)^p + (d^2 chi(kji)/dz^2)^p
        for(int a = 0; a < NDIM; ++a) {
          derive_kji += std::pow(pz4c->FD.Dxx(a, z4c.chi(k,j,i)),p);
        }
        err_pnt[kji] = std::abs(derive_kji);
        kji++;
      }
    }
  }
  else
  {
    std::stringstream msg;
    msg << "No such derivative" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  auto max_err = *std::max_element(err_pnt.cbegin(),err_pnt.cend());
  max_err     *= std::pow(ref_hmax,6);
  
  return max_err;
}
  
Z4c_AMR::~Z4c_AMR()
{
}
