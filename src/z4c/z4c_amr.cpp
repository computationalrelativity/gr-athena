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
#include "../z4c/puncture_tracker.hpp"
#include "z4c.hpp"
#include "z4c_amr.hpp"

// print the results
// note: at 'if (Verbose)' when Verbose = 0, the if block is ignored by the compiler
#define Verbose (0)

// set some parameters
Z4c_AMR::Z4c_AMR(Z4c *z4c, MeshBlock *pmb, ParameterInput *pin):
pz4c(z4c),
pin(pin)
{
  const Real dmax =  std::numeric_limits<Real>::max();
  Real h1, h2, h3; // grid space
  
  // available methods: "Linf_box_in_box", "L2_sphere_in_sphere", and "fd_truncation_error"
  ref_method = pin->GetOrAddString("z4c_amr","method","Linf_box_in_box");
  ref_x1min = pin->GetOrAddReal("z4c_amr","x1min",pin->GetReal("mesh","x1min")/2.);
  ref_x1max = pin->GetOrAddReal("z4c_amr","x1max",pin->GetReal("mesh","x1max")/2.);
  ref_x2min = pin->GetOrAddReal("z4c_amr","x2min",pin->GetReal("mesh","x2min")/2.);
  ref_x2max = pin->GetOrAddReal("z4c_amr","x2max",pin->GetReal("mesh","x2max")/2.);
  ref_x3min = pin->GetOrAddReal("z4c_amr","x3min",pin->GetReal("mesh","x3min")/2.);
  ref_x3max = pin->GetOrAddReal("z4c_amr","x3max",pin->GetReal("mesh","x3max")/2.);
  // specify how to compute truncation error
  ref_deriv = pin->GetOrAddReal("z4c_amr","deriv_order",7);
  ref_pow   = pin->GetOrAddReal("z4c_amr","deriv_power",1);
  
  // ensure the grid-space doesn't fall below specific number for GW extraction
  ref_gwh   = pin->GetOrAddReal("z4c_amr","gw_gridspace",dmax);// can be an order of total mass
  // where to check the above criterion
  ref_gwr   = pin->GetOrAddReal("z4c_amr","gw_radius",dmax);// can be the largest gw extraction radius
  
  // one can specify different criteria for different annuli using finite difference
  // truncation error
  ref_FD_r1_inn  = pin->GetOrAddReal("z4c_amr","FD_radius1_inn", 0.);
  ref_FD_r1_out  = pin->GetOrAddReal("z4c_amr","FD_radius1_out",
                        2.2*pin->GetOrAddReal("problem", "par_b", 1.) );
  ref_FD_r2_inn  = pin->GetOrAddReal("z4c_amr","FD_radius2_inn", ref_FD_r1_out );
  ref_FD_r2_out  = pin->GetOrAddReal("z4c_amr","FD_radius2_out",
                        std::max( 
                                  std::abs(ref_x3max-ref_x3min),
                                  std::max( 
                                            std::abs(ref_x1max-ref_x1min),
                                            std::abs(ref_x2max-ref_x2min)
                                          )
                                )*0.5);
  // assigning a radius to the center of the meshblock
  mb_radius = std::sqrt( POW2(pmb->block_size.x3max + pmb->block_size.x3min) + 
                         POW2(pmb->block_size.x2max + pmb->block_size.x2min) + 
                         POW2(pmb->block_size.x1max + pmb->block_size.x1min))/2.;
  // pre-refine method
  ref_IsPreref_Linf = pin->GetOrAddBoolean("z4c_amr","preref_Linf",0);
  ref_IsPreref_L2   = pin->GetOrAddBoolean("z4c_amr","preref_L2",1);
  // using the pre-refine method till this time (less than or equal to this time)
  ref_PrerefTime = pin->GetOrAddReal("z4c_amr","preref_time_lt",20);
  
  assert(NDIM == 3);// the subsequent calculation may get affected if N!=3.
  h1 = pmb->pcoord->x1f(1)-pmb->pcoord->x1f(0);
  h2 = pmb->pcoord->x2f(1)-pmb->pcoord->x2f(0);
  h3 = pmb->pcoord->x3f(1)-pmb->pcoord->x3f(0);
  ref_hmax = std::max(h1,h2);
  ref_hmax = std::max(ref_hmax,h3);
  
  // power of grid-space to compare the truncation error with.
  // 4 is chosen as we use 4-th order Runge-Kutta time integration
  ref_hpow  = pin->GetOrAddReal("z4c_amr","h_power",4.);
  Real hp   = pow(ref_hmax,ref_hpow);
  // the range [lowb1,uppb1] for the first annular region
  dref_tol1 = pin->GetOrAddReal("z4c_amr","lowb1",1.)*hp;
  ref_tol1  = pin->GetOrAddReal("z4c_amr","uppb1",10.)*hp;
  // the range [lowb2,uppb2] for the second annular region
  dref_tol2 = pin->GetOrAddReal("z4c_amr","lowb2",1.)*hp;
  ref_tol2  = pin->GetOrAddReal("z4c_amr","uppb2",10.)*hp;
}

// using the FD error as an approximation of the error in the meshblock.
// this error compares agains the range (dref_tol,ref_tol)
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
  // err = FDTruncErrorChiL2(pmb,ref_deriv,ref_pow);
  // err = FDTruncErrorChiLinf(pmb,ref_deriv,ref_pow);
  err = FDTruncErrorChiLinfComponentWise(pmb,ref_deriv,ref_pow);
  
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
// \!fn void Z4c:: FDTruncErrorChiL2(MeshBlock *const pmy_block, const int p)
// \brief returning the L2 norm of error based on some derivative of chi
Real Z4c_AMR::FDTruncErrorChiL2(MeshBlock *const pmy_block, 
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
// \!fn void Z4c:: FDTruncErrorChiLinf(MeshBlock *const pmy_block, const int p)
// \brief returning the point-wise max error of a derivative of chi in a meshblock
//
Real Z4c_AMR::FDTruncErrorChiLinf(MeshBlock *const pmy_block, 
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

//----------------------------------------------------------------------------------------
// \!fn void Z4c:: FDTruncErrorChiLinfComponentWise(MeshBlock *const pmy_block, const int p)
// \brief returning the maximum truncation error of chi among x, y, and z directions 
// and for all points in the given meshblock.
Real Z4c_AMR::FDTruncErrorChiLinfComponentWise(MeshBlock *const pmy_block, 
                                        const int deriv_order, const int p)
{
  Z4c::Z4c_vars z4c;
  double max_err = 0.;

  z4c.chi.InitWithShallowSlice(pz4c->storage.u, pz4c->I_Z4c_chi);
  
  // calc. 7th derivative in all dirs
  if (deriv_order == 7)
  {
    assert(NGHOST > 3 && p == 1);// as p = 1 is optimized
    ILOOP2(k,j) {
      ILOOP1(i) {
        Real der_max_comp = 0.;
        // max { |d^7 chi(kji)/dx^7)^p|, |(d^7 chi(kji)/dy^7)^p|, |(d^7 chi(kji)/dz^7)^p| }
        for(int a = 0; a < NDIM; ++a) {
          Real der_abs = std::abs(pz4c->FD.Dx7(a, z4c.chi(k,j,i))); // p = 1 (optimization)
          der_max_comp = der_max_comp > der_abs ? der_max_comp : der_abs;
        }
        max_err = der_max_comp > max_err ? der_max_comp : max_err;
      }
    }
  }
  else
  {
    std::stringstream msg;
    msg << "No such derivative order developed yet!" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  max_err *= std::pow(ref_hmax,6);
  
  return max_err;
}

  
Z4c_AMR::~Z4c_AMR()
{
}

// 1: refines, -1: de-refines, 0: does nothing
int Z4c_AMR::ShouldIRefine(MeshBlock *pmb)
{
  int ret = 0;
  
  // use box in box method
  if (ref_method  == "Linf_box_in_box")
  {
    ret = LinfBoxInBox(pmb);
  }
  // use L-2 norm as a criteria for refinement
  else if (ref_method == "L2_sphere_in_sphere")
  {
    ret = L2SphereInSphere(pmb);
  }
  // finite difference truncation error must fall less that a prescribed value.
  else if (ref_method == "fd_truncation_error")
  {
    ret = FDTruncError(pmb);
  }
  else
  {
    std::stringstream msg;
    msg << "No such option for z4c/refinement" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  return ret;
}  

// Mimicking box in box refinement with Linf
int Z4c_AMR::LinfBoxInBox(MeshBlock *pmb)
{  
  int root_lev = pmb->pmy_mesh->GetRootLevel();
  int level = pmb->loc.level - root_lev;

  // Box in box ---------------------------------------------------------------
#ifdef Z4C_REF_BOX_IN_BOX
  //Initial distance between one of the punctures and the edge of the full mesh, needed to
  //calculate the box-in-box grid structure
  Real par_b = pin->GetOrAddReal("problem", "par_b", 1.);
  Real L = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2. - par_b; 
  Real xv[24];

  //Needed to calculate coordinates of vertices of a block with same center but
  //edge of 1/8th of the original size
  Real x1sum_sup = (5*pmb->block_size.x1max+3*pmb->block_size.x1min)/8.;
  Real x1sum_inf = (3*pmb->block_size.x1max+5*pmb->block_size.x1min)/8.;
  Real x2sum_sup = (5*pmb->block_size.x2max+3*pmb->block_size.x2min)/8.;
  Real x2sum_inf = (3*pmb->block_size.x2max+5*pmb->block_size.x2min)/8.;
  Real x3sum_sup = (5*pmb->block_size.x3max+3*pmb->block_size.x3min)/8.;
  Real x3sum_inf = (3*pmb->block_size.x3max+5*pmb->block_size.x3min)/8.;

  xv[0] = x1sum_sup;
  xv[1] = x2sum_sup;
  xv[2] = x3sum_sup;

  xv[3] = x1sum_sup;
  xv[4] = x2sum_sup;
  xv[5] = x3sum_inf;

  xv[6] = x1sum_sup;
  xv[7] = x2sum_inf;
  xv[8] = x3sum_sup;

  xv[9] = x1sum_sup;
  xv[10] = x2sum_inf;
  xv[11] = x3sum_inf;

  xv[12] = x1sum_inf;
  xv[13] = x2sum_sup;
  xv[14] = x3sum_sup;

  xv[15] = x1sum_inf;
  xv[16] = x2sum_sup;
  xv[17] = x3sum_inf;

  xv[18] = x1sum_inf;
  xv[19] = x2sum_inf;
  xv[20] = x3sum_sup;

  xv[21] = x1sum_inf;
  xv[22] = x2sum_inf;
  xv[23] = x3sum_inf;

  // Min distance between the two punctures
  Real d = std::numeric_limits<Real>::max();
  for (auto ptracker : pmb->pmy_mesh->pz4c_tracker) {
    // Abs difference
    Real diff;
    // Max norm_inf
    Real dmin_punct = std::numeric_limits<Real>::max();
    for (int i_vert = 0; i_vert < 8; ++i_vert) {
      // Norm_inf
      Real norm_inf = -1;
      for (int i_diff = 0; i_diff < 3; ++ i_diff) {
        diff = std::abs(ptracker->GetPos(i_diff) - xv[i_vert*3+i_diff]);
        if (diff > norm_inf) {
          norm_inf = diff;
        }
      }
      //Calculate minimum of the distances of the 8 vertices above
      if (dmin_punct > norm_inf) {
        dmin_punct = norm_inf;
      }
    }
    //Calculate minimum of the closest between the n punctures
    if (d > dmin_punct) {
      d = dmin_punct;
    }
  }
  Real ratio = L/d;

  if (ratio < 1) {
    return -1;
  }

  //Calculate level that the block should be in, given a box-in-box theoretical structure of the grid
  Real th_level = std::floor(std::log2(ratio));
  if (th_level > level) {
    return 1;
  } else if (th_level < level) {
    return -1;
  }
  return 0;

#endif // Z4C_REF_BOX_IN_BOX

#ifdef Z4C_REF_SPHERES
  for (int six=0; six<pmb->pz4c->opt.sphere_zone_number; ++six) {
    Real xyz_wz[3] = {0., 0., 0.};

    // use tracker if we can and if it is relevant
    int const pix = pmb->pz4c->opt.sphere_zone_puncture(six);
    if (pix != -1) {
      xyz_wz[0] = pmb->pmy_mesh->pz4c_tracker[pix].pos[0];
      xyz_wz[1] = pmb->pmy_mesh->pz4c_tracker[pix].pos[1];
      xyz_wz[2] = pmb->pmy_mesh->pz4c_tracker[pix].pos[2];
    } else {
      xyz_wz[0] = pmb->pz4c->opt.sphere_zone_center1(six);
      xyz_wz[1] = pmb->pz4c->opt.sphere_zone_center2(six);
      xyz_wz[2] = pmb->pz4c->opt.sphere_zone_center3(six);
    }

    int const lev_wz = pmb->pz4c->opt.sphere_zone_levels(six);
    Real const R_wz = pmb->pz4c->opt.sphere_zone_radii(six);

    if (lev_wz > 0) { // ensure currently iterated sphere actually has non-trivial level
      if (pmb->SphereIntersects(xyz_wz[0], xyz_wz[1], xyz_wz[2], R_wz)) {
        need_ref = need_ref or (level < lev_wz);
        satisfied_ref = satisfied_ref or (level == lev_wz);
      }
    }
  }

  if (need_ref) {
    return 1;
  } else if (satisfied_ref) {
    return 0;
  }
  // force de-refine if no condition satisfied
  return -1;
#endif // Z4C_REF_SPHERES

  return 0;

}


// refine based on a finite difference truncation error
int Z4c_AMR::FDTruncError(MeshBlock *pmb)
{
  int ret = 0;
  Real time = pmb->pmy_mesh->time;
  
  // note: the order of ifs matters
  if (ref_IsPreref_Linf && time <= ref_PrerefTime)
  {
    if (Verbose)
      std::cout << "calling Linf AMR for pre-refined" << std::endl;
    
    ret = LinfBoxInBox(pmb);
  }
  else if (ref_IsPreref_L2 && time <= ref_PrerefTime)
  {
    if (Verbose)
      std::cout << "calling L2 AMR for pre-refined" << std::endl;
    
    ret = L2SphereInSphere(pmb);
  }
  else if (ref_FD_r1_inn <= mb_radius && mb_radius <= ref_FD_r1_out)
  {
    if (Verbose)
      printf("Mb_radius = %g ==> calling FD AMR for the ring = [%g,%g], tol=[%g,%g]\n", 
              mb_radius,ref_FD_r1_inn,ref_FD_r1_out,dref_tol1,ref_tol1);
    
    ret = FDErrorApprox(pmb,dref_tol1,ref_tol1);
  }
  else if (ref_FD_r2_inn <= mb_radius && mb_radius <= ref_FD_r2_out)
  {
    if (Verbose)
      printf("Mb_radius = %g ==> calling FD AMR for the ring = [%g,%g], tol=[%g,%g]\n", 
              mb_radius,ref_FD_r2_inn,ref_FD_r2_out,dref_tol2,ref_tol2);
    
    ret = FDErrorApprox(pmb,dref_tol2,ref_tol2);
  }
  else
  {
    if (Verbose)
      std::cout << "Do nothing" << std::endl;
    
    ret = 0;
  }
  
  return ret;
}
 
// L-2 norm for refinement kind of like sphere in sphere
int Z4c_AMR::L2SphereInSphere(MeshBlock *pmb)
{
  int root_lev = pmb->pmy_mesh->GetRootLevel();
  int level = pmb->loc.level - root_lev;
  
  //Initial distance between one of the punctures and the edge of the full mesh
  Real par_b = pin->GetOrAddReal("problem", "par_b", 1.);
  Real L = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2. - par_b; 
  Real xv[24];

  //Needed to calculate coordinates of vertices of a block with same center but
  //edge of 1/8th of the original size
  Real x1sum_sup = (5*pmb->block_size.x1max+3*pmb->block_size.x1min)/8.;
  Real x1sum_inf = (3*pmb->block_size.x1max+5*pmb->block_size.x1min)/8.;
  Real x2sum_sup = (5*pmb->block_size.x2max+3*pmb->block_size.x2min)/8.;
  Real x2sum_inf = (3*pmb->block_size.x2max+5*pmb->block_size.x2min)/8.;
  Real x3sum_sup = (5*pmb->block_size.x3max+3*pmb->block_size.x3min)/8.;
  Real x3sum_inf = (3*pmb->block_size.x3max+5*pmb->block_size.x3min)/8.;

  xv[0] = x1sum_sup;
  xv[1] = x2sum_sup;
  xv[2] = x3sum_sup;

  xv[3] = x1sum_sup;
  xv[4] = x2sum_sup;
  xv[5] = x3sum_inf;

  xv[6] = x1sum_sup;
  xv[7] = x2sum_inf;
  xv[8] = x3sum_sup;

  xv[9] = x1sum_sup;
  xv[10] = x2sum_inf;
  xv[11] = x3sum_inf;

  xv[12] = x1sum_inf;
  xv[13] = x2sum_sup;
  xv[14] = x3sum_sup;

  xv[15] = x1sum_inf;
  xv[16] = x2sum_sup;
  xv[17] = x3sum_inf;

  xv[18] = x1sum_inf;
  xv[19] = x2sum_inf;
  xv[20] = x3sum_sup;

  xv[21] = x1sum_inf;
  xv[22] = x2sum_inf;
  xv[23] = x3sum_inf;

  // Min distance between the two punctures
  Real d = std::numeric_limits<Real>::max();
  for (auto ptracker : pmb->pmy_mesh->pz4c_tracker) {
    // square difference
    Real diff;
    
    Real dmin_punct = std::numeric_limits<Real>::max();
    for (int i_vert = 0; i_vert < 8; ++i_vert) {
      // Norm_L-2
      Real norm_L2 = 0;
      for (int i_diff = 0; i_diff < 3; ++ i_diff) {
        diff = (ptracker->GetPos(i_diff) - xv[i_vert*3+i_diff])*(ptracker->GetPos(i_diff) - xv[i_vert*3+i_diff]);
        norm_L2 += diff;
      }
      // Compute the L-2 norm
      norm_L2 = std::sqrt(norm_L2);
      
      //Calculate minimum of the distances of the 8 vertices above
      if (dmin_punct > norm_L2) {
        dmin_punct = norm_L2;
      }
    }
    //Calculate minimum of the closest between the n punctures
    if (d > dmin_punct) {
      d = dmin_punct;
    }
  }
  Real ratio = L/d;

  if (ratio < 1) {
    return -1;
  }

  //Calculate level that the block should be in, given a box-in-box theoretical structure of the grid
  Real th_level = std::floor(std::log2(ratio));
  if (th_level > level) {
    return 1;
  } else if (th_level < level) {
    return -1;
  }
  return 0;

}

