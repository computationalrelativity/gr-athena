//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//! \file gr_Lorene_BinNSs.cpp
//  \brief Initial conditions for binary neutron stars.
//         Interpolation of Lorene initial data.
//         Requires the library:
//         https://lorene.obspm.fr/

#include <cassert>
#include <iostream>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../z4c/z4c.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../bvals/bvals.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"
#include "elliptica_id_reader_lib.h"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

namespace {
  int RefinementCondition(MeshBlock *pmb);

#if MAGNETIC_FIELDS_ENABLED
  Real DivBface(MeshBlock *pmb, int iout);
#endif

  Real max_rho(      MeshBlock *pmb, int iout);
  Real min_alpha(    MeshBlock *pmb, int iout);
  Real max_abs_con_H(MeshBlock *pmb, int iout);

std::string checkpoint_file;
Elliptica_ID_Reader_T *idr;

}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);


  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, max_rho,   "max-rho",
    UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, min_alpha, "min-alpha",
    UserHistoryOperation::min);
  EnrollUserHistoryOutput(2, max_abs_con_H, "max-abs-con.H",
    UserHistoryOperation::max);

#if MAGNETIC_FIELDS_ENABLED
  // AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(3, DivBface, "divBface");
#endif

  checkpoint_file = pin->GetOrAddString("problem", "filename", "checkpoint.dat");
  idr = elliptica_id_reader_init(checkpoint_file.c_str(),"generic_MT_safe");
  // this is needed only for BHNS system
  idr->set_param("BH_filler_method","ChebTn_Ylm_perfect_s2",idr);

  // this is needed for both BHNS and BNS systems
  idr->set_param("ADM_B1I_form","zero",idr);

  // preparation and setting some interpolation settings (not thread safe)
  elliptica_id_reader_interpolate(idr);

  return;
}

void MeshBlock::UserWorkAfterOutput(ParameterInput *pin) {
  // Reset the status
  phydro->c2p_status.Fill(0);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  using namespace LinearAlgebra;

  // Interpolate data onto the grid.
  
  // settings -----------------------------------------------------------------
  //std::string checkpoint_file = pin->GetOrAddString("problem", "filename", "checkpoint.dat");
  Real const tol_det_zero =  pin->GetOrAddReal("problem","tolerance_det_zero",1e-10);
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  // check ID is accessible
  if (!file_exists(checkpoint_file.c_str()))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR problem/filename: " << checkpoint_file << " "
        << " could not be accessed.";
    ATHENA_ERROR(msg);
  }

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pz4c->mbi);

  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha( pz4c->storage.u,   Z4c::I_Z4c_alpha);
  AT_N_vec beta_u(pz4c->storage.u,   Z4c::I_Z4c_betax);
  AT_N_sym g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);

  
  // --------------------------------------------------------------------------
//  #pragma omp critical
//  {

    // prepare geometry grid --------------------------------------------------
    int npoints_gs = 0;

    for (int k=0; k<mbi->nn3; ++k)
    for (int j=0; j<mbi->nn2; ++j)
    for (int i=0; i<mbi->nn1; ++i)
    {
    	++npoints_gs;
    }

    Real * xx_gs = new Real[npoints_gs];
    Real * yy_gs = new Real[npoints_gs];
    Real * zz_gs = new Real[npoints_gs];

    int I = 0;  // collapsed ijk index

    for (int k=0; k<mbi->nn3; ++k)
    for (int j=0; j<mbi->nn2; ++j)
    for (int i=0; i<mbi->nn1; ++i)
    {
      zz_gs[I] = mbi->x3(k);
      yy_gs[I] = mbi->x2(j);
      xx_gs[I] = mbi->x1(i);

      ++I;
    }
    // ------------------------------------------------------------------------

    // prepare Elliptica interpolator for geometry -------------------------------
        
    // assert(idr->np == npoints_gs);

    I = 0;      // reset

    for (int k=0; k<mbi->nn3; ++k)
    for (int j=0; j<mbi->nn2; ++j)
    for (int i=0; i<mbi->nn1; ++i)
    {

      double x = xx_gs[I];
      double y = yy_gs[I];
      double z = zz_gs[I];
      
      alpha(k, j, i)     =  idr->fieldx(idr,"alpha",x,y,z);
      beta_u(0, k, j, i) =  -idr->fieldx(idr,"betax",x,y,z);
      beta_u(1, k, j, i) =  -idr->fieldx(idr,"betay",x,y,z);
      beta_u(2, k, j, i) =  -idr->fieldx(idr,"betaz",x,y,z);

      g_dd(0, 0, k, j, i) = idr->fieldx(idr,"adm_gxx",x,y,z);
      K_dd(0, 0, k, j, i) = idr->fieldx(idr,"adm_Kxx",x,y,z);

      g_dd(0, 1, k, j, i) = idr->fieldx(idr,"adm_gxy",x,y,z);
      K_dd(0, 1, k, j, i) = idr->fieldx(idr,"adm_Kxy",x,y,z);

      g_dd(0, 2, k, j, i) = idr->fieldx(idr,"adm_gxz",x,y,z);
      K_dd(0, 2, k, j, i) = idr->fieldx(idr,"adm_Kxz",x,y,z);

      g_dd(1, 1, k, j, i) = idr->fieldx(idr,"adm_gyy",x,y,z);
      K_dd(1, 1, k, j, i) = idr->fieldx(idr,"adm_Kyy",x,y,z);

      g_dd(1, 2, k, j, i) = idr->fieldx(idr,"adm_gyz",x,y,z);
      K_dd(1, 2, k, j, i) = idr->fieldx(idr,"adm_Kyz",x,y,z);

      g_dd(2, 2, k, j, i) = idr->fieldx(idr,"adm_gzz",x,y,z);
      K_dd(2, 2, k, j, i) = idr->fieldx(idr,"adm_Kzz",x,y,z);

      const Real det = Det3Metric(g_dd, k, j, i);
      assert(std::fabs(det) > tol_det_zero);

      ++I;
    }
    // ------------------------------------------------------------------------

    // clean up
    delete[] xx_gs;
    delete[] yy_gs;
    delete[] zz_gs;

    // ------------------------------------------------------------------------

    // prepare matter grid ----------------------------------------------------
    int npoints_cc = 0;

    const int il = 0;
    const int iu = ncells1-1;

    const int jl = 0;
    const int ju = ncells2-1;

    const int kl = 0;
    const int ku = ncells3-1;

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {
      ++npoints_cc;
    }

    Real * xx_cc = new Real[npoints_cc];
    Real * yy_cc = new Real[npoints_cc];
    Real * zz_cc = new Real[npoints_cc];

    I = 0;      // reset

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {
      zz_cc[I] = pcoord->x3v(k);
      yy_cc[I] = pcoord->x2v(j);
      xx_cc[I] = pcoord->x1v(i);

      ++I;
    }

    // ------------------------------------------------------------------------
    // prepare interpolator for matter ---------------------------------
    
    I = 0;      // reset

    Real k_adi = pin->GetReal("hydro", "k_adi");
    Real gamma_adi = pin->GetReal("hydro","gamma");


    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
    {

      double xc = xx_cc[I];
      double yc = yy_cc[I];
      double zc = zz_cc[I];
      
      const Real w_rho = idr->fieldx(idr,"grhd_rho",xc,yc,zc);
      const Real w_p = k_adi*pow(w_rho,gamma_adi);

      const Real v_u_x = idr->fieldx(idr,"grhd_vx",xc,yc,zc);
      const Real v_u_y = idr->fieldx(idr,"grhd_vy",xc,yc,zc);
      const Real v_u_z = idr->fieldx(idr,"grhd_vz",xc,yc,zc);

      const Real vsq = (
        2.0*(v_u_x * v_u_y * idr->fieldx(idr,"adm_gxy",xc,yc,zc) +
             v_u_x * v_u_z * idr->fieldx(idr,"adm_gxz",xc,yc,zc) +
             v_u_y * v_u_z * idr->fieldx(idr,"adm_gyz",xc,yc,zc))+
        v_u_x * v_u_x * idr->fieldx(idr,"adm_gxx",xc,yc,zc) +
        v_u_y * v_u_y * idr->fieldx(idr,"adm_gyy",xc,yc,zc) +
        v_u_z * v_u_z * idr->fieldx(idr,"adm_gzz",xc,yc,zc) 
      );

      const Real W = 1.0 / std::sqrt(1.0 - vsq);

      phydro->w(IDN, k, j, i) = w_rho;
      phydro->w(IVX, k, j, i) = W * v_u_x;
      phydro->w(IVY, k, j, i) = W * v_u_y;
      phydro->w(IVZ, k, j, i) = W * v_u_z;
      phydro->w(IPR, k, j, i) = w_p;

      ++I;
    }

    phydro->w1 = phydro->w;

    // clean up
    delete[] xx_cc;
    delete[] yy_cc;
    delete[] zz_cc;

    //delete pmy_mesh->bns;

  //} // OMP Critical

  //elliptica_id_reader_free(idr);
  // --------------------------------------------------------------------------

  /*
  if (MAGNETIC_FIELDS_ENABLED)
  {
    // B field ------------------------------------------------------------------
    // Assume stars are located on x axis

    Real pcut = pin->GetReal("problem","pcut") * pgasmax;
    Real b_amp = pin->GetReal("problem","b_amp");
    int magindex = pin->GetInteger("problem","magindex");

    int nx1 = (ie-is)+1 + 2*(NGHOST); //TODO Shouldn't this be ncell[123]?
    int nx2 = (je-js)+1 + 2*(NGHOST);
    int nx3 = (ke-ks)+1 + 2*(NGHOST);

    pfield->b.x1f.ZeroClear();
    pfield->b.x2f.ZeroClear();
    pfield->b.x3f.ZeroClear();
    pfield->bcc.ZeroClear();

    AthenaArray<Real> bxcc,bycc,bzcc;
    bxcc.NewAthenaArray(nx3,nx2,nx1);
    bycc.NewAthenaArray(nx3,nx2,nx1);
    bzcc.NewAthenaArray(nx3,nx2,nx1);

    AthenaArray<Real> Atot;
    Atot.NewAthenaArray(3,nx3,nx2,nx1);

    for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    for (int i = il; i <= iu; ++i)
      {
        if(pcoord->x1v(i) > 0){
    Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(1,k,j,i) = (pcoord->x1v(i) - 0.5*sep) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(2,k,j,i) = 0.0;
        } else {
    Atot(0,k,j,i) = -pcoord->x2v(j) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(1,k,j,i) = (pcoord->x1v(i) + 0.5*sep) * b_amp * std::max(phydro->w(IPR,k,j,i) - pcut, 0.0);
    Atot(2,k,j,i) = 0.0;
        }
      }

    for(int k = ks-1; k<=ke+1; k++)
    for(int j = js-1; j<=je+1; j++)
    for(int i = is-1; i<=ie+1; i++)
      {

      bxcc(k,j,i) = - ((Atot(1,k+1,j,i) - Atot(1,k-1,j,i))/(2.0*pcoord->dx3v(k)));
      bycc(k,j,i) =  ((Atot(0,k+1,j,i) - Atot(0,k-1,j,i))/(2.0*pcoord->dx3v(k)));
      bzcc(k,j,i) = ( (Atot(1,k,j,i+1) - Atot(1,k,j,i-1))/(2.0*pcoord->dx1v(i))
                    - (Atot(0,k,j+1,i) - Atot(0,k,j-1,i))/(2.0*pcoord->dx2v(j)));
      }

    for(int k = ks; k<=ke; k++)
    for(int j = js; j<=je; j++)
    for(int i = is; i<=ie+1; i++)
      {

      pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
      }

    for(int k = ks; k<=ke; k++)
    for(int j = js; j<=je+1; j++)
    for(int i = is; i<=ie; i++)
      {
      pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
      }

    for(int k = ks; k<=ke+1; k++)
    for(int j = js; j<=je; j++)
    for(int i = is; i<=ie; i++)
      {
        pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
      }

  } // MAGNETIC_FIELDS_ENABLED
  */
  //  -------------------------------------------------------------------------

  // Construct Z4c vars from ADM vars ------------------------------------------
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  // pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);

  // Allow override of Lorene gauge -------------------------------------------
  bool fix_gauge_precollapsed = pin->GetOrAddBoolean(
    "problem", "fix_gauge_precollapsed", false);

  if (fix_gauge_precollapsed)
  {
    // to construct psi4
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);
  }
  // --------------------------------------------------------------------------

  // consistent pressure atmosphere -------------------------------------------
  bool id_floor_primitives = pin->GetOrAddBoolean(
    "problem", "id_floor_primitives", false);

  if (id_floor_primitives)
  {

    for (int k = 0; k <= ncells3-1; ++k)
    for (int j = 0; j <= ncells2-1; ++j)
    for (int i = 0; i <= ncells1-1; ++i)
    {
      peos->ApplyPrimitiveFloors(phydro->w, k, j, i);
    }

  }
  // --------------------------------------------------------------------------


  // Initialise conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                             0, ncells1,
                             0, ncells2,
                             0, ncells3);

  //TODO Check if the momentum and velocity are finite.

  // Set up the matter tensor in the Z4c variables.
  // TODO: BD - this needs to be fixed properly
  // No magnetic field, pass dummy or fix with overload
  //  AthenaArray<Real> null_bb_cc;
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pscalars->r,
                  pfield->bcc);

  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);

  // --------------------------------------------------------------------------
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, int res_flag)
//  \brief Free Elliptica memory 
//========================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  elliptica_id_reader_free(idr);
  idr = 0;
  return;
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: extrema based
// 1: refines, -1: de-refines, 0: does nothing
int RefinementCondition(MeshBlock *pmb)
{
  /*
  // BD: TODO in principle this should be possible
  Z4c_AMR *const pz4c_amr = pmb->pz4c->pz4c_amr;

  // ensure we actually have a tracker
  if (pmb->pmy_mesh->ptracker_extrema->N_tracker > 0)
  {
    return 0;
  }

  return pz4c_amr->ShouldIRefine(pmb);
  */

  Mesh * pmesh = pmb->pmy_mesh;
  ExtremaTracker * ptracker_extrema = pmesh->ptracker_extrema;

  int root_level = ptracker_extrema->root_level;
  int mb_physical_level = pmb->loc.level - root_level;


  // Iterate over refinement levels offered by trackers.
  //
  // By default if a point is not in any sphere, completely de-refine.
  int req_level = 0;

  for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained = false;
    int cur_req_level = ptracker_extrema->ref_level(n-1);

    {
      if (ptracker_extrema->ref_type(n-1) == 0)
      {
        is_contained = pmb->PointContained(
          ptracker_extrema->c_x1(n-1),
          ptracker_extrema->c_x2(n-1),
          ptracker_extrema->c_x3(n-1)
        );
      }
      else if (ptracker_extrema->ref_type(n-1) == 1)
      {
        is_contained = pmb->SphereIntersects(
          ptracker_extrema->c_x1(n-1),
          ptracker_extrema->c_x2(n-1),
          ptracker_extrema->c_x3(n-1),
          ptracker_extrema->ref_zone_radius(n-1)
        );
      }
    }

    if (is_contained)
    {
      req_level = std::max(cur_req_level, req_level);
    }

  }

  if (req_level > mb_physical_level)
  {
    return 1;  // currently too coarse, refine
  }
  else if (req_level == mb_physical_level)
  {
    return 0;  // level satisfied, do nothing
  }

  // otherwise de-refine
  return -1;

}

#if MAGNETIC_FIELDS_ENABLED
Real DivBface(MeshBlock *pmb, int iout) {
  Real divB = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        divB += ((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/ dx +
                 (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/ dy +
                 (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/ dz) * vol;
      }
    }
  }
  return divB;
}
#endif

Real max_rho(MeshBlock *pmb, int iout)
{
  Real max_rho = -std::numeric_limits<Real>::infinity();
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
  }

  return max_rho;
}

Real min_alpha(MeshBlock *pmb, int iout)
{
  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca alpha(pmb->pz4c->storage.u, Z4c::I_Z4c_alpha);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_alpha = std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_alpha = std::min(alpha(k,j,i), m_alpha);
  }

  return m_alpha;
}

Real max_abs_con_H(MeshBlock *pmb, int iout)
{
  // --------------------------------------------------------------------------
  // Set some aliases for the variables.
  AT_N_sca con_H(pmb->pz4c->storage.con, Z4c::I_CON_H);

  // container with idx / grids pertaining z4c
  MB_info* mbi = &(pmb->pz4c->mbi);

  Real m_abs_con_H = -std::numeric_limits<Real>::infinity();

  for (int k=mbi->kl; k<=mbi->ku; k++)
  for (int j=mbi->jl; j<=mbi->ju; j++)
  for (int i=mbi->il; i<=mbi->iu; i++)
  {
    m_abs_con_H = std::max(std::abs(con_H(k,j,i)), m_abs_con_H);
  }

  return m_abs_con_H;
}

}
