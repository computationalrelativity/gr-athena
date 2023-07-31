//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file awa_test.cpp
//  \brief Initial conditions for Apples with Apples Test

#include <cassert> // assert
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"


// twopuncturesc: Stand-alone library ripped from Cactus
#include "RNS.h"

//using namespace std;

namespace{
  Real gamma_adi, k_adi;  // hydro EOS parameters
  Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i);
  Real DivBface(MeshBlock *pmb, int iout);


int RefinementCondition(MeshBlock *pmb);
// QUESTION: is it better to setup two different problems instead of using ifdef?
static ini_data *data;
  Real Maxrho(MeshBlock *pmb, int iout);
  Real Minalp(MeshBlock *pmb, int iout);
  Real L1rhodiff(MeshBlock *pmb, int iout);
}
//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin, int res_flag)
{
  k_adi = pin->GetReal("hydro", "k_adi");
  gamma_adi = pin->GetReal("hydro","gamma");
  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0, Maxrho, "max-rho", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, L1rhodiff, "L1rhodiff");
  EnrollUserHistoryOutput(2, DivBface, "divBface");
  EnrollUserHistoryOutput(3, Minalp, "min-alp", UserHistoryOperation::min);

    if (!res_flag) {     
      std::string set_name = "problem";
      //printf("BeforeSetDefault\n");
      RNS_params_set_default();
      //printf("AfterSetDefault\n");

      std::string inputfile = pin->GetOrAddString("problem", "filename", "tovgamma2.par");


      RNS_params_set_inputfile((char *) inputfile.c_str());

      //printf("AfterParSetting\n");
      data = RNS_make_initial_data();
      //printf("AfterMakeInitData\n");
    }
//    if(adaptive==true)
//      EnrollUserRefinementCondition(RefinementCondition);

    return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin, int res_flag)
{
  if (!res_flag)
    RNS_finalise(data);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

#ifdef Z4C_ASSERT_FINITE
  // as a sanity check (these should be over-written)
  pz4c->adm.psi4.Fill(NAN);
  pz4c->adm.g_dd.Fill(NAN);
  pz4c->adm.K_dd.Fill(NAN);

  pz4c->z4c.chi.Fill(NAN);
  pz4c->z4c.Khat.Fill(NAN);
  pz4c->z4c.Theta.Fill(NAN);
  pz4c->z4c.alpha.Fill(NAN);
  pz4c->z4c.Gam_u.Fill(NAN);
  pz4c->z4c.beta_u.Fill(NAN);
  pz4c->z4c.g_dd.Fill(NAN);
  pz4c->z4c.A_dd.Fill(NAN);

  /*
  pz4c->con.C.Fill(NAN);
  pz4c->con.H.Fill(NAN);
  pz4c->con.M.Fill(NAN);
  pz4c->con.Z.Fill(NAN);
  pz4c->con.M_d.Fill(NAN);

  */
#endif //Z4C_ASSERT_FINITE
  //---------------------------------------------------------------------------


  //Interpolate Metric quantities to VC
  phydro->RNS_Metric(pin, pz4c->storage.adm, pz4c->storage.u, pz4c->storage.u1, data);

  //Interpolate primitives to CC
  phydro->RNS_Hydro(pin, phydro->w, phydro->w1, phydro->w_init, data);

  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u1);






    pcoord->UpdateMetric();

    if(pmy_mesh->multilevel){
        pmr->pcoarsec->UpdateMetric();
          }
  // Prepare index bounds
     int il = is - NGHOST;
     int iu = ie + NGHOST;
     int jl = js;
     int ju = je;
     if (block_size.nx2 > 1) {
        jl -= NGHOST;
        ju += NGHOST;
     }
     int kl = ks;
     int ku = ke;
     if (block_size.nx3 > 1) {
        kl -= NGHOST;
        ku += NGHOST;
     }
  Real rhoc = pin->GetReal("problem","rhoc");
  Real pgasmax = k_adi*pow(rhoc,gamma_adi); 
  Real pcut = pin->GetReal("problem","pcut")*pgasmax;
  Real amp = pin->GetReal("problem","b_amp");
  int magindex=pin->GetInteger("problem","magindex");
  AthenaArray<Real> ax,ay,az,bxcc,bycc,bzcc;
  int nx1 = (ie-is)+1 + 2*(NGHOST);
  int nx2 = (je-js)+1 + 2*(NGHOST);
  int nx3 = (ke-ks)+1 + 2*(NGHOST);
// should be athena tensors if we merge w/ dynamical metric
pfield->b.x1f.ZeroClear();
pfield->b.x2f.ZeroClear();
pfield->b.x3f.ZeroClear();
pfield->bcc.ZeroClear();
  ax.NewAthenaArray(nx3,nx2,nx1);
  ay.NewAthenaArray(nx3,nx2,nx1);
  az.NewAthenaArray(nx3,nx2,nx1);
  bxcc.NewAthenaArray(nx3,nx2,nx1);
  bycc.NewAthenaArray(nx3,nx2,nx1);
  bzcc.NewAthenaArray(nx3,nx2,nx1);
//parameter defns from 
//Openmp - pragma/ GLOOP macros?
      AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
      AthenaArray<Real> vcgamma_yz,vcgamma_zz;
      AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd; 
      gamma_dd.NewAthenaTensor(iu+1);
 vcgamma_xx.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxx,1);
      vcgamma_xy.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxy,1);
      vcgamma_xz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gxz,1);
      vcgamma_yy.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gyy,1);
      vcgamma_yz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gyz,1);
      vcgamma_zz.InitWithShallowSlice(pz4c->storage.adm,Z4c::I_ADM_gzz,1);

//Initialize field at xfaces:
//Real prx1f, rhox1f, prx2f, rhox2f, prx3f, rhox3f, div1, div2, div3;
//printf("kl= %d, ku=%d, jl = %d, ju = %d, il = %d, iu = %d, NGHOST= %d",kl,ku,jl,ju,il,iu,NGHOST); 
  for (int k=kl; k<=ku; k++) {
  for (int j=jl; j<=ju; j++) {
  for (int i=il; i<=iu; i++) {
// ay at x faces - need w at xface

      ax(k,j,i) = -pcoord->x2v(j)*amp*std::max(phydro->w(IPR,k,j,i) - pcut,0.0)*pow((1.0 - phydro->w(IDN,k,j,i)/rhoc),magindex);
      ay(k,j,i) = pcoord->x1v(i)*amp*std::max(phydro->w(IPR,k,j,i) - pcut,0.0)*pow((1.0 - phydro->w(IDN,k,j,i)/rhoc),magindex);


//     prx1f = (phydro->w(IPR,k,j,i-1) + phydro->w(IPR,k,j,i))/2.0;
//      rhox1f = (phydro->w(IDN,k,j,i-1) + phydro->w(IDN,k,j,i))/2.0;
//     if(isnan(prx1f)==1){printf("prx1f isnan i=%d,j=%d,k=%d\n",i,j,k);}
//     if(isnan(rhox1f)==1){printf("rhox1f isnan i=%d,j=%d,k=%d\n",i,j,k);}
//      ay(k,j,i) = pcoord->x1f(i)*amp*std::max(prx1f - pcut,0.0)*pow((1 - rhox1f/rhomax),magindex);
  }}}

  for(int k = ks-1; k<=ke+1; k++){
  for(int j = js-1; j<=je+1; j++){
  for(int i = is-1; i<=ie+1; i++){
          gamma_dd(0,0,i) = pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
          gamma_dd(0,1,i) = pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
          gamma_dd(0,2,i) = pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
          gamma_dd(1,1,i) = pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
          gamma_dd(1,2,i) = pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
          gamma_dd(2,2,i) = pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));
}
   for(int i = is-1; i<=ie+1; i++){
    Real detgamma = std::sqrt(Det3Metric(gamma_dd,i));

    bxcc(k,j,i) = - ((ay(k+1,j,i) - ay(k-1,j,i))/(2.0*pcoord->dx3v(k)));
    bycc(k,j,i) =  ((ax(k+1,j,i) - ax(k-1,j,i))/(2.0*pcoord->dx3v(k)));
    bzcc(k,j,i) = ( (ay(k,j,i+1) - ay(k,j,i-1))/(2.0*pcoord->dx1v(i))
                   - (ax(k,j+1,i) - ax(k,j-1,i))/(2.0*pcoord->dx2v(j)));
//    bxcc(k,j,i) = - (ay(k+1,j,i) - ay(k,j,i))/(pcoord->dx3v(k));
//    bycc(k,j,i) =  (ax(k+1,j,i) - ax(k,j,i))/(pcoord->dx3v(k));
//    bzcc(k,j,i) =  (ay(k,j,i+1) - ay(k,j,i))/(pcoord->dx1v(i))
//                   - (ax(k,j+1,i) - ax(k,j,i))/(pcoord->dx2v(j));
}}}

  for(int k = ks; k<=ke; k++){
  for(int j = js; j<=je; j++){
  for(int i = is; i<=ie+1; i++){

  pfield->b.x1f(k,j,i) = 0.5*(bxcc(k,j,i-1) + bxcc(k,j,i));
}}}
  for(int k = ks; k<=ke; k++){
  for(int j = js; j<=je+1; j++){
  for(int i = is; i<=ie; i++){
  pfield->b.x2f(k,j,i) = 0.5*(bycc(k,j-1,i) + bycc(k,j,i));
}}}
  for(int k = ks; k<=ke+1; k++){
  for(int j = js; j<=je; j++){
  for(int i = is; i<=ie; i++){

  pfield->b.x3f(k,j,i) = 0.5*(bzcc(k-1,j,i) + bzcc(k,j,i));
}}}

pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il,iu,jl,ju,kl,ku);
                                         
            // Initialise conserved variables
              peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  
                // Initialise VC matter
                  //TODO(WC) (don't strictly need this here, will be caught in task list before used
                    pz4c->GetMatter(pz4c->storage.mat, pz4c->storage.adm, phydro->w,pfield->bcc);
                      pz4c->ADMConstraints(pz4c->storage.con,pz4c->storage.adm,pz4c->storage.mat,pz4c->storage.u);
  //










  //std::cout << "Two punctures initialized." << std::endl;

  // debug!
  /*
  pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);

  pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm,
                       pz4c->storage.mat, pz4c->storage.u);

  */

#ifdef Z4C_ASSERT_FINITE
  pz4c->assert_is_finite_adm();
  pz4c->assert_is_finite_con();
  pz4c->assert_is_finite_mat();
  pz4c->assert_is_finite_z4c();
#endif //Z4C_ASSERT_FINITE

  return;
}
/*
int RefinementCondition(MeshBlock *pmb)
{
#ifdef Z4C_TRACKER
  //Initial distance between one of the punctures and the edge of the full mesh, needed to
  //calculate the box-in-box grid structure
  Real L = pmb->pmy_mesh->pz4c_tracker->L_grid;
  int root_lev = pmb->pmy_mesh->pz4c_tracker->root_lev;
#ifdef DEBUG
  printf("Root lev = %d\n", root_lev);
#endif
  Real xv[24];
#ifdef DEBUG
  printf("Max x = %g\n", pmb->block_size.x1max);
  printf("Min x = %g\n", pmb->block_size.x1min);
#endif
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

  //Level of current block
  int level = pmb->loc.level-root_lev;
#ifdef DEBUG
  printf("\n<===================================================>\n");
  printf("L = %g\n",L);
  printf("lev = %d\n",level);
#endif
  // Min distance between the two punctures
  Real d = 1000000;
  for (int i_punct = 0; i_punct < NPUNCT; ++i_punct) {
    // Abs difference
    Real diff;
    // Max norm_inf
    Real dmin_punct = 1000000;
#ifdef DEBUG
    printf("==> Punc = %d\n", i_punct);
#endif
    for (int i_vert = 0; i_vert < 8; ++i_vert) {
      // Norm_inf
      Real norm_inf = -1;
      for (int i_diff = 0; i_diff < 3; ++ i_diff) {
        diff = std::abs(pmb->pmy_mesh->pz4c_tracker->pos_body[i_punct].pos[i_diff] - xv[i_vert*3+i_diff]);
#ifdef DEBUG
        printf("======> Coordpos = %g, coordblock = %g\n",pmb->pmy_mesh->pz4c_tracker->pos_body[i_punct].pos[i_diff], xv[i_vert*3+i_diff]);
#endif
        if (diff > norm_inf) {
          norm_inf = diff;
        }
#ifdef DEBUG
	printf("======> Dist = %g\n", diff);
#endif
      }
#ifdef DEBUG
      printf("====> Inf norm = %g\n", norm_inf);
#endif
      //Calculate minimum of the distances of the 8 vertices above
      if (dmin_punct > norm_inf) {
        dmin_punct = norm_inf;
      }
    }
#ifdef DEBUG
    printf("====> dmin_punct = %g\n", dmin_punct);
#endif
    //Calculate minimum of the distances between the n punctures
    if (d > dmin_punct) {
      d = dmin_punct;
    }
  }
#ifdef DEBUG
  printf("Min dist = %g\n", d);
#endif
  Real ratio = L/d;
  if (ratio < 1) return -1;
  //Calculate level that the block should be in, given a box-in-box theoretical structure of the grid
  Real th_level = std::floor(std::log2(ratio));
#ifdef DEBUG
  printf("Level = %d, th_level = %g\n", level, th_level);
  printf("<===================================================>\n");
#endif
  if (th_level > level) {
#ifdef DEBUG
    printf("Refine\n");
#endif
    return 1;
  } else if (th_level < level) {
#ifdef DEBUG
    printf("Derefine\n");
#endif
    return -1;
  } else
#ifdef DEBUG
    printf("Do nothing\n");
#endif
    return 0;

#else // Z4C_TRACKER
  return 0;
#endif // Z4C_TRACKER
}
*/
namespace{
Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i)
{
  return - SQR(gamma(0,2,i))*gamma(1,1,i) +
          2*gamma(0,1,i)*gamma(0,2,i)*gamma(1,2,i) -
          gamma(0,0,i)*SQR(gamma(1,2,i)) - SQR(gamma(0,1,i))*gamma(2,2,i) +
          gamma(0,0,i)*gamma(1,1,i)*gamma(2,2,i);
}

Real SpatialDet(Real gxx, Real gxy, Real gxz, Real gyy, Real gyz, Real gzz)
{
  return - SQR(gxz)*gyy+
          2*gxy*gxz*gyz -
          gxx*SQR(gyz) - SQR(gxy)*gzz +
          gxx*gyy*gzz;
}
Real Maxrho(MeshBlock *pmb, int iout) {
  Real max_rho = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
      }
    }
  }
  return max_rho;
}

Real Minalp(MeshBlock *pmb, int iout) {
  Real min_alp = 1.0e100;
  int is = pmb->is, ie = pmb->ie+1, js = pmb->js, je = pmb->je+1, ks = pmb->ks, ke = pmb->ke+1;
  AthenaArray<Real> alpha;
  alpha.InitWithShallowSlice(pmb->pz4c->storage.u,Z4c::I_Z4c_alpha,1);
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        min_alp = std::min(std::abs(alpha(k,j,i)), min_alp);
      }
    }
  }
  return min_alp;
}

Real L1rhodiff(MeshBlock *pmb, int iout) {
  Real L1rho = 0.0;
  Real vol,dx,dy,dz;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        dx = pmb->pcoord->dx1v(i);
        dy = pmb->pcoord->dx2v(j);
        dz = pmb->pcoord->dx3v(k);
        vol = dx*dy*dz;
        L1rho += std::abs(pmb->phydro->w(IDN,k,j,i) - pmb->phydro->w_init(IDN,k,j,i))*vol;
      }
    }
  }
  return L1rho;
}

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
        divB += ((pmb->pfield->b.x1f(k,j,i+1) - pmb->pfield->b.x1f(k,j,i))/dx + (pmb->pfield->b.x2f(k,j+1,i) - pmb->pfield->b.x2f(k,j,i))/(dy) + (pmb->pfield->b.x3f(k+1,j,i) - pmb->pfield->b.x3f(k,j,i))/(dz))*vol;
      }
    }
  }
  return divB;
}
}
