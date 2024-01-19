//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ahf.cpp
//  \brief implementation of the apparent horizon finder class
// Developed from BAM's AHmod, see also
//  https://git.tpi.uni-jena.de/sbernuzzi/ahfpy

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <cmath> // NAN

#define DEBUG_OUTPUT 0 

#include "ejecta.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/tensor.hpp"
#include "../coordinates/coordinates.hpp"

using namespace utils::tensor;

//----------------------------------------------------------------------------------------
//! \fn Ejecta::Ejecta(Mesh * pmesh, ParameterInput * pin, int n)
//  \brief class for ejecta extraction class
Ejecta::Ejecta(Mesh * pmesh, ParameterInput * pin, int n):
  pmesh(pmesh) 
{
  nrad = pin->GetOrAddInteger("ejecta", "nrad", 1);
  radius = pin->GetOrAddReal("ejecta", "radius", 1.0); 
  nr = n;
  std::string parname;
  std::string n_str = std::to_string(nr);
  
  ntheta = pin->GetOrAddInteger("ejecta", "ntheta", 10);
  nphi = pin->GetOrAddInteger("ejecta", "nphi", 6);
  if ((nphi+1)%2==0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Ejecta setup" << std::endl
        << "nphi must be even " << nphi << std::endl;
    ATHENA_ERROR(msg);
  }

  verbose = pin->GetOrAddBoolean("ejecta", "verbose", 0);
  root = pin->GetOrAddInteger("ejejcta", "mpi_root", 0);
  bitant = pin->GetOrAddBoolean("z4c", "bitant", false);
  
  parname = "compute_every_iter_";
  parname += n_str;
  compute_every_iter = pin->GetOrAddInteger("ejecta", parname, 1);

  parname = "start_time_";
  parname += n_str;
  start_time = pin->GetOrAddReal("ejecta", parname, std::numeric_limits<double>::max());

  start_time = 0.;
  parname = "stop_time_";
  parname += n_str;
  stop_time = pin->GetOrAddReal("ejecta", parname, -1.0);
  stop_time = 1000.;
  // the spherical grid is the same for all surfaces

  theta.NewAthenaArray(ntheta);
  phi.NewAthenaArray(nphi);
  rho.NewAthenaArray(ntheta, nphi);
  press.NewAthenaArray(ntheta, nphi);
  Y.NewAthenaArray(ntheta, nphi);
  vx.NewAthenaArray(ntheta, nphi);
  vy.NewAthenaArray(ntheta, nphi);
  vz.NewAthenaArray(ntheta, nphi);
  Bx.NewAthenaArray(ntheta, nphi);
  By.NewAthenaArray(ntheta, nphi);
  Bz.NewAthenaArray(ntheta, nphi);

  for (int i=0; i<ntheta; ++i) {
    theta(i)   = th_grid(i);
  }

  for (int j=0; j<nphi; ++j) {
    phi(j)   = ph_grid(j);
  }

  
  // Flag points existing on this mesh
  havepoint.NewAthenaArray(ntheta, nphi);

  // Prepare output
  parname = "ejecta_file_summary_";
  parname += n_str;
  ofname_summary = pin->GetString("job", "problem_id") + ".";
  ofname_summary += pin->GetOrAddString("ejecta", parname, "ejecta_summary_"+n_str);
  ofname_summary += ".txt";

#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ioproc = (root == rank);
#else
  ioproc = true;
#endif
  if (ioproc) {
    // Summary file
    bool new_file = true;
    if (access(ofname_summary.c_str(), F_OK) == 0) {
      new_file = false;
    }
    pofile_summary = fopen(ofname_summary.c_str(), "a");
    if (NULL == pofile_summary) {
      std::stringstream msg;
      msg << "### FATAL ERROR in AHF constructor" << std::endl;
      msg << "Could not open file '" << pofile_summary << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    if (new_file) {
      fprintf(pofile_summary, "# 1:iter 2:time 3:rho 4:press 5:Ye 6:vx 7:vy 8:vz 9:Bx 10:By 11:Bz\n");
      fflush(pofile_summary);
    }
  }
}

Ejecta::~Ejecta() {

  rho.DeleteAthenaArray();
  press.DeleteAthenaArray();
  Y.DeleteAthenaArray();
  vx.DeleteAthenaArray();
  vy.DeleteAthenaArray();
  vz.DeleteAthenaArray();
  Bx.DeleteAthenaArray();
  By.DeleteAthenaArray();
  Bz.DeleteAthenaArray();  
  // Flag points existing on this mesh
  havepoint.DeleteAthenaArray();

  // Close files
  if (ioproc) {
    fclose(pofile_summary);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Interp(MeshBlock * pmb)
// \brief interpolate quantities on the surface n
// Flag here the surface points contained in the MB
void Ejecta::Interp(MeshBlock * pmb)
{

  LagrangeInterpND<2*NGHOST-1, 3> * pinterp3 = nullptr;
  AthenaArray<Real> rho_, press_, Y_, vx_, vy_, vz_, Bx_, By_, Bz_;
  rho_.InitWithShallowSlice(pmb->phydro->w, IDN, 1);
  press_.InitWithShallowSlice(pmb->phydro->w, IPR, 1);
  //Y_.InitWithShallowSlice(pmb->pscalars->r, IYF, 1);
  vx_.InitWithShallowSlice(pmb->phydro->w, IVX, 1);
  vy_.InitWithShallowSlice(pmb->phydro->w, IVY, 1);
  vz_.InitWithShallowSlice(pmb->phydro->w, IVZ, 1);

  if (MAGNETIC_FIELDS_ENABLED) {
    Bx_.InitWithShallowSlice(pmb->pfield->bcc, IB1, 1);
    By_.InitWithShallowSlice(pmb->pfield->bcc, IB2, 1);
    Bz_.InitWithShallowSlice(pmb->pfield->bcc, IB3, 1);
  }

  // For interp
  Real origin[NDIM];
  Real delta[NDIM];
  int size[NDIM];
  Real coord[NDIM];
  
  for (int i=0; i<ntheta; i++) {
    Real sinth = std::sin(theta(i));
    Real costh = std::cos(theta(i));
    for (int j=0; j<nphi; j++) {
      Real sinph = std::sin(phi(j));
      Real cosph = std::cos(phi(j));
      
      // Global coordinates of the surface
      Real x = radius * sinth * cosph;
      Real y = radius * sinth * sinph;
      Real z = radius * costh;

      // Impose bitant symmetry below
      bool bitant_sym = ( bitant && z < 0 ) ? true : false;
      // Associate z -> -z if bitant
      if (bitant) z = std::abs(z);

      if (!pmb->PointContained(x,y,z)) continue;

      // this surface point is in this MB
      havepoint(i,j) += 1;
      
      // Interpolate
      origin[0] = pmb->pcoord->x1v(0);
      size[0]   = pmb->ncells1;
      delta[0]  = pmb->pcoord->dx1v(0);
      coord[0]  = x;
      
      origin[1] = pmb->pcoord->x2v(0);
      size[1]   = pmb->ncells2;
      delta[1]  = pmb->pcoord->dx2v(0);
      coord[1]  = y;
      
      origin[2] = pmb->pcoord->x3v(0);
      size[2]   = pmb->ncells3;
      delta[2]  = pmb->pcoord->dx3v(0);
      coord[2]  = z;
        
      pinterp3 =  new LagrangeInterpND<2*NGHOST-1, 3>(origin, delta, size, coord);

      rho(i,j) = pinterp3->eval(&(rho_(0,0,0)));
      press(i,j) = pinterp3->eval(&(press_(0,0,0)));
      //Y(i,j) = pinterp3->eval(&(Y_(0,0,0)));
      vx(i,j) = pinterp3->eval(&(vx_(0,0,0)));
      vy(i,j) = pinterp3->eval(&(vy_(0,0,0)));
      vz(i,j) = pinterp3->eval(&(vz_(0,0,0)));
      if (MAGNETIC_FIELDS_ENABLED) {
        Bx(i,j) = pinterp3->eval(&(Bx_(0,0,0)));
        By(i,j) = pinterp3->eval(&(By_(0,0,0)));
        Bz(i,j) = pinterp3->eval(&(Bz_(0,0,0)));
      }
    
      delete pinterp3;
    } // phi loop
  } // theta loop
  
  rho_.DeleteAthenaArray();
  press_.DeleteAthenaArray();
  //Y_.DeleteAthenaArray();
  vx_.DeleteAthenaArray();
  vy_.DeleteAthenaArray();
  vz_.DeleteAthenaArray();
  Bx_.DeleteAthenaArray();
  By_.DeleteAthenaArray();
  Bz_.DeleteAthenaArray();

}

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Calculate(int iter, Real time)
// \brief Calculate ejecta quantities
void Ejecta::Calculate(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return;
  if (iter % compute_every_iter != 0) return;
  MeshBlock * pmb = pmesh->pblock;
  while (pmb != nullptr) {
    Interp(pmb);
    pmb = pmb->next;
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Write(int iter, Real time)
// \brief Output ejecta quantities
// void Ejecta::Write_txt(int iter, Real time)
// {
//   if((time < start_time) || (time > stop_time)) return;
//   if (iter % compute_every_iter != 0) return;
//   MeshBlock * pmb = pmesh->pblock;

//   for (int i=0; i<ntheta; i++) {
//     Real theta = th_grid(i);
//     Real sinth = std::sin(theta);
//     Real costh = std::cos(theta);
//     for (int j=0; j<nphi; j++) {
//       Real phi   = ph_grid(j);
      
//       std::stringstream ss_i, ss_j;
//       ss_i << std::setw(3) << std::setfill('0') << i;
//       ss_j << std::setw(3) << std::setfill('0') << j;
//       std::string s_i = ss_i.str();
//       std::string s_j = ss_j.str();
//       std::ofstream outfile;
//       outfile.open("quantities_" + s_i + "_" + s_j + ".dat", std::ios_base::app);
//       Real tt = pmb->pmy_mesh->time;
//       outfile << tt << " " << theta << " " << phi << " "
//                     << rho(i,j) << " "
//                     << press(i,j) << " "
//                     << vx(i, j) << " " 
//                     << vy(i, j) << " "
//                     << vz(i, j) << " "
//                     << Bx(i, j) << " "
//                     << By(i, j) << " "
//                     << Bz(i, j) << " " << std::endl;
//       outfile.close();
//     }
//   }
// }

//----------------------------------------------------------------------------------------
// \!fn void Ejecta::Write(int iter, Real time)
// \brief Output ejecta quantities
void Ejecta::Write(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return;
  if (iter % compute_every_iter != 0) return;
  MeshBlock * pmb = pmesh->pblock;

  std::stringstream ss_i;
  ss_i << std::setw(6) << std::setfill('0') << iter;
  std::string s_i = ss_i.str();
  std::string filename = "ejecta_" + s_i + ".h5";
  hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t prim_id = H5Gcreate1(file, "prim", H5P_DEFAULT);
  //Create the data space for the data set//
  hsize_t dim_1[1], dim_2[2];
  hid_t dataset, dataspace;

  // Time
  dim_1[0] = 1;
  dataspace = H5Screate_simple(1, dim_1, NULL);
  Real tvec[1] = {time};
  dataset = H5Dcreate(file, "time", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tvec);
  H5Dclose(dataset);

  // Theta
  dim_1[0] = ntheta;
  dataspace = H5Screate_simple(1, dim_1, NULL);
  dataset = H5Dcreate(file, "theta", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, theta.data());
  H5Dclose(dataset);

  // Phi
  dim_1[0] = nphi;
  dataspace = H5Screate_simple(1, dim_1, NULL);
  dataset = H5Dcreate(file, "phi", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, phi.data());
  H5Dclose(dataset);

  // Primitive variables
  dim_2[0] = ntheta;
  dim_2[1] = nphi;
  dataspace = H5Screate_simple(2, dim_2, NULL);

  // rho
  dataset = H5Dcreate(file, "prim/rho", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rho.data());
  H5Dclose(dataset);

  // press
  dataset = H5Dcreate(file, "prim/press", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, press.data());
  H5Dclose(dataset);

  // vx
  dataset = H5Dcreate(file, "prim/vx", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vx.data());
  H5Dclose(dataset);

  // vy
  dataset = H5Dcreate(file, "prim/vy", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vy.data());
  H5Dclose(dataset);

  // vz
  dataset = H5Dcreate(file, "prim/vz", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vz.data());
  H5Dclose(dataset);

  // Bx
  dataset = H5Dcreate(file, "prim/Bx", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Bx.data());
  H5Dclose(dataset);

  // By
  dataset = H5Dcreate(file, "prim/By", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, By.data());
  H5Dclose(dataset);

  // Bz
  dataset = H5Dcreate(file, "prim/Bz", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Bz.data());
  H5Dclose(dataset);

  H5Sclose(dataspace);
  H5Gclose(prim_id);
  H5Fclose(file);
}


//----------------------------------------------------------------------------------------
// \!fn int AHF::tpindex(const int i, const int j)
// \brief spherical grid single index (i,j) -> index
int Ejecta::tpindex(const int i, const int j)
{
  return i*nphi + j; 
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::th_grid(const int i)
// \brief theta coordinate from index
Real Ejecta::th_grid(const int i)
{
  Real dtheta = dth_grid();
  return dtheta*(0.5 + i);
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::ph_grid(const int i)
// \brief phi coordinate from index
Real Ejecta::ph_grid(const int j)
{
  Real dphi = dph_grid();
  return dphi*(0.5 + j);
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::dth_grid()
// \brief compute spacing dtheta 
Real Ejecta::dth_grid()
{
  return PI/ntheta;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::dph_grid()
// \brief compute spacing dphi

Real Ejecta::dph_grid()
{
  return 2.0*PI/nphi;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::SpatialDet(Real gxx, ... , Real gzz)
// \brief returns determinant of 3-metric
// Taken from class Z4c 

Real Ejecta::SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz)
{
  return - SQR(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQR(gyz) - SQR(gxy)*gzz + gxx*gyy*gzz;
}

//----------------------------------------------------------------------------------------
// \!fn void AHF::SpatialInv(Real const detginv,
//           Real const gxx, Real const gxy, Real const gxz,
//           Real const gyy, Real const gyz, Real const gzz,
//           Real * uxx, Real * uxy, Real * uxz,
//           Real * uyy, Real * uyz, Real * uzz)
// \brief returns inverse of 3-metric
// Taken from class Z4c

void Ejecta::SpatialInv(Real const detginv,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real * uxx, Real * uxy, Real * uxz,
                     Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}

//----------------------------------------------------------------------------------------
// \!fn Real AHF::Trace(Real detginv, Real gxx, ... , Real gzz, Real Axx, ..., Real Azz)
// \brief returns Trace of extrinsic curvature
// Taken from class Z4c

Real Ejecta::Trace(Real const detginv,
                Real const gxx, Real const gxy, Real const gxz,
                Real const gyy, Real const gyz, Real const gzz,
                Real const Axx, Real const Axy, Real const Axz,
                Real const Ayy, Real const Ayz, Real const Azz)
{
  return (detginv*(
       - 2.*Ayz*gxx*gyz + Axx*gyy*gzz +  gxx*(Azz*gyy + Ayy*gzz)
       + 2.*(gxz*(Ayz*gxy - Axz*gyy + Axy*gyz) + gxy*(Axz*gyz - Axy*gzz))
       - Azz*SQR(gxy) - Ayy*SQR(gxz) - Axx*SQR(gyz)
       ));
}
