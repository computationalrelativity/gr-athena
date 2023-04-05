#include <algorithm> // for fill
#include <cmath>     // for NAN
#include <stdexcept>
#include <sstream>
#include <unistd.h> // for F_OK
#include <fstream> 

#define H5_USE_16_API (1)
#include <hdf5.h>


#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "cce.hpp"
#include "matrix.hpp"
#include "sYlm.hpp"
#include "myassert.hpp"
#include "decomp.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../utils/lagrange_interp.hpp"
#include "../../globals.hpp"
#include "../z4c.hpp"

#define BUFFSIZE  (1024)
#define MAX_RADII (100)
#define BOOKKEEPING_NAME "cce_bookkeeping.txt"
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

#define HDF5_ERROR(fn_call)                                           \
{                                                                     \
  /* ex: error_code = group_id = H5Gcreate(file_id, metaname, 0) */   \
  hid_t _error_code = fn_call;                                        \
  if (_error_code < 0)                                                \
  {                                                                   \
    cerr << "File: " << __FILE__ << "\n"                              \
         << "line: " << __LINE__ << "\n"                              \
         << "HDF5 call " << #fn_call << ", "                          \
         << "returned error code: " << (int)_error_code << ".\n";     \
         exit((int)_error_code);                                      \
  }                                                                   \
}

using namespace decomp_matrix_class;
using namespace decomp_sYlm;
using namespace decomp_decompose;


static int output_3Dmodes(const int iter,
       const char *dir,
       const char* name,
       const int obs, double time,
       int s, int nl,
       int nn, double rin, double rout,
       const double *re, const double *im);


CCE::CCE(Mesh *const pm, ParameterInput *const pin, std::string name, int rn):
    pm(pm),
    pin(pin),
    fieldname(name),
    spin(0),
    dinfo_pp(nullptr),
    rn(rn)
{
  output_dir = pin->GetString("cce","output_dir");
  bfname = output_dir + "/" + BOOKKEEPING_NAME;
  rin  = pin->GetReal("cce", "rin_"  + std::to_string(rn));
  rout = pin->GetReal("cce", "rout_" + std::to_string(rn));
  num_mu_points  = pin->GetOrAddInteger("cce","num_theta",41);
  num_phi_points = pin->GetOrAddInteger("cce","num_phi",82);
  num_x_points   = pin->GetOrAddInteger("cce","num_r_inshell",28);
  num_l_modes    = pin->GetOrAddInteger("cce","num_l_modes",7);
  num_n_modes    = pin->GetOrAddInteger("cce","num_radial_modes",7);
  nangle = num_mu_points*num_phi_points;
  npoint = nangle*num_x_points;

  double *radius    = nullptr;
  double *mucolloc  = nullptr;
  double *phicolloc = nullptr;
  myassert (ABS(spin) <= MAX_SPIN);

  nlmmodes = num_l_modes*(num_l_modes+2*ABS(MAX_SPIN));
  dinfo_pp = new const decomp_info* [2*MAX_SPIN+1];
  myassert(dinfo_pp);
  for (int s=-MAX_SPIN; s<=MAX_SPIN; s++)
  {
    dinfo_pp[s+MAX_SPIN] = nullptr;
  }

  // alloc
  radius = new double [num_x_points];
  xb = new double [nangle*num_x_points];
  yb = new double [nangle*num_x_points];
  zb = new double [nangle*num_x_points];
  mucolloc = new double [nangle];
  phicolloc = new double [nangle];
  ifield = new double [nangle*num_x_points]();
 
  myassert(radius);
  myassert(xb);
  myassert(yb);
  myassert(zb);
  myassert(mucolloc);
  myassert(phicolloc);
  myassert(ifield);

  std::fill(radius, radius + (num_x_points),NAN); // init to nan
  std::fill(xb, xb + (nangle*num_x_points),NAN); // init to nan
  std::fill(yb, yb + (nangle*num_x_points),NAN); // init to nan
  std::fill(zb, zb + (nangle*num_x_points),NAN); // init to nan
  std::fill(mucolloc,  mucolloc  + (nangle),NAN); // init to nan
  std::fill(phicolloc, phicolloc + (nangle),NAN); // init to nan

  if (! dinfo_pp[MAX_SPIN + spin])
  {
    dinfo_pp[MAX_SPIN + spin] =  initialize(spin, num_l_modes, num_n_modes,
                                   num_mu_points, num_phi_points, num_x_points);
    myassert(dinfo_pp[MAX_SPIN + spin]);
  }

  dinfo_pp[MAX_SPIN + spin]->get_ncolloc(num_x_points, radius);
  dinfo_pp[MAX_SPIN + spin]->get_mucolloc(nangle, mucolloc);
  dinfo_pp[MAX_SPIN + spin]->get_phicolloc(nangle, phicolloc);

  for (int k=0; k < num_x_points; k++)
  {
    double xk = radius[k];
    radius[k] = 0.5 * ( (rout - rin) * xk + (rout + rin) );
  }

  for (int k=0; k < num_x_points; k++)
  {
    for (int i=0; i < nangle; i++)
    {
      const double phi = phicolloc[i];
      const double mu = mucolloc[i];

      const double sph = sin(phi);
      const double cph = cos(phi);
      const double cth = mu;
      const double sth = sqrt(1.0 - mu*mu);

      const int indx = i + k*nangle;
      
      xb[indx] = radius[k] * sth*cph;
      yb[indx] = radius[k] * sth*sph;
      zb[indx] = radius[k] * cth;
    
    }
  }
    
  // free mem.
  delete [] mucolloc;
  delete [] phicolloc;
  delete [] radius;
}

CCE::~CCE()
{
  delete [] ifield;
  delete [] xb;
  delete [] yb;
  delete [] zb;

  for (int i = 0; i < 2*MAX_SPIN+1; ++i)
    delete [] dinfo_pp[i];
  delete [] dinfo_pp;
}

void CCE::InterpolateSphToCart(MeshBlock *const pmb)
{
  const int Npoints = nangle*num_x_points;
  Real const origin[3] = {pmb->pcoord->x1f(0),
                          pmb->pcoord->x2f(0),
                          pmb->pcoord->x3f(0)};
  Real const delta[3]  = {pmb->pcoord->dx1f(0),
                          pmb->pcoord->dx2f(0),
                          pmb->pcoord->dx3f(0)};
  int const size[3]    = {pmb->nverts1,
                          pmb->nverts2,
                          pmb->nverts3};

  // find the src field
  Real *src_field = nullptr;
  Z4c *const pz4c = pmb->pz4c;
  Z4c::Z4c_vars z4c;
  Z4c::ADM_vars adm;
  pz4c->SetZ4cAliases(pz4c->storage.u, z4c);
  pz4c->SetADMAliases(pz4c->storage.adm, adm);
  
  if (fieldname == "gxx")
  {
    src_field = &adm.g_dd(0,0, 0,0,0);
  }
  else if (fieldname == "gxy")
  {
    src_field = &adm.g_dd(0,1, 0,0,0);
  }
  else if (fieldname == "gxz")
  {
    src_field = &adm.g_dd(0,2, 0,0,0);
  }
  else if (fieldname == "gyy")
  {
    src_field = &adm.g_dd(1,1, 0,0,0);
  }
  else if (fieldname == "gyz")
  {
    src_field = &adm.g_dd(1,2, 0,0,0);
  }
  else if (fieldname == "gzz")
  {
    src_field = &adm.g_dd(2,2, 0,0,0);
  }
  else if (fieldname == "betax")
  {
    src_field = &z4c.beta_u(0, 0, 0, 0);
  }
  else if (fieldname == "betay")
  {
    src_field = &z4c.beta_u(1, 0, 0, 0);
  }
  else if (fieldname == "betaz")
  {
    src_field = &z4c.beta_u(2, 0, 0, 0);
  }
  else if (fieldname == "alp")
  {
    src_field = &z4c.alpha(0, 0, 0);
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in CCE interpolation" << std::endl;
    msg << "Could not find '" << fieldname << "' for interpolation!";
    throw std::runtime_error(msg.str().c_str());
  }
  
  
  for (int p = 0; p < Npoints; ++p)
  {
    double coord[3] = {xb[p], yb[p], zb[p]};
    if (pmb->PointContained(coord[0], coord[1], coord[2]))
    {
      LagrangeInterpND<2*NGHOST-1, 3> linterp(origin, delta, size, coord);

// note: the point may take place at the boundary of two meshblocks
#pragma omp atomic write
      ifield[p] = linterp.eval(src_field);
      
      // printf("f(%g,%g,%g) = %g\n",coord[0],coord[1],coord[2],ifield[p]);
    }
  }
}

void CCE::ReduceInterpolation(){}

void CCE::DecomposeAndWrite()
{
  if (0 == Globals::my_rank)
  {
    // which write iter I am;
    int iter = 0;//pin->GetOrAddInteger("cce","write_iter",0);
    // create workspace
    Real *re_m = new Real [nlmmodes*num_x_points];
    Real *im_m = new Real [nlmmodes*num_x_points];
    Real *im_f = new Real [nangle*num_x_points](); // init to zero
    myassert(re_m);
    myassert(im_m);
    myassert(im_f);
    std::fill(re_m, re_m + (nlmmodes*num_x_points),NAN); // init to nan
    std::fill(im_m, im_m + (nlmmodes*num_x_points),NAN); // init to nan
    
    // decompose the re_f, note im_f is zero 
    Real *const re_f = ifield;
    decompose3D(dinfo_pp[MAX_SPIN + spin], re_f, im_f, re_m, im_m);

    // dump the modes into an h5 file
    output_3Dmodes(iter, output_dir.c_str(), fieldname.c_str(), rn, pm->time, 
       spin, num_l_modes, num_n_modes, rin, rout, re_m, im_m);

    // free workspace
    delete [] re_m;
    delete [] im_m;
    delete [] im_f;
    
    // increase write iter
    iter++;
    
  }
}

// write the decomposed field in an h5 file.
static int output_3Dmodes(const int iter/* output iteration */, const char *dir,
  const char* name, const int obs, double time,
  int s, int nl,
  int nn, double rin, double rout,
  const double *re, const double *im)
{
  char filename[BUFFSIZE];
  hid_t   file_id;
  hsize_t dims[2];
  herr_t  status;

  snprintf(filename, sizeof filename, "%s/cce_decomp_shell_%d.h5", dir, obs);

  const int nlmmodes = nl*(nl+2*ABS(s));
  dims[0] = nn;
  dims[1] = nlmmodes;

  static int FirstCall = 1;
  static int last_dump[MAX_RADII];

  const int dump_it = iter;
  hid_t dataset_id, attribute_id, dataspace_id, group_id;

  if (FirstCall)
  {
    FirstCall = 0;
    for (unsigned int i=0; i < sizeof(last_dump) / sizeof(*last_dump); i++)
    {
      last_dump[i] = -1000;
    }
  }

  bool file_exists = false;
  H5E_BEGIN_TRY {
     file_exists = H5Fis_hdf5(filename) > 0;
  } H5E_END_TRY;

  if(file_exists)
  {
    file_id = H5Fopen (filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0)
    {
      cerr << "Failed to open hdf5 file";
      exit((int)file_id);
    }
  }
  else
  {
    file_id = H5Fcreate (filename, H5F_ACC_TRUNC,
           H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
      cerr << "Failed to create hdf5 file";
      exit((int)file_id);
    }

    char metaname[]="/metadata";
    HDF5_ERROR(group_id = H5Gcreate(file_id, metaname, 0));

    int ds[2] = {nn, nlmmodes};
    hsize_t oD2 = 2;
    hsize_t oD1 = 1;

    HDF5_ERROR(dataspace_id =  H5Screate_simple(1, &oD2, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(group_id, "dim", H5T_NATIVE_INT,
                   dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(status = H5Awrite(attribute_id, H5T_NATIVE_INT, ds));
    HDF5_ERROR(status = H5Aclose(attribute_id));
    HDF5_ERROR(status = H5Sclose(dataspace_id));

    HDF5_ERROR(dataspace_id =  H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(group_id, "spin", H5T_NATIVE_INT,
                    dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(status = H5Awrite(attribute_id, H5T_NATIVE_INT, &s));
    HDF5_ERROR(status = H5Aclose(attribute_id));
    HDF5_ERROR(status = H5Sclose(dataspace_id));

    HDF5_ERROR(dataspace_id =  H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(group_id, "Rin", H5T_NATIVE_DOUBLE,
                    dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &rin));
    HDF5_ERROR(status = H5Aclose(attribute_id));
    HDF5_ERROR(status = H5Sclose(dataspace_id));

    HDF5_ERROR(dataspace_id =  H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(group_id, "Rout", H5T_NATIVE_DOUBLE,
                    dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &rout));
    HDF5_ERROR(status = H5Aclose(attribute_id));
    HDF5_ERROR(status = H5Sclose(dataspace_id));

    HDF5_ERROR(H5Gclose(group_id));

  }

  char buff[BUFFSIZE];
  // NOTE: the dump_it should be the same for all vars that's why we need this
  if (dump_it > last_dump[obs])
  {
    hsize_t oneD = 1;
    snprintf(buff, sizeof buff, "/%d", dump_it);
    HDF5_ERROR(group_id = H5Gcreate(file_id, buff, 0));
    HDF5_ERROR(dataspace_id =  H5Screate_simple(1, &oneD, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(group_id, "Time", H5T_NATIVE_DOUBLE,
      dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &time));
    HDF5_ERROR(status = H5Aclose(attribute_id));
    HDF5_ERROR(status = H5Sclose(dataspace_id));
    HDF5_ERROR(H5Gclose(group_id));

  }
  last_dump[obs] = dump_it;

  snprintf(buff, sizeof buff, "/%d/%s", dump_it, name);
  HDF5_ERROR(group_id = H5Gcreate(file_id, buff, 0));
  HDF5_ERROR(H5Gclose(group_id));


  snprintf(buff, sizeof buff, "/%d/%s/re", dump_it, name);
  HDF5_ERROR(dataspace_id =  H5Screate_simple(2, dims, NULL));
  HDF5_ERROR(dataset_id =  H5Dcreate(file_id, buff, H5T_NATIVE_DOUBLE,
         dataspace_id, H5P_DEFAULT));
  HDF5_ERROR(status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,
                   H5S_ALL, H5P_DEFAULT, re));
  HDF5_ERROR(status = H5Dclose(dataset_id));
  HDF5_ERROR(status = H5Sclose(dataspace_id));

  
  snprintf(buff, sizeof buff, "/%d/%s/im", dump_it, name);
  HDF5_ERROR(dataspace_id =  H5Screate_simple(2, dims, NULL));
  HDF5_ERROR(dataset_id =  H5Dcreate(file_id, buff, H5T_NATIVE_DOUBLE,
         dataspace_id, H5P_DEFAULT));
  HDF5_ERROR(status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,
                   H5S_ALL, H5P_DEFAULT, im));
  HDF5_ERROR(status = H5Dclose(dataset_id));
  HDF5_ERROR(status = H5Sclose(dataspace_id));

  HDF5_ERROR(H5Fclose(file_id));

  return 0;
}

// save the last iteration in a text file to prevent error of duplicated entries in
// the h5 files. 
void CCE::BookKeeping(ParameterInput *const pin)
{
  if (0 != Globals::my_rank) return;
    
  std::string fname = pin->GetString("cce","output_dir") + "/" + BOOKKEEPING_NAME;

  // if this is the first time, create a bookkeeping file
  if (access(fname.c_str(), F_OK) != 0) 
  {
    FILE *file = fopen(fname.c_str(), "w");
    if (file == nullptr) 
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in CCE " << std::endl;
      msg << "Could not open file '" << fname << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    fprintf(file, "0");
    fclose(file);
  }
  // update the iter
  else
  {
    std::fstream file;
    int iter;

    // first read the iter value
    file.open(fname, std::ios::in);
    if (!file) 
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in CCE " << std::endl;
      msg << "Could not open file '" << fname << "' for reading!";
      throw std::runtime_error(msg.str().c_str());
    }
    file >> iter;
    file.close();
    
    // now open a fresh file and update iter
    // first read the iter value
    file.open(fname, std::ios::out);
    if (!file) 
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in CCE " << std::endl;
      msg << "Could not open file '" << fname << "' for writing!";
      throw std::runtime_error(msg.str().c_str());
    }
    iter++;
    file << iter << std::endl;
    file.close();
  }
}

