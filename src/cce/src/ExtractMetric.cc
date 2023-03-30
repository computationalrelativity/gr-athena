//#include "cctk.h"
//#include "cctk_Arguments.h"
//#include "cctk_Parameters.h"
//#include "util_Table.h"
//#include "util_ErrorCodes.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#define H5_USE_16_API 1
#include <hdf5.h>
#include "myassert.h"
#include "sYlm.hh"
#include "SphericalHarmonicDecomp.h"
#include "decomp.hh"

/* vars and params to change for each code

- all //! comments
- Code_* macros

*/

#define Code_mesh       const MeshBlock *mb
#define Code_time       1 // ex: mb->pmy_mesh->time
#define Code_iteration  1 // ex: mb->pmy_mesh->ncycle
#define Code_write_freq 1 // ex: from param file
#define Code_max_spin   0 // ex: from param file
#define Code_proc_rank  0 // ex: Globals::my_rank
#define Code_num_radii  2 // ex: from param file. number of extraction radii
#define Code_out_dir    nullptr // ex: path/to/output dir
#define Code_num_mu_points  1 // ex: form param file
#define Code_num_phi_points 1 // ex: from param file
#define Code_num_x_points   1 // ex: from param file
#define Code_num_l_modes    1 // ex: from param file
#define Code_num_n_modes    1 // ex: from param file
#define Code_Rout(i)        1.0 // ex: from param file getd("Rout"#i);
#define Code_Rin(i)         0.5 // ex: from param file getd("Rin"#i);

class MeshBlock;

#define HDF5_ERROR(fn_call)                                           \
{                                                                     \
  /* ex: error_code = group_id = H5Gcreate(file_id, metaname, 0) */   \
  hid_t _error_code = fn_call;                                        \
  if (_error_code < 0)                                                \
  {                                                                   \
    cerr << "File: " << __FILE__ << "\n"                              \
         << "line: " << __LINE__ << "\n"                              \
         << "HDF5 call " << #fn_call << ","                           \
         << "returned error code: " << (int)_error_code << ".\n";     \
         exit((int)_error_code);                                      \
  }                                                                   \
}


#ifdef USE_LEGENDRE
#  include "Legendre.hh"
#endif

// instead of interpolation, fill the data for test purposes
#undef TEST_DECOMP

#define Max(a_,b_) ((a_)>(b_)? (a_):(b_))
#define Min(a_,b_) ((a_)<(b_)? (a_):(b_))
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))


#ifdef TEST_DECOMP
static void fill_in_data(double time, int s, int nl, int nn,
    int npoints, double Rin, double Rout,
    const double xb[], const double yb[],
    const double zb[], double re[], double im[]);
#endif

/////////////////
#if 0
 static int interp_fields(Code_mesh,
          int MyProc,
          int re_inxd, int im_indx,
          int num_points, 
          const double * xc,
          const double * yc,
          const double * zc,
          double *re_f,
          double *im_f);
#endif 
/////////////////

static int output_3Dmodes(const int iter,
  const char *dir,
  const char* name,
       const int obs,  int it, double time,
       int s, int nl,
       int nn, double rin, double rout,
       const double *re, const double *im);

static int Decompose3D (
       Code_mesh,
       const char *name,
       int re_gindx,
       const int iter)
{
  using namespace decomp_matrix_class;
  using namespace decomp_sYlm;
  using namespace decomp_decompose;
  using namespace std;

  const double evo_time = Code_time;
  const int iteration   = Code_iteration;
  const char *const out_dir = Code_out_dir;
  const int max_spin  = Code_max_spin;
  const int num_radii = Code_num_radii;
  const int num_mu_points  = Code_num_mu_points;
  const int num_phi_points = Code_num_phi_points;
  const int num_x_points   = Code_num_x_points;
  const int num_l_modes    = Code_num_l_modes;
  const int num_n_modes    = Code_num_n_modes;
  
  const int spin = 0;
  const int im_gindx = -1;

  static const decomp_info **dinfo_pp = NULL;
  static int FirstTime = 1;

#define MAX_RADII  100
  static double *xb[MAX_RADII];
  static double *yb[MAX_RADII];
  static double *zb[MAX_RADII];
  static double *radius[MAX_RADII];

  static double *re_f = NULL;
  static double *im_f = NULL;

  static double *re_m = NULL;
  static double *im_m = NULL;

  static double *mucolloc = NULL;
  static double *phicolloc = NULL;


  myassert (ABS(spin) <= max_spin);
  myassert (num_radii <= int(sizeof(radius) / sizeof(*radius)));

  static int MyProc = Code_proc_rank;
  const char *outdir = out_dir;

  const int nangle = num_mu_points*num_phi_points;
  
  if (FirstTime)
  {
    for (unsigned int i=0; i < sizeof(radius) / sizeof(*radius); i++)
    {
      xb[i] = NULL;
      yb[i] = NULL;
      zb[i] = NULL;
      radius[i] = NULL;
    }
  }

  if (!MyProc)
  {
    if (FirstTime)
    {
      const int nlmmodes = num_l_modes*(num_l_modes+2*ABS(max_spin));
      FirstTime = 0;
      dinfo_pp = new  const decomp_info* [2*max_spin+1];
      myassert(dinfo_pp);
      for (int s=-max_spin; s<=max_spin; s++)
      {
	dinfo_pp[s+max_spin] = NULL;
      }

      for (int i=0; i < num_radii; i++)
      {
        radius[i] = new double [num_x_points];
        xb[i] = new double [nangle*num_x_points];
        yb[i] = new double [nangle*num_x_points];
        zb[i] = new double [nangle*num_x_points];

        myassert(radius[i]);
        myassert(xb[i]);
        myassert(yb[i]);
        myassert(zb[i]);
      }

      re_f = new double [nangle*num_x_points];
      im_f = new double [nangle*num_x_points];

      mucolloc = new double [nangle];
      phicolloc = new double [nangle];
      re_m = new double [nlmmodes*num_x_points];
      im_m = new double [nlmmodes*num_x_points];


      myassert(re_f && im_f &&
             mucolloc && phicolloc && re_m && im_m);
    }


    if (! dinfo_pp[max_spin + spin])
    {
      dinfo_pp[max_spin + spin] = 
         initialize(spin, num_l_modes, num_n_modes,
         num_mu_points, num_phi_points, num_x_points);
      myassert(dinfo_pp[max_spin + spin]);
    }

    dinfo_pp[max_spin + spin]->get_ncolloc(num_x_points, radius[0]);
    dinfo_pp[max_spin + spin]->get_mucolloc(nangle, mucolloc);
    dinfo_pp[max_spin + spin]->get_phicolloc(nangle, phicolloc);

    for (int k=0; k < num_x_points; k++)
    {
      double xk = radius[0][k];
      for (int i=0; i < num_radii; i++)
      {
        radius[i][k] = 0.5 * ((Code_Rout(i) - Code_Rin(i)) * xk +
            (Code_Rout(i) + Code_Rin(i)));
      }
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
        for (int l = 0; l < num_radii; l++)
        {
          xb[l][indx] = radius[l][k] * sth*cph;
          yb[l][indx] = radius[l][k] * sth*sph;
          zb[l][indx] = radius[l][k] * cth;
        }
      }
    }
  }

  for (int obs = 0; obs < num_radii; obs++)
  {

#ifdef TEST_DECOMP
    if (!MyProc)
    {
      fill_in_data(evo_time, spin, num_l_modes, num_n_modes,
         nangle*num_x_points, Code_Rin(obs), Code_Rout(obs), xb[obs], yb[obs],
       zb[obs], re_f, im_f);
    }
#else
////////
    //interp_fields(cctkGH, MyProc, re_gindx, im_gindx,
      //  nangle*num_x_points, xb[obs], yb[obs], zb[obs], re_f, im_f);
////////
#endif

    if (!MyProc)
    {
      decompose3D(dinfo_pp[max_spin + spin], re_f, im_f, re_m, im_m);

      output_3Dmodes(iter, outdir, name, obs,
         iteration, evo_time, 
         spin, num_l_modes, num_n_modes, Code_Rin(obs), Code_Rout(obs),
         re_m, im_m);
    }
  }
  return 0;
}



#define BUFFSIZE 1024

static int output_3Dmodes(const int iter,
  const char *dir,
  const char* name, const int obs,  int it, double time,
       int s, int nl,
       int nn, double rin, double rout,
       const double *re, const double *im)
{
  char filename[BUFFSIZE];
  hid_t   file_id;
  hsize_t dims[2];
  herr_t  status;


  snprintf(filename, sizeof filename,
        "%s/%s_obs_%d_Decomp.h5", dir, "metric", obs);

  const int nlmmodes = nl*(nl+2*ABS(s));
  dims[0] = nn;
  dims[1] = nlmmodes;

  int error_count = 0;
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
      exit(file_id);
    }
  }
  else
  {
    file_id = H5Fcreate (filename, H5F_ACC_TRUNC,
           H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
      cerr << "Failed to create hdf5 file";
      exit(file_id);
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

///////////// 
#if 0
static int interp_fields(Code_mesh,
          int MyProc,
          int num_points, 
          const double * xc,
          const double * yc,
          const double * zc,
          double *re_f,
          double *im_f)
{
  static int operator_handle = -1;
  static int coord_handle = -1;
  static int param_table_handle = -1;

  int N_dims = 3;
  int N_interp_points    = MyProc ? 0: num_points; /* only proc 0 requests points*/
  int N_input_arrays     = im_indx < 0 ? 1 : 2;
  int N_output_arrays    = im_indx < 0 ? 1 : 2;

  const int interp_coords_type_code = CCTK_VARIABLE_REAL;
  const void *const interp_coords[3] =
	{(void *)xc, (void *)yc, (void *)zc};

  const int input_array_variable_indices[2]=
      {re_indx, im_indx};
  const int output_array_type_codes[2]=
	{CCTK_VARIABLE_REAL, CCTK_VARIABLE_REAL};
  void *const output_arrays[2]={(void *)re_f, (void*)im_f};

  if (operator_handle < 0)
  {
    operator_handle = CCTK_InterpHandle(
           "Lagrange polynomial interpolation (tensor product)");

    if (operator_handle < 0 )
      CCTK_WARN(0, "cound not get interpolator handle");

    param_table_handle = Util_TableCreateFromString("order=3"); /*4th order error*/
    if (param_table_handle < 0 )
      CCTK_WARN(0, "cound not get parameter table handle");

    coord_handle = CCTK_CoordSystemHandle ("cart3d");

   if (coord_handle < 0 )
      CCTK_WARN(0, "could net get coord handle");
  }

  if(CCTK_InterpGridArrays(cctkGH, N_dims, operator_handle,
			   param_table_handle,
			   coord_handle, N_interp_points,
			   interp_coords_type_code,
			   interp_coords,
			   N_input_arrays, input_array_variable_indices,
			   N_output_arrays, output_array_type_codes,
			   output_arrays))
  {
    CCTK_WARN(0, "Interpolation error ");
    return -1;
  }

  if ((!MyProc) && N_input_arrays ==1)
  {
    for (int i=0; i < num_points; i++)
    {
      im_f[i] = 0.0;
    }
  }
  
  return 0;
}

#endif 
////////////////////
#ifdef TEST_DECOMP
static void fill_in_data(double time, int s, int nl, int nn,
    int npoints, double Rin, double Rout,
    const double xb[], const double yb[],
    const double zb[], double re[], double im[])
{
  using namespace decomp_sYlm;
#ifdef USE_LEGENDRE
  using namespace decomp_Legendre;
#else
  using namespace decomp_Chebyshev;
CCTK_WARN(CCTK_WARN_ALERT, "using chebyshev");
#endif

  for (int i=0; i < npoints; i++)
  {
    const double r = sqrt(xb[i]*xb[i] + yb[i]*yb[i] + zb[i]*zb[i]);
    const double X = -1.0 + (r-Rin) * 2.0/(Rout - Rin);
    const double phi = atan2(yb[i], xb[i]);
    const double mu = zb[i] / (r+1.0e-100);
    complex<double> val = 0;

    for (int n=0; n < nn; n++)
    {
#ifdef USE_LEGENDRE
      const double Pn = LegendreP(n, X);
#else
      const double Pn = ChebyshevU(n, X);
#endif
      for (int ll=0; ll < nl; ll++)
      {
        const int l = abs(s) + ll;
        for (int m = -l; m <=l; m++)
        {
          const complex<double> coef(l*(n+1), m*(n+1));
          complex<double> ylm = Pn*coef * sYlm_mu(s,l,m,mu,phi) * (2*time+1);
          val += ylm;
        }
      }
    }
    re[i] = val.real();
    im[i] = val.imag();
  }
}
#endif

void SphericalHarmonicDecomp_DumpMetric(Code_mesh)
{
  const int iteration = Code_iteration;
  const int extract_spacetime_metric_every = Code_write_freq;
  
  if (iteration % extract_spacetime_metric_every == 0)
  {
    return;
  }
  const int iter = iteration / extract_spacetime_metric_every;
  
  Decompose3D(mb, "gxx", CCTK_VarIndex("ADMBase::gxx"), iter);
  Decompose3D(mb, "gxy", CCTK_VarIndex("ADMBase::gxy"), iter);
  Decompose3D(mb, "gxz", CCTK_VarIndex("ADMBase::gxz"), iter);
  Decompose3D(mb, "gyy", CCTK_VarIndex("ADMBase::gyy"), iter);
  Decompose3D(mb, "gyz", CCTK_VarIndex("ADMBase::gyz"), iter);
  Decompose3D(mb, "gzz", CCTK_VarIndex("ADMBase::gzz"), iter);
  Decompose3D(mb, "betax", CCTK_VarIndex("ADMBase::betax"), iter);
  Decompose3D(mb, "betay", CCTK_VarIndex("ADMBase::betay"), iter);
  Decompose3D(mb, "betaz", CCTK_VarIndex("ADMBase::betaz"), iter);
  Decompose3D(mb, "alp", CCTK_VarIndex("ADMBase::alp"), iter);
}

