#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "util_Table.h"
#include "util_ErrorCodes.h"
#include "SphericalHarmonicDecomp.h"
#include <stdio.h>
#include <stdlib.h>
#include "myassert.h"

#include <iostream>
#include <complex>
#include "sYlm.hh"

static int hdf5_warn_level = CCTK_WARN_ALERT;

#define HDF5_ERROR(fn_call)                                                   \
        do {                                                                  \
          hid_t _error_code = fn_call;                                        \
                                                                              \
                                                                              \
          if (_error_code < 0)                                                \
          {                                                                   \
            CCTK_VWarn (hdf5_warn_level, __LINE__, __FILE__, CCTK_THORNSTRING,\
                        "HDF5 call '%s' returned error code %d",              \
                        #fn_call, (int)_error_code);                          \
            error_count++;                                                    \
          }                                                                   \
        } while (0)

#ifdef USE_LEGENDRE
#  include "Legendre.hh"
#endif

#include "decomp.hh"

#define H5_USE_16_API 1
#include <hdf5.h>

#undef TEST_DECOMP


#define Max(a_,b_) ((a_)>(b_)? (a_):(b_))
#define Min(a_,b_) ((a_)<(b_)? (a_):(b_))
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))



#ifdef TEST_DECOMP
static void fill_in_data(CCTK_REAL time, int s, int nl, int nn,
    int npoints, CCTK_REAL Rin, CCTK_REAL Rout,
    const CCTK_REAL xb[], const CCTK_REAL yb[],
    const CCTK_REAL zb[], CCTK_REAL re[], CCTK_REAL im[]);
#endif

 
static int interp_fields(const cGH* cctkGH,
          CCTK_INT MyProc,
          CCTK_INT re_inxd, CCTK_INT im_indx,
          CCTK_INT num_points, 
          const CCTK_REAL * xc,
          const CCTK_REAL * yc,
          const CCTK_REAL * zc,
          CCTK_REAL *re_f,
          CCTK_REAL *im_f);

static int output_3Dmodes(const int iter,
  const char *dir,
  const char* name,
       const int obs,  CCTK_INT it, CCTK_REAL time,
       CCTK_INT s, CCTK_INT nl,
       CCTK_INT nn, CCTK_REAL rin, CCTK_REAL rout,
       const CCTK_REAL *re, const CCTK_REAL *im);

static int Decompose3D (
       const cGH* cctkGH,
       const char *name,
       CCTK_INT re_gindx,
       const int iter)
{
  using namespace decomp_matrix_class;
  using namespace decomp_sYlm;
  using namespace decomp_decompose;
  using namespace std;

  const int spin = 0;
  const int im_gindx = -1;

  static const decomp_info **dinfo_pp = NULL;
  static int FirstTime = 1;

#define MAX_RADII  100
  static CCTK_REAL *xb[MAX_RADII];
  static CCTK_REAL *yb[MAX_RADII];
  static CCTK_REAL *zb[MAX_RADII];
  static double *radius[MAX_RADII];

  static CCTK_REAL *re_f = NULL;
  static CCTK_REAL *im_f = NULL;

  static CCTK_REAL *re_m = NULL;
  static CCTK_REAL *im_m = NULL;

  static CCTK_REAL *mucolloc = NULL;
  static CCTK_REAL *phicolloc = NULL;


  DECLARE_CCTK_PARAMETERS;


  myassert (ABS(spin) <= max_spin);
  myassert (num_radii <= CCTK_INT(sizeof(radius) / sizeof(*radius)));

  static int MyProc = CCTK_MyProc(cctkGH);
  const char *outdir = *out_dir ? out_dir : io_out_dir;


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
        radius[i] = new CCTK_REAL [num_x_points];
        xb[i] = new CCTK_REAL [nangle*num_x_points];
        yb[i] = new CCTK_REAL [nangle*num_x_points];
        zb[i] = new CCTK_REAL [nangle*num_x_points];

        myassert(radius[i]);
        myassert(xb[i]);
        myassert(yb[i]);
        myassert(zb[i]);
      }

      re_f = new CCTK_REAL [nangle*num_x_points];
      im_f = new CCTK_REAL [nangle*num_x_points];

      mucolloc = new CCTK_REAL [nangle];
      phicolloc = new CCTK_REAL [nangle];
      re_m = new CCTK_REAL [nlmmodes*num_x_points];
      im_m = new CCTK_REAL [nlmmodes*num_x_points];


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
      CCTK_REAL xk = radius[0][k];
      for (int i=0; i < num_radii; i++)
      {
        radius[i][k] = 0.5 * ((EM_Rout[i] - EM_Rin[i]) * xk +
            (EM_Rout[i] + EM_Rin[i]));
      }
    }

    for (int k=0; k < num_x_points; k++)
    {
      for (int i=0; i < nangle; i++)
      {
	const CCTK_REAL phi = phicolloc[i];
	const CCTK_REAL mu = mucolloc[i];

	const CCTK_REAL sph = sin(phi);
	const CCTK_REAL cph = cos(phi);
        const CCTK_REAL cth = mu;
        const CCTK_REAL sth = sqrt(1.0 - mu*mu);

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
      fill_in_data(cctkGH->cctk_time, spin, num_l_modes, num_n_modes,
         nangle*num_x_points, EM_Rin[obs], EM_Rout[obs], xb[obs], yb[obs],
       zb[obs], re_f, im_f);
    }
#else
    interp_fields(cctkGH, MyProc, re_gindx, im_gindx,
        nangle*num_x_points, xb[obs], yb[obs], zb[obs], re_f, im_f);
#endif

    if (!MyProc)
    {
      decompose3D(dinfo_pp[max_spin + spin], re_f, im_f, re_m, im_m);

      output_3Dmodes(iter, outdir, name, obs,
         cctkGH->cctk_iteration, cctkGH->cctk_time, 
         spin, num_l_modes, num_n_modes, EM_Rin[obs], EM_Rout[obs],
         re_m, im_m);
    }
  }
  return 0;
}



#define BUFFSIZE 1024

static int output_3Dmodes(const int iter,
  const char *dir,
  const char* name, const int obs,  CCTK_INT it, CCTK_REAL time,
       CCTK_INT s, CCTK_INT nl,
       CCTK_INT nn, CCTK_REAL rin, CCTK_REAL rout,
       const CCTK_REAL *re, const CCTK_REAL *im)
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
      CCTK_WARN(hdf5_warn_level, "Failed to open hdf5 file");
    }
  }
  else
  {
    file_id = H5Fcreate (filename, H5F_ACC_TRUNC,
           H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
    {
      CCTK_WARN(hdf5_warn_level, "Failed to create hdf5 file");
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

    if (error_count > 0)
    {
      CCTK_WARN(hdf5_warn_level, "Failed to initialize hdf5 file");
    }
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

    if (error_count > 0)
    {
      CCTK_WARN(hdf5_warn_level, "Failed to write to hdf5 file");
    }
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

  if (error_count>0)
  {
    CCTK_WARN(hdf5_warn_level, "Failed to write data to hdf5 file");
  }

  return 0;
}
 
static int interp_fields(const cGH* cctkGH,
          CCTK_INT MyProc,
          CCTK_INT re_indx, CCTK_INT im_indx,
          CCTK_INT num_points, 
          const CCTK_REAL * xc,
          const CCTK_REAL * yc,
          const CCTK_REAL * zc,
          CCTK_REAL *re_f,
          CCTK_REAL *im_f)
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

  const CCTK_INT input_array_variable_indices[2]=
      {re_indx, im_indx};
  const CCTK_INT output_array_type_codes[2]=
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

#ifdef TEST_DECOMP
static void fill_in_data(CCTK_REAL time, int s, int nl, int nn,
    int npoints, CCTK_REAL Rin, CCTK_REAL Rout,
    const CCTK_REAL xb[], const CCTK_REAL yb[],
    const CCTK_REAL zb[], CCTK_REAL re[], CCTK_REAL im[])
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
    const CCTK_REAL r = sqrt(xb[i]*xb[i] + yb[i]*yb[i] + zb[i]*zb[i]);
    const CCTK_REAL X = -1.0 + (r-Rin) * 2.0/(Rout - Rin);
    const CCTK_REAL phi = atan2(yb[i], xb[i]);
    const CCTK_REAL mu = zb[i] / (r+1.0e-100);
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

extern "C"
{
  void SphericalHarmonicDecomp_DumpMetric(CCTK_ARGUMENTS)
  {
    DECLARE_CCTK_ARGUMENTS;
    DECLARE_CCTK_PARAMETERS;

    static int last_iter = -10000;


    if ((!extract_spacetime_metric_every) ||
         (cctk_iteration % extract_spacetime_metric_every))
    {
      return;
    }
    const int iter = cctk_iteration / extract_spacetime_metric_every;

    if (iter <= last_iter)
    {
      return;
    }

    
    if (CCTK_Equals(action_on_hdf5_error, "abort"))
    {
      hdf5_warn_level = CCTK_WARN_ABORT;
    }
    else if  (CCTK_Equals(action_on_hdf5_error, "alert"))
    {
      hdf5_warn_level = CCTK_WARN_ALERT;
    }

    Decompose3D(cctkGH, "gxx", CCTK_VarIndex("ADMBase::gxx"), iter);
    Decompose3D(cctkGH, "gxy", CCTK_VarIndex("ADMBase::gxy"), iter);
    Decompose3D(cctkGH, "gxz", CCTK_VarIndex("ADMBase::gxz"), iter);
    Decompose3D(cctkGH, "gyy", CCTK_VarIndex("ADMBase::gyy"), iter);
    Decompose3D(cctkGH, "gyz", CCTK_VarIndex("ADMBase::gyz"), iter);
    Decompose3D(cctkGH, "gzz", CCTK_VarIndex("ADMBase::gzz"), iter);
    Decompose3D(cctkGH, "betax", CCTK_VarIndex("ADMBase::betax"), iter);
    Decompose3D(cctkGH, "betay", CCTK_VarIndex("ADMBase::betay"), iter);
    Decompose3D(cctkGH, "betaz", CCTK_VarIndex("ADMBase::betaz"), iter);
    Decompose3D(cctkGH, "alp", CCTK_VarIndex("ADMBase::alp"), iter);

    last_iter = iter;
  }
}
