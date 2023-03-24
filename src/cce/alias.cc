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

#ifdef USE_LEGENDRE
#  include "Legendre.hh"
#endif

#include "decomp.hh"

#define H5_USE_16_API 1
#include <hdf5.h>

// check return code of HDF5 call and print a warning in case of an error
#define HDF5_ERROR(fn_call) HDF5_ERROR_FUNC((fn_call), #fn_call, \
                                            __LINE__, __FILE__)
static hid_t HDF5_ERROR_FUNC(hid_t error_code, const char* fn_call,
                             const int line, const char *file)
{
  if (error_code < 0)
  {
    CCTK_VError (__LINE__, __FILE__, CCTK_THORNSTRING,
                "HDF5 call '%s' returned error code %d",
                fn_call, (int)error_code);
  }
  return error_code;
}

#undef TEST_DECOMP


#define Max(a_,b_) ((a_)>(b_)? (a_):(b_))
#define Min(a_,b_) ((a_)<(b_)? (a_):(b_))
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))



#ifdef TEST_DECOMP
static void fill_in_data(int s, int nl, int nn,
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

static int output_modes(const char *dir,
  const char* name, CCTK_REAL time,
       CCTK_INT s, CCTK_INT nl,
       CCTK_REAL *re, CCTK_REAL *im);

static int output_3Dmodes(const char *dir,
  const char* name, CCTK_INT it, CCTK_REAL time,
       CCTK_INT s, CCTK_INT nl,
       CCTK_INT nn, CCTK_REAL rin, CCTK_REAL rout,
       const CCTK_REAL *re, const CCTK_REAL *im);

class output_info
{
  private:
    output_info *next;
    string name;
    int filecreated;
  public:
    void add_info(const string nam) { output_info *nw = new output_info(nam);
      next = nw;};
    void add_info(const char* nam) { output_info *nw = new output_info(nam);
      next = nw;};
    output_info(const char *nam) { name = nam; next = NULL; filecreated=0; };
    output_info(const string nam) { name = nam; next = NULL; filecreated=0; };
    int thatsme(const string nam) const
    {
      if (name == nam)
      {
        return 1;
      }
      else
      {
       return 0;
      }
    };    

    const output_info *locate_info (const string nam) const
    {
      for (const output_info *oi = this; oi != NULL; oi = oi->next)
      {
        if (oi->thatsme(nam))
        {
          return oi;
        }
      }
      return NULL;
    };
};

extern "C"
{
CCTK_INT SphericalHarmonicDecomp_DecomposeField (
       CCTK_POINTER_TO_CONST _GHp,
       CCTK_POINTER_TO_CONST _name,
       CCTK_INT re_gindx,
       CCTK_INT im_gindx,
       CCTK_REAL radius,
       CCTK_INT spin )
{
  using namespace decomp_matrix_class;
  using namespace decomp_sYlm;
  using namespace decomp_decompose;
  using namespace std;

  const cGH* cctkGH = (const cGH*) _GHp;
  const char *name = (const char *) _name;

  static const decomp_info **dinfo_pp = NULL;
  static int FirstTime = 1;

  static CCTK_REAL *xb = NULL;
  static CCTK_REAL *yb = NULL;
  static CCTK_REAL *zb = NULL;
  static CCTK_REAL *mucolloc = NULL;
  static CCTK_REAL *phicolloc = NULL;

  static CCTK_REAL *re_f = NULL;
  static CCTK_REAL *im_f = NULL;

  static CCTK_REAL *re_m = NULL;
  static CCTK_REAL *im_m = NULL;

  DECLARE_CCTK_PARAMETERS;

  myassert(ABS(spin) <= max_spin);

  static int MyProc = CCTK_MyProc(cctkGH);
  const char *outdir = *out_dir ? out_dir : io_out_dir;

  const int nangle = num_mu_points * num_phi_points;

  if (!MyProc)
  {
    if (FirstTime)
    {
      FirstTime = 0;
      dinfo_pp = new  const decomp_info* [2*max_spin+1];
      myassert(dinfo_pp);
      for (int s=-max_spin; s<=max_spin; s++)
      {
	dinfo_pp[s+max_spin] = NULL;
      }

      xb = new CCTK_REAL [nangle];
      yb = new CCTK_REAL [nangle];
      zb = new CCTK_REAL [nangle];

      re_f = new CCTK_REAL [nangle];
      im_f = new CCTK_REAL [nangle];
      mucolloc = new CCTK_REAL [nangle];
      phicolloc = new CCTK_REAL [nangle];
      re_m = new CCTK_REAL [num_l_modes*(num_l_modes+2*ABS(max_spin))];
      im_m = new CCTK_REAL [num_l_modes*(num_l_modes+2*ABS(max_spin))];

      myassert(xb && yb && zb && re_f && im_f && re_m && im_m &&
           mucolloc && phicolloc);
    }

    if (! dinfo_pp[max_spin + spin])
    {
      dinfo_pp[max_spin + spin] = 
          initialize(spin, num_l_modes, num_n_modes,
          num_mu_points, num_phi_points, num_x_points);
      myassert(dinfo_pp[max_spin + spin]);
    }

    dinfo_pp[max_spin + spin]->get_mucolloc(
        num_mu_points*num_phi_points, mucolloc);
    dinfo_pp[max_spin + spin]->get_phicolloc(
       num_mu_points*num_phi_points, phicolloc);

    for (int i = 0; i < nangle; i++)
    {
      const CCTK_REAL phi = phicolloc[i];
      const CCTK_REAL mu = mucolloc[i];
      const CCTK_REAL sph = sin(phi);
      const CCTK_REAL cph = cos(phi);
      const CCTK_REAL cth = mu; 
      const CCTK_REAL sth = sqrt(1.0 - cth*cth); 

      xb[i] = radius * sth*cph;
      yb[i] = radius * sth*sph;
      zb[i] = radius * cth;
    }
  }

  interp_fields(cctkGH, MyProc, re_gindx, im_gindx,
        nangle, xb, yb, zb, re_f, im_f);

  if (!MyProc)
  {
    decompose2D(dinfo_pp[max_spin + spin], re_f, im_f, re_m, im_m);
 
    output_modes(outdir, name, cctkGH->cctk_time, 
       spin, num_l_modes, re_m, im_m);
  }
  return 0;
}
}

extern "C"
{
CCTK_INT SphericalHarmonicDecomp_3D_Decompose (
       CCTK_POINTER_TO_CONST _GHp,
       CCTK_POINTER_TO_CONST _name,
       CCTK_INT re_gindx,
       CCTK_INT im_gindx,
       CCTK_INT spin )
{
  using namespace decomp_matrix_class;
  using namespace decomp_sYlm;
  using namespace decomp_decompose;
  using namespace std;

  const cGH* cctkGH = (const cGH*) _GHp;
  const char *name = (const char *) _name;

  static const decomp_info **dinfo_pp = NULL;
  static int FirstTime = 1;

  static CCTK_REAL *xb = NULL;
  static CCTK_REAL *yb = NULL;
  static CCTK_REAL *zb = NULL;

  static CCTK_REAL *re_f = NULL;
  static CCTK_REAL *im_f = NULL;

  static CCTK_REAL *re_m = NULL;
  static CCTK_REAL *im_m = NULL;

  static CCTK_REAL *mucolloc = NULL;
  static CCTK_REAL *phicolloc = NULL;


  DECLARE_CCTK_PARAMETERS;


  myassert(ABS(spin) <= max_spin);

  static int MyProc = CCTK_MyProc(cctkGH);
  const char *outdir = *out_dir ? out_dir : io_out_dir;

  static double *radius = NULL;

  const int nangle = num_mu_points*num_phi_points;

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

      radius = new CCTK_REAL [num_x_points];

      xb = new CCTK_REAL [nangle*num_x_points];
      yb = new CCTK_REAL [nangle*num_x_points];
      zb = new CCTK_REAL [nangle*num_x_points];

      re_f = new CCTK_REAL [nangle*num_x_points];
      im_f = new CCTK_REAL [nangle*num_x_points];

      mucolloc = new CCTK_REAL [nangle];
      phicolloc = new CCTK_REAL [nangle];
      re_m = new CCTK_REAL [nlmmodes*num_x_points];
      im_m = new CCTK_REAL [nlmmodes*num_x_points];


      myassert(xb && yb && zb && re_f && im_f &&
             mucolloc && phicolloc && re_m && im_m);
    }


    if (! dinfo_pp[max_spin + spin])
    {
      dinfo_pp[max_spin + spin] = 
         initialize(spin, num_l_modes, num_n_modes,
         num_mu_points, num_phi_points, num_x_points);
      myassert(dinfo_pp[max_spin + spin]);
    }

    dinfo_pp[max_spin + spin]->get_ncolloc(num_x_points, radius);
    dinfo_pp[max_spin + spin]->get_mucolloc(nangle, mucolloc);
    dinfo_pp[max_spin + spin]->get_phicolloc(nangle, phicolloc);

    for (int k=0; k < num_x_points; k++)
    {
      CCTK_REAL xk = radius[k];
      radius[k] = 0.5 * ((Rout - Rin) * xk + (Rout + Rin));
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
        xb[indx] = radius[k] * sth*cph;
        yb[indx] = radius[k] * sth*sph;
        zb[indx] = radius[k] * cth;
      }
    }
  }


#ifdef TEST_DECOMP
  if (!MyProc)
  {
    fill_in_data(spin, num_l_modes, num_n_modes,
       nangle*num_x_points, Rin, Rout, xb, yb, zb, re_f, im_f);
  }
#else
  interp_fields(cctkGH, MyProc, re_gindx, im_gindx,
        nangle*num_x_points, xb, yb, zb, re_f, im_f);
#endif

  if (!MyProc)
  {
    decompose3D(dinfo_pp[max_spin + spin], re_f, im_f, re_m, im_m);

    output_3Dmodes(outdir, name, cctkGH->cctk_iteration, cctkGH->cctk_time, 
       spin, num_l_modes, num_n_modes, Rin, Rout,
       re_m, im_m);
  }
  return 0;
}
}

extern "C"
{
void SphericalHarmonicDecomp_Test (CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;

#ifdef TEST_DECOMP
  
  SphericalHarmonicDecomp_3D_Decompose ( 
           (CCTK_POINTER_TO_CONST) cctkGH,
           (CCTK_POINTER_TO_CONST) "gxx",
           -1, -1, 0);
{
  CCTK_REAL time, rin, rout;
  CCTK_INT lmax, nmax;
  CCTK_REAL *re = NULL;
  CCTK_REAL *im = NULL;
 SphericalHarmonicDecomp_Read("gxx_Decomp.h5", cctk_iteration, &time, &rin, &rout,
        &lmax, &nmax, &re, &im);

}
#else
  CCTK_WARN(CCTK_WARN_ABORT, "The test was not compiled in");
#endif
}
}

#define BUFFSIZE 1024
static int output_modes(const char *dir,
  const char* name, CCTK_REAL time,
       CCTK_INT s, CCTK_INT nl,
       CCTK_REAL *re, CCTK_REAL *im)
{
  char filename[BUFFSIZE];

  DECLARE_CCTK_PARAMETERS;

  int sfun = 0;
  for (int ll=0; ll < num_l_modes; ll++)
  {
    const int l = ABS(s) + ll;
   
    for (int m=-l; m <= l; m++, sfun++)
    {
      if (ABS(m) > output_m_max)
      {
        continue;
      }
      snprintf(filename, BUFFSIZE-1,"%s/%s_l_%d_m_%d.asc", dir, name, l, m);
      FILE *file = fopen(filename, "a");
      myassert(file);
      fprintf(file, "%20.16e  %20.16e %20.16e\n", time,
          re[sfun], im[sfun]);
      fclose(file);
      file = NULL;
    }
  }
  return 0;
}

static int output_3Dmodes(const char *dir,
  const char* name, CCTK_INT it, CCTK_REAL time,
       CCTK_INT s, CCTK_INT nl,
       CCTK_INT nn, CCTK_REAL rin, CCTK_REAL rout,
       const CCTK_REAL *re, const CCTK_REAL *im)
{
  char filename[BUFFSIZE];
  hid_t   file_id;
  hsize_t dims[2];


  DECLARE_CCTK_PARAMETERS;

  snprintf(filename, BUFFSIZE-1,"%s/%s_Decomp.h5", dir, name);

  const int nlmmodes = nl*(nl+2*ABS(s));
  dims[0] = nn;
  dims[1] = nlmmodes;

  {
    /* OK there must be a better to test if the file exists */
    FILE *file = fopen(filename, "rb");

    if (!file)
    {
      file_id = HDF5_ERROR(H5Fcreate (filename, H5F_ACC_TRUNC,
           H5P_DEFAULT, H5P_DEFAULT));

      {
        hid_t attribute_id, dataspace_id;;
        int ds[2] = {nn, nlmmodes};
        hsize_t oD2 = 2;
        hsize_t oD1 = 1;

        dataspace_id =  HDF5_ERROR(H5Screate_simple(1, &oD2, NULL));
        attribute_id = HDF5_ERROR(H5Acreate(file_id, "dim", H5T_NATIVE_INT,
                       dataspace_id, H5P_DEFAULT));
        HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_INT, ds));
        HDF5_ERROR(H5Aclose(attribute_id));
        HDF5_ERROR(H5Sclose(dataspace_id));

        dataspace_id =  HDF5_ERROR(H5Screate_simple(1, &oD1, NULL));
        attribute_id = HDF5_ERROR(H5Acreate(file_id, "spin", H5T_NATIVE_INT,
                        dataspace_id, H5P_DEFAULT));
        HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_INT, &s));
        HDF5_ERROR(H5Aclose(attribute_id));
        HDF5_ERROR(H5Sclose(dataspace_id));

        dataspace_id =  HDF5_ERROR(H5Screate_simple(1, &oD1, NULL));
        attribute_id = HDF5_ERROR(H5Acreate(file_id, "Rin", H5T_NATIVE_DOUBLE,
                        dataspace_id, H5P_DEFAULT));
        HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Rin));
        HDF5_ERROR(H5Aclose(attribute_id));
        HDF5_ERROR(H5Sclose(dataspace_id));

        dataspace_id =  HDF5_ERROR(H5Screate_simple(1, &oD1, NULL));
        attribute_id = HDF5_ERROR(H5Acreate(file_id, "Rout", H5T_NATIVE_DOUBLE,
                        dataspace_id, H5P_DEFAULT));
        HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &Rout));
        HDF5_ERROR(H5Aclose(attribute_id));
        HDF5_ERROR(H5Sclose(dataspace_id));

      }
    }
    else
    {
      fclose(file);
      file_id = HDF5_ERROR(H5Fopen (filename, H5F_ACC_RDWR, H5P_DEFAULT));
    }
  }

  {
    hid_t group_id, dataset_id, attribute_id, dataspace_id;;
    char buff[BUFFSIZE];
    hsize_t oneD = 1;
    snprintf(buff, BUFFSIZE-1, "/%d", (int)it);
    group_id = HDF5_ERROR(H5Gcreate(file_id, buff, 0));
    dataspace_id =  HDF5_ERROR(H5Screate_simple(1, &oneD, NULL));
    attribute_id = HDF5_ERROR(H5Acreate(group_id, "Time", H5T_NATIVE_DOUBLE,
        dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &time));
    HDF5_ERROR(H5Aclose(attribute_id));
    HDF5_ERROR(H5Sclose(dataspace_id));
    HDF5_ERROR(H5Gclose(group_id));

    snprintf(buff, BUFFSIZE-1, "/%d/re", (int)it);
    dataspace_id =  HDF5_ERROR(H5Screate_simple(2, dims, NULL));
    dataset_id =  HDF5_ERROR(H5Dcreate(file_id, buff, H5T_NATIVE_DOUBLE,
           dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, re));
    HDF5_ERROR(H5Dclose(dataset_id));
    HDF5_ERROR(H5Sclose(dataspace_id));

    
    snprintf(buff, BUFFSIZE-1, "/%d/im", (int)it);
    dataspace_id =  HDF5_ERROR(H5Screate_simple(2, dims, NULL));
    dataset_id =  HDF5_ERROR(H5Dcreate(file_id, buff, H5T_NATIVE_DOUBLE,
           dataspace_id, H5P_DEFAULT));
    HDF5_ERROR(H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, im));
    HDF5_ERROR(H5Dclose(dataset_id));
    HDF5_ERROR(H5Sclose(dataspace_id));

  }
  HDF5_ERROR(H5Fclose(file_id));

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
  int N_input_arrays     = 2;
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

  if (MyProc && N_input_arrays ==1)
  {
    for (int i=0; i < num_points; i++)
    {
      im_f[i] = 0.0;
    }
  }
  
  return 0;
}

#ifdef TEST_DECOMP
static void fill_in_data(int s, int nl, int nn,
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
          complex<double> ylm = Pn*coef * sYlm_mu(s,l,m,mu,phi);
          val += ylm;
        }
      }
    }
    re[i] = val.real();
    im[i] = val.imag();
  }
}
#endif
