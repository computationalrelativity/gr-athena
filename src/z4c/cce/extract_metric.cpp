#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include <algorithm> // for fill
#include <cmath>     // for NAN

#define H5_USE_16_API (1)
#include <hdf5.h>


#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../globals.hpp"
#include "../../parameter_input.hpp"
#include "../../mesh/mesh.hpp"
#include "../z4c.hpp"


#include "myassert.hpp"
#include "sYlm.hpp"
#include "h5read.hpp"
#include "decomp.hpp"

#ifdef USE_LEGENDRE
# ERROR: do not use this option.
#include "Legendre.hpp"
#endif

/* vars and params to change for each code

- all //! comments
- Code_* macros

*/


/*
Example:

SphericalHarmonicDecomp::extract_spacetime_metric_every=32
SphericalHarmonicDecomp::num_radii=3
SphericalHarmonicDecomp::EM_Rin[0]=18
SphericalHarmonicDecomp::EM_Rout[0]=22
SphericalHarmonicDecomp::EM_Rin[1]=47
SphericalHarmonicDecomp::EM_Rout[1]=53
SphericalHarmonicDecomp::EM_Rin[2]=94
SphericalHarmonicDecomp::EM_Rout[2]=106
SphericalHarmonicDecomp::num_l_modes=7
SphericalHarmonicDecomp::num_n_modes=7
SphericalHarmonicDecomp::num_mu_points=41
SphericalHarmonicDecomp::num_phi_points=82
SphericalHarmonicDecomp::num_x_points=28

In this example we will extract the metric on 3 different shells
 (num\_radii), the
first between r=18 and r=22, the second between r=47 and r=53, and
the third between r=94 and 106. The idea here is to make the shell
small enough that we can accurately calculate the radial derivatives
of the metric function, while also large enough that we can smooth out
the grid noise. We decompose the metric functions in terms of
7 $\ell$ modes (num\_l\_modes=7 or all modes from $\ell=0$ to $\ell=6$,
 the m modes are automatically set) 
and 7 radial modes (num\_n\_mode=7). The grid functions are evaluated
at 41 points in mu (mu=cos(theta)),  82 points in phi, and 28 points
in radius. Minimally, we would need the number of angular points to be
equal to the number of angular spectral functions $\ell^2 + 2\ell$,
in this case we have many more angular modes ($41*82$) than angular
spectral functions. Similarly num\_x\_points must be greater than
num\_n\_modes. The number of n modes is set by the need to accurately
model the radial derivative of the mertric functions in the spherical
shell. The larger the difference between EM\_Rin[] and EM\_Rout[] the
more points required. The number of l modes is determined by the accuracy
requiremnts of the final CCE waveoform. in this case, choosin
num\_n\_modes=7 is marginally acceptable.

*/


#define Code_mesh       Z4c *const pz4c

#define Code_field(x_)  0 // ex: ??

#define Code_field_t    int re_gindx // field type. ex: ??

#define Code_proc_rank  (Globals::my_rank) // ex: Globals::my_rank

#define Code_interpolate(x_,y_,z_,N)  // function to call for interpolation at given points

#define Code_time       1 // code evolution time; ex: mb->pmy_mesh->time

#define Code_iteration  1 // code iteration number; ex: mb->pmy_mesh->ncycle

#define Code_write_freq 1 // after how many iteration dumping metrics; ex: get from param file

#define Code_max_spin  (2) // max spin of spin weighted Ylm, 
                           // maximum absolute spin of fields". Set it to 2 for now.

#define Code_out_dir    "./" // ex: path/to/output dir
                           
#define Code_num_radii  1 // number of extraction radii; ex: get it from param file.

#define Code_num_mu_points  41 // number of points in theta direction(polar); 
                               // ex: get it from param file

#define Code_num_phi_points 82 // number of points in phi direction(azimuthal); 
                               // ex: get it from param file

#define Code_num_x_points   28 // number of points in radius between the two shells
                               // ex: get it from param file

#define Code_num_l_modes    7 // number of l modes in Ylm (m modes calculated automatically)
                              // ex: get it from param file

#define Code_num_n_modes    7 // radial modes; ex: get it from param file


#define Code_Rout(i_)  30 // outer radius for decomp; 
                          // ex: get it from param file getd("Rout"#i);

#define Code_Rin(i_)   20 // inner radius for decomp;
                          // ex: get it from param file getd("Rin"#i);

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

#define Max(a_,b_) ((a_)>(b_)? (a_):(b_))
#define Min(a_,b_) ((a_)<(b_)? (a_):(b_))
#define ABS(x_) ((x_)>0 ? (x_) : (-(x_)))

#define BUFFSIZE  (1024)
#define MAX_RADII (100)

void CCEDumpMetric(Code_mesh);

static int output_3Dmodes(const int iter,
       const char *dir,
       const char* name,
       const int obs, double time,
       int s, int nl,
       int nn, double rin, double rout,
       const double *re, const double *im);

static int Decompose3D(Code_mesh,
       const char *name,
       Code_field_t,
       const int iter);

#ifdef TEST_DECOMP
// instead of interpolation, fill the data for test purposes
static void fill_in_data(double time, int s, int nl, int nn,
    int npoints, double Rin, double Rout,
    const double xb[], const double yb[],
    const double zb[], double re[], double im[]);
#endif

// this is where the magic happens, it decomposes the field in Ylm for the angular 
// directions and in Legendre or Chebyshev for the radial direction.
// Note: for each shell with an inner radius and outer radius, it samples the 
// given field within these two radii by an interpolation and then 3-D decomposes 
// them.
static int Decompose3D (
       Code_mesh,
       const char *name,
       Code_field_t,
       const int iter)
{
  using namespace decomp_matrix_class;
  using namespace decomp_sYlm;
  using namespace decomp_decompose;
  using namespace std;

  const double evo_time = Code_time;
  const char *const out_dir = Code_out_dir;
  const int max_spin  = Code_max_spin;
  const int num_radii = Code_num_radii;
  const int num_mu_points  = Code_num_mu_points;
  const int num_phi_points = Code_num_phi_points;
  const int num_x_points   = Code_num_x_points;
  const int num_l_modes    = Code_num_l_modes;
  const int num_n_modes    = Code_num_n_modes;
  
  const int spin = 0;

  static const decomp_info **dinfo_pp = NULL;
  static int FirstTime = 1;

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
      std::fill(re_f, re_f+(nangle*num_x_points),NAN); // init to nan
      im_f = new double [nangle*num_x_points](); // init to zero

      mucolloc = new double [nangle];
      phicolloc = new double [nangle];
      re_m = new double [nlmmodes*num_x_points];
      im_m = new double [nlmmodes*num_x_points];

      myassert(re_f && im_f && mucolloc && phicolloc && re_m && im_m);
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
    //! interpolate for a given field
    // array length = nangle*num_x_points for each x,y,z
    // coords = xb[obs], yb[obs], zb[obs]
    // interpolated values will be returned to = re_f
    Code_interpolate(xb[obs], yb[obs], zb[obs],nangle*num_x_points);
#endif

    if (!MyProc)
    {
      decompose3D(dinfo_pp[max_spin + spin], re_f, im_f, re_m, im_m);

      output_3Dmodes(iter, outdir, name, obs, evo_time, 
         spin, num_l_modes, num_n_modes, Code_Rin(obs), Code_Rout(obs),
         re_m, im_m);
    }
  }
  return 0;
  
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

  snprintf(filename, sizeof filename,
        "%s/%s_obs_%d_Decomp.h5", dir, "metric", obs);

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

#ifdef TEST_DECOMP
static void fill_in_data(double time, int s, int nl, int nn,
    int npoints, double Rin, double Rout,
    const double xb[], const double yb[],
    const double zb[], double re[], double im[])
{
  using namespace decomp_sYlm;
#ifdef USE_LEGENDRE
  using namespace decomp_Legendre;
  // cout << "using Legendre" << endl;
#else
  using namespace decomp_Chebyshev;
  // cout << "using chebyshev" << endl;
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

// this function writes the pertinent metric variables in a specific format 
// readable for pittnull
void CCEDumpMetric(Code_mesh)
{
  const int iteration = Code_iteration;
  const int extract_spacetime_metric_every = Code_write_freq;
  
  if (iteration % extract_spacetime_metric_every)
  {
    return;
  }
  const int iter = iteration / extract_spacetime_metric_every;
  //! pass the pertinent field
  // We need all the following fields
  Decompose3D(pz4c, "gxx", Code_field("ADM::gxx"), iter);
  Decompose3D(pz4c, "gxy", Code_field("ADM::gxy"), iter);
  Decompose3D(pz4c, "gxz", Code_field("ADM::gxz"), iter);
  Decompose3D(pz4c, "gyy", Code_field("ADM::gyy"), iter);
  Decompose3D(pz4c, "gyz", Code_field("ADM::gyz"), iter);
  Decompose3D(pz4c, "gzz", Code_field("ADM::gzz"), iter);
  Decompose3D(pz4c, "betax", Code_field("ADM::betax"), iter);
  Decompose3D(pz4c, "betay", Code_field("ADM::betay"), iter);
  Decompose3D(pz4c, "betaz", Code_field("ADM::betaz"), iter);
  Decompose3D(pz4c, "alp", Code_field("ADM::alp"), iter);
}
