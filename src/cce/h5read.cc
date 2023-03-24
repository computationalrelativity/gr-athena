#include "cctk.h"
#include "cctk_Arguments.h"

#define H5_USE_16_API 1
#include <hdf5.h>
#include "hdf5_hl.h"


#include <stdlib.h>
#include <iostream>

#define BUFFSIZE 1024

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

using namespace std;

extern "C"
{
void SphericalHarmonicDecomp_Read(
     const char *name,
     const int iteration,
     CCTK_REAL *p_time,
     CCTK_REAL *p_Rin,
     CCTK_REAL *p_Rout,
     CCTK_INT *p_lmax,
     CCTK_INT *p_nmax,
     CCTK_REAL **p_re,
     CCTK_REAL **p_im)
{
  char buff[BUFFSIZE];
  double Rin, Rout;
  int spin;
  int dim[2];

  hid_t       file_id;

  file_id = HDF5_ERROR(H5Fopen(name, H5F_ACC_RDONLY, H5P_DEFAULT));

  HDF5_ERROR(H5LTget_attribute_int(file_id,"/","spin",&spin));
  HDF5_ERROR(H5LTget_attribute_int(file_id,"/","dim",dim));
  HDF5_ERROR(H5LTget_attribute_double(file_id,"/","Rin",&Rin));
  HDF5_ERROR(H5LTget_attribute_double(file_id,"/","Rout",&Rout));

   /* the order of the dimensions has changed to C
     so dim[0] = nn
        dim[1] = na
  */
  const int nn = dim[0];
  const int na = dim[1];
  const int nl = static_cast<int>(round(sqrt(double(na+spin*spin))-abs(double(spin))));

  cout << "# Spin = " << spin << endl;
  cout << "# Rin = " << Rin << endl;
  cout << "# Rout = " << Rout << endl;
  cout << "# NA = " << na << endl;
  cout << "# NL = " << nl << endl;
  cout << "# NN = " << nn << endl;


  snprintf(buff, BUFFSIZE-1, "/%d", iteration);
  HDF5_ERROR(H5LTget_attribute_double(file_id,buff, "Time", p_time));
  cout << "# Time = " << *p_time << endl;

  if (!*p_re)
  {
    *p_re = (CCTK_REAL *)malloc(sizeof(CCTK_REAL)*na*nn*2);
    *p_im = *p_re + na*nn;
  }

  snprintf(buff, BUFFSIZE-1, "/%d/re", iteration);
  HDF5_ERROR(H5LTread_dataset_double(file_id, buff, *p_re));

  snprintf(buff, BUFFSIZE-1, "/%d/im", iteration);
  HDF5_ERROR(H5LTread_dataset_double(file_id, buff, *p_im));

  HDF5_ERROR(H5Fclose(file_id));

  for (int n=0; n < nn; n++)
  {
    int indx = n*na;
    for (int l=0; l < nl; l++)
    {
      for (int m=-l; m<=l; m++, indx++)
      {

        cout  << indx << endl;
        cout << "C(" << n << ", " << l << ", " << m
                << ") = (" << (*p_re)[indx] << ", "
                << (*p_im)[indx] << ")" << endl;
      }
    }
  }
}
}
