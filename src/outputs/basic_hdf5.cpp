// C++ standard headers
#include <vector>

// Athena++ headers
#include "../utils/utils.hpp"
#include "outputs.hpp"

// External libraries

// ----------------------------------------------------------------------------
#ifdef HDF5OUTPUT

hid_t hdf5_touch_file(const std::string & filename)
{
  return H5Fcreate(filename.c_str(),
                    H5F_ACC_TRUNC,
                    H5P_DEFAULT,
                    H5P_DEFAULT);
}
// ----------------------------------------------------------------------------

void hdf5_write_arr_nd(
  hid_t & id_file,
  const std::string & full_path,
  const AthenaArray<Real> & arr
)
{
  // Ensure we have any nested group structure
  _hdf5_prepare_path(id_file, full_path);

  const int ndim = arr.GetNumDim();

  hsize_t dim[ndim];
  for (int n=0; n<ndim; ++n)
  {
    dim[n] = arr.GetDim(ndim-n);
  }

  hid_t dataset;
  hid_t dataspace;
  dataspace = H5Screate_simple(ndim, dim, NULL);

  dataset = H5Dcreate(id_file, full_path.c_str(),
                      H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT,
                      H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           arr.data());
  H5Dclose(dataset);
  H5Sclose(dataspace);
}

void hdf5_write_scalar(
  hid_t & id_file, const std::string & full_path, Real scalar
)
{
  AthenaArray<Real> arr(1);
  arr(0) = scalar;
  hdf5_write_arr_nd(id_file, full_path, arr);
}

void _hdf5_prepare_path(
  hid_t & id_file,
  const std::string & full_path
)
{
  // extract path
  std::vector<std::string> vs;
  tokenize(full_path, '/', vs);

  if (vs.size() == 1)
  {
    // only have file-name
    return;
  }

  std::string name_dataset { vs.back() };
  vs.pop_back();

  std::stringstream ss;
  std::string path_group;

  for (auto grp : vs)
  {
    // create group structure if not extant:
    ss << grp << "/";
    path_group = ss.str();

    if( !H5Lexists(id_file, path_group.c_str(), H5P_DEFAULT) )
    {
      hid_t id_group;
      id_group = H5Gcreate2(id_file,
                            path_group.c_str(),
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Gclose(id_group);
    }
  }
}

void hdf5_close_file(hid_t & id_file)
{
  H5Fclose(id_file);
}

#endif
// ----------------------------------------------------------------------------

//
// :D
//