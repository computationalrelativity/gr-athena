//! \file coldeos_compose.cpp
//  \brief Implementation of ColdColdEOSCompOSE

#include "cold_eos_compose.hpp"

#include <algorithm>
#include <hdf5.h>
#include <hdf5_hl.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

namespace
{

[[noreturn]] void ColdTableError(const string& message)
{
  throw runtime_error("ColdEOSCompOSE: " + message);
}

void CheckHDFStatus(herr_t status, const string& operation)
{
  if (status < 0)
  {
    ColdTableError(operation);
  }
}

void CheckColdAxisDataset(hid_t group_id, hsize_t* extent)
{
  int rank = 0;
  CheckHDFStatus(H5LTget_dataset_ndims(group_id, "nb", &rank),
                 "cannot read rank for dataset 'cold_slice/nb'");
  if (rank != 1)
  {
    ColdTableError("dataset 'cold_slice/nb' must have rank 1");
  }

  hsize_t dimensions[1];
  CheckHDFStatus(H5LTget_dataset_info(group_id, "nb", dimensions, NULL, NULL),
                 "cannot read shape for dataset 'cold_slice/nb'");
  if (dimensions[0] < 4)
  {
    ColdTableError(
      "dataset 'cold_slice/nb' must contain at least four values");
  }
  if (dimensions[0] > static_cast<hsize_t>(numeric_limits<int>::max()))
  {
    ColdTableError(
      "dataset 'cold_slice/nb' exceeds the supported index range");
  }
  *extent = dimensions[0];
}

void CheckColdScalarDataset(hid_t group_id, const char* name)
{
  int rank = 0;
  CheckHDFStatus(
    H5LTget_dataset_ndims(group_id, name, &rank),
    string("cannot read rank for dataset 'cold_slice/") + name + "'");
  if (rank == 0)
  {
    return;
  }

  if (rank == 1)
  {
    hsize_t dimensions[1];
    CheckHDFStatus(
      H5LTget_dataset_info(group_id, name, dimensions, NULL, NULL),
      string("cannot read shape for dataset 'cold_slice/") + name + "'");
    if (dimensions[0] == 1)
    {
      return;
    }
  }
  else if (rank == 3)
  {
    hsize_t dimensions[3];
    CheckHDFStatus(
      H5LTget_dataset_info(group_id, name, dimensions, NULL, NULL),
      string("cannot read shape for dataset 'cold_slice/") + name + "'");
    if (dimensions[0] == 1 && dimensions[1] == 1 && dimensions[2] == 1)
    {
      return;
    }
  }

  ColdTableError(string("dataset 'cold_slice/") + name +
                 "' must contain exactly one value");
}

void CheckColdFieldDataset(hid_t group_id, const char* name, hsize_t np)
{
  int rank = 0;
  CheckHDFStatus(
    H5LTget_dataset_ndims(group_id, name, &rank),
    string("cannot read rank for dataset 'cold_slice/") + name + "'");
  if (rank == 1)
  {
    hsize_t dimensions[1];
    CheckHDFStatus(
      H5LTget_dataset_info(group_id, name, dimensions, NULL, NULL),
      string("cannot read shape for dataset 'cold_slice/") + name + "'");
    if (dimensions[0] == np)
    {
      return;
    }
  }
  else if (rank == 3)
  {
    hsize_t dimensions[3];
    CheckHDFStatus(
      H5LTget_dataset_info(group_id, name, dimensions, NULL, NULL),
      string("cannot read shape for dataset 'cold_slice/") + name + "'");
    if (dimensions[0] == np && dimensions[1] == 1 && dimensions[2] == 1)
    {
      return;
    }
  }

  ColdTableError(string("dataset 'cold_slice/") + name +
                 "' must have shape (N), or (N, 1, 1)");
}

string ColdAxisError(int index, const char* condition)
{
  ostringstream stream;
  stream << "dataset 'cold_slice/nb' at index " << index << " " << condition;
  return stream.str();
}

Real InverseSpacing(Real lower, Real upper)
{
  const Real inverse = 1.0 / (upper - lower);
  if (!isfinite(inverse) || inverse <= 0.0)
  {
    ColdTableError("invalid inverse spacing for cold log(nb)");
  }
  return inverse;
}

}  // namespace

ColdEOSCompOSE::ColdEOSCompOSE()
    : m_np(0),
      m_table(nullptr),
      m_initialized(false),
      m_id_log_nb(numeric_limits<Real>::quiet_NaN())
{
  n_species = NSCALARS;
  eos_units = &Nuclear;
}

ColdEOSCompOSE::~ColdEOSCompOSE()
{
  delete[] m_table;
}

Real ColdEOSCompOSE::Pressure(Real n)
{
  assert(m_initialized);
  return exp(eval_at_n<0>(ECLOGP, n));
}

Real ColdEOSCompOSE::Energy(Real n)
{
  assert(m_initialized);
  return exp(eval_at_n<0>(ECLOGE, n));
}

Real ColdEOSCompOSE::dPdn(Real n)
{
  assert(m_initialized);
  return eval_at_n<2>(ECDPDN, n);
}

Real ColdEOSCompOSE::SpecificInternalEnergy(Real n)
{
  return Energy(n) / (mb * n) - 1;
}

Real ColdEOSCompOSE::Y(Real n, int iy)
{
  assert(m_initialized);
  return eval_at_n<0>(ECY + iy, n);
}

Real ColdEOSCompOSE::Enthalpy(Real n)
{
  return (Pressure(n) + Energy(n)) / n;
}

Real ColdEOSCompOSE::DensityFromPressure(Real P)
{
  return exp(eval_at_general(ECLOGP, ECLOGN, log(P)));
}

Real ColdEOSCompOSE::DensityFromEnergy(Real E)
{
  return exp(eval_at_general(ECLOGE, ECLOGN, log(E)));
}

void ColdEOSCompOSE::ReadColdSliceFromFile(std::string fname,
                                           std::string species_names[NSCALARS],
                                           bool uniformize_axes)
{
  hid_t file_id = H5I_INVALID_HID;
  hid_t group_id = H5I_INVALID_HID;

  try
  {
    file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
      ColdTableError("cannot open cold-slice input file");
    }
    group_id = H5Gopen(file_id, "cold_slice", H5P_DEFAULT);
    if (group_id < 0)
    {
      ColdTableError("cannot open group 'cold_slice'");
    }

    hsize_t snb;
    CheckColdAxisDataset(group_id, &snb);
    const int np = static_cast<int>(snb);
    const auto table_index = [np](int iv, int in) -> ptrdiff_t
    {
      return static_cast<ptrdiff_t>(in) + static_cast<ptrdiff_t>(np) * iv;
    };
    const size_t ntable = static_cast<size_t>(ECNVARS) * static_cast<size_t>(np);
    vector<Real> table(ntable);
    vector<double> scratch(np);

    CheckHDFStatus(H5LTread_dataset_double(group_id, "nb", scratch.data()),
                   "cannot read dataset 'cold_slice/nb'");
    const Real new_min_n = static_cast<Real>(scratch[0]);
    const Real new_max_n = static_cast<Real>(scratch[np - 1]);
    for (int in = 0; in < np; ++in)
    {
      const Real n = static_cast<Real>(scratch[in]);
      if (!isfinite(n) || n <= 0.0)
      {
        ColdTableError(ColdAxisError(in, "is not finite and positive"));
      }
      const Real log_n = log(n);
      if (!isfinite(log_n))
      {
        ColdTableError(ColdAxisError(in, "does not produce a finite log(nb)"));
      }
      if (in > 0 && !(log_n > table[table_index(ECLOGN, in - 1)]))
      {
        ColdTableError(ColdAxisError(in, "is not strictly increasing in log(nb)"));
      }
      table[table_index(ECLOGN, in)] = log_n;
    }

    CheckColdScalarDataset(group_id, "t");
    CheckHDFStatus(H5LTread_dataset_double(group_id, "t", scratch.data()),
                   "cannot read dataset 'cold_slice/t'");
    const Real new_temperature = static_cast<Real>(scratch[0]);
    if (!isfinite(new_temperature) || new_temperature <= 0.0)
    {
      ColdTableError("dataset 'cold_slice/t' is not finite and positive");
    }

    // The neutron mass is used as the baryon mass in CompOSE.
    CheckColdScalarDataset(group_id, "mn");
    CheckHDFStatus(H5LTread_dataset_double(group_id, "mn", scratch.data()),
                   "cannot read dataset 'cold_slice/mn'");
    const Real new_mb = static_cast<Real>(scratch[0]);
    if (!isfinite(new_mb) || new_mb <= 0.0)
    {
      ColdTableError("dataset 'cold_slice/mn' is not finite and positive");
    }

    int new_i_lorene_cut = 0;
    const htri_t has_lorene_cut = H5LTfind_dataset(group_id, "lorene_cut");
    if (has_lorene_cut < 0)
    {
      ColdTableError("cannot check for dataset 'cold_slice/lorene_cut'");
    }
    if (has_lorene_cut > 0)
    {
      CheckColdScalarDataset(group_id, "lorene_cut");
      CheckHDFStatus(H5LTread_dataset_int(group_id, "lorene_cut", &new_i_lorene_cut),
                     "cannot read dataset 'cold_slice/lorene_cut'");
    }
    else if (Globals::my_rank == 0)
    {
      std::printf("lorene_cut dataset not found; setting i_lorene_cut=0\n");
    }

    CheckColdFieldDataset(group_id, "Q1", snb);
    CheckHDFStatus(H5LTread_dataset_double(group_id, "Q1", scratch.data()),
                   "cannot read dataset 'cold_slice/Q1'");
    for (int in = 0; in < np; ++in)
    {
      const Real q1 = static_cast<Real>(scratch[in]);
      if (!isfinite(q1) || q1 <= 0.0)
      {
        ColdTableError("dataset 'cold_slice/Q1' is not finite and positive");
      }
      table[table_index(ECLOGP, in)] = log(q1) + table[table_index(ECLOGN, in)];
    }

    CheckColdFieldDataset(group_id, "Q7", snb);
    CheckHDFStatus(H5LTread_dataset_double(group_id, "Q7", scratch.data()),
                   "cannot read dataset 'cold_slice/Q7'");
    for (int in = 0; in < np; ++in)
    {
      const Real q7 = static_cast<Real>(scratch[in]);
      if (!isfinite(q7) || q7 <= -1.0)
      {
        ColdTableError("dataset 'cold_slice/Q7' is not finite and greater than -1");
      }
      table[table_index(ECLOGE, in)] =
        log(new_mb * (q7 + 1.0)) + table[table_index(ECLOGN, in)];
    }

    for (int iy = 0; iy < n_species; ++iy)
    {
      ostringstream stream;
      stream << "Y[" << species_names[iy] << "]";
      const string name = stream.str();
      CheckColdFieldDataset(group_id, name.c_str(), snb);
      CheckHDFStatus(H5LTread_dataset_double(group_id, name.c_str(), scratch.data()),
                     string("cannot read dataset 'cold_slice/") + name + "'");
      for (int in = 0; in < np; ++in)
      {
        const Real abundance = static_cast<Real>(scratch[in]);
        if (!isfinite(abundance))
        {
          ColdTableError(string("dataset 'cold_slice/") + name +
                         "' contains a nonfinite value");
        }
        table[table_index(ECY + iy, in)] = abundance;
      }
    }

    // Fill enthalpy (per baryon): (e + p) / n.
    for (int in = 0; in < np; ++in)
    {
      const Real n = exp(table[table_index(ECLOGN, in)]);
      table[table_index(ECH, in)] =
        (exp(table[table_index(ECLOGE, in)]) +
         exp(table[table_index(ECLOGP, in)])) / n;
    }

    // D0_x_2 yields d(logP)/d(logN); convert via
    // dP/dn = exp(logP - logN) * d(logP)/d(logN).
    D0_x_2(&table[table_index(ECLOGP, 0)],
           &table[table_index(ECLOGN, 0)],
           np,
           &table[table_index(ECDPDN, 0)]);
    for (int in = 1; in < np; ++in)
    {
      table[table_index(ECDPDN, in)] *=
        exp(table[table_index(ECLOGP, in)] - table[table_index(ECLOGN, in)]);
    }

    for (int iv = 0; iv < ECNVARS; ++iv)
    {
      for (int in = 0; in < np; ++in)
      {
        if (!isfinite(table[table_index(iv, in)]))
        {
          ostringstream stream;
          stream << "nonfinite stored lane " << iv << " at index " << in;
          ColdTableError(stream.str());
        }
      }
    }

    CheckHDFStatus(H5Gclose(group_id), "cannot close group 'cold_slice'");
    group_id = H5I_INVALID_HID;
    CheckHDFStatus(H5Fclose(file_id), "cannot close cold-slice input file");
    file_id = H5I_INVALID_HID;

    Real new_id_log_nb;
    if (uniformize_axes)
    {
      UniformizeAxis(table.data(), np, &new_id_log_nb);
    }
    else
    {
      new_id_log_nb = InverseSpacing(table[table_index(ECLOGN, 0)],
                                     table[table_index(ECLOGN, 1)]);
    }

    Real* const new_table = new Real[table.size()];
    copy(table.begin(), table.end(), new_table);
    Real* const old_table = m_table;
    m_table = new_table;
    m_np = np;
    m_id_log_nb = new_id_log_nb;
    min_n = new_min_n;
    max_n = new_max_n;
    T = new_temperature;
    mb = new_mb;
    i_lorene_cut = new_i_lorene_cut;
    m_initialized = true;
    delete[] old_table;
  }
  catch (...)
  {
    if (group_id != H5I_INVALID_HID)
    {
      H5Gclose(group_id);
    }
    if (file_id != H5I_INVALID_HID)
    {
      H5Fclose(file_id);
    }
    throw;
  }
}

void ColdEOSCompOSE::UniformizeAxis(Real* table, int np, Real* inverse_spacing)
{
  const auto table_index = [np](int iv, int in) -> ptrdiff_t
  {
    return static_cast<ptrdiff_t>(in) + static_cast<ptrdiff_t>(np) * iv;
  };
  const Real* const source_log_nb = &table[table_index(ECLOGN, 0)];
  vector<Real> target_log_nb(np);
  for (int in = 0; in < np; ++in)
  {
    target_log_nb[in] = source_log_nb[0] + static_cast<Real>(in) *
                         (source_log_nb[np - 1] - source_log_nb[0]) /
                         static_cast<Real>(np - 1);
  }
  target_log_nb[0] = source_log_nb[0];
  target_log_nb[np - 1] = source_log_nb[np - 1];
  for (int in = 0; in < np; ++in)
  {
    if (!isfinite(target_log_nb[in]))
    {
      ColdTableError("nonfinite target cold log(nb) coordinate");
    }
    if (in > 0 && !(target_log_nb[in] > target_log_nb[in - 1]))
    {
      ColdTableError("target cold log(nb) coordinates are not strictly increasing");
    }
  }

  struct CellWeight
  {
    int lower;
    Real w0;
    Real w1;
  };
  const auto find_cell = [source_log_nb, np](Real log_n) -> CellWeight
  {
    CellWeight result;
    if (log_n == source_log_nb[0])
    {
      result.lower = 0;
      result.w1 = 0.0;
    }
    else if (log_n == source_log_nb[np - 1])
    {
      result.lower = np - 2;
      result.w1 = 1.0;
    }
    else
    {
      const Real* const upper =
        std::upper_bound(source_log_nb, source_log_nb + np, log_n);
      if (upper == source_log_nb || upper == source_log_nb + np)
      {
        ColdTableError("target cold log(nb) coordinate is outside the source axis");
      }
      result.lower = static_cast<int>(upper - source_log_nb) - 1;
      result.w1 = (log_n - source_log_nb[result.lower]) /
                  (source_log_nb[result.lower + 1] - source_log_nb[result.lower]);
    }
    result.w0 = 1.0 - result.w1;
    if (!isfinite(result.w0) || !isfinite(result.w1))
    {
      ColdTableError("nonfinite cold source-cell interpolation weight");
    }
    return result;
  };

  vector<Real> remapped(static_cast<size_t>(ECNVARS) * static_cast<size_t>(np));
  for (int in = 0; in < np; ++in)
  {
    remapped[table_index(ECLOGN, in)] = target_log_nb[in];
    const CellWeight cell = find_cell(target_log_nb[in]);
    for (int iv = ECLOGP; iv < ECNVARS; ++iv)
    {
      const Real value = cell.w0 * table[table_index(iv, cell.lower)] +
                         cell.w1 * table[table_index(iv, cell.lower + 1)];
      if (!isfinite(value))
      {
        ostringstream stream;
        stream << "nonfinite remapped cold lane " << iv << " at index " << in;
        ColdTableError(stream.str());
      }
      remapped[table_index(iv, in)] = value;
    }
  }

  copy(remapped.begin(), remapped.end(), table);
  *inverse_spacing = InverseSpacing(target_log_nb[0], target_log_nb[1]);
}

void ColdEOSCompOSE::DumpLoreneEOSFile(std::string fname)
{
  // Dump the eos_akmalpr.d file that lorene routines expect
  // Lorene units are n [fm^-3], e [g/cm^3], p [erg/cm^3]
  Real n_conv = eos_units->DensityConversion(Nuclear);
  Real e_conv =
    eos_units->DensityConversion(CGS) * eos_units->MassConversion(CGS);
  Real p_conv = eos_units->PressureConversion(CGS);

  std::ofstream lorenefile(fname.c_str());
  lorenefile << std::scientific << std::setprecision(15);

  lorenefile << "#\n#\n#\n#\n#\n" << m_np - i_lorene_cut << "\n#\n#\n#\n";

  for (int i = i_lorene_cut; i < m_np; ++i)
  {
    Real nb = n_conv * exp(m_table[index(ECLOGN, i)]);
    Real e  = e_conv * exp(m_table[index(ECLOGE, i)]);
    Real p  = p_conv * exp(m_table[index(ECLOGP, i)]);
    lorenefile << i - i_lorene_cut + 1 << " " << nb << " " << e << " " << p
               << std::endl;
  }
}

template <int LIX_EXTRAPOLATE>
void ColdEOSCompOSE::weight_idx_ln(Real* w0,
                                   Real* w1,
                                   int* in,
                                   Real log_n) const
{
  *in = (log_n - m_table[index(ECLOGN, 0)]) * m_id_log_nb;

  // if outside table limits, linearly extrapolate
  if (*in > m_np - 2)
  {
    *in = m_np - 2;
  }
  else if (*in < LIX_EXTRAPOLATE)
  {
    *in = LIX_EXTRAPOLATE;
  }

  *w1 = (log_n - m_table[index(ECLOGN, *in)]) * m_id_log_nb;
  *w0 = 1.0 - (*w1);
}

template <int LIX_EXTRAPOLATE>
Real ColdEOSCompOSE::eval_at_ln(int iv, Real log_n) const
{
  int in;
  Real wn0, wn1;

  weight_idx_ln<LIX_EXTRAPOLATE>(&wn0, &wn1, &in, log_n);

  const int ix1 = index(iv, in + 0);
  const int ix2 = index(iv, in + 1);
  const Real m1 = m_table[ix1];
  const Real m2 = m_table[ix2];
  return wn0 * m1 + wn1 * m2;
}

template <int LIX_EXTRAPOLATE>
Real ColdEOSCompOSE::eval_at_n(int iv, Real n) const
{
  return eval_at_ln<LIX_EXTRAPOLATE>(iv, log(n));
}

Real ColdEOSCompOSE::eval_at_general(int iv_in, int iv_out, Real v) const
{
  return linterp1d(v, &m_table[index(iv_in, 0)], &m_table[index(iv_out, 0)]);
}

//--------------------------------------------------------------------------------------
//! \fn int D0_x_2(double *f, double *x, int n, double *df)
// \brief 1st order centered stencil first derivative, nonuniform grids
int ColdEOSCompOSE::D0_x_2(double* f, double* x, int n, double* df)
{
  int i;
  for (i = 1; i < n - 1; i++)
  {
    df[i] = (f[i + 1] - f[i - 1]) / (x[i + 1] - x[i - 1]);
  }
  i     = 0;
  df[i] = (f[i] - f[i + 1]) / (x[i] - x[i + 1]);
  i     = n - 1;
  df[i] = (f[i] - f[i - 1]) / (x[i] - x[i - 1]);
  return 0;
}

Real ColdEOSCompOSE::linterp1d(Real x, Real* xp, Real* fp) const
{
  int i;
  for (i = 1; i < m_np; i++)
  {
    if (x < xp[i])
    {
      break;
    }
  }
  if (i == m_np)
  {
    i = m_np - 1;
  }
  Real w1 = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
  Real w0 = 1.0 - w1;
  return w0 * fp[i - 1] + w1 * fp[i];
}
