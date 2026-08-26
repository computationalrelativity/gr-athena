//! \file eos_compose.cpp
//  \brief Implementation of EOSCompose

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// #ifdef HDF5OUTPUT
#include <hdf5.h>
#include <hdf5_hl.h>

#include "eos_compose.hpp"
#include "numtools_root.hpp"
#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

namespace
{

[[noreturn]] void TableError(const string& message)
{
  throw runtime_error("EOSCompOSE: " + message);
}

void CheckHDFStatus(herr_t status, const string& operation)
{
  if (status < 0)
  {
    TableError(operation);
  }
}

size_t CheckedProduct(size_t left, size_t right, const char* quantity)
{
  if (left != 0 && right > numeric_limits<size_t>::max() / left)
  {
    TableError(string("overflow computing ") + quantity);
  }
  return left * right;
}

void CheckAllocationBytes(size_t count, size_t element_size, const char* quantity)
{
  if (count > numeric_limits<size_t>::max() / element_size)
  {
    TableError(string("byte-count overflow allocating ") + quantity);
  }
}

void CheckAxisDataset(hid_t file_id, const char* name, hsize_t* extent)
{
  int rank = 0;
  CheckHDFStatus(H5LTget_dataset_ndims(file_id, name, &rank),
                 string("cannot read rank for dataset '") + name + "'");
  if (rank != 1)
  {
    TableError(string("dataset '") + name + "' must have rank 1");
  }

  hsize_t dimensions[1];
  CheckHDFStatus(H5LTget_dataset_info(file_id, name, dimensions, NULL, NULL),
                 string("cannot read shape for dataset '") + name + "'");
  if (dimensions[0] < 2)
  {
    TableError(string("dataset '") + name + "' must contain at least two values");
  }
  *extent = dimensions[0];
}

void CheckFieldDataset(hid_t file_id,
                       const char* name,
                       hsize_t nn,
                       hsize_t ny,
                       hsize_t nt)
{
  int rank = 0;
  CheckHDFStatus(H5LTget_dataset_ndims(file_id, name, &rank),
                 string("cannot read rank for dataset '") + name + "'");
  if (rank != 3)
  {
    TableError(string("dataset '") + name + "' must have rank 3");
  }

  hsize_t dimensions[3];
  CheckHDFStatus(H5LTget_dataset_info(file_id, name, dimensions, NULL, NULL),
                 string("cannot read shape for dataset '") + name + "'");
  if (dimensions[0] != nn || dimensions[1] != ny || dimensions[2] != nt)
  {
    ostringstream stream;
    stream << "dataset '" << name << "' must have shape (" << nn << ", " << ny
           << ", " << nt << ")";
    TableError(stream.str());
  }
}

void CheckMassDataset(hid_t file_id, const char* name)
{
  int rank = 0;
  CheckHDFStatus(H5LTget_dataset_ndims(file_id, name, &rank),
                 string("cannot read rank for dataset '") + name + "'");
  if (rank == 0)
  {
    return;
  }
  if (rank != 1)
  {
    TableError(string("dataset '") + name + "' must be scalar or rank 1");
  }

  hsize_t dimensions[1];
  CheckHDFStatus(H5LTget_dataset_info(file_id, name, dimensions, NULL, NULL),
                 string("cannot read shape for dataset '") + name + "'");
  if (dimensions[0] != 1)
  {
    TableError(string("dataset '") + name + "' must contain one value");
  }
}

string AxisError(const char* name, int index, const char* condition)
{
  ostringstream stream;
  stream << "dataset '" << name << "' at index " << index << " " << condition;
  return stream.str();
}

string FieldError(const char* name, int in, int iy, int it, const char* condition)
{
  ostringstream stream;
  stream << "dataset '" << name << "' at (" << in << ", " << iy << ", " << it
         << ") " << condition;
  return stream.str();
}

void CheckStoredFinite(Real value,
                       const char* name,
                       int in,
                       int iy,
                       int it)
{
  if (!isfinite(value))
  {
    TableError(FieldError(name, in, iy, it, "does not produce a finite stored value"));
  }
}

Real InverseSpacing(Real lower, Real upper, const char* name)
{
  const Real inverse = 1.0 / (upper - lower);
  if (!isfinite(inverse) || inverse <= 0.0)
  {
    TableError(string("invalid inverse spacing for ") + name);
  }
  return inverse;
}

}  // namespace

EOSCompOSE::EOSCompOSE()
    : m_id_log_nb(numeric_limits<Real>::quiet_NaN()),
      m_id_log_t(numeric_limits<Real>::quiet_NaN()),
      m_id_yq(numeric_limits<Real>::quiet_NaN()),
      m_nn(0),
      m_nt(0),
      m_ny(0),
      m_min_h(numeric_limits<Real>::max())
{
  n_species = 1;
  eos_units = &Nuclear;
}
// These are static now, so are defined separately below
/*
m_log_nb(nullptr),
m_log_t(nullptr),
m_yq(nullptr),
m_table(nullptr),
m_initialized(false)
*/
EOSCompOSE::~EOSCompOSE()
{
  // These are static variables now, so no need to delete
  /*
  if (m_initialized) {
    delete[] m_log_nb;
    delete[] m_log_t;
    delete[] m_yq;
    delete[] m_table;
  }
  */
}

// Definitions for static members
Real* EOSCompOSE::m_log_nb     = nullptr;
Real* EOSCompOSE::m_log_t      = nullptr;
Real* EOSCompOSE::m_yq         = nullptr;
Real* EOSCompOSE::m_table      = nullptr;
bool EOSCompOSE::m_initialized = false;

Real EOSCompOSE::sm_id_log_nb = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::sm_id_log_t  = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::sm_id_yq     = numeric_limits<Real>::quiet_NaN();

int EOSCompOSE::sm_nn = 0;
int EOSCompOSE::sm_nt = 0;
int EOSCompOSE::sm_ny = 0;

Real EOSCompOSE::sm_min_h = numeric_limits<Real>::max();

Real EOSCompOSE::s_mb                 = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_max_n              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_min_n              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_max_T              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_min_T              = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_max_Y[MAX_SPECIES] = { 0 };
Real EOSCompOSE::s_min_Y[MAX_SPECIES] = { 0 };

Real EOSCompOSE::s_mn = numeric_limits<Real>::quiet_NaN();
Real EOSCompOSE::s_mp = numeric_limits<Real>::quiet_NaN();

Real EOSCompOSE::TemperatureFromE(Real n, Real e, Real* Y)
{
  assert(m_initialized);
  // Hoist density and composition weights: computed once, reused for
  // bounds checks and the inner root-find.
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  // Evaluate log(e) at the table boundaries using 4 lookups each
  // (the temperature weight is trivially 1 at grid endpoints).
  Real loge_min = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, 0);
  Real loge_max = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
  Real e_min    = exp(loge_min);
  Real e_max    = exp(loge_max);

  if (e <= e_min)
    return min_T;
  if (e >= e_max)
    return max_T;
  return temperature_from_var_precomp(
    loge_min, loge_max, ECLOGE, log(e), wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::TemperatureFromP(Real n, Real p, Real* Y)
{
  assert(m_initialized);
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  Real logp_min = eval_at_it(ECLOGP, wn0, wn1, in, wy0, wy1, iy, 0);
  Real logp_max = eval_at_it(ECLOGP, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
  Real p_min    = exp(logp_min);
  Real p_max    = exp(logp_max);

  if (p <= p_min)
    return min_T;
  if (p >= p_max)
    return max_T;
  return temperature_from_var_precomp(
    logp_min, logp_max, ECLOGP, log(p), wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::TemperatureFromEntropy(Real n, Real s, Real* Y)
{
  assert(m_initialized);
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  // Entropy is stored directly (not in log space).
  Real s_min = eval_at_it(ECENT, wn0, wn1, in, wy0, wy1, iy, 0);
  Real s_max = eval_at_it(ECENT, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);

  if (s <= s_min)
    return min_T;
  if (s >= s_max)
    return max_T;
  return temperature_from_var_precomp(
    s_min, s_max, ECENT, s, wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::Energy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return exp(eval_at_nty(ECLOGE, n, T, Y[0]));
}

Real EOSCompOSE::Pressure(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return exp(eval_at_nty(ECLOGP, n, T, Y[0]));
}

Real EOSCompOSE::Entropy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECENT, n, T, Y[0]);
}

Real EOSCompOSE::Enthalpy(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  Real const P = Pressure(n, T, Y);
  Real const e = Energy(n, T, Y);
  return (P + e) / n;
}

void EOSCompOSE::PressureAndEnthalpy(Real n, Real T, Real* Y, Real* P, Real* h)
{
  assert(m_initialized);
  Real log_n = log(n);
  Real log_t = log(T);
  Real Yq    = Y[0];

  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1;
  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, Yq);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

  // Interpolate log(P) and log(e) with shared weights.
  ptrdiff_t bp00 = index(ECLOGP, in, iy, it);
  ptrdiff_t bp01 = index(ECLOGP, in, iy + 1, it);
  ptrdiff_t bp10 = index(ECLOGP, in + 1, iy, it);
  ptrdiff_t bp11 = index(ECLOGP, in + 1, iy + 1, it);

  Real logP = wn0 * (wy0 * (wt0 * m_table[bp00] + wt1 * m_table[bp00 + 1]) +
                     wy1 * (wt0 * m_table[bp01] + wt1 * m_table[bp01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[bp10] + wt1 * m_table[bp10 + 1]) +
                     wy1 * (wt0 * m_table[bp11] + wt1 * m_table[bp11 + 1]));

  // ECLOGE base offsets: same (in, iy, it) cell, different variable slice.
  ptrdiff_t be00 = index(ECLOGE, in, iy, it);
  ptrdiff_t be01 = index(ECLOGE, in, iy + 1, it);
  ptrdiff_t be10 = index(ECLOGE, in + 1, iy, it);
  ptrdiff_t be11 = index(ECLOGE, in + 1, iy + 1, it);

  Real logE = wn0 * (wy0 * (wt0 * m_table[be00] + wt1 * m_table[be00 + 1]) +
                     wy1 * (wt0 * m_table[be01] + wt1 * m_table[be01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[be10] + wt1 * m_table[be10 + 1]) +
                     wy1 * (wt0 * m_table[be11] + wt1 * m_table[be11 + 1]));

  Real Pval = exp(logP);
  Real eval = exp(logE);
  *P        = Pval;
  *h        = (Pval + eval) / n;
}

void EOSCompOSE::FindTBracketAndWeights(Real n,
                                        Real e,
                                        Real* Y,
                                        int* guess_it,
                                        int& in,
                                        int& iy,
                                        int& it,
                                        Real& wn0,
                                        Real& wn1,
                                        Real& wy0,
                                        Real& wy1,
                                        Real& wt0,
                                        Real& wt1,
                                        Real& lt,
                                        bool& boundary_lo,
                                        bool& boundary_hi) const
{
  boundary_lo = false;
  boundary_hi = false;

  Real log_n = log(n);
  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, Y[0]);

  ptrdiff_t const be00 = index(ECLOGE, in, iy, 0);
  ptrdiff_t const be01 = index(ECLOGE, in, iy + 1, 0);
  ptrdiff_t const be10 = index(ECLOGE, in + 1, iy, 0);
  ptrdiff_t const be11 = index(ECLOGE, in + 1, iy + 1, 0);

  Real var = log(e);

  auto f = [=](int iti) -> Real
  {
    Real var_pt =
      wn0 * (wy0 * m_table[be00 + iti] + wy1 * m_table[be01 + iti]) +
      wn1 * (wy0 * m_table[be10 + iti] + wy1 * m_table[be11 + iti]);
    return var - var_pt;
  };

  int ilo        = 0;
  int ihi        = m_nt - 1;
  Real flo       = 0.0;
  Real fhi       = 0.0;
  bool bracketed = false;

  // Hunt locally first
  if (guess_it && *guess_it >= 0 && *guess_it < m_nt - 1)
  {
    int itg = *guess_it;
    Real fl = f(itg);
    Real fh = f(itg + 1);
    if (fl * fh <= 0)
    {
      ilo       = itg;
      ihi       = itg + 1;
      flo       = fl;
      fhi       = fh;
      bracketed = true;
    }
    else if (fl < 0 && itg > 0)  // Try shifting left
    {
      Real fl_minus = f(itg - 1);
      if (fl_minus * fl <= 0)
      {
        ilo       = itg - 1;
        ihi       = itg;
        flo       = fl_minus;
        fhi       = fl;
        bracketed = true;
      }
    }
    else if (fh > 0 && itg + 2 < m_nt)  // Try shifting right
    {
      Real fh_plus = f(itg + 2);
      if (fh * fh_plus <= 0)
      {
        ilo       = itg + 1;
        ihi       = itg + 2;
        flo       = fh;
        fhi       = fh_plus;
        bracketed = true;
      }
    }
  }

  if (!bracketed)
  {
    // Evaluate log(e) at the table boundaries using 4 lookups each
    Real loge_min = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, 0);
    Real loge_max = eval_at_it(ECLOGE, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);

    // Here log-space using the same representation as f(iti) below,
    // so the bracket invariant flo > 0, fhi < 0 holds strictly and no
    // FP gap can appear between an exp(loge_min) round-trip.
    if (var <= loge_min)
    {
      boundary_lo = true;
      return;
    }
    if (var >= loge_max)
    {
      boundary_hi = true;
      return;
    }

    flo = var - loge_min;  // > 0 strictly
    fhi = var - loge_max;  // < 0 strictly

    // Log-space guarantees flo > 0 and fhi < 0; equality cases are collapesd
    // onto the boundary_lo/hi returns above.
    assert(flo * fhi <= 0);

    int it_guess =
      static_cast<int>(static_cast<Real>(m_nt - 1) * flo / (flo - fhi));
    it_guess = std::max(0, std::min(m_nt - 2, it_guess));

    Real fg = f(it_guess);
    if (fg * flo <= 0)
    {
      ihi = it_guess;
      fhi = fg;
    }
    else
    {
      ilo = it_guess;
      flo = fg;
    }

    while (ihi - ilo > 1)
    {
      int ip  = ilo + (ihi - ilo) / 2;
      Real fp = f(ip);
      if (fp * flo <= 0)
      {
        ihi = ip;
        fhi = fp;
      }
      else
      {
        ilo = ip;
        flo = fp;
      }
    }
  }

  assert(ihi - ilo == 1 && flo * fhi <= 0);
  if (guess_it)
  {
    *guess_it = ilo;
  }
  Real ltlo = m_log_t[ilo];
  Real lthi = m_log_t[ihi];

  if (flo == 0)
  {
    lt  = ltlo;
    wt0 = 1.0;
    wt1 = 0.0;
  }
  else if (fhi == 0)
  {
    lt  = lthi;
    wt0 = 0.0;
    wt1 = 1.0;
  }
  else
  {
    lt  = ltlo - flo * (lthi - ltlo) / (fhi - flo);
    wt1 = (lt - ltlo) * m_id_log_t;
    wt0 = 1.0 - wt1;
  }

  it = ilo;
}

void EOSCompOSE::TemperaturePressureAndEnthalpyFromE(Real n,
                                                     Real e,
                                                     Real* Y,
                                                     Real* T,
                                                     Real* P,
                                                     Real* h,
                                                     int* guess_it)
{
  assert(m_initialized);
  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1, lt;
  bool boundary_lo, boundary_hi;

  FindTBracketAndWeights(n,
                         e,
                         Y,
                         guess_it,
                         in,
                         iy,
                         it,
                         wn0,
                         wn1,
                         wy0,
                         wy1,
                         wt0,
                         wt1,
                         lt,
                         boundary_lo,
                         boundary_hi);

  if (boundary_lo)
  {
    *T = min_T;
    PressureAndEnthalpy(n, *T, Y, P, h);
    return;
  }
  if (boundary_hi)
  {
    *T = max_T;
    PressureAndEnthalpy(n, *T, Y, P, h);
    return;
  }

  *T = exp(lt);

  ptrdiff_t bp00 = index(ECLOGP, in, iy, it);
  ptrdiff_t bp01 = index(ECLOGP, in, iy + 1, it);
  ptrdiff_t bp10 = index(ECLOGP, in + 1, iy, it);
  ptrdiff_t bp11 = index(ECLOGP, in + 1, iy + 1, it);

  Real logP = wn0 * (wy0 * (wt0 * m_table[bp00] + wt1 * m_table[bp00 + 1]) +
                     wy1 * (wt0 * m_table[bp01] + wt1 * m_table[bp01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[bp10] + wt1 * m_table[bp10 + 1]) +
                     wy1 * (wt0 * m_table[bp11] + wt1 * m_table[bp11 + 1]));

  *P = exp(logP);
  *h = (*P + e) / n;
}

void EOSCompOSE::PressureAndEnthalpyFromE(Real n,
                                          Real e,
                                          Real* Y,
                                          Real* P,
                                          Real* h,
                                          int* guess_it)
{
  assert(m_initialized);
  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1, lt;
  bool boundary_lo, boundary_hi;

  FindTBracketAndWeights(n,
                         e,
                         Y,
                         guess_it,
                         in,
                         iy,
                         it,
                         wn0,
                         wn1,
                         wy0,
                         wy1,
                         wt0,
                         wt1,
                         lt,
                         boundary_lo,
                         boundary_hi);

  if (boundary_lo)
  {
    PressureAndEnthalpy(n, min_T, Y, P, h);
    return;
  }
  if (boundary_hi)
  {
    PressureAndEnthalpy(n, max_T, Y, P, h);
    return;
  }

  ptrdiff_t bp00 = index(ECLOGP, in, iy, it);
  ptrdiff_t bp01 = index(ECLOGP, in, iy + 1, it);
  ptrdiff_t bp10 = index(ECLOGP, in + 1, iy, it);
  ptrdiff_t bp11 = index(ECLOGP, in + 1, iy + 1, it);

  Real logP = wn0 * (wy0 * (wt0 * m_table[bp00] + wt1 * m_table[bp00 + 1]) +
                     wy1 * (wt0 * m_table[bp01] + wt1 * m_table[bp01 + 1])) +
              wn1 * (wy0 * (wt0 * m_table[bp10] + wt1 * m_table[bp10 + 1]) +
                     wy1 * (wt0 * m_table[bp11] + wt1 * m_table[bp11 + 1]));

  *P = exp(logP);
  *h = (*P + e) / n;
}

Real EOSCompOSE::SoundSpeed(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECCS, n, T, Y[0]);
}

Real EOSCompOSE::FrYn(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECYN, n, T, Y[0]);
}

Real EOSCompOSE::FrYp(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECYP, n, T, Y[0]);
}

Real EOSCompOSE::FrXh(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECXH, n, T, Y[0]);
}

Real EOSCompOSE::AN(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECAN, n, T, Y[0]);
}

Real EOSCompOSE::ZN(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECZN, n, T, Y[0]);
}

Real EOSCompOSE::SpecificInternalEnergy(Real n, Real T, Real* Y)
{
  return Energy(n, T, Y) / (mb * n) - 1;
}

Real EOSCompOSE::BaryonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECMUB, n, T, Y[0]);
}

Real EOSCompOSE::ChargeChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECMUQ, n, T, Y[0]);
}

Real EOSCompOSE::ElectronLeptonChemicalPotential(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  return eval_at_nty(ECMUL, n, T, Y[0]);
}

Real EOSCompOSE::InteractionPotentialDifference(Real n, Real T, Real* Y)
{
  assert(m_initialized);
  if (!m_has_dU)
  {
    std::cerr << "EOSCompOSE: dU dataset not present in loaded table; "
                 "regenerate CompOSE HDF5 with dU populated."
              << std::endl;
    assert(false);
  }
  return eval_at_nty(ECDU, n, T, Y[0]);
}

Real EOSCompOSE::MinimumEnthalpy()
{
  return m_min_h;
}

Real EOSCompOSE::MinimumPressure(Real n, Real* Y)
{
  return Pressure(n, min_T, Y);
}

Real EOSCompOSE::MaximumPressure(Real n, Real* Y)
{
  return Pressure(n, max_T, Y);
}

Real EOSCompOSE::MinimumEnergy(Real n, Real* Y)
{
  return Energy(n, min_T, Y);
}

Real EOSCompOSE::MaximumEnergy(Real n, Real* Y)
{
  return Energy(n, max_T, Y);
}

Real EOSCompOSE::MinimumEntropy(Real n, Real* Y)
{
  return Entropy(n, min_T, Y);
}

Real EOSCompOSE::MaximumEntropy(Real n, Real* Y)
{
  return Entropy(n, max_T, Y);
}

void EOSCompOSE::UniformizeAxes()
{
  vector<Real> target_log_nb(m_nn);
  vector<Real> target_log_t(m_nt);
  vector<Real> target_yq(m_ny);

  for (int in = 0; in < m_nn; ++in)
  {
    target_log_nb[in] = m_log_nb[0] + static_cast<Real>(in) *
                                        (m_log_nb[m_nn - 1] - m_log_nb[0]) /
                                        static_cast<Real>(m_nn - 1);
  }
  for (int it = 0; it < m_nt; ++it)
  {
    target_log_t[it] = m_log_t[0] + static_cast<Real>(it) *
                                      (m_log_t[m_nt - 1] - m_log_t[0]) /
                                      static_cast<Real>(m_nt - 1);
  }
  for (int iy = 0; iy < m_ny; ++iy)
  {
    target_yq[iy] = m_yq[0] + static_cast<Real>(iy) *
                                (m_yq[m_ny - 1] - m_yq[0]) /
                                static_cast<Real>(m_ny - 1);
  }

  target_log_nb[0]        = m_log_nb[0];
  target_log_nb[m_nn - 1] = m_log_nb[m_nn - 1];
  target_log_t[0]         = m_log_t[0];
  target_log_t[m_nt - 1]  = m_log_t[m_nt - 1];
  target_yq[0]            = m_yq[0];
  target_yq[m_ny - 1]     = m_yq[m_ny - 1];

  auto check_target_axis = [](const vector<Real>& axis, const char* name)
  {
    for (size_t i = 0; i < axis.size(); ++i)
    {
      if (!isfinite(axis[i]))
      {
        TableError(string("nonfinite target ") + name + " coordinate");
      }
      if (i > 0 && !(axis[i] > axis[i - 1]))
      {
        TableError(string("target ") + name +
                   " coordinates are not strictly increasing");
      }
    }
  };

  check_target_axis(target_log_nb, "log(nb)");
  check_target_axis(target_log_t, "log(T)");
  check_target_axis(target_yq, "Yq");

  struct CellWeight
  {
    int lower;
    Real w0;
    Real w1;
  };

  auto find_cell = [](const Real* axis, int n, Real x) -> CellWeight
  {
    CellWeight result;
    if (x == axis[0])
    {
      result.lower = 0;
      result.w1    = 0.0;
    }
    else if (x == axis[n - 1])
    {
      result.lower = n - 2;
      result.w1    = 1.0;
    }
    else
    {
      const Real* upper = upper_bound(axis, axis + n, x);
      if (upper == axis || upper == axis + n)
      {
        TableError("target coordinate is outside the source axis");
      }
      result.lower = static_cast<int>(upper - axis) - 1;
      result.w1    = (x - axis[result.lower]) /
                     (axis[result.lower + 1] - axis[result.lower]);
    }
    result.w0 = 1.0 - result.w1;
    if (!isfinite(result.w0) || !isfinite(result.w1))
    {
      TableError("nonfinite source-cell interpolation weight");
    }
    return result;
  };

  const size_t ncell = CheckedProduct(
    CheckedProduct(
      static_cast<size_t>(m_nn), static_cast<size_t>(m_ny), "cell count"),
    static_cast<size_t>(m_nt),
    "cell count");
  const size_t ntable =
    CheckedProduct(static_cast<size_t>(ECNVARS), ncell, "table count");
  CheckAllocationBytes(ntable, sizeof(Real), "uniformized EOS table");
  if (ntable > static_cast<size_t>(numeric_limits<ptrdiff_t>::max()))
  {
    TableError("uniformized EOS table exceeds index range");
  }
  vector<Real> remapped(ntable);

  for (int in = 0; in < m_nn; ++in)
  {
    const CellWeight wn = find_cell(m_log_nb, m_nn, target_log_nb[in]);
    for (int iy = 0; iy < m_ny; ++iy)
    {
      const CellWeight wy = find_cell(m_yq, m_ny, target_yq[iy]);
      for (int it = 0; it < m_nt; ++it)
      {
        const CellWeight wt = find_cell(m_log_t, m_nt, target_log_t[it]);
        for (int iv = 0; iv < ECNVARS; ++iv)
        {
          const Real v000 = m_table[index(iv, wn.lower, wy.lower, wt.lower)];
          const Real v001 =
            m_table[index(iv, wn.lower, wy.lower, wt.lower + 1)];
          const Real v010 =
            m_table[index(iv, wn.lower, wy.lower + 1, wt.lower)];
          const Real v011 =
            m_table[index(iv, wn.lower, wy.lower + 1, wt.lower + 1)];
          const Real v100 =
            m_table[index(iv, wn.lower + 1, wy.lower, wt.lower)];
          const Real v101 =
            m_table[index(iv, wn.lower + 1, wy.lower, wt.lower + 1)];
          const Real v110 =
            m_table[index(iv, wn.lower + 1, wy.lower + 1, wt.lower)];
          const Real v111 =
            m_table[index(iv, wn.lower + 1, wy.lower + 1, wt.lower + 1)];
          const Real value = wn.w0 * (wy.w0 * (wt.w0 * v000 + wt.w1 * v001) +
                                      wy.w1 * (wt.w0 * v010 + wt.w1 * v011)) +
                             wn.w1 * (wy.w0 * (wt.w0 * v100 + wt.w1 * v101) +
                                      wy.w1 * (wt.w0 * v110 + wt.w1 * v111));
          if (!isfinite(value))
          {
            ostringstream stream;
            stream << "nonfinite remapped lane " << iv << " at (" << in << ", "
                   << iy << ", " << it << ")";
            TableError(stream.str());
          }
          remapped[index(iv, in, iy, it)] = value;
        }
      }
    }
  }

  copy(remapped.begin(), remapped.end(), m_table);
  copy(target_log_nb.begin(), target_log_nb.end(), m_log_nb);
  copy(target_log_t.begin(), target_log_t.end(), m_log_t);
  copy(target_yq.begin(), target_yq.end(), m_yq);

  m_id_log_nb = InverseSpacing(m_log_nb[0], m_log_nb[1], "log(nb)");
  m_id_log_t  = InverseSpacing(m_log_t[0], m_log_t[1], "log(T)");
  m_id_yq     = InverseSpacing(m_yq[0], m_yq[1], "Yq");
}

void EOSCompOSE::ReadTableFromFile(std::string fname, bool uniformize_axes)
{
#pragma omp critical(EOSCompose_ReadTable)
  {
    if (m_initialized == false)
    {
      hid_t file_id   = H5I_INVALID_HID;
      double* scratch = nullptr;
      try
      {
        hsize_t snb, st, syq;

        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0)
        {
          TableError(string("cannot open EOS table '") + fname + "'");
        }

        CheckAxisDataset(file_id, "nb", &snb);
        CheckAxisDataset(file_id, "t", &st);
        CheckAxisDataset(file_id, "yq", &syq);
        if (snb > static_cast<hsize_t>(numeric_limits<int>::max()) ||
            st > static_cast<hsize_t>(numeric_limits<int>::max()) ||
            syq > static_cast<hsize_t>(numeric_limits<int>::max()))
        {
          TableError("axis extent exceeds int index range");
        }

        const char* const fields[] = { "Q1",   "Q2",   "Q3",   "Q4",
                                       "Q5",   "Q7",   "cs2",  "Y[n]",
                                       "Y[p]", "Y[N]", "A[N]", "Z[N]" };
        for (size_t i = 0; i < sizeof(fields) / sizeof(*fields); ++i)
        {
          CheckFieldDataset(file_id, fields[i], snb, syq, st);
        }
        const htri_t dU_exists = H5Lexists(file_id, "dU", H5P_DEFAULT);
        if (dU_exists < 0)
        {
          TableError("cannot determine whether dataset 'dU' exists");
        }
        if (dU_exists > 0)
        {
          CheckFieldDataset(file_id, "dU", snb, syq, st);
        }
        CheckMassDataset(file_id, "mn");
        CheckMassDataset(file_id, "mp");

        m_nn = static_cast<int>(snb);
        m_nt = static_cast<int>(st);
        m_ny = static_cast<int>(syq);
        const size_t ncell =
          CheckedProduct(CheckedProduct(static_cast<size_t>(m_nn),
                                        static_cast<size_t>(m_ny),
                                        "cell count"),
                         static_cast<size_t>(m_nt),
                         "cell count");
        const size_t ntable =
          CheckedProduct(static_cast<size_t>(ECNVARS), ncell, "table count");
        CheckAllocationBytes(ncell, sizeof(double), "EOS scratch array");
        CheckAllocationBytes(ntable, sizeof(Real), "EOS table");
        CheckAllocationBytes(
          static_cast<size_t>(m_nn), sizeof(Real), "log(nb) axis");
        CheckAllocationBytes(
          static_cast<size_t>(m_nt), sizeof(Real), "log(T) axis");
        CheckAllocationBytes(
          static_cast<size_t>(m_ny), sizeof(Real), "Yq axis");
        if (ntable > static_cast<size_t>(numeric_limits<ptrdiff_t>::max()))
        {
          TableError("EOS table exceeds index range");
        }

        m_log_nb = new Real[m_nn];
        m_log_t  = new Real[m_nt];
        m_yq     = new Real[m_ny];
        m_table  = new Real[ntable];
        scratch  = new double[ncell];
        m_has_dU = false;

        CheckHDFStatus(H5LTread_dataset_double(file_id, "nb", scratch),
                       "cannot read dataset 'nb'");
        double previous_log_nb = 0.0;
        for (int in = 0; in < m_nn; ++in)
        {
          const double value = scratch[in];
          if (!isfinite(value) || value <= 0.0)
          {
            TableError(AxisError("nb", in, "must be finite and positive"));
          }
          const double log_value = log(value);
          if (!isfinite(log_value))
          {
            TableError(AxisError("nb", in, "has a nonfinite logarithm"));
          }
          if (in > 0 && !(log_value > previous_log_nb))
          {
            TableError(
              AxisError("nb", in, "is not strictly increasing in log(nb)"));
          }
          const Real stored = static_cast<Real>(log_value);
          if (!isfinite(stored) || (in > 0 && !(stored > m_log_nb[in - 1])))
          {
            TableError(AxisError(
              "nb", in, "cannot form a strictly increasing stored log axis"));
          }
          m_log_nb[in]    = stored;
          previous_log_nb = log_value;
        }
        min_n = scratch[0];
        max_n = scratch[m_nn - 1];

        CheckHDFStatus(H5LTread_dataset_double(file_id, "t", scratch),
                       "cannot read dataset 't'");
        double previous_log_t = 0.0;
        for (int it = 0; it < m_nt; ++it)
        {
          const double value = scratch[it];
          if (!isfinite(value) || value <= 0.0)
          {
            TableError(AxisError("t", it, "must be finite and positive"));
          }
          const double log_value = log(value);
          if (!isfinite(log_value))
          {
            TableError(AxisError("t", it, "has a nonfinite logarithm"));
          }
          if (it > 0 && !(log_value > previous_log_t))
          {
            TableError(
              AxisError("t", it, "is not strictly increasing in log(T)"));
          }
          const Real stored = static_cast<Real>(log_value);
          if (!isfinite(stored) || (it > 0 && !(stored > m_log_t[it - 1])))
          {
            TableError(AxisError(
              "t", it, "cannot form a strictly increasing stored log axis"));
          }
          m_log_t[it]    = stored;
          previous_log_t = log_value;
        }
        min_T = scratch[0];
        max_T = scratch[m_nt - 1];

        CheckHDFStatus(H5LTread_dataset_double(file_id, "yq", scratch),
                       "cannot read dataset 'yq'");
        double previous_yq = 0.0;
        for (int iy = 0; iy < m_ny; ++iy)
        {
          const double value = scratch[iy];
          if (!isfinite(value))
          {
            TableError(AxisError("yq", iy, "must be finite"));
          }
          if (iy > 0 && !(value > previous_yq))
          {
            TableError(AxisError("yq", iy, "is not strictly increasing"));
          }
          const Real stored = static_cast<Real>(value);
          if (!isfinite(stored) || (iy > 0 && !(stored > m_yq[iy - 1])))
          {
            TableError(AxisError(
              "yq", iy, "cannot form a strictly increasing stored axis"));
          }
          m_yq[iy]    = stored;
          previous_yq = value;
        }
        min_Y[0] = scratch[0];
        max_Y[0] = scratch[m_ny - 1];

        CheckHDFStatus(H5LTread_dataset_double(file_id, "mn", scratch),
                       "cannot read dataset 'mn'");
        if (!isfinite(scratch[0]) || scratch[0] <= 0.0)
        {
          TableError("dataset 'mn' must be finite and positive");
        }
        mb = static_cast<Real>(scratch[0]);
        if (!isfinite(mb) || mb <= 0.0)
        {
          TableError(
            "dataset 'mn' does not produce a finite positive stored value");
        }
        const Real mn = mb;

        CheckHDFStatus(H5LTread_dataset_double(file_id, "mp", scratch),
                       "cannot read dataset 'mp'");
        if (!isfinite(scratch[0]) || scratch[0] <= 0.0)
        {
          TableError("dataset 'mp' must be finite and positive");
        }
        const Real mp = static_cast<Real>(scratch[0]);
        if (!isfinite(mp) || mp <= 0.0)
        {
          TableError(
            "dataset 'mp' does not produce a finite positive stored value");
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Q1", scratch),
                       "cannot read dataset 'Q1'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw) || raw <= 0.0)
              {
                TableError(
                  FieldError("Q1", in, iy, it, "must be finite and positive"));
              }
              const Real stored = static_cast<Real>(log(raw) + m_log_nb[in]);
              CheckStoredFinite(stored, "Q1", in, iy, it);
              m_table[index(ECLOGP, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Q2", scratch),
                       "cannot read dataset 'Q2'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw))
              {
                TableError(FieldError("Q2", in, iy, it, "must be finite"));
              }
              const Real stored = static_cast<Real>(raw);
              CheckStoredFinite(stored, "Q2", in, iy, it);
              m_table[index(ECENT, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Q3", scratch),
                       "cannot read dataset 'Q3'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw))
              {
                TableError(FieldError("Q3", in, iy, it, "must be finite"));
              }
              const Real stored = static_cast<Real>(mb * (raw + 1.0));
              CheckStoredFinite(stored, "Q3", in, iy, it);
              m_table[index(ECMUB, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Q4", scratch),
                       "cannot read dataset 'Q4'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw))
              {
                TableError(FieldError("Q4", in, iy, it, "must be finite"));
              }
              const Real stored = static_cast<Real>(mb * raw);
              CheckStoredFinite(stored, "Q4", in, iy, it);
              m_table[index(ECMUQ, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Q5", scratch),
                       "cannot read dataset 'Q5'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw))
              {
                TableError(FieldError("Q5", in, iy, it, "must be finite"));
              }
              const Real stored = static_cast<Real>(mb * raw);
              CheckStoredFinite(stored, "Q5", in, iy, it);
              m_table[index(ECMUL, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Q7", scratch),
                       "cannot read dataset 'Q7'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw) || raw <= -1.0)
              {
                TableError(FieldError(
                  "Q7", in, iy, it, "must be finite and greater than -1"));
              }
              const Real stored =
                static_cast<Real>(log(mb * (raw + 1.0)) + m_log_nb[in]);
              CheckStoredFinite(stored, "Q7", in, iy, it);
              m_table[index(ECLOGE, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "cs2", scratch),
                       "cannot read dataset 'cs2'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw) || raw < 0.0)
              {
                TableError(FieldError(
                  "cs2", in, iy, it, "must be finite and nonnegative"));
              }
              const Real stored = static_cast<Real>(sqrt(raw));
              CheckStoredFinite(stored, "cs2", in, iy, it);
              m_table[index(ECCS, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Y[n]", scratch),
                       "cannot read dataset 'Y[n]'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw) || raw < 0.0)
              {
                TableError(FieldError(
                  "Y[n]", in, iy, it, "must be finite and nonnegative"));
              }
              const Real stored = static_cast<Real>(raw);
              CheckStoredFinite(stored, "Y[n]", in, iy, it);
              m_table[index(ECYN, in, iy, it)] = stored;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Y[p]", scratch),
                       "cannot read dataset 'Y[p]'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const double raw = scratch[index(0, in, iy, it)];
              if (!isfinite(raw) || raw < 0.0)
              {
                TableError(FieldError(
                  "Y[p]", in, iy, it, "must be finite and nonnegative"));
              }
              const Real stored = static_cast<Real>(raw);
              CheckStoredFinite(stored, "Y[p]", in, iy, it);
              m_table[index(ECYP, in, iy, it)] = stored;
            }
          }
        }

        vector<double> raw_yn(ncell);
        vector<double> raw_an(ncell);
        CheckHDFStatus(H5LTread_dataset_double(file_id, "Y[N]", scratch),
                       "cannot read dataset 'Y[N]'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const ptrdiff_t source = index(0, in, iy, it);
              const double raw       = scratch[source];
              if (!isfinite(raw) || raw < 0.0)
              {
                TableError(FieldError(
                  "Y[N]", in, iy, it, "must be finite and nonnegative"));
              }
              const Real stored = static_cast<Real>(raw);
              CheckStoredFinite(stored, "Y[N]", in, iy, it);
              m_table[index(ECXH, in, iy, it)]    = stored;
              raw_yn[static_cast<size_t>(source)] = raw;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "A[N]", scratch),
                       "cannot read dataset 'A[N]'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const ptrdiff_t source = index(0, in, iy, it);
              const double raw       = scratch[source];
              if (!isfinite(raw) || raw < 0.0)
              {
                TableError(FieldError(
                  "A[N]", in, iy, it, "must be finite and nonnegative"));
              }
              const Real stored = static_cast<Real>(raw);
              CheckStoredFinite(stored, "A[N]", in, iy, it);
              m_table[index(ECAN, in, iy, it)]    = stored;
              raw_an[static_cast<size_t>(source)] = raw;
            }
          }
        }

        CheckHDFStatus(H5LTread_dataset_double(file_id, "Z[N]", scratch),
                       "cannot read dataset 'Z[N]'");
        for (int in = 0; in < m_nn; ++in)
        {
          for (int iy = 0; iy < m_ny; ++iy)
          {
            for (int it = 0; it < m_nt; ++it)
            {
              const ptrdiff_t source = index(0, in, iy, it);
              const double raw       = scratch[source];
              if (!isfinite(raw) || raw < 0.0)
              {
                TableError(FieldError(
                  "Z[N]", in, iy, it, "must be finite and nonnegative"));
              }
              const Real zn = static_cast<Real>(raw);
              CheckStoredFinite(zn, "Z[N]", in, iy, it);
              const Real an = m_table[index(ECAN, in, iy, it)];
              if (raw_yn[static_cast<size_t>(source)] > 0.0 &&
                  (!(raw_an[static_cast<size_t>(source)] > 0.0) ||
                   !(raw <= raw_an[static_cast<size_t>(source)])))
              {
                TableError(
                  FieldError("Z[N]",
                             in,
                             iy,
                             it,
                             "must satisfy 0 <= Z[N] <= A[N] when Y[N] > 0"));
              }
              m_table[index(ECZN, in, iy, it)] = zn;
              const Real xh =
                static_cast<Real>(an * raw_yn[static_cast<size_t>(source)]);
              CheckStoredFinite(xh, "Y[N]", in, iy, it);
              m_table[index(ECXH, in, iy, it)] = xh;
            }
          }
        }

        vector<double>().swap(raw_yn);
        vector<double>().swap(raw_an);

        if (dU_exists > 0)
        {
          CheckHDFStatus(H5LTread_dataset_double(file_id, "dU", scratch),
                         "cannot read dataset 'dU'");
          for (int in = 0; in < m_nn; ++in)
          {
            for (int iy = 0; iy < m_ny; ++iy)
            {
              for (int it = 0; it < m_nt; ++it)
              {
                const double raw = scratch[index(0, in, iy, it)];
                if (!isfinite(raw))
                {
                  TableError(FieldError("dU", in, iy, it, "must be finite"));
                }
                const Real stored = static_cast<Real>(raw);
                CheckStoredFinite(stored, "dU", in, iy, it);
                m_table[index(ECDU, in, iy, it)] = stored;
              }
            }
          }
          m_has_dU = true;
        }
        else
        {
          fill(m_table + index(ECDU, 0, 0, 0),
               m_table + index(ECDU, 0, 0, 0) + ncell,
               Real(0.0));
          m_has_dU = false;
        }

        delete[] scratch;
        scratch                   = nullptr;
        const herr_t close_status = H5Fclose(file_id);
        file_id                   = H5I_INVALID_HID;
        CheckHDFStatus(close_status, "cannot close EOS table");

        if (uniformize_axes)
        {
          UniformizeAxes();
        }
        else
        {
          m_id_log_nb = InverseSpacing(m_log_nb[0], m_log_nb[1], "log(nb)");
          m_id_log_t  = InverseSpacing(m_log_t[0], m_log_t[1], "log(T)");
          m_id_yq     = InverseSpacing(m_yq[0], m_yq[1], "Yq");
        }

        m_initialized = true;
        m_min_h       = numeric_limits<Real>::max();
        for (int in = 0; in < m_nn; ++in)
        {
          Real const nb = exp(m_log_nb[in]);
          for (int it = 0; it < m_nt; ++it)
          {
            Real const t = exp(m_log_t[it]);
            for (int iy = 0; iy < m_ny; ++iy)
            {
              m_min_h = min(m_min_h, Enthalpy(nb, t, &m_yq[iy]));
            }
          }
        }

        sm_id_log_nb = m_id_log_nb;
        sm_id_log_t  = m_id_log_t;
        sm_id_yq     = m_id_yq;

        sm_nn = m_nn;
        sm_nt = m_nt;
        sm_ny = m_ny;

        sm_min_h = m_min_h;

        s_mb       = mb;
        s_max_n    = max_n;
        s_min_n    = min_n;
        s_max_T    = max_T;
        s_min_T    = min_T;
        s_max_Y[0] = max_Y[0];
        s_min_Y[0] = min_Y[0];

        s_mn = mn;
        s_mp = mp;
      }
      catch (...)
      {
        if (m_initialized)
        {
          throw;
        }
        if (scratch != nullptr)
        {
          delete[] scratch;
          scratch = nullptr;
        }
        if (file_id != H5I_INVALID_HID)
        {
          H5Fclose(file_id);
          file_id = H5I_INVALID_HID;
        }
        delete[] m_log_nb;
        delete[] m_log_t;
        delete[] m_yq;
        delete[] m_table;
        m_log_nb      = nullptr;
        m_log_t       = nullptr;
        m_yq          = nullptr;
        m_table       = nullptr;
        m_nn          = 0;
        m_nt          = 0;
        m_ny          = 0;
        m_id_log_nb   = numeric_limits<Real>::quiet_NaN();
        m_id_log_t    = numeric_limits<Real>::quiet_NaN();
        m_id_yq       = numeric_limits<Real>::quiet_NaN();
        m_min_h       = numeric_limits<Real>::max();
        m_has_dU      = false;
        m_initialized = false;
        throw;
      }
    }  // if (sm_initialized==false)
  }  // omp critical (EOSCompOSE_ReadTable)

  // Disseminate applicable static variables to local memory
  m_id_log_nb = sm_id_log_nb;
  m_id_log_t  = sm_id_log_t;
  m_id_yq     = sm_id_yq;

  m_nn = sm_nn;
  m_nt = sm_nt;
  m_ny = sm_ny;

  m_min_h = sm_min_h;

  mb       = s_mb;
  max_n    = s_max_n;
  min_n    = s_min_n;
  max_T    = s_max_T;
  min_T    = s_min_T;
  max_Y[0] = s_max_Y[0];
  min_Y[0] = s_min_Y[0];
}

Real EOSCompOSE::temperature_from_var(int iv, Real var, Real n, Real Yq) const
{
  int in, iy;
  Real wn0, wn1, wy0, wy1;
  weight_idx_ln(&wn0, &wn1, &in, log(n));
  weight_idx_yq(&wy0, &wy1, &iy, Yq);
  Real var_min = eval_at_it(iv, wn0, wn1, in, wy0, wy1, iy, 0);
  Real var_max = eval_at_it(iv, wn0, wn1, in, wy0, wy1, iy, m_nt - 1);
  return temperature_from_var_precomp(
    var_min, var_max, iv, var, wn0, wn1, in, wy0, wy1, iy);
}

Real EOSCompOSE::temperature_from_var_precomp(Real var_min,
                                              Real var_max,
                                              int iv,
                                              Real var,
                                              Real wn0,
                                              Real wn1,
                                              int in,
                                              Real wy0,
                                              Real wy1,
                                              int iy) const
{
  // Pre-compute the four base offsets for the (iv, in, iy) cell.
  // Temperature indices are contiguous, so f(it) = m_table[base + it].
  ptrdiff_t const b00 = index(iv, in, iy, 0);
  ptrdiff_t const b01 = index(iv, in, iy + 1, 0);
  ptrdiff_t const b10 = index(iv, in + 1, iy, 0);
  ptrdiff_t const b11 = index(iv, in + 1, iy + 1, 0);

  // Lambda: evaluate the bilinear interpolant at temperature index it,
  // return residual (var - interpolated_value).
  auto f = [=](int it) -> Real
  {
    Real var_pt = wn0 * (wy0 * m_table[b00 + it] + wy1 * m_table[b01 + it]) +
                  wn1 * (wy0 * m_table[b10 + it] + wy1 * m_table[b11 + it]);
    return var - var_pt;
  };

  int ilo  = 0;
  int ihi  = m_nt - 1;
  Real flo = var - var_min;
  Real fhi = var - var_max;

  // Binary search for the sign change.
  // The table variable is monotone in T at fixed (n, Yq), so there is
  // at most one sign change.  Binary search finds it in O(log m_nt).
  if (flo * fhi > 0)
  {
    // Should not happen after the caller's bounds check, but handle
    // gracefully: bracket already at adjacent points is the best we can do.
  }
  else
  {
    // Use the boundary residuals to estimate the root location via
    // false-position on the full index range.  For log-stored variables
    // (ECLOGP, ECLOGE) the interpolant is roughly linear in the uniform
    // log-T index, so this guess typically lands within a few cells of
    // the true root, shrinking the bracket from m_nt to O(1) and saving
    // most of the subsequent binary-search iterations.
    int it_guess =
      static_cast<int>(static_cast<Real>(m_nt - 1) * flo / (flo - fhi));
    it_guess = std::max(0, std::min(m_nt - 2, it_guess));

    Real fg = f(it_guess);
    if (fg * flo <= 0)
    {
      ihi = it_guess;
      fhi = fg;
    }
    else
    {
      ilo = it_guess;
      flo = fg;
    }

    // Refine the bracket with standard binary search.
    while (ihi - ilo > 1)
    {
      int ip  = ilo + (ihi - ilo) / 2;
      Real fp = f(ip);
      if (fp * flo <= 0)
      {
        ihi = ip;
        fhi = fp;
      }
      else
      {
        ilo = ip;
        flo = fp;
      }
    }
  }

  assert(ihi - ilo == 1 || flo * fhi <= 0);
  Real ltlo = m_log_t[ilo];
  Real lthi = m_log_t[ihi];

  if (flo == 0)
  {
    return exp(ltlo);
  }
  if (fhi == 0)
  {
    return exp(lthi);
  }

  // False-position interpolation in log-T for sub-cell accuracy.
  Real lt = ltlo - flo * (lthi - ltlo) / (fhi - flo);
  return exp(lt);
}

Real EOSCompOSE::eval_at_nty(int vi, Real n, Real T, Real Yq) const
{
  return eval_at_lnty(vi, log(n), log(T), Yq);
}

void EOSCompOSE::weight_idx_ln(Real* w0, Real* w1, int* in, Real log_n) const
{
  *in = (log_n - m_log_nb[0]) * m_id_log_nb;
  // if outside table limits, linearly extrapolate
  if (*in > m_nn - 2)
  {
    *in = m_nn - 2;
  }
  else if (*in < 0)
  {
    *in = 0;
  }

  *w1 = (log_n - m_log_nb[*in]) * m_id_log_nb;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSE::weight_idx_yq(Real* w0, Real* w1, int* iy, Real yq) const
{
  *iy = (yq - m_yq[0]) * m_id_yq;
  // if outside table limits, linearly extrapolate
  if (*iy > m_ny - 2)
  {
    *iy = m_ny - 2;
  }
  else if (*iy < 0)
  {
    *iy = 0;
  }

  *w1 = (yq - m_yq[*iy]) * m_id_yq;
  *w0 = 1.0 - (*w1);
}

void EOSCompOSE::weight_idx_lt(Real* w0, Real* w1, int* it, Real log_t) const
{
  *it = (log_t - m_log_t[0]) * m_id_log_t;
  // if outside table limits, linearly extrapolate
  if (*it > m_nt - 2)
  {
    *it = m_nt - 2;
  }
  else if (*it < 0)
  {
    *it = 0;
  }
  *w1 = (log_t - m_log_t[*it]) * m_id_log_t;
  *w0 = 1.0 - (*w1);
}

Real EOSCompOSE::eval_at_lnty(int iv, Real log_n, Real log_t, Real yq) const
{
  int in, iy, it;
  Real wn0, wn1, wy0, wy1, wt0, wt1;

  weight_idx_ln(&wn0, &wn1, &in, log_n);
  weight_idx_yq(&wy0, &wy1, &iy, yq);
  weight_idx_lt(&wt0, &wt1, &it, log_t);

  // Pre-compute the four base offsets; it and it+1 are contiguous.
  ptrdiff_t const b00 = index(iv, in, iy, it);
  ptrdiff_t const b01 = index(iv, in, iy + 1, it);
  ptrdiff_t const b10 = index(iv, in + 1, iy, it);
  ptrdiff_t const b11 = index(iv, in + 1, iy + 1, it);

  return wn0 * (wy0 * (wt0 * m_table[b00] + wt1 * m_table[b00 + 1]) +
                wy1 * (wt0 * m_table[b01] + wt1 * m_table[b01 + 1])) +
         wn1 * (wy0 * (wt0 * m_table[b10] + wt1 * m_table[b10 + 1]) +
                wy1 * (wt0 * m_table[b11] + wt1 * m_table[b11 + 1]));
}

// #else //HDF5OUTPUT
//  Consider adding no-ops here?
// #endif
