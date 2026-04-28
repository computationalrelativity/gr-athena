#include <algorithm>  // std::min
#include <cstring>    // std::memcpy
#include <limits>
#include <map>
#include <string>
#include <utility>  // std::pair
#include <vector>

#include "surfaces.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../m1/m1.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/interp_barycentric.hpp"
#include "../utils/lagrange_interp.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/utils.hpp"
#include "../z4c/z4c.hpp"

// hdf5 / mpi macros
#include "../outputs/outputs.hpp"

// ============================================================================
namespace gra::mesh::surfaces
{
// ============================================================================

SurfacesCylindrical::SurfacesCylindrical(Mesh* pm,
                                         ParameterInput* pin,
                                         const int par_ix)
    : Surfaces(pm, pin, par_ix)
{
  num_surf = pin->GetOrAddRealArray(par_block_name, "radii", -1, 0).GetSize();
  for (int surf_ix = 0; surf_ix < num_surf; ++surf_ix)
  {
    psurf.push_back(
      new SurfaceCylindrical(pm, pin, dynamic_cast<Surfaces*>(this), surf_ix));
  }
}

void SurfaceCylindrical::write_hdf5_coordinates(hid_t& id_file,
                                                const std::string& six)
{
#ifdef HDF5OUTPUT
  // scalars [grid]
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/R", rad);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_min", z_min);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_max", z_max);

  // 1d arrays [grid]
  hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/ph", ph);
  hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/z", z);
#endif
}

SurfaceCylindrical::SurfaceCylindrical(Mesh* pm,
                                       ParameterInput* pin,
                                       Surfaces* psurfs,
                                       const int surf_ix)
    : Surface(pm, pin, psurfs, surf_ix)
{
  // extract [target] sampling variety ----------------------------------------
  {
    static const std::map<std::string, variety_sampling> opt_vs{
      { "uniform", variety_sampling::uniform }
    };

    const std::string par_name = "sampling";

    auto itr = opt_vs.find(pin->GetString(psurfs->par_block_name, par_name));

    if (itr != opt_vs.end())
    {
      vs = itr->second;
    }
    else
    {
      std::ostringstream msg;
      msg << psurfs->par_block_name << "/" << par_name << " unknown"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // extract interpolater variety ---------------------------------------------
  {
    static const std::map<std::string, variety_interpolator> opt_vi{
      { "Lagrange", variety_interpolator::Lagrange },
      { "LagrangeLinear", variety_interpolator::LagrangeLinear }
    };

    const std::string par_name = "interpolator";

    auto itr = opt_vi.find(pin->GetString(psurfs->par_block_name, par_name));

    if (itr != opt_vi.end())
    {
      vi = itr->second;
    }
    else
    {
      std::ostringstream msg;
      msg << psurfs->par_block_name << "/" << par_name << " unknown"
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // get other parameters -----------------------------------------------------
  aliases::AA radii =
    pin->GetOrAddRealArray(psurfs->par_block_name, "radii", -1, 0);
  aliases::AA zmin =
    pin->GetOrAddRealArray(psurfs->par_block_name, "z_min", -1, 0);
  aliases::AA zmax =
    pin->GetOrAddRealArray(psurfs->par_block_name, "z_max", -1, 0);
  aliases::AA nz = pin->GetOrAddRealArray(psurfs->par_block_name, "nz", -1, 0);
  aliases::AA nph =
    pin->GetOrAddRealArray(psurfs->par_block_name, "nph", -1, 0);

  if ((radii.GetSize() == 0) || (zmin.GetSize() == 0) ||
      (zmax.GetSize() == 0) || (nz.GetSize() == 0) || (nph.GetSize() == 0))
  {
    std::ostringstream msg;
    msg << psurfs->par_block_name << "/radii,zmin,zmax,nph,nz ";
    msg << "length must be greater then zero.\n";
    ATHENA_ERROR(msg);
  }

  const int sz_radii = radii.GetSize();
  const int sz_zmin  = zmin.GetSize();
  const int sz_zmax  = zmax.GetSize();
  const int sz_nz    = nz.GetSize();
  const int sz_nph   = nph.GetSize();

  if (sz_zmin == 1)
  {
    this->z_min = zmin(0);
  }
  else if (sz_zmin == sz_radii)
  {
    this->z_min = zmin(surf_ix);
  }
  else
  {
    // unequal sizes
    assert(false);
  }

  if (sz_zmax == 1)
  {
    this->z_max = zmax(0);
  }
  else if (sz_zmax == sz_radii)
  {
    this->z_max = zmax(surf_ix);
  }
  else
  {
    // unequal sizes
    assert(false);
  }

  if (sz_nz == 1)
  {
    this->N_z = nz(0);
  }
  else if (sz_nz == sz_radii)
  {
    this->N_z = nz(surf_ix);
  }
  else
  {
    // unequal sizes
    assert(false);
  }

  if (sz_nph == 1)
  {
    this->N_ph = nph(0);
  }
  else if (sz_nph == sz_radii)
  {
    this->N_ph = nph(surf_ix);
  }
  else
  {
    // unequal sizes
    assert(false);
  }

  if ((this->N_ph + 1) % 2 == 0)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in surfaces setup" << std::endl
        << "nph must be even " << this->N_ph << std::endl;
    ATHENA_ERROR(msg);
  }

  rad   = radii(surf_ix);
  N_pts = this->N_z * this->N_ph;

  // prepare grids ------------------------------------------------------------
  ph.NewAthenaArray(this->N_ph);
  z.NewAthenaArray(this->N_z);
  x_o_ph_z.NewAthenaArray(N, this->N_ph, this->N_z);

  switch (vs)
  {
    case variety_sampling::uniform:
    {
      gr_z(z);
      gr_ph(ph);

      for (int i = 0; i < this->N_ph; ++i)
      {
        const Real sin_ph = std::sin(ph(i));
        const Real cos_ph = std::cos(ph(i));

        for (int j = 0; j < this->N_z; ++j)
        {
          x_o_ph_z(0, i, j) = rad * cos_ph;
          x_o_ph_z(1, i, j) = rad * sin_ph;
          x_o_ph_z(2, i, j) = z(j);
        }
      }

      break;
    }
    default:
    {
      assert(false);
    }
  }

  // index arrays for interpolators -------------------------------------------
  mask_mb.NewAthenaArray(N_ph, N_z);
  mask_mb.Fill(nullptr);

  mask_interp_idx_cc.NewAthenaArray(N_ph, N_z);
  mask_interp_idx_vc.NewAthenaArray(N_ph, N_z);
  mask_interp_idx_cc.Fill(-1);
  mask_interp_idx_vc.Fill(-1);

  // finally allocate storage for result of interpolation ---------------------
  int N_cpts_total = 0;
  for (int i = 0; i < N_cpts.GetSize(); ++i)
  {
    N_cpts_total += N_cpts(i);
  }

  u_vars.NewAthenaArray(N_cpts_total, N_ph, N_z);
}

void SurfaceCylindrical::PrepareInterpolators()
{
  if (prepared)
  {
    return;
  }

  // Reserve pool capacity (upper bound: every grid point gets an interpolator)
  const int N_pts_max = N_ph * N_z;
  switch (vi)
  {
    case variety_interpolator::Lagrange:
      interp_pool_Lag_cc.reserve(N_pts_max);
      interp_pool_Lag_vc.reserve(N_pts_max);
      break;
    case variety_interpolator::LagrangeLinear:
      interp_pool_LagLinear_cc.reserve(N_pts_max);
      interp_pool_LagLinear_vc.reserve(N_pts_max);
      break;
    default:
      assert(false);
  }

    // --- Per-thread storage for interpolators built in the parallel region
    // -----
#ifdef OPENMP_PARALLEL
  const int nthreads = pm->GetNumMeshThreads();
#else
  const int nthreads = 1;
#endif

  struct ThreadLocalInterps
  {
    std::vector<LagInterp> lag_cc, lag_vc;
    std::vector<LagInterpLinear> laglin_cc, laglin_vc;
    // Grid indices of occupied points (for writing mask_interp_idx later)
    std::vector<std::pair<int, int>> occupied;
  };
  std::vector<ThreadLocalInterps> tls(nthreads);

  // Connect target (ph_i, z_j) to salient MeshBlock pointer -----------------
  const auto& pmb_array = pm->GetMeshBlocksCached();

  for (auto* pmb : pmb_array)
  {
    // Grid origin / spacing are constant for this MeshBlock
    const Real origin_cc[N] = { pmb->pcoord->x1v(0),
                                pmb->pcoord->x2v(0),
                                pmb->pcoord->x3v(0) };
    const Real delta_cc[N]  = { pmb->pcoord->dx1v(0),
                                pmb->pcoord->dx2v(0),
                                pmb->pcoord->dx3v(0) };
    const int size_cc[N]    = { pmb->ncells1, pmb->ncells2, pmb->ncells3 };

    const Real origin_vc[N] = { pmb->pcoord->x1f(0),
                                pmb->pcoord->x2f(0),
                                pmb->pcoord->x3f(0) };
    const Real delta_vc[N]  = { pmb->pcoord->dx1f(0),
                                pmb->pcoord->dx2f(0),
                                pmb->pcoord->dx3f(0) };
    const int size_vc[N]    = { pmb->nverts1, pmb->nverts2, pmb->nverts3 };

#pragma omp parallel for num_threads(nthreads) collapse(2)
    for (int i = 0; i < N_ph; ++i)
    {
      for (int j = 0; j < N_z; ++j)
      {
        const Real x_1 = x_o_ph_z(0, i, j);
        const Real x_2 = x_o_ph_z(1, i, j);
        const Real x_3 = x_o_ph_z(2, i, j);

        if (pmb->PointContainedExclusive(x_1, x_2, x_3))
        {
          // Each (i,j) is unique per iteration - safe to write without lock
          mask_mb(i, j) = pmb;

#ifdef OPENMP_PARALLEL
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          ThreadLocalInterps& tl = tls[tid];

          const Real tar_coord[N] = { x_1, x_2, x_3 };

          switch (vi)
          {
            case variety_interpolator::Lagrange:
            {
              tl.lag_cc.emplace_back(origin_cc, delta_cc, size_cc, tar_coord);
              tl.lag_vc.emplace_back(origin_vc, delta_vc, size_vc, tar_coord);
              tl.occupied.emplace_back(i, j);
              break;
            }
            case variety_interpolator::LagrangeLinear:
            {
              tl.laglin_cc.emplace_back(
                origin_cc, delta_cc, size_cc, tar_coord);
              tl.laglin_vc.emplace_back(
                origin_vc, delta_vc, size_vc, tar_coord);
              tl.occupied.emplace_back(i, j);
              break;
            }
            default:
              assert(false);
          }
        }
      }
    }
  }

  // --- Serial merge: move thread-local interpolators into global pools ------
  for (int t = 0; t < nthreads; ++t)
  {
    ThreadLocalInterps& tl = tls[t];

    switch (vi)
    {
      case variety_interpolator::Lagrange:
      {
        const int base_cc = static_cast<int>(interp_pool_Lag_cc.size());
        const int base_vc = static_cast<int>(interp_pool_Lag_vc.size());

        for (size_t k = 0; k < tl.lag_cc.size(); ++k)
        {
          interp_pool_Lag_cc.push_back(std::move(tl.lag_cc[k]));
          interp_pool_Lag_vc.push_back(std::move(tl.lag_vc[k]));

          const int gi               = tl.occupied[k].first;
          const int gj               = tl.occupied[k].second;
          mask_interp_idx_cc(gi, gj) = base_cc + static_cast<int>(k);
          mask_interp_idx_vc(gi, gj) = base_vc + static_cast<int>(k);
        }
        break;
      }
      case variety_interpolator::LagrangeLinear:
      {
        const int base_cc = static_cast<int>(interp_pool_LagLinear_cc.size());
        const int base_vc = static_cast<int>(interp_pool_LagLinear_vc.size());

        for (size_t k = 0; k < tl.laglin_cc.size(); ++k)
        {
          interp_pool_LagLinear_cc.push_back(std::move(tl.laglin_cc[k]));
          interp_pool_LagLinear_vc.push_back(std::move(tl.laglin_vc[k]));

          const int gi               = tl.occupied[k].first;
          const int gj               = tl.occupied[k].second;
          mask_interp_idx_cc(gi, gj) = base_cc + static_cast<int>(k);
          mask_interp_idx_vc(gi, gj) = base_vc + static_cast<int>(k);
        }
        break;
      }
      default:
        assert(false);
    }
  }
  // --------------------------------------------------------------------------

  prepared = true;
}

Real SurfaceCylindrical::InterpolateAtPoint(aliases::AA& raw_cpt,
                                            Surfaces::variety_base_grid vs,
                                            const int tar_i,
                                            const int tar_j)
{
  Real res = 0;

  const int idx_cc = mask_interp_idx_cc(tar_i, tar_j);
  const int idx_vc = mask_interp_idx_vc(tar_i, tar_j);

  switch (vi)
  {
    case (variety_interpolator::Lagrange):
    {
      // call suitable interpolator
      if (vs == Surfaces::variety_base_grid::cc)
      {
        if (idx_cc >= 0)
          res = interp_pool_Lag_cc[idx_cc].eval(raw_cpt.data());
      }
      else if (vs == Surfaces::variety_base_grid::vc)
      {
        if (idx_vc >= 0)
          res = interp_pool_Lag_vc[idx_vc].eval(raw_cpt.data());
      }
      else
      {
        assert(false);
      }
      break;
    }
    case (variety_interpolator::LagrangeLinear):
    {
      if (vs == Surfaces::variety_base_grid::cc)
      {
        if (idx_cc >= 0)
        {
          res = interp_pool_LagLinear_cc[idx_cc].eval(raw_cpt.data());
        }
      }
      else if (vs == Surfaces::variety_base_grid::vc)
      {
        if (idx_vc >= 0)
        {
          res = interp_pool_LagLinear_vc[idx_vc].eval(raw_cpt.data());
        }
      }
      else
      {
        assert(false);
      }
      break;
    }
    default:
    {
      assert(false);
    }
  }

  return res;
}

void SurfaceCylindrical::DoInterpolations()
{
  u_vars.Fill(0);
  const int N_vars = N_cpts.GetSize();

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  // precompute loop-invariant lookups ----------------------------------------
  // variable_sampling(vix) and GetRemappedFieldIndex(vd, cix) do not depend
  // on the grid point (i,j), so we hoist them out of the parallel region.
  std::vector<Surfaces::variety_base_grid> vbg_pre(N_vars);
  int total_cpts = 0;
  for (int vix = 0; vix < N_vars; ++vix)
  {
    vbg_pre[vix] = psurfs->variable_sampling(vix);
    total_cpts += N_cpts(vix);
  }

  // flattened array of remapped component indices, indexed by ix_dump
  std::vector<int> mapped_cix_pre(total_cpts);
  {
    int ix = 0;
    for (int vix = 0; vix < N_vars; ++vix)
    {
      Surfaces::variety_data vd = psurfs->variables(vix);
      for (int cix = 0; cix < N_cpts(vix); ++cix)
      {
        mapped_cix_pre[ix++] = GetRemappedFieldIndex(vd, cix);
      }
    }
  }

// deal with fields & their components --------------------------------------
#pragma omp parallel for num_threads(nthreads) collapse(2)
  for (int i = 0; i < N_ph; ++i)
    for (int j = 0; j < N_z; ++j)
    {
      if (mask_mb(i, j) == nullptr)
        continue;

      int ix_dump = 0;
      for (int vix = 0; vix < N_vars; ++vix)
      {
        // given target (ph_i, z_j) get pointer to data on relevant MeshBlock
        AA& raw_var = *GetRawData(psurfs->variables(vix), mask_mb(i, j));

        const Surfaces::variety_base_grid vbg = vbg_pre[vix];

        // interpolate field component to the specified target point
        for (int cix = 0; cix < N_cpts(vix); ++cix)
        {
          // slice to current function component for interp
          AA sl_u(raw_var, mapped_cix_pre[ix_dump], 1);

          u_vars(ix_dump, i, j) = InterpolateAtPoint(sl_u, vbg, i, j);
          ix_dump++;
        }
      }
    }

  // assemble frame vectors ---------------------------------------------------
  // BD: TODO ...
}

// ============================================================================
}  // namespace gra::mesh::surfaces
// ============================================================================

//
// :D
//
