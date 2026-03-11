#include "surfaces.hpp"
#include <algorithm> // std::min
#include <cstring>   // std::memcpy
#include <limits>
#include <map>
#include <string>
#include <tuple>   // std::tuple
#include <vector>

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#include "../coordinates/coordinates.hpp"
#include "../utils/linear_algebra.hpp"
#include "../utils/lagrange_interp.hpp"
#include "../utils/interp_barycentric.hpp"
#include "../utils/utils.hpp"

#include "../m1/m1.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../scalars/scalars.hpp"
#include "../z4c/z4c.hpp"

// hdf5 / mpi macros
#include "../outputs/outputs.hpp"

// ============================================================================
namespace gra::mesh::surfaces {
// ============================================================================

SurfacesCartesian::SurfacesCartesian(
  Mesh *pm,
  ParameterInput *pin,
  const int par_ix)
  : Surfaces(pm, pin, par_ix)
{
  num_surf = pin->GetOrAddRealArray(par_block_name, "nx", -1, 0).GetSize();
  for (int surf_ix=0; surf_ix<num_surf; ++surf_ix)
  {
    psurf.push_back(new SurfaceCartesian(pm,
                                         pin,
                                         dynamic_cast<Surfaces*>(this),
                                         surf_ix));
  }
}

void SurfaceCartesian::write_hdf5_coordinates(hid_t & id_file,
                                               const std::string & six)
{
#ifdef HDF5OUTPUT
  // scalars [grid]
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/x_min", x_min);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/x_max", x_max);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/y_min", y_min);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/y_max", y_max);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_min", z_min);
  hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_max", z_max);

  // 1d arrays [grid]
  hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/x", x);
  hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/y", y);
  hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/z", z);
#endif
}

SurfaceCartesian::SurfaceCartesian(
  Mesh *pm,
  ParameterInput *pin,
  Surfaces *psurfs,
  const int surf_ix)
  : Surface(pm, pin, psurfs, surf_ix)
{
  // extract [target] sampling variety ----------------------------------------
  {
    static const std::map<std::string, variety_sampling> opt_vs {
      {"uniform", variety_sampling::uniform},
      {"cgl",     variety_sampling::cgl},
    };

    const std::string par_name = "sampling";

    auto itr = opt_vs.find(
      pin->GetString(psurfs->par_block_name, par_name)
    );

    if (itr != opt_vs.end())
    {
      vs = itr->second;
    }
    else
    {
      std::ostringstream msg;
      msg << psurfs->par_block_name
          << "/" << par_name << " unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // extract interpolater variety ---------------------------------------------
  {
    static const std::map<std::string, variety_interpolator> opt_vi {
      {"Lagrange", variety_interpolator::Lagrange},
      {"LagrangeLinear", variety_interpolator::LagrangeLinear}
    };

    const std::string par_name = "interpolator";

    auto itr = opt_vi.find(
      pin->GetString(psurfs->par_block_name, par_name)
    );

    if (itr != opt_vi.end())
    {
      vi = itr->second;
    }
    else
    {
      std::ostringstream msg;
      msg << psurfs->par_block_name
          << "/" << par_name << " unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // get other parameters -----------------------------------------------------
  aliases::AA xmin = pin->GetOrAddRealArray(
    psurfs->par_block_name, "x_min", -1, 0
  );
  aliases::AA xmax = pin->GetOrAddRealArray(
    psurfs->par_block_name, "x_max", -1, 0
  );

  aliases::AA ymin = pin->GetOrAddRealArray(
    psurfs->par_block_name, "y_min", -1, 0
  );
  aliases::AA ymax = pin->GetOrAddRealArray(
    psurfs->par_block_name, "y_max", -1, 0
  );

  aliases::AA zmin = pin->GetOrAddRealArray(
    psurfs->par_block_name, "z_min", -1, 0
  );
  aliases::AA zmax = pin->GetOrAddRealArray(
    psurfs->par_block_name, "z_max", -1, 0
  );

  aliases::AA nx = pin->GetOrAddRealArray(
    psurfs->par_block_name, "nx", -1, 0
  );

  aliases::AA ny = pin->GetOrAddRealArray(
    psurfs->par_block_name, "ny", -1, 0
  );

  aliases::AA nz = pin->GetOrAddRealArray(
    psurfs->par_block_name, "nz", -1, 0
  );

  const int sz_xmin = xmin.GetSize();
  const int sz_xmax = xmax.GetSize();
  const int sz_ymin = ymin.GetSize();
  const int sz_ymax = ymax.GetSize();
  const int sz_zmin = zmin.GetSize();
  const int sz_zmax = zmax.GetSize();

  const int sz_nx = nx.GetSize();
  const int sz_ny = ny.GetSize();
  const int sz_nz = nz.GetSize();

  int sizes[] = {
    sz_xmin, sz_xmax,
    sz_ymin, sz_ymax,
    sz_zmin, sz_zmax,
    sz_nx,   sz_ny,   sz_nz
  };

  bool zero_found = false;
  bool mismatch_found = false;

  int expected_size = sizes[0];

  for (int i = 0; i < 9; ++i)
  {
    if (sizes[i] == 0)
    {
      zero_found = true;
    }
    if (sizes[i] != expected_size)
    {
      mismatch_found = true;
    }
  }

  if (zero_found || mismatch_found) {
    std::ostringstream msg;
    msg << psurfs->par_block_name << "/xmin,..zmax,nx,..,nz ";
    msg << "length must be greater than zero and all equal.\n";
    ATHENA_ERROR(msg);
  }

  // all the same size
  const int use_ix = (sz_xmin == 1) ? 0 : surf_ix;

  this->x_min = xmin(use_ix);
  this->x_max = xmax(use_ix);

  this->y_max = ymax(use_ix);
  this->y_min = ymin(use_ix);

  this->z_max = zmax(use_ix);
  this->z_min = zmin(use_ix);

  this->N_x = nx(use_ix);
  this->N_y = ny(use_ix);
  this->N_z = nz(use_ix);

  if ((this->N_x < 1) || (this->N_y < 1) || (this->N_z < 1))
  {
    std::ostringstream msg;
    msg << psurfs->par_block_name << "/nx,ny,nz "
        << "entries must each be >= 1.\n";
    ATHENA_ERROR(msg);
  }

  N_pts = this->N_x * this->N_y * this->N_z;

  // prepare grids ------------------------------------------------------------
  x.NewAthenaArray(this->N_x);
  y.NewAthenaArray(this->N_y);
  z.NewAthenaArray(this->N_z);

  switch (vs)
  {
    case variety_sampling::uniform:
    {
      uniform_gr_x(x);
      uniform_gr_y(y);
      uniform_gr_z(z);
      break;
    }
    case variety_sampling::cgl:
    {
      cgl_gr_x(x);
      cgl_gr_y(y);
      cgl_gr_z(z);
      break;
    }
    default:
    {
      assert(false);
    }
  }

  // index arrays for interpolators -------------------------------------------
  mask_mb.NewAthenaArray(N_x, N_y, N_z);
  mask_mb.Fill(nullptr);

  mask_interp_idx_cc.NewAthenaArray(N_x, N_y, N_z);
  mask_interp_idx_vc.NewAthenaArray(N_x, N_y, N_z);
  mask_interp_idx_cc.Fill(-1);
  mask_interp_idx_vc.Fill(-1);


  // finally allocate storage for result of interpolation ---------------------
  int N_cpts_total = 0;
  for (int i=0; i<N_cpts.GetSize(); ++i)
  {
    N_cpts_total += N_cpts(i);
  }

  u_vars.NewAthenaArray(N_cpts_total, N_x, N_y, N_z);
}

void SurfaceCartesian::PrepareInterpolators()
{
  if (prepared)
  {
    return;
  }

  // Reserve pool capacity (upper bound: every grid point gets an interpolator)
  const int N_pts_max = N_x * N_y * N_z;
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

  // --- Per-thread storage for interpolators built in the parallel region -----
#ifdef OPENMP_PARALLEL
  const int nthreads = pm->GetNumMeshThreads();
#else
  const int nthreads = 1;
#endif

  struct ThreadLocalInterps {
    std::vector<LagInterp>       lag_cc, lag_vc;
    std::vector<LagInterpLinear> laglin_cc, laglin_vc;
    // Grid indices of occupied points (for writing mask_interp_idx later)
    std::vector<std::tuple<int,int,int>> occupied;
  };
  std::vector<ThreadLocalInterps> tls(nthreads);

  // Connect target (x_i, y_j, z_k) to salient MeshBlock pointer -------------
  MeshBlock * pmb = pm->pblock;

  while (pmb != nullptr)
  {
    // Grid origin / spacing are constant for this MeshBlock
    const Real origin_cc[N] = {
      pmb->pcoord->x1v(0), pmb->pcoord->x2v(0), pmb->pcoord->x3v(0)
    };
    const Real delta_cc[N] = {
      pmb->pcoord->dx1v(0), pmb->pcoord->dx2v(0), pmb->pcoord->dx3v(0)
    };
    const int size_cc[N] = {
      pmb->ncells1, pmb->ncells2, pmb->ncells3
    };

    const Real origin_vc[N] = {
      pmb->pcoord->x1f(0), pmb->pcoord->x2f(0), pmb->pcoord->x3f(0)
    };
    const Real delta_vc[N] = {
      pmb->pcoord->dx1f(0), pmb->pcoord->dx2f(0), pmb->pcoord->dx3f(0)
    };
    const int size_vc[N] = {
      pmb->nverts1, pmb->nverts2, pmb->nverts3
    };

    #pragma omp parallel for num_threads(nthreads) collapse(3)
    for (int i = 0; i < N_x; ++i)
    {
      for (int j = 0; j < N_y; ++j)
      {
        for (int k = 0; k < N_z; ++k)
        {
          const Real x_1 = x(i);
          const Real x_2 = y(j);
          const Real x_3 = z(k);

          if (pmb->PointContainedExclusive(x_1, x_2, x_3))
          {
            // Each (i,j,k) is unique per iteration - safe to write without lock
            mask_mb(i, j, k) = pmb;

#ifdef OPENMP_PARALLEL
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            ThreadLocalInterps & tl = tls[tid];

            const Real tar_coord[N] = { x_1, x_2, x_3 };

            switch (vi)
            {
              case variety_interpolator::Lagrange:
              {
                tl.lag_cc.emplace_back(origin_cc, delta_cc, size_cc, tar_coord);
                tl.lag_vc.emplace_back(origin_vc, delta_vc, size_vc, tar_coord);
                tl.occupied.emplace_back(i, j, k);
                break;
              }
              case variety_interpolator::LagrangeLinear:
              {
                tl.laglin_cc.emplace_back(origin_cc, delta_cc, size_cc, tar_coord);
                tl.laglin_vc.emplace_back(origin_vc, delta_vc, size_vc, tar_coord);
                tl.occupied.emplace_back(i, j, k);
                break;
              }
              default:
                assert(false);
            }
          }
        }
      }
    }

    pmb = pmb->next;
  }

  // --- Serial merge: move thread-local interpolators into global pools ------
  for (int t = 0; t < nthreads; ++t)
  {
    ThreadLocalInterps & tl = tls[t];

    switch (vi)
    {
      case variety_interpolator::Lagrange:
      {
        const int base_cc = static_cast<int>(interp_pool_Lag_cc.size());
        const int base_vc = static_cast<int>(interp_pool_Lag_vc.size());

        for (size_t n = 0; n < tl.lag_cc.size(); ++n)
        {
          interp_pool_Lag_cc.push_back(std::move(tl.lag_cc[n]));
          interp_pool_Lag_vc.push_back(std::move(tl.lag_vc[n]));

          const int gi = std::get<0>(tl.occupied[n]);
          const int gj = std::get<1>(tl.occupied[n]);
          const int gk = std::get<2>(tl.occupied[n]);
          mask_interp_idx_cc(gi, gj, gk) = base_cc + static_cast<int>(n);
          mask_interp_idx_vc(gi, gj, gk) = base_vc + static_cast<int>(n);
        }
        break;
      }
      case variety_interpolator::LagrangeLinear:
      {
        const int base_cc = static_cast<int>(interp_pool_LagLinear_cc.size());
        const int base_vc = static_cast<int>(interp_pool_LagLinear_vc.size());

        for (size_t n = 0; n < tl.laglin_cc.size(); ++n)
        {
          interp_pool_LagLinear_cc.push_back(std::move(tl.laglin_cc[n]));
          interp_pool_LagLinear_vc.push_back(std::move(tl.laglin_vc[n]));

          const int gi = std::get<0>(tl.occupied[n]);
          const int gj = std::get<1>(tl.occupied[n]);
          const int gk = std::get<2>(tl.occupied[n]);
          mask_interp_idx_cc(gi, gj, gk) = base_cc + static_cast<int>(n);
          mask_interp_idx_vc(gi, gj, gk) = base_vc + static_cast<int>(n);
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



Real SurfaceCartesian::InterpolateAtPoint(
  aliases::AA & raw_cpt, Surfaces::variety_base_grid vs,
  const int tar_i, const int tar_j, const int tar_k)
{
  Real res = 0;

  const int idx_cc = mask_interp_idx_cc(tar_i, tar_j, tar_k);
  const int idx_vc = mask_interp_idx_vc(tar_i, tar_j, tar_k);

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

void SurfaceCartesian::DoInterpolations()
{
  u_vars.Fill(0);
  const int N_vars = N_cpts.GetSize();

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  // precompute loop-invariant lookups ----------------------------------------
  // variable_sampling(vix) and GetRemappedFieldIndex(vd, cix) do not depend
  // on the grid point (i,j,k), so we hoist them out of the parallel region.
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
  #pragma omp parallel for num_threads(nthreads) collapse(3)
  for (int i=0; i<N_x; ++i)
  for (int j=0; j<N_y; ++j)
  for (int k=0; k<N_z; ++k)
  {
    if (mask_mb(i,j,k) == nullptr)
      continue;

    int ix_dump = 0;
    for (int vix=0; vix<N_vars; ++vix)
    {
      // given target (x_i, y_j, z_k) get pointer to data on relevant MeshBlock
      AA & raw_var = *GetRawData(psurfs->variables(vix),
                                 mask_mb(i,j,k));

      const Surfaces::variety_base_grid vbg = vbg_pre[vix];

      // interpolate field component to the specified target point
      for (int cix=0; cix<N_cpts(vix); ++cix)
      {
        // slice to current function component for interp
        AA sl_u(raw_var, mapped_cix_pre[ix_dump], 1);

        u_vars(ix_dump,i,j,k) = InterpolateAtPoint(sl_u, vbg, i, j, k);
        ix_dump++;
      }
    }
  }

  // assemble frame vectors ---------------------------------------------------
  // BD: TODO ...
}

// ============================================================================
} // namespace gra::mesh::surfaces
// ============================================================================

//
// :D
//
