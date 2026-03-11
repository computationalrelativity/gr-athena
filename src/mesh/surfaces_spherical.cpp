#include "surfaces.hpp"
#include <limits>
#include <map>
#include <string>
#include <utility>  // std::pair
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

SurfacesSpherical::SurfacesSpherical(
  Mesh *pm,
  ParameterInput *pin,
  const int par_ix)
  : Surfaces(pm, pin, par_ix),
    num_radii( pin->GetOrAddRealArray(par_block_name, "radii", -1, 0).GetSize() )
{
  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf.push_back(new SurfaceSpherical(pm,
                                         pin,
                                         dynamic_cast<Surfaces*>(this),
                                         surf_ix));
  }
}

SurfacesSpherical::~SurfacesSpherical()
{
  if (can_async)
  {
    // Ensure any current writes complete before changing data
    WriteBlock();
  }

  for (auto surf : psurf) {
    delete surf;
  }
  psurf.resize(0);
}

bool SurfacesSpherical::IsActive(const Real time)
{
  if ((start_time >= 0) && time < start_time)
  {
    return false;
  }

  if ((stop_time >= 0) && time > stop_time)
  {
    return false;
  }
  return true;
};

void SurfacesSpherical::WriteBlock()
{
  if (write_future.valid())
  {
    write_future.get();
  }
}

void SurfacesSpherical::WriteAllSurfaces(const Real time)
{
  // launch async write in background
  write_future = std::async(std::launch::async, [this, time]()
  {
    // each surface can write its own contribution
    for (auto &surf : psurf)
    {
      surf->write_hdf5(time);
    }

    // debug
    std::printf("%s @ time = %.3e async out!\n", par_block_name.c_str(), time);
  });
}

void SurfacesSpherical::Reduce(const int ncycle, const Real time,
                               const bool is_final)
{
  // do not perform reduction if we are outside specified ranges --------------
  if (!IsActive(time))
  {
    return;
  }
  // --------------------------------------------------------------------------

  // filename update for final write
  this->is_final = is_final;
  if (is_final && !write_final)
  {
    return;
  }

  if (can_async)
  {
    // Ensure any current writes complete before changing data
    WriteBlock();
  }

  // perform all the reductions
  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf[surf_ix]->Reduce(ncycle, time);
  }

  if (dump_data)
  {
    // immediate, blocking write complete, update file_number here;
    // in the case of async this is also safe as it starts reduced by 1
    if (!is_final)
    {
      file_number++;
      pin->OverwriteParameter(par_block_name, "file_number", file_number);
    }

    // launch writes in the background asynchronously
    if (can_async)
    {
      // only write-rank actually does the write
      if (Globals::my_rank == write_rank)
      {
        WriteAllSurfaces(time);
      }
    }
  }
}

void SurfacesSpherical::ReinitializeSurfaces(const int ncycle, const Real time)
{
  if (can_async)
  {
    // Ensure any current writes complete before changing data
    WriteBlock();
  }

  // After AMR, MeshBlock pointers held by surfaces are stale.
  // Always tear down; only re-prepare if the surface is active.
  const bool active = IsActive(time);

  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf[surf_ix]->ReinitializeSurface(active);
  }
}

void SurfaceSpherical::write_hdf5(const Real T)
{
#ifdef HDF5OUTPUT
  if (Globals::my_rank == psurfs->write_rank)
  {
    std::string filename;
    hdf5_get_next_filename(filename);

    // write to existing file if multiple-radii being dumped
    const bool use_existing = surf_ix > 0;
    hid_t id_file = hdf5_touch_file(filename, use_existing);


    /*
    // Write attributes:
    if (ix_rad == 0)
    {
      hdf5_write_attribute(id_file, "test_string", "test");
      hdf5_write_attribute(id_file, "test_integer", 123);
      hdf5_write_attribute(id_file, "test_Real", 3.21);
    }
    */

    std::stringstream ssix;
    ssix << std::setw(2) << std::setfill('0') << std::to_string(surf_ix);
    const std::string six { ssix.str() };

    // scalars [grid] -----------------------------------------------------------
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/T", T);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/R", rad);

    // 1d arrays [grid] ---------------------------------------------------------
    hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/th", th);
    hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/ph", ph);

    // 2d arrays [geometry] -----------------------------------------------------
    int ix_dump = 0; // for offset in u_dump

    std::string full_path_base { "/fields/" + six };

    for (int v=0; v<psurfs->variables.GetSize(); ++v)
    {
      Surfaces::variety_data vd = psurfs->variables(v);

      // get name prior to initial point i.e. from "M1.lab" extract "M1"
      std::string var_type;
      for (auto it = psurfs->map_to_variety_data.begin();
                it != psurfs->map_to_variety_data.end(); ++it)
      {
        if (it->second == vd)
        {
          var_type = it->first;
          break;
        }
      }
      var_type = var_type.substr(0, var_type.find("."));

      // DEBUG
      /*
      #pragma omp critical
      if (vd == Surfaces::variety_data::M1_lab)
      {
        for (int n=0; n<N_cpts(v); ++n)
        {
          std::string var_name = GetNameFieldComponent(vd, n);
          std::printf("%d %s \n", n, var_name.c_str());
        }
        std::exit(0);
      }
      */

      for (int n=0; n<N_cpts(v); ++n)
      {
        // slice into next field component to dump
        AA sl_u (u_vars, ix_dump, 1);

        std::string var_name = GetNameFieldComponent(vd, n);
        std::string full_path = full_path_base + "/" + var_type + "/";
        full_path += var_name;

        hdf5_write_arr_nd(id_file, full_path, sl_u);
        ix_dump++;
      }
    }

    // Finally close
    hdf5_close_file(id_file);
  }

#endif
}

SurfaceSpherical::SurfaceSpherical(
  Mesh *pm,
  ParameterInput *pin,
  Surfaces *psurfs,
  const int surf_ix)
  : Surface(pm, pin, psurfs, surf_ix)
{
  // extract [target] sampling variety ----------------------------------------
  {
    static const std::map<std::string, variety_sampling> opt_vs {
      {"uniform", variety_sampling::uniform}
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
  aliases::AA radii = pin->GetOrAddRealArray(
    psurfs->par_block_name, "radii", -1, 0
  );
  aliases::AA nth = pin->GetOrAddRealArray(
    psurfs->par_block_name, "nth", -1, 0
  );
  aliases::AA nph = pin->GetOrAddRealArray(
    psurfs->par_block_name, "nph", -1, 0
  );

  if ((radii.GetSize() == 0) ||
      (nth.GetSize() == 0) ||
      (nph.GetSize() == 0))
  {
    std::ostringstream msg;
    msg << psurfs->par_block_name << "/radii,nth,nph ";
    msg << "length must be greater then zero.\n";
    ATHENA_ERROR(msg);
  }

  const int sz_radii = radii.GetSize();
  const int sz_nth = nth.GetSize();
  const int sz_nph = nph.GetSize();

  if (sz_nth == 1)
  {
    this->N_th = nth(0);
  }
  else if (sz_nth ==sz_radii)
  {
    this->N_th = nth(surf_ix);
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
  else if (sz_nph ==sz_radii)
  {
    this->N_ph = nph(surf_ix);
  }
  else
  {
    // unequal sizes
    assert(false);
  }

  if ((this->N_ph + 1) % 2 == 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in surfaces setup" << std::endl
        << "nph must be even " << this->N_ph << std::endl;
    ATHENA_ERROR(msg);
  }

  rad = radii(surf_ix);
  N_pts = this->N_th * this->N_ph;


  // prepare grids ------------------------------------------------------------
  th.NewAthenaArray(this->N_th);
  ph.NewAthenaArray(this->N_ph);
  x_o_th_ph.NewAthenaArray(N, this->N_th, this->N_ph);

  switch (vs)
  {
    case variety_sampling::uniform:
    {
      gr_th(th);
      gr_ph(ph);

      for (int i=0; i<this->N_th; ++i)
      {
        const Real sin_th = std::sin(th(i));
        const Real cos_th = std::cos(th(i));

        for (int j=0; j<this->N_ph; ++j)
        {
          const Real sin_ph = std::sin(ph(j));
          const Real cos_ph = std::cos(ph(j));

          x_o_th_ph(0,i,j) = rad * sin_th * cos_ph;
          x_o_th_ph(1,i,j) = rad * sin_th * sin_ph;
          x_o_th_ph(2,i,j) = rad * cos_th;
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
  mask_mb.NewAthenaArray(N_th, N_ph);
  mask_mb.Fill(nullptr);

  mask_interp_idx_cc.NewAthenaArray(N_th, N_ph);
  mask_interp_idx_vc.NewAthenaArray(N_th, N_ph);
  mask_interp_idx_cc.Fill(-1);
  mask_interp_idx_vc.Fill(-1);


  // finally allocate storage for result of interpolation ---------------------
  int N_cpts_total = 0;
  for (int i=0; i<N_cpts.GetSize(); ++i)
  {
    N_cpts_total += N_cpts(i);
  }

  u_vars.NewAthenaArray(N_cpts_total, N_th, N_ph);
}

SurfaceSpherical::~SurfaceSpherical()
{
  TearDownInterpolators();
}

void SurfaceSpherical::PrepareInterpolators()
{
  if (prepared)
  {
    return;
  }

  // Reserve pool capacity (upper bound: every grid point gets an interpolator)
  const int N_pts_max = N_th * N_ph;
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
    std::vector<std::pair<int,int>> occupied;
  };
  std::vector<ThreadLocalInterps> tls(nthreads);

  // Connect target (th_i, ph_j) to salient MeshBlock pointer -----------------
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

    #pragma omp parallel for num_threads(nthreads) collapse(2)
    for (int i = 0; i < N_th; ++i)
    {
      for (int j = 0; j < N_ph; ++j)
      {
        const Real x_1 = x_o_th_ph(0,i,j);
        const Real x_2 = x_o_th_ph(1,i,j);
        const Real x_3 = x_o_th_ph(2,i,j);

        if (pmb->PointContainedExclusive(x_1, x_2, x_3))
        {
          // Each (i,j) is unique per iteration - safe to write without lock
          mask_mb(i,j) = pmb;

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
              tl.occupied.emplace_back(i, j);
              break;
            }
            case variety_interpolator::LagrangeLinear:
            {
              tl.laglin_cc.emplace_back(origin_cc, delta_cc, size_cc, tar_coord);
              tl.laglin_vc.emplace_back(origin_vc, delta_vc, size_vc, tar_coord);
              tl.occupied.emplace_back(i, j);
              break;
            }
            default:
              assert(false);
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

        for (size_t k = 0; k < tl.lag_cc.size(); ++k)
        {
          interp_pool_Lag_cc.push_back(std::move(tl.lag_cc[k]));
          interp_pool_Lag_vc.push_back(std::move(tl.lag_vc[k]));

          const int gi = tl.occupied[k].first;
          const int gj = tl.occupied[k].second;
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

          const int gi = tl.occupied[k].first;
          const int gj = tl.occupied[k].second;
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

void SurfaceSpherical::TearDownInterpolators()
{
  if (!prepared)
  {
    return;
  }

  // clean up mask containing salient MeshBlock
  mask_mb.Fill(nullptr);

  // clear interpolator pools and reset index arrays
  interp_pool_Lag_cc.clear();
  interp_pool_Lag_vc.clear();
  interp_pool_LagLinear_cc.clear();
  interp_pool_LagLinear_vc.clear();
  mask_interp_idx_cc.Fill(-1);
  mask_interp_idx_vc.Fill(-1);

  prepared = false;
}

void SurfaceSpherical::ReinitializeSurface(const bool active)
{
  TearDownInterpolators();
  if (active)
  {
    PrepareInterpolators();
  }
}

void SurfaceSpherical::Reduce(const int ncycle, const Real time)
{
  // ensure we have prepared the interpolators --------------------------------
  if (!prepared)
  {
    PrepareInterpolators();

    /*
    // DEBUG:
    bool have_point = false;
    for (int i=0; i<N_th; ++i)
    for (int j=0; j<N_ph; ++j)
    {
      have_point = have_point || (mask_mb(i,j) != nullptr);
    }
    if (have_point)
      mask_mb.print_all("%p");
    */
  }

  // use pre-allocated interpolators on desired data --------------------------
  DoInterpolations();

  // --------------------------------------------------------------------------

  // MPI logic for surface reduction to all ranks -----------------------------
  MPI_Reduce();
  // --------------------------------------------------------------------------

  // finally write ------------------------------------------------------------
  if (psurfs->dump_data && !psurfs->can_async)
  {
    if (Globals::my_rank == psurfs->write_rank)
    {
      write_hdf5(time);
    }
  }
  // --------------------------------------------------------------------------
};

void SurfaceSpherical::MPI_Reduce()
{
#ifdef MPI_PARALLEL
  size_t total_bytes = u_vars.GetSizeInBytes();

  // max chunk size in bytes
  size_t max_chunk_bytes = DBG_SURF_CHUNK * 1024 * 1024;
  size_t start_byte = 0;

  // Pointer to the data
  Real* data_ptr = &(u_vars(0,0,0));

  while (start_byte < total_bytes) {
    // Compute number of elements for this chunk
    size_t bytes_left = total_bytes - start_byte;
    size_t chunk_bytes = std::min(bytes_left, max_chunk_bytes);
    int count = static_cast<int>(chunk_bytes / sizeof(Real));

    MPI_Allreduce(MPI_IN_PLACE,
                  data_ptr + start_byte / sizeof(Real),
                  count,
                  MPI_ATHENA_REAL,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    start_byte += count * sizeof(Real);
  }
#endif
}


Real SurfaceSpherical::InterpolateAtPoint(
  aliases::AA & raw_cpt, Surfaces::variety_base_grid vs,
  const int tar_i, const int tar_j)
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

void SurfaceSpherical::DoInterpolations()
{
  u_vars.Fill(0);
  const int N_vars = N_cpts.GetSize();

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  // deal with fields & their components --------------------------------------
  #pragma omp parallel for num_threads(nthreads) collapse(2)
  for (int i=0; i<N_th; ++i)
  for (int j=0; j<N_ph; ++j)
  {
    if (mask_mb(i,j) == nullptr)
      continue;

    int ix_dump = 0;
    for (int vix=0; vix<N_vars; ++vix)
    {
      // given target (th_i, th_j) get pointer to data on relevant MeshBlock
      AA & raw_var = *GetRawData(psurfs->variables(vix),
                                 mask_mb(i,j));

      Surfaces::variety_base_grid vbg = psurfs->variable_sampling(vix);

      // interpolate field component to the specified target point
      for (int cix=0; cix<N_cpts(vix); ++cix)
      {
        // slice to current function component for interp
        const int mapped_cix = GetRemappedFieldIndex(
          psurfs->variables(vix), cix
        );
        AA sl_u(raw_var, mapped_cix, 1);

        // const int ix_dump = cix + vix * N_cpts(vix);
        u_vars(ix_dump,i,j) = InterpolateAtPoint(sl_u, vbg, i, j);
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
