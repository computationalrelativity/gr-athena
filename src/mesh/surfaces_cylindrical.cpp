#include "surfaces.hpp"
#include <algorithm> // std::min
#include <cstring>   // std::memcpy
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

SurfacesCylindrical::SurfacesCylindrical(
  Mesh *pm,
  ParameterInput *pin,
  const int par_ix)
  : Surfaces(pm, pin, par_ix),
    num_radii( pin->GetOrAddRealArray(par_block_name, "radii", -1, 0).GetSize() )
{
  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf.push_back(new SurfaceCylindrical(pm,
                                           pin,
                                           dynamic_cast<Surfaces*>(this),
                                           surf_ix));
  }
}

SurfacesCylindrical::~SurfacesCylindrical()
{
  if (can_async)
  {
    // Ensure any current writes complete before changing data
    WriteBlock();
  }

  for (auto surf : psurf)
  {
    delete surf;
  }
  psurf.resize(0);
}

bool SurfacesCylindrical::IsActive(const Real time)
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

void SurfacesCylindrical::WriteBlock()
{
  if (write_future.valid())
  {
    write_future.get();
  }
}

void SurfacesCylindrical::WriteAllSurfaces(const Real time)
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

void SurfacesCylindrical::Reduce(const int ncycle, const Real time,
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

  // Phase 1: compute interpolations on all surfaces --------------------------
  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf[surf_ix]->Reduce_Compute();
  }

  // Phase 2: batched MPI_Allreduce across all radii --------------------------
#ifdef MPI_PARALLEL
  {
    // compute total element count across all surfaces
    size_t total_elems = 0;
    for (int surf_ix = 0; surf_ix < num_radii; ++surf_ix)
    {
      total_elems += static_cast<size_t>(psurf[surf_ix]->u_vars.GetSize());
    }

    // pack all u_vars into one contiguous buffer
    std::vector<Real> buf(total_elems);
    size_t offset = 0;
    for (int surf_ix = 0; surf_ix < num_radii; ++surf_ix)
    {
      const size_t n = static_cast<size_t>(psurf[surf_ix]->u_vars.GetSize());
      std::memcpy(buf.data() + offset,
                  psurf[surf_ix]->u_vars.data(),
                  n * sizeof(Real));
      offset += n;
    }

    // single chunked MPI_Allreduce on the contiguous buffer
    const size_t total_bytes = total_elems * sizeof(Real);
    const size_t max_chunk_bytes =
      static_cast<size_t>(DBG_SURF_CHUNK) * 1024 * 1024;
    size_t start_byte = 0;

    while (start_byte < total_bytes)
    {
      size_t bytes_left = total_bytes - start_byte;
      size_t chunk_bytes = std::min(bytes_left, max_chunk_bytes);
      int count = static_cast<int>(chunk_bytes / sizeof(Real));

      MPI_Allreduce(MPI_IN_PLACE,
                    buf.data() + start_byte / sizeof(Real),
                    count,
                    MPI_ATHENA_REAL,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      start_byte += count * sizeof(Real);
    }

    // scatter results back to each surface's u_vars
    offset = 0;
    for (int surf_ix = 0; surf_ix < num_radii; ++surf_ix)
    {
      const size_t n = static_cast<size_t>(psurf[surf_ix]->u_vars.GetSize());
      std::memcpy(psurf[surf_ix]->u_vars.data(),
                  buf.data() + offset,
                  n * sizeof(Real));
      offset += n;
    }
  }
#endif

  // Phase 3: synchronous writes ----------------------------------------------
  if (dump_data && !can_async)
  {
    if (Globals::my_rank == write_rank)
    {
      for (int surf_ix = 0; surf_ix < num_radii; ++surf_ix)
      {
        psurf[surf_ix]->write_hdf5(time);
      }
    }
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

void SurfacesCylindrical::ReinitializeSurfaces(const int ncycle, const Real time)
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

void SurfaceCylindrical::write_hdf5(const Real T)
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
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_min", z_min);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_max", z_max);

    // 1d arrays [grid] ---------------------------------------------------------
    hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/ph", ph);
    hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/z", z);

    // 2d arrays [geometry] -----------------------------------------------------
    int ix_dump = 0; // for offset in u_dump

    std::string full_path_base { "/fields/" + six };

    for (int v=0; v<psurfs->variables.GetSize(); ++v)
    {
      Surfaces::variety_data vd = psurfs->variables(v);

      // get HDF5 group prefix e.g. from "M1.lab" extract "M1"
      const std::string & var_type =
        psurfs->map_to_variety_prefix.at(vd);

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

SurfaceCylindrical::SurfaceCylindrical(
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
  aliases::AA zmin = pin->GetOrAddRealArray(
    psurfs->par_block_name, "z_min", -1, 0
  );
  aliases::AA zmax = pin->GetOrAddRealArray(
    psurfs->par_block_name, "z_max", -1, 0
  );
  aliases::AA nz = pin->GetOrAddRealArray(
    psurfs->par_block_name, "nz", -1, 0
  );
  aliases::AA nph = pin->GetOrAddRealArray(
    psurfs->par_block_name, "nph", -1, 0
  );

  if ((radii.GetSize() == 0) ||
      (zmin.GetSize() == 0) ||
      (zmax.GetSize() == 0) ||
      (nz.GetSize() == 0) ||
      (nph.GetSize() == 0))
  {
    std::ostringstream msg;
    msg << psurfs->par_block_name << "/radii,zmin,zmax,nph,nz ";
    msg << "length must be greater then zero.\n";
    ATHENA_ERROR(msg);
  }

  const int sz_radii = radii.GetSize();
  const int sz_zmin = zmin.GetSize();
  const int sz_zmax = zmax.GetSize();
  const int sz_nz = nz.GetSize();
  const int sz_nph = nph.GetSize();

  if (sz_zmin == 1)
  {
    this->z_min = zmin(0);
  }
  else if (sz_zmin ==sz_radii)
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
  else if (sz_zmax ==sz_radii)
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
  else if (sz_nz ==sz_radii)
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

      for (int i=0; i<this->N_ph; ++i)
      {
        const Real sin_ph = std::sin(ph(i));
        const Real cos_ph = std::cos(ph(i));

        for (int j=0; j<this->N_z; ++j)
        {
          x_o_ph_z(0,i,j) = rad * cos_ph;
          x_o_ph_z(1,i,j) = rad * sin_ph;
          x_o_ph_z(2,i,j) = z(j);
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
  for (int i=0; i<N_cpts.GetSize(); ++i)
  {
    N_cpts_total += N_cpts(i);
  }

  u_vars.NewAthenaArray(N_cpts_total, N_ph, N_z);
}

SurfaceCylindrical::~SurfaceCylindrical()
{
  TearDownInterpolators();
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

  // Connect target (ph_i, z_j) to salient MeshBlock pointer -----------------
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
    for (int i = 0; i < N_ph; ++i)
    {
      for (int j = 0; j < N_z; ++j)
      {
        const Real x_1 = x_o_ph_z(0,i,j);
        const Real x_2 = x_o_ph_z(1,i,j);
        const Real x_3 = x_o_ph_z(2,i,j);

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

void SurfaceCylindrical::TearDownInterpolators()
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

void SurfaceCylindrical::ReinitializeSurface(const bool active)
{
  TearDownInterpolators();
  if (active)
  {
    PrepareInterpolators();
  }
}

void SurfaceCylindrical::Reduce_Compute()
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
}

void SurfaceCylindrical::Reduce_Communicate()
{
  MPI_Reduce();
}

void SurfaceCylindrical::Reduce(const int ncycle, const Real time)
{
  Reduce_Compute();

  // MPI logic for surface reduction to all ranks -----------------------------
  Reduce_Communicate();
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

void SurfaceCylindrical::MPI_Reduce()
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

Real SurfaceCylindrical::InterpolateAtPoint(
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
  for (int i=0; i<N_ph; ++i)
  for (int j=0; j<N_z; ++j)
  {
    if (mask_mb(i,j) == nullptr)
      continue;

    int ix_dump = 0;
    for (int vix=0; vix<N_vars; ++vix)
    {
      // given target (th_i, th_j) get pointer to data on relevant MeshBlock
      AA & raw_var = *GetRawData(psurfs->variables(vix),
                                 mask_mb(i,j));

      const Surfaces::variety_base_grid vbg = vbg_pre[vix];

      // interpolate field component to the specified target point
      for (int cix=0; cix<N_cpts(vix); ++cix)
      {
        // slice to current function component for interp
        AA sl_u(raw_var, mapped_cix_pre[ix_dump], 1);

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
