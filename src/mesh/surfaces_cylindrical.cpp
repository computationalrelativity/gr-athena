#include "surfaces.hpp"
#include <limits>
#include <map>
#include <string>

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
    // save incremented file number (in case rst is written during this async)
    pin->OverwriteParameter(par_block_name, "file_number", file_number+1);

    // each surface can write its own contribution
    for (auto &surf : psurf)
    {
      surf->write_hdf5(time);
    }

    file_number++;

    // debug
    std::printf("%s @ time = %.3e async out!\n", par_block_name.c_str(), time);
  });
}

void SurfacesCylindrical::Reduce(const int ncycle, const Real time)
{
  // do not perform reduction if we are outside specified ranges --------------
  if (!IsActive(time))
  {
    return;
  }
  // --------------------------------------------------------------------------

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
    // launch writes in the background asynchronously
    if (can_async)
    {
      // file_number needs pre-increment internally
      if (Globals::my_rank == 0)
        WriteAllSurfaces(time);
    }
    else
    {
      // immediate, blocking write complete, update file_number here
      file_number++;
      pin->OverwriteParameter(par_block_name, "file_number", file_number);
    }
  }
}

void SurfacesCylindrical::ReinitializeSurfaces(const int ncycle, const Real time)
{
  // do not reinitialize if we are outside specified ranges -------------------
  if (!IsActive(time))
  {
    return;
  }
  // --------------------------------------------------------------------------

  if (can_async)
  {
    // Ensure any current writes complete before changing data
    WriteBlock();
  }

  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf[surf_ix]->ReinitializeSurface();
  }
}

void SurfaceCylindrical::write_hdf5(const Real T)
{
#ifdef HDF5OUTPUT
  if (Globals::my_rank == 0)
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

      // get name prior to initial point i.e. from "M1.lab" extract "M1"
      std::string var_type;
      for (auto it = psurfs->map_to_variety_data.begin();
                it != psurfs->map_to_variety_data.end(); ++it)
      {
        if (it->second == vd)
        {
          var_type = it->first;
          continue;
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
      {"Lagrange", variety_interpolator::Lagrange}
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

  // pointer arrays for interpolators -----------------------------------------
  mask_mb.NewAthenaArray(N_ph, N_z);
  mask_mb.Fill(nullptr);

  switch (vi)
  {
    case variety_interpolator::Lagrange:
    {
      mask_pinterp_Lag_cc.NewAthenaArray(N_ph, N_z);
      mask_pinterp_Lag_vc.NewAthenaArray(N_ph, N_z);

      mask_pinterp_Lag_cc.Fill(nullptr);
      mask_pinterp_Lag_vc.Fill(nullptr);
      break;
    }
    default:
    {
      assert(false);
    }
  }


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

void SurfaceCylindrical::PrepareInterpolatorAtPoint(
  MeshBlock * pmb, const int i, const int j)
{
  switch (vi)
  {
    case variety_interpolator::Lagrange:
    {
      if (mask_pinterp_Lag_cc(i,j) != nullptr)
      {
        delete mask_pinterp_Lag_cc(i,j);
        mask_pinterp_Lag_cc(i,j) = nullptr;
      }

      if (mask_pinterp_Lag_vc(i,j) != nullptr)
      {
        delete mask_pinterp_Lag_vc(i,j);
        mask_pinterp_Lag_vc(i,j) = nullptr;
      }

      // CC vars
      const Real origin_cc[N] = {
        pmb->pcoord->x1v(0), pmb->pcoord->x2v(0), pmb->pcoord->x3v(0)
      };
      const Real delta_cc[N] = {
        pmb->pcoord->dx1v(0), pmb->pcoord->dx2v(0), pmb->pcoord->dx3v(0)
      };
      const int size_cc[N] = {
        pmb->ncells1, pmb->ncells2, pmb->ncells3
      };

      // VC
      const Real origin_vc[N] = {
        pmb->pcoord->x1f(0), pmb->pcoord->x2f(0), pmb->pcoord->x3f(0)
      };
      const Real delta_vc[N] = {
        pmb->pcoord->dx1f(0), pmb->pcoord->dx2f(0), pmb->pcoord->dx3f(0)
      };
      int size_vc[N] = {
        pmb->nverts1, pmb->nverts2, pmb->nverts3
      };

      Real tar_coord[N] = {
        x_o_ph_z(0,i,j), x_o_ph_z(1,i,j), x_o_ph_z(2,i,j)
      };

      mask_pinterp_Lag_cc(i,j) = new LagInterp(origin_cc,
                                               delta_cc,
                                               size_cc,
                                               tar_coord);

      mask_pinterp_Lag_vc(i,j) = new LagInterp(origin_vc,
                                               delta_vc,
                                               size_vc,
                                               tar_coord);

      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void SurfaceCylindrical::PrepareInterpolators()
{
  // BD: TODO - could be optimized further
  if (prepared)
  {
    return;
  }

  // Connect target (th_i, ph_j) to salient MeshBlock pointer -----------------
  MeshBlock * pmb = pm->pblock;

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  while (pmb != nullptr)
  {
    // Given current MeshBlock check whether grid intersects (excluding ghosts)
    [&] {
      #pragma omp parallel for collapse(2) num_threads(nthreads)
      for (int i=0; i<N_ph; ++i)
      {
        for (int j=0; j<N_z; ++j)
        {
          const Real x_1 = x_o_ph_z(0,i,j);
          const Real x_2 = x_o_ph_z(1,i,j);
          const Real x_3 = x_o_ph_z(2,i,j);

          if (pmb->PointContainedExclusive(x_1, x_2, x_3))
          {
            mask_mb(i,j) = pmb;
            PrepareInterpolatorAtPoint(pmb, i, j);
          }
        }
      }
    }();

    pmb = pmb->next;
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

  for (int i=0; i<N_ph; ++i)
  for (int j=0; j<N_z; ++j)
  {
    switch (vi)
    {
      case variety_interpolator::Lagrange:
      {
        if (mask_pinterp_Lag_cc(i,j) != nullptr)
        {
          delete mask_pinterp_Lag_cc(i,j);
          mask_pinterp_Lag_cc(i,j) = nullptr;
        }

        if (mask_pinterp_Lag_vc(i,j) != nullptr)
        {
          delete mask_pinterp_Lag_vc(i,j);
          mask_pinterp_Lag_vc(i,j) = nullptr;
        }
        break;
      }
      default:
      {
        assert(false);
      }
    }
  }

  prepared = false;
}

void SurfaceCylindrical::ReinitializeSurface()
{
  TearDownInterpolators();
  PrepareInterpolators();
}

void SurfaceCylindrical::Reduce(const int ncycle, const Real time)
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
    if (Globals::my_rank == 0)
    {
      write_hdf5(time);
    }
  }
  // --------------------------------------------------------------------------
};

/*
void SurfaceCylindrical::MPI_Reduce()
{
#ifdef MPI_PARALLEL

  int N_cpts_total = 0;
  for (int cix=0; cix<N_cpts.GetSize(); ++cix)
  {
    N_cpts_total += N_cpts(cix);
  }

  int rank;
  static const int root = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Useful to have the data on all ranks
  MPI_Allreduce(MPI_IN_PLACE,
                &(u_vars(0,0,0)),
                N_cpts_total * N_ph * N_z,
                MPI_ATHENA_REAL,
                MPI_SUM,
                MPI_COMM_WORLD);
#endif
}
*/

void SurfaceCylindrical::MPI_Reduce()
{
#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    start_byte += chunk_bytes;
  }
#endif
}

Real SurfaceCylindrical::InterpolateAtPoint(
  aliases::AA & raw_cpt, Surfaces::variety_base_grid vs,
  const int tar_i, const int tar_j)
{
  Real res = 0;

  switch (vi)
  {
    case (variety_interpolator::Lagrange):
    {
      // call suitable interpolator
      if (vs == Surfaces::variety_base_grid::cc)
      {
        if (mask_pinterp_Lag_cc(tar_i,tar_j) != nullptr)
          res = mask_pinterp_Lag_cc(tar_i,tar_j)->eval(raw_cpt.data());
      }
      else if (vs == Surfaces::variety_base_grid::vc)
      {
        if (mask_pinterp_Lag_vc(tar_i,tar_j) != nullptr)
          res = mask_pinterp_Lag_vc(tar_i,tar_j)->eval(raw_cpt.data());
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
