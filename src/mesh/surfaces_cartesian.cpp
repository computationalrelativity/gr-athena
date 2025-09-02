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

SurfacesCartesian::SurfacesCartesian(
  Mesh *pm,
  ParameterInput *pin,
  const int par_ix)
  : Surfaces(pm, pin, par_ix),
    num_surf( pin->GetOrAddRealArray(par_block_name, "nx", -1, 0).GetSize() )
{
  for (int surf_ix=0; surf_ix<num_surf; ++surf_ix)
  {
    psurf.push_back(new SurfaceCartesian(pm,
                                         pin,
                                         dynamic_cast<Surfaces*>(this),
                                         surf_ix));
  }
}

SurfacesCartesian::~SurfacesCartesian()
{
  for (auto surf : psurf) {
    delete surf;
  }
  psurf.resize(0);
}

bool SurfacesCartesian::IsActive(const Real time)
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

void SurfacesCartesian::Reduce(const int ncycle, const Real time)
{
  // do not perform reduction if we are outside specified ranges --------------
  if (!IsActive(time))
  {
    return;
  }
  // --------------------------------------------------------------------------

  for (int surf_ix=0; surf_ix<num_surf; ++surf_ix)
  {
    psurf[surf_ix]->Reduce(ncycle, time);
  }

  // Update file number for next write if required
  if (dump_data)
  {
    file_number++;
    pin->OverwriteParameter(par_block_name, "file_number", file_number);
  }
}

void SurfacesCartesian::ReinitializeSurfaces(const int ncycle, const Real time)
{
  // do not reinitialize if we are outside specified ranges -------------------
  if (!IsActive(time))
  {
    return;
  }
  // --------------------------------------------------------------------------

  for (int surf_ix=0; surf_ix<num_surf; ++surf_ix)
  {
    psurf[surf_ix]->ReinitializeSurface();
  }
}

void SurfaceCartesian::write_hdf5(const Real T)
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
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/x_min", x_min);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/x_max", x_max);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/y_min", y_min);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/y_max", y_max);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_min", z_min);
    hdf5_write_scalar(id_file, "/coordinates/" + six + "/z_max", z_max);

    // 1d arrays [grid] ---------------------------------------------------------
    hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/x", x);
    hdf5_write_arr_nd(id_file, "/coordinates/" + six + "/y", y);
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

  // pointer arrays for interpolators -----------------------------------------
  mask_mb.NewAthenaArray(N_x, N_y, N_z);
  mask_mb.Fill(nullptr);

  switch (vi)
  {
    case variety_interpolator::Lagrange:
    {
      mask_pinterp_Lag_cc.NewAthenaArray(N_x, N_y, N_z);
      mask_pinterp_Lag_vc.NewAthenaArray(N_x, N_y, N_z);

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

  u_vars.NewAthenaArray(N_cpts_total, N_x, N_y, N_z);
}

SurfaceCartesian::~SurfaceCartesian()
{
  TearDownInterpolators();
}

void SurfaceCartesian::PrepareInterpolatorAtPoint(
  MeshBlock * pmb, const int i, const int j, const int k)
{
  switch (vi)
  {
    case variety_interpolator::Lagrange:
    {
      if (mask_pinterp_Lag_cc(i,j,k) != nullptr)
      {
        delete mask_pinterp_Lag_cc(i,j,k);
        mask_pinterp_Lag_cc(i,j,k) = nullptr;
      }

      if (mask_pinterp_Lag_vc(i,j,k) != nullptr)
      {
        delete mask_pinterp_Lag_vc(i,j,k);
        mask_pinterp_Lag_vc(i,j,k) = nullptr;
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
        x(i), y(j), z(k)
      };

      mask_pinterp_Lag_cc(i,j,k) = new LagInterp(origin_cc,
                                                 delta_cc,
                                                 size_cc,
                                                 tar_coord);

      mask_pinterp_Lag_vc(i,j,k) = new LagInterp(origin_vc,
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

void SurfaceCartesian::PrepareInterpolators()
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
      #pragma omp parallel for collapse(3) num_threads(nthreads)
      for (int i=0; i<N_x; ++i)
      for (int j=0; j<N_y; ++j)
      for (int k=0; k<N_z; ++k)
      {
        const Real x_1 = x(i);
        const Real x_2 = y(j);
        const Real x_3 = z(k);

        if (pmb->PointContainedExclusive(x_1, x_2, x_3))
        {
          mask_mb(i, j, k) = pmb;
          PrepareInterpolatorAtPoint(pmb, i, j, k);
        }
      }
    }();

    pmb = pmb->next;
  }
  // --------------------------------------------------------------------------

  prepared = true;
}

void SurfaceCartesian::TearDownInterpolators()
{
  if (!prepared)
  {
    return;
  }

  // clean up mask containing salient MeshBlock
  mask_mb.Fill(nullptr);

  for (int i=0; i<N_x; ++i)
  for (int j=0; j<N_y; ++j)
  for (int k=0; k<N_z; ++k)
  {
    switch (vi)
    {
      case variety_interpolator::Lagrange:
      {
        if (mask_pinterp_Lag_cc(i,j,k) != nullptr)
        {
          delete mask_pinterp_Lag_cc(i,j,k);
          mask_pinterp_Lag_cc(i,j,k) = nullptr;
        }

        if (mask_pinterp_Lag_vc(i,j,k) != nullptr)
        {
          delete mask_pinterp_Lag_vc(i,j,k);
          mask_pinterp_Lag_vc(i,j,k) = nullptr;
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

void SurfaceCartesian::ReinitializeSurface()
{
  TearDownInterpolators();
  PrepareInterpolators();
}

void SurfaceCartesian::Reduce(const int ncycle, const Real time)
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
  if (psurfs->dump_data)
  {
    if (Globals::my_rank == 0)
    {
      write_hdf5(time);
    }
  }
};

/*
void SurfaceCartesian::MPI_Reduce()
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
                &(u_vars(0,0,0,0)),
                N_cpts_total * N_x * N_y * N_z,
                MPI_ATHENA_REAL,
                MPI_SUM,
                MPI_COMM_WORLD);
#endif
}
*/

void SurfaceCartesian::MPI_Reduce()
{
#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t total_bytes = u_vars.GetSizeInBytes();

  // max chunk size in bytes
  size_t max_chunk_bytes = DBG_SURF_CHUNK * 1024 * 1024;
  size_t start_byte = 0;

  // Pointer to the data
  Real* data_ptr = &(u_vars(0,0,0,0));

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

Real SurfaceCartesian::InterpolateAtPoint(
  aliases::AA & raw_cpt, Surfaces::variety_base_grid vs,
  const int tar_i, const int tar_j, const int tar_k)
{
  Real res = 0;

  switch (vi)
  {
    case (variety_interpolator::Lagrange):
    {
      // call suitable interpolator
      if (vs == Surfaces::variety_base_grid::cc)
      {
        if (mask_pinterp_Lag_cc(tar_i,tar_j,tar_k) != nullptr)
          res = mask_pinterp_Lag_cc(tar_i,tar_j,tar_k)->eval(raw_cpt.data());
      }
      else if (vs == Surfaces::variety_base_grid::vc)
      {
        if (mask_pinterp_Lag_vc(tar_i,tar_j,tar_k) != nullptr)
          res = mask_pinterp_Lag_vc(tar_i,tar_j,tar_k)->eval(raw_cpt.data());
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
      // given target (th_i, th_j) get pointer to data on relevant MeshBlock
      AA & raw_var = *GetRawData(psurfs->variables(vix),
                                 mask_mb(i,j,k));

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
