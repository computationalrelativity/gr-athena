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
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"

// hdf5 / mpi macros
#include "../outputs/outputs.hpp"

// ============================================================================
namespace gra::mesh::surfaces {
// ============================================================================

void InitSurfaces(Mesh *pm, ParameterInput *pin)
{
  // find salient params specifying surfaces
  InputBlock *pib = pin->pfirst_block;
  while (pib != nullptr)
  {
    if (pib->block_name.compare(0, 7, "surface")  == 0)
    {
      // extract integer number of surface.
      std::string surfn = pib->block_name.substr(7); // counting starts at 0!
      const int par_ix = atoi(surfn.c_str());

      // extract surface variety ----------------------------------------------
      typedef Surfaces::variety_surface variety_surface;
      variety_surface vs;

      const std::string block_name = "surface" + surfn;
      {

        static const std::map<std::string, variety_surface> opt_sv {
          {"spherical", variety_surface::spherical}
        };

        const std::string par_name = "surface";

        auto itr = opt_sv.find(
          pin->GetString(block_name, par_name)
        );

        if (itr != opt_sv.end())
        {
          vs = itr->second;
        }
        else
        {
          std::ostringstream msg;
          msg << block_name << "/" << par_name << " unknown" << std::endl;
          ATHENA_ERROR(msg);
        }
      }

      Surfaces * psurf;
      switch (vs)
      {
        case variety_surface::spherical:
        {
          psurf = dynamic_cast<Surfaces*>(
            new SurfacesSpherical(pm, pin, par_ix)
          );
          break;
        }
        default:
        {
          assert(false);
        }
      }

      psurf->vs = vs;
      pm->psurfs.push_back(psurf);
    }

    pib = pib->pnext;
  }
}

void InitSurfaceTriggers(gra::triggers::Triggers & trgs,
                         std::vector<gra::mesh::surfaces::Surfaces *> &psurfs)
{
  using namespace gra::triggers;
  typedef Triggers::TriggerVariant tvar;
  typedef Triggers::OutputVariant ovar;

  static const bool force_first_iter = false;

  for (auto psurf : psurfs)
  {
    trgs.Add(tvar::Surfaces,
             ovar::user,
             force_first_iter,
             psurf->adjust_mesh_dt,
             psurf->par_ix);
  }
}

// Surface collection classes -------------------------------------------------
Surfaces::Surfaces(Mesh *pm, ParameterInput *pin, const int par_ix)
  : pmesh(pm),
    pin(pin),
    par_ix(par_ix),
    par_block_name{"surface" + std::to_string(par_ix)},
    file_basename{pin->GetString("job", "problem_id")},
    file_number(pin->GetOrAddInteger(par_block_name, "file_number", 0))
{
  dt = pin->GetReal(par_block_name, "dt");
  adjust_mesh_dt = pin->GetOrAddBoolean(par_block_name,
                                        "adjust_mesh_dt",
                                        false);

  start_time = pin->GetReal(par_block_name, "start_time");
  stop_time = pin->GetReal(par_block_name, "stop_time");

  dump_data = pin->GetOrAddBoolean(par_block_name, "dump_data", false);
  prepared = false;

  // extract variables that are to be reduced ---------------------------------
  AthenaArray<std::string> str_vars = pin->GetOrAddStringArray(
    par_block_name, "variables", "", 0
  );

  const int N_vars = str_vars.GetSize();
  if (N_vars == 0)
  {
    std::ostringstream msg;
    msg << par_block_name
        << "/variables" << " not specified" << std::endl;
    ATHENA_ERROR(msg);
  }

  // reduce to enum elements
  variables.NewAthenaArray(N_vars);
  variable_sampling.NewAthenaArray(N_vars);

  for (int v=0; v<N_vars; ++v)
  {
    auto itr = map_to_variety_data.find(str_vars(v));

    if (itr != map_to_variety_data.end())
    {
      variables(v) = itr->second;
      variable_sampling(v) = GetDataBaseGrid(variables(v));
    }
    else
    {
      std::ostringstream msg;
      msg << par_block_name
          << "/variables/" << str_vars(v) << " unknown" << std::endl;
      ATHENA_ERROR(msg);
    }

  }

}

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
  for (auto surf : psurf) {
    delete surf;
  }
  psurf.resize(0);
}

void SurfacesSpherical::Reduce(const int ncycle, const Real time)
{
  // do not perform reduction if we are outside specified ranges --------------
  if ((start_time >= 0) && time < start_time)
  {
    return;
  }

  if ((stop_time >= 0) && time > stop_time)
  {
    return;
  }
  // --------------------------------------------------------------------------

  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
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

void SurfacesSpherical::ReinitializeSurfaces()
{
  for (int surf_ix=0; surf_ix<num_radii; ++surf_ix)
  {
    psurf[surf_ix]->ReinitializeSurface();
  }
}

// Surface implementations ----------------------------------------------------
Surface::Surface(
  Mesh *pm,
  ParameterInput *pin,
  Surfaces *psurfs,
  const int surf_ix)
  : pm(pm),
    psurfs(psurfs),
    surf_ix(surf_ix)
{
  // Number of field components that we wish to dump for each var
  const int num_vars = psurfs->variables.GetSize();
  N_cpts.NewAthenaArray(num_vars);

  for (int v=0; v<num_vars; ++v)
  {
    N_cpts(v) = GetNumFieldComponents(psurfs->variables(v));
  }
}

// accessors for raw data -----------------------------------------------------
AA * Surface::GetRawData(Surfaces::variety_data vd, MeshBlock * pmb)
{
  typedef Surfaces::variety_data variety_data;
  switch(vd)
  {
    case variety_data::geom_Z4c:
    {
      return &pmb->pz4c->storage.u;
    }
    case variety_data::geom_ADM:
    {
      return &pmb->pz4c->storage.adm;
    }
    case variety_data::hydro_cons:
    {
      return &pmb->phydro->u;
    }
    case variety_data::hydro_prim:
    {
      return &pmb->phydro->w;
    }
    case variety_data::hydro_aux:
    {
      return &pmb->phydro->derived_ms;
    }
    case variety_data::M1_lab:
    {
      return &pmb->pm1->storage.u;
    }
    case variety_data::M1_geom_sc_alpha:
    {
      return &pmb->pm1->geom.sc_alpha.array();
    }
    case variety_data::M1_geom_sp_beta_u:
    {
      return &pmb->pm1->geom.sp_beta_u.array();
    }
    case variety_data::M1_geom_sp_g_dd:
    {
      return &pmb->pm1->geom.sp_g_dd.array();
    }
    case variety_data::M1_geom_sp_K_dd:
    {
      return &pmb->pm1->geom.sp_K_dd.array();
    }
    case variety_data::M1_radmat:
    {
      return &pmb->pm1->storage.radmat;
    }
    case variety_data::M1_radmat_sc_avg_nrg_00:
    {
      return &pmb->pm1->radmat.sc_avg_nrg(0,0).array();
    }
    case variety_data::M1_radmat_sc_avg_nrg_01:
    {
      return &pmb->pm1->radmat.sc_avg_nrg(0,1).array();
    }
    case variety_data::M1_radmat_sc_avg_nrg_02:
    {
      return &pmb->pm1->radmat.sc_avg_nrg(0,2).array();
    }
    default:
    {
      assert(false);
    }
  }

  return nullptr;
}

int Surface::GetNumFieldComponents(Surfaces::variety_data vd)
{
  typedef Surfaces::variety_data variety_data;
  switch(vd)
  {
    case variety_data::geom_Z4c:
    {
      return Z4c::N_Z4c;
    }
    case variety_data::geom_ADM:
    {
      return Z4c::N_ADM;
    }
    case variety_data::hydro_cons:
    {
      return NHYDRO;
    }
    case variety_data::hydro_prim:
    {
      return NHYDRO;
    }
    case variety_data::hydro_aux:
    {
      return NDRV_HYDRO;
    }
    case variety_data::M1_lab:
    {
      // multiple groups / species per these vars
      const int N_GRPS = pm->pblock->pm1->N_GRPS;
      const int N_SPCS = pm->pblock->pm1->N_SPCS;
      const int N_VARS = M1::M1::ixn_Lab::N;
      return N_VARS * N_GRPS * N_SPCS;
    }
    case variety_data::M1_geom_sc_alpha:
    {
      return 1;
    }
    case variety_data::M1_geom_sp_beta_u:
    {
      return 3;
    }
    case variety_data::M1_geom_sp_g_dd:
    {
      return 6;
    }
    case variety_data::M1_geom_sp_K_dd:
    {
      return 6;
    }
    case variety_data::M1_radmat:
    {
      const int N_GRPS = pm->pblock->pm1->N_GRPS;
      const int N_SPCS = pm->pblock->pm1->N_SPCS;
      const int N_VARS = M1::M1::ixn_RaM::N;
      return N_VARS * N_GRPS * N_SPCS;
    }
    case variety_data::M1_radmat_sc_avg_nrg_00:
    {
      return 1;
    }
    case variety_data::M1_radmat_sc_avg_nrg_01:
    {
      return 1;
    }
    case variety_data::M1_radmat_sc_avg_nrg_02:
    {
      return 1;
    }
    default:
    {
      assert(false);
    }
  }

  return -1;
}

std::string Surface::GetNameFieldComponent(Surfaces::variety_data vd,
                                           const int nix)
{
  std::string ret {};
  typedef Surfaces::variety_data variety_data;

  // Map 1d index to 3d (needed for M1) - python scratch
  /*
  N_GRPS = 2
  N_SPCS = 3
  N_VARS = 5

  n = 0
  for g in range(N_GRPS):
    for s in range(N_SPCS):
      for v in range(N_VARS):
        # Indicial structure:
        # n_ix = v + N_VARS * (s + N_SPCS * g)
        #        v + s * N_VARS + g * N_VARS * N_SPCS

        # group index
        g_ix = n // (N_VARS * N_SPCS)

        # species index
        s_ix = (n - g_ix * N_VARS * N_SPCS) // N_VARS

        # variable index
        v_ix = (n - g_ix * N_VARS * N_SPCS - s_ix * N_VARS)

        print(n, v + N_VARS * (s + N_SPCS * g),
              g_ix - g, s_ix - s, v_ix - v)
        n += 1
  */

  // flat idx, total num vars, var idx, grp idx, sps idx
  auto m1_vgs_idx = [&](const int n, const int N, int & v, int & g, int & s)
  {
    // See indicial structure above
    const int N_VARS = N;
    const int N_GRPS = pm->pblock->pm1->N_GRPS;
    const int N_SPCS = pm->pblock->pm1->N_SPCS;

    // group index
    g = n / (N_VARS * N_SPCS);

    // species index
    s = (n - g * N_VARS * N_SPCS) / N_VARS;

    // variable index
    v = (n - g * N_VARS * N_SPCS - s * N_VARS);
  };

  switch(vd)
  {
    case variety_data::geom_Z4c:
    {
      ret = Z4c::Z4c_names[nix];
      break;
    }
    case variety_data::geom_ADM:
    {
      ret = Z4c::ADM_names[nix];
      break;
    }
    case variety_data::hydro_cons:
    {
      ret = Hydro::ixn_cons::names[nix];
      break;
    }
    case variety_data::hydro_prim:
    {
      ret = Hydro::ixn_prim::names[nix];
      break;
    }
    case variety_data::hydro_aux:
    {
      ret = Hydro::ixn_derived_ms::names[nix];
      break;
    }
    case variety_data::M1_lab:
    {
      int v, g, s;
      m1_vgs_idx(nix, M1::M1::ixn_Lab::N, v, g, s);
      ret += M1::M1::ixn_Lab::names[v];
      ret += "_" + std::to_string(g) + std::to_string(s);
      break;
    }
    case variety_data::M1_geom_sc_alpha:
    {
      ret = "M1.geom.sc_alpha";
      break;
    }
    case variety_data::M1_geom_sp_beta_u:
    {
      static const char * const names[] {
        "M1.geom.sp_beta_u_x",
        "M1.geom.sp_beta_u_y",
        "M1.geom.sp_beta_u_z"
      };
      ret = names[nix];
      break;
    }
    case variety_data::M1_geom_sp_g_dd:
    {
      static const char * const names[] {
        "M1.geom.sp_g_dd_xx",
        "M1.geom.sp_g_dd_xy",
        "M1.geom.sp_g_dd_xz",
        "M1.geom.sp_g_dd_yy",
        "M1.geom.sp_g_dd_yz",
        "M1.geom.sp_g_dd_zz"
      };
      ret = names[nix];
      break;
    }
    case variety_data::M1_geom_sp_K_dd:
    {
      static const char * const names[] {
        "M1.geom.sp_K_dd_xx",
        "M1.geom.sp_K_dd_xy",
        "M1.geom.sp_K_dd_xz",
        "M1.geom.sp_K_dd_yy",
        "M1.geom.sp_K_dd_yz",
        "M1.geom.sp_K_dd_zz"
      };
      ret = names[nix];
      break;
    }
    case variety_data::M1_radmat:
    {
      int v, g, s;
      m1_vgs_idx(nix, M1::M1::ixn_RaM::N, v, g, s);
      ret += M1::M1::ixn_RaM::names[v];
      ret += "__" + std::to_string(g) + std::to_string(s);
      break;
    }
    case variety_data::M1_radmat_sc_avg_nrg_00:
    {
      ret = "M1.radmat.sc_avg_nrg_00";
      break;
    }
    case variety_data::M1_radmat_sc_avg_nrg_01:
    {
      ret = "M1.radmat.sc_avg_nrg_01";
      break;
    }
    case variety_data::M1_radmat_sc_avg_nrg_02:
    {
      ret = "M1.radmat.sc_avg_nrg_02";
      break;
    }
    default:
    {
      assert(false);
    }
  }

  return ret;
}

void Surface::hdf5_get_next_filename(std::string & filename)
{
  const int iter = psurfs->file_number;

  std::stringstream ss_i;

  // One file per shell:
  // ss_i << std::setw(2) << std::setfill('0') << std::to_string(ix_rad);
  // ss_i << "_";
  // filename = file_basename + ".shell." + ss_i.str() + ".hdf5";

  // Coalesce all shells into single file:
  ss_i << std::setw(6) << std::setfill('0') << iter;
  // filename = psurfs->file_basename + ".shells." + ss_i.str() + ".hdf5";
  filename = psurfs->file_basename + "." + psurfs->par_block_name + ".";
  filename += ss_i.str() + ".hdf5";
}

void SurfaceSpherical::write_hdf5(const Real T)
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

// ----------------------------------------------------------------------------

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

  // pointer arrays for interpolators -----------------------------------------
  mask_mb.NewAthenaArray(N_th, N_ph);
  mask_mb.Fill(nullptr);

  switch (vi)
  {
    case variety_interpolator::Lagrange:
    {
      mask_pinterp_Lag_cc.NewAthenaArray(N_th, N_ph);
      mask_pinterp_Lag_vc.NewAthenaArray(N_th, N_ph);

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

  u_vars.NewAthenaArray(N_cpts_total, N_th, N_ph);
}

SurfaceSpherical::~SurfaceSpherical()
{
  TearDownInterpolators();
  prepared = false;
  // DEBUG
  // std::printf("Killed s @ rad=%.3g, N_th=%d, N_ph=%d \n", rad, N_th, N_ph);
}

void SurfaceSpherical::PrepareInterpolatorAtPoint(
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
        x_o_th_ph(0,i,j), x_o_th_ph(1,i,j), x_o_th_ph(2,i,j)
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

void SurfaceSpherical::PrepareInterpolators()
{
  // BD: TODO - could be optimized further

  // Connect target (th_i, ph_j) to salient MeshBlock pointer -----------------
  MeshBlock * pmb = pm->pblock;

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  while (pmb != nullptr)
  {
    // Given current MeshBlock check whether grid intersects (excluding ghosts)
    [&] {
      #pragma omp parallel for collapse(2) num_threads(nthreads)
      for (int i=0; i<N_th; ++i)
      {
        for (int j=0; j<N_ph; ++j)
        {
          const Real x_1 = x_o_th_ph(0,i,j);
          const Real x_2 = x_o_th_ph(1,i,j);
          const Real x_3 = x_o_th_ph(2,i,j);

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
}

void SurfaceSpherical::TearDownInterpolators()
{
  // clean up mask containing salient MeshBlock
  mask_mb.Fill(nullptr);

  for (int i=0; i<N_th; ++i)
  for (int j=0; j<N_ph; ++j)
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

void SurfaceSpherical::ReinitializeSurface()
{
  TearDownInterpolators();
  PrepareInterpolators();
  prepared = true;
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

    prepared = true;
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

void SurfaceSpherical::MPI_Reduce()
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
                N_cpts_total * N_th * N_ph,
                MPI_ATHENA_REAL,
                MPI_SUM,
                MPI_COMM_WORLD);
#endif
}

Real SurfaceSpherical::InterpolateAtPoint(
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
        AA sl_u(raw_var, cix, 1);

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