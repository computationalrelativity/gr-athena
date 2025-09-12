#ifndef MESH_SURFACES_HPP
#define MESH_SURFACES_HPP
// C++ standard headers
#include <future>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <numeric>

// Athena++ headers
#include "../athena_aliases.hpp"
#include "../parameter_input.hpp"
#include "mesh.hpp"
#include "../outputs/hdf5_guard.hpp"

#include "../utils/lagrange_interp.hpp"
#include "../utils/interp_barycentric.hpp"
#include "../main_triggers.hpp"

// ============================================================================
namespace gra::mesh::surfaces {
// ============================================================================

// Forward declare
class SurfaceCartesian;
class SurfaceCylindrical;
class SurfaceSpherical;

void InitSurfaces(Mesh *pm, ParameterInput *pin);

// Add a trigger for each surface
void InitSurfaceTriggers(gra::triggers::Triggers & trgs,
                         std::vector<gra::mesh::surfaces::Surfaces *> &psurfs);


// Surface collection classes -------------------------------------------------
class Surfaces
{
  public:
    enum class variety_surface { cartesian,
                                 cylindrical,
                                 spherical };

    // Data that can be reduced to surface
    enum class variety_data {
      geom_Z4c,
      geom_ADM,
      // fluid
      hydro_cons,
      hydro_prim,
      hydro_aux,
      // B-field [aux]
      field_aux,
      // tracer quantities
      tracer_vel,
      tracer_rho,
      tracer_ye,
      tracer_aux_T,
      tracer_aux_U_d_0,
      tracer_aux_HU_d_0,
      tracer_aux_SPB,
      // scalars
      passive_scalars_cons,
      passive_scalars_prim,
      // magnetic fields
      B,
      // radiation
      M1_lab,
      M1_geom_sc_sqrt_det_g,
      M1_geom_sc_alpha,
      M1_geom_sp_beta_u,
      M1_geom_sp_g_dd,
      M1_geom_sp_K_dd,
      M1_rad,
      M1_radmat,
      M1_radmat_sc_avg_nrg_00,
      M1_radmat_sc_avg_nrg_01,
      M1_radmat_sc_avg_nrg_02,
    };

    // N.B. variables must contain a "."; it is used in dump-naming
    const std::map<std::string, variety_data> map_to_variety_data {
#if Z4C_ENABLED
      {"geom.Z4c",   variety_data::geom_Z4c},
      {"geom.ADM",   variety_data::geom_ADM},
#endif
#if FLUID_ENABLED
      {"hydro.cons", variety_data::hydro_cons},
      {"hydro.prim", variety_data::hydro_prim},
      {"hydro.aux",  variety_data::hydro_aux},
      {"tracer.vel",  variety_data::tracer_vel},
      {"tracer.rho",  variety_data::tracer_rho},
      {"tracer.aux.T",       variety_data::tracer_aux_T},
      {"tracer.aux.U_d_0",   variety_data::tracer_aux_U_d_0},
      {"tracer.aux.HU_d_0",  variety_data::tracer_aux_HU_d_0},
      {"tracer.aux.SPB",     variety_data::tracer_aux_SPB},
#endif
#if NSCALARS > 0
      {"tracer.ye",  variety_data::tracer_ye},
      {"passive_scalars.cons",  variety_data::passive_scalars_cons},
      {"passive_scalars.prim",  variety_data::passive_scalars_prim},
#endif
#if MAGNETIC_FIELDS_ENABLED
      {"field.aux",  variety_data::field_aux},
      {"B",          variety_data::B},
#endif
#if M1_ENABLED
      // non-contiguous arrays
      {"M1.lab",               variety_data::M1_lab},
      {"M1.geom.sc_sqrt_det_g",variety_data::M1_geom_sc_sqrt_det_g},
      {"M1.geom.sc_alpha",     variety_data::M1_geom_sc_alpha},
      {"M1.geom.sp_beta_u",    variety_data::M1_geom_sp_beta_u},
      {"M1.geom.sp_g_dd",      variety_data::M1_geom_sp_g_dd},
      {"M1.geom.sp_K_dd",      variety_data::M1_geom_sp_K_dd},
      {"M1.rad",               variety_data::M1_rad},
      {"M1.radmat",            variety_data::M1_radmat},
      {"M1.radmat.sc_avg_nrg_00", variety_data::M1_radmat_sc_avg_nrg_00},
      {"M1.radmat.sc_avg_nrg_01", variety_data::M1_radmat_sc_avg_nrg_01},
      {"M1.radmat.sc_avg_nrg_02", variety_data::M1_radmat_sc_avg_nrg_02},
#endif
    };

    enum class variety_base_grid {
      cc, vc
    };

  private:
    // Get the base grid sampling (e.g. Z4c/ADM depend on macros etc)
    inline variety_base_grid GetDataBaseGrid(variety_data vd)
    {
#ifdef Z4C_VC_ENABLED
      if ((vd == variety_data::geom_Z4c) ||
          (vd == variety_data::geom_ADM))
      {
        return variety_base_grid::vc;
      }
#endif // Z4C_VC_ENABLED
      return variety_base_grid::cc;
    }

  public:
    Surfaces(Mesh *pm, ParameterInput *pin, const int par_ix);
    virtual ~Surfaces() = default;

  public:
    Mesh *pmesh;
    ParameterInput *pin;

    const int par_ix;
    const std::string par_block_name;

    const std::string file_basename;
    int file_number;

    Real dt;
    bool adjust_mesh_dt;

    Real start_time;
    Real stop_time;

    bool dump_data;
    bool prepared;
    variety_surface vs;

    // variables that are to be reduced
    AthenaArray<variety_data>      variables;
    // their associated samplings
    AthenaArray<variety_base_grid> variable_sampling;

    // only use asynchronous writes with thread-safe library
    const bool can_async = is_hdf5_threadsafe();

    // by default we use root rank for writing
    const bool use_multiple_ranks = true;
    int write_rank = 0;

  public:

    // Check whether a surface is active
    virtual bool IsActive(const Real time) { return true; };
    // Reduction (call on each surface in Surfaces collection)
    virtual void Reduce(const int ncycle, const Real time) { };
    // Teardown and prepare interpolators on each Surface
    virtual void ReinitializeSurfaces(const int ncycle,
                                      const Real time) { };
};

class SurfacesCartesian : public Surfaces
{
  public:
    SurfacesCartesian(Mesh *pm, ParameterInput *pin, const int par_ix);
    ~SurfacesCartesian();

  public:
    const int num_surf;

    std::vector<SurfaceCartesian *> psurf;

  public:
    virtual bool IsActive(const Real time) override;
    virtual void Reduce(const int ncycle, const Real time) override;
    virtual void ReinitializeSurfaces(const int ncycle,
                                      const Real time) override;

    // finish writing operations
    void WriteBlock();
    // write all surfaces asynchronously
    void WriteAllSurfaces(const Real time);

  private:
    std::future<void> write_future;
};

class SurfacesCylindrical : public Surfaces
{
  public:
    SurfacesCylindrical(Mesh *pm, ParameterInput *pin, const int par_ix);
    ~SurfacesCylindrical();

  public:
    const int num_radii;

    std::vector<SurfaceCylindrical *> psurf;

  public:
    virtual bool IsActive(const Real time) override;
    virtual void Reduce(const int ncycle, const Real time) override;
    virtual void ReinitializeSurfaces(const int ncycle,
                                      const Real time) override;

    // finish writing operations
    void WriteBlock();
    // write all surfaces asynchronously
    void WriteAllSurfaces(const Real time);

  private:
    std::future<void> write_future;
};

class SurfacesSpherical : public Surfaces
{
  public:
    SurfacesSpherical(Mesh *pm, ParameterInput *pin, const int par_ix);
    ~SurfacesSpherical();

  public:
    const int num_radii;

    std::vector<SurfaceSpherical *> psurf;

  public:
    virtual bool IsActive(const Real time) override;
    virtual void Reduce(const int ncycle, const Real time) override;
    virtual void ReinitializeSurfaces(const int ncycle,
                                      const Real time) override;

    // finish writing operations
    void WriteBlock();
    // write all surfaces asynchronously
    void WriteAllSurfaces(const Real time);

  private:
    std::future<void> write_future;
};


// single surface class -------------------------------------------------------
class Surface
{
  public:
    Surface(Mesh *pm,
            ParameterInput *pin,
            Surfaces *psurfs,
            const int surf_ix);
    virtual ~Surface() = default;

    // General interface for the reduction
    virtual void Reduce(const int ncycle, const Real time) { };
    // Set up from scratch (cleaning up internally) surface interp etc
    virtual void ReinitializeSurface() {};

  public:

    // Number of field components that we wish to dump for each var
    AthenaArray<int> N_cpts;

    // collective array storing all interpolated data
    aliases::AA u_vars;

    // Get ptr to the data based on variety
    aliases::AA * GetRawData(Surfaces::variety_data vd, MeshBlock * pmb);

    // Total number of field components to reduce
    int GetNumFieldComponents(Surfaces::variety_data vd);

    // For non-contiguous data remap idx based on variety of data
    int GetRemappedFieldIndex(Surfaces::variety_data vd, const int nix);

    // Pointer to array of field component names
    std::string GetNameFieldComponent(Surfaces::variety_data vd,
                                      const int nix);

  protected:
    virtual void write_hdf5(const Real time) { };
    void hdf5_get_next_filename(std::string & filename);

  protected:
    Mesh * pm;
    Surfaces *psurfs;

    const int surf_ix;
};

class SurfaceCylindrical : public Surface
{
  friend SurfacesCylindrical;

  public:
    enum class variety_sampling { uniform };
    enum class variety_interpolator { Lagrange };

    SurfaceCylindrical(Mesh *pm,
                       ParameterInput *pin,
                       Surfaces *psurfs,
                       const int surf_ix);

    ~SurfaceCylindrical();

    virtual void Reduce(const int ncycle, const Real time) override;
    virtual void ReinitializeSurface() override;

  private:

    Real rad;
    Real z_min, z_max;
    int N_ph;
    int N_z;
    int N_pts;

    // For storage of grids
    aliases::AA ph;
    aliases::AA z;
    aliases::AA x_o_ph_z;  // (x1(rad,ph,z), x2(rad,ph,z), x3(rad,ph,z))

    variety_sampling vs;
    variety_interpolator vi;

    // For storage of interpolators / target point masks
    typedef LagrangeInterpND<2 * NGHOST - 1, 3> LagInterp;

    // (i,j) = pointer to MeshBlock (if it exists) within Mesh
    // that contains (th_i, th_j)
    AthenaArray<MeshBlock *> mask_mb;

    AthenaArray<LagInterp *> mask_pinterp_Lag_cc;
    AthenaArray<LagInterp *> mask_pinterp_Lag_vc;

    // have we allocated interpolators for a given grid structure?
    bool prepared = false;

  private:

    inline void gr_z(aliases::AA & z_in)
    {
      const Real dz = (z_max-z_min) / static_cast<Real>(N_z - 1);
      for (int n=0; n<N_z; ++n)
      {
        z_in(n) = z_min + dz * n;
      }
    }

    inline void gr_ph(aliases::AA & ph_in)
    {
      const Real dph = 2.0 * PI / static_cast<Real>(N_ph);
      for (int n=0; n<N_ph; ++n)
      {
        ph_in(n) = dph * (0.5 + n);
      }
    }

  // interpolator specific ----------------------------------------------------
  private:
    void PrepareInterpolators();
    void PrepareInterpolatorAtPoint(MeshBlock * pmb, const int i, const int j);
    void TearDownInterpolators();

    void DoInterpolations();
    Real InterpolateAtPoint(aliases::AA & raw_cpt,
                            Surfaces::variety_base_grid vs,
                            const int tar_i, const int tar_j);

    void MPI_Reduce();

  // output specific ----------------------------------------------------------
  protected:
    virtual void write_hdf5(const Real time) override;

};


class SurfaceCartesian : public Surface
{
  friend SurfacesCartesian;

  public:
    enum class variety_sampling { uniform, cgl };
    enum class variety_interpolator { Lagrange };

    SurfaceCartesian(Mesh *pm,
                     ParameterInput *pin,
                     Surfaces *psurfs,
                     const int surf_ix);

    ~SurfaceCartesian();

    virtual void Reduce(const int ncycle, const Real time) override;
    virtual void ReinitializeSurface() override;

  private:

    Real x_min, x_max;
    Real y_min, y_max;
    Real z_min, z_max;

    int N_x;
    int N_y;
    int N_z;
    int N_pts;

    // For storage of grids
    aliases::AA x;
    aliases::AA y;
    aliases::AA z;

    variety_sampling vs;
    variety_interpolator vi;

    // For storage of interpolators / target point masks
    typedef LagrangeInterpND<2 * NGHOST - 1, 3> LagInterp;

    // (i,j) = pointer to MeshBlock (if it exists) within Mesh
    // that contains (th_i, th_j)
    AthenaArray<MeshBlock *> mask_mb;

    AthenaArray<LagInterp *> mask_pinterp_Lag_cc;
    AthenaArray<LagInterp *> mask_pinterp_Lag_vc;

    // have we allocated interpolators for a given grid structure?
    bool prepared = false;

  private:

    inline void uniform_gr_x(aliases::AA & x_in)
    {
      const Real dx = (x_max-x_min) / static_cast<Real>(N_x - 1);
      for (int n=0; n<N_x; ++n)
      {
        x_in(n) = x_min + dx * n;
      }
    }

    inline void uniform_gr_y(aliases::AA & y_in)
    {
      const Real dy = (y_max-y_min) / static_cast<Real>(N_y - 1);
      for (int n=0; n<N_y; ++n)
      {
        y_in(n) = y_min + dy * n;
      }
    }

    inline void uniform_gr_z(aliases::AA & z_in)
    {
      const Real dz = (z_max-z_min) / static_cast<Real>(N_z - 1);
      for (int n=0; n<N_z; ++n)
      {
        z_in(n) = z_min + dz * n;
      }
    }

    inline void cgl_gr_x(aliases::AA & x_in)
    {
      const Real mi = 0.5 * (x_min + x_max);
      const Real hr = 0.5 * (x_max - x_min);

      for (int n=0; n<N_x; ++n)
      {
        x_in(n) = mi + hr * std::cos(PI * n / (N_x - 1));
      }
    }

    inline void cgl_gr_y(aliases::AA & y_in)
    {
      const Real mi = 0.5 * (y_min + y_max);
      const Real hr = 0.5 * (y_max - y_min);

      for (int n=0; n<N_y; ++n)
      {
        y_in(n) = mi + hr * std::cos(PI * n / (N_y - 1));
      }
    }

    inline void cgl_gr_z(aliases::AA & z_in)
    {
      const Real mi = 0.5 * (z_min + z_max);
      const Real hr = 0.5 * (z_max - z_min);

      for (int n=0; n<N_z; ++n)
      {
        z_in(n) = mi + hr * std::cos(PI * n / (N_z - 1));
      }
    }

  // interpolator specific ----------------------------------------------------
  private:
    void PrepareInterpolators();
    void PrepareInterpolatorAtPoint(MeshBlock * pmb,
                                    const int i, const int j, const int k);
    void TearDownInterpolators();

    void DoInterpolations();
    Real InterpolateAtPoint(aliases::AA & raw_cpt,
                            Surfaces::variety_base_grid vs,
                            const int tar_i, const int tar_j, const int tar_k);

    void MPI_Reduce();

  // output specific ----------------------------------------------------------
  private:
    virtual void write_hdf5(const Real time) override;

};

class SurfaceSpherical : public Surface
{
  friend SurfacesSpherical;

  public:
    enum class variety_sampling { uniform };
    enum class variety_interpolator { Lagrange };

    SurfaceSpherical(Mesh *pm,
                     ParameterInput *pin,
                     Surfaces *psurfs,
                     const int surf_ix);

    ~SurfaceSpherical();

    virtual void Reduce(const int ncycle, const Real time) override;
    virtual void ReinitializeSurface() override;

  private:

    Real rad;
    int N_th;
    int N_ph;
    int N_pts;

    // For storage of grids
    aliases::AA th;
    aliases::AA ph;
    aliases::AA x_o_th_ph;  // (x1(rad,th,ph), x2(rad,th,ph), x3(rad,th,ph))

    variety_sampling vs;
    variety_interpolator vi;

    // For storage of interpolators / target point masks
    typedef LagrangeInterpND<2 * NGHOST - 1, 3> LagInterp;

    // (i,j) = pointer to MeshBlock (if it exists) within Mesh
    // that contains (th_i, th_j)
    AthenaArray<MeshBlock *> mask_mb;

    AthenaArray<LagInterp *> mask_pinterp_Lag_cc;
    AthenaArray<LagInterp *> mask_pinterp_Lag_vc;

    // have we allocated interpolators for a given grid structure?
    bool prepared = false;

  private:

    inline void gr_th(aliases::AA & th_in)
    {
      const Real dth = PI / static_cast<Real>(N_th);
      for (int n=0; n<N_th; ++n)
      {
        th_in(n) = dth * (0.5 + n);
      }
    }

    inline void gr_ph(aliases::AA & ph_in)
    {
      const Real dph = 2.0 * PI / static_cast<Real>(N_ph);
      for (int n=0; n<N_ph; ++n)
      {
        ph_in(n) = dph * (0.5 + n);
      }
    }

  // interpolator specific ----------------------------------------------------
  private:
    void PrepareInterpolators();
    void PrepareInterpolatorAtPoint(MeshBlock * pmb, const int i, const int j);
    void TearDownInterpolators();

    void DoInterpolations();
    Real InterpolateAtPoint(aliases::AA & raw_cpt,
                            Surfaces::variety_base_grid vs,
                            const int tar_i, const int tar_j);

    void MPI_Reduce();

  // output specific ----------------------------------------------------------
  private:
    virtual void write_hdf5(const Real time) override;

};

// ============================================================================
} // namespace gra::mesh::surfaces
// ============================================================================

#endif  // MESH_SURFACES_HPP
