#ifndef MESH_SURFACES_HPP
#define MESH_SURFACES_HPP
// C++ standard headers
#include <future>
#include <iomanip>
#include <map>
#include <sstream>
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
#include "../utils/grid_theta_phi.hpp"
#include "../utils/spherical_harmonics.hpp"
#include "../main_triggers.hpp"

// ============================================================================
namespace gra::mesh::surfaces {
// ============================================================================

// Forward declare
class Surface;
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
      geom_aux,
      geom_weyl,
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

  private:
    // Build reverse map: variety_data enum -> HDF5 group prefix
    // (substring before the first '.', or the full key if no '.')
    static std::map<variety_data, std::string> BuildVarietyPrefixMap(
      const std::map<std::string, variety_data> & fwd)
    {
      std::map<variety_data, std::string> rev;
      for (auto const & kv : fwd)
      {
        auto dot = kv.first.find('.');
        std::string prefix = (dot != std::string::npos)
          ? kv.first.substr(0, dot)
          : kv.first;
        // first entry wins - all entries sharing an enum value have the
        // same prefix, so order does not matter
        rev.emplace(kv.second, std::move(prefix));
      }
      return rev;
    }

  public:
    // N.B. variables must contain a "."; it is used in dump-naming
    const std::map<std::string, variety_data> map_to_variety_data {
#if Z4C_ENABLED
      {"geom.Z4c",   variety_data::geom_Z4c},
      {"geom.ADM",   variety_data::geom_ADM},
      {"geom.aux",   variety_data::geom_aux},
      {"geom.weyl",   variety_data::geom_weyl},
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

    // Reverse map: variety_data enum -> HDF5 group prefix string
    // (built once from map_to_variety_data at construction time)
    const std::map<variety_data, std::string> map_to_variety_prefix {
      BuildVarietyPrefixMap(map_to_variety_data)
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
          (vd == variety_data::geom_ADM) ||
	  (vd == variety_data::geom_aux) ||
	  (vd == variety_data::geom_weyl))
      {
        return variety_base_grid::vc;
      }
#endif // Z4C_VC_ENABLED
      return variety_base_grid::cc;
    }

  public:
    Surfaces(Mesh *pm, ParameterInput *pin, const int par_ix);
    virtual ~Surfaces();

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

    // variables that are (additionally, independently) to be projected onto
    // scalar spherical harmonics, producing a_lm coefficients.
    // Populated from the optional "variables_sh" input list; only meaningful
    // for spherical surfaces. Empty (default) disables SH projection.
    AthenaArray<variety_data>      variables_sh;
    // their associated samplings
    AthenaArray<variety_base_grid> variable_sh_sampling;

    // maximum angular momentum number for scalar spherical-harmonic
    // projection of variables_sh; < 0 disables projection entirely.
    int lmax_sh = -1;

    // only use asynchronous writes with thread-safe library
    const bool can_async = is_hdf5_threadsafe();

    // by default we use root rank for writing
    const bool use_multiple_ranks = true;
    int write_rank = 0;

    bool write_final;
    bool is_final = false;

    // number of individual surfaces in this collection
    int num_surf = 0;

    // per-surface pointers (owned by this collection, deleted in destructor)
    std::vector<Surface *> psurf;

  public:

    // Check whether a surface is active
    bool IsActive(const Real time);
    // Reduction (call on each surface in Surfaces collection)
    void Reduce(const int ncycle, const Real time,
                const bool is_final);
    // Teardown and prepare interpolators on each Surface
    void ReinitializeSurfaces(const int ncycle,
                              const Real time);

    // finish writing operations
    void WriteBlock();
    // write all surfaces asynchronously
    void WriteAllSurfaces(const Real time);

  private:
    std::future<void> write_future;
};

class SurfacesCartesian : public Surfaces
{
  public:
    SurfacesCartesian(Mesh *pm, ParameterInput *pin, const int par_ix);
};

class SurfacesCylindrical : public Surfaces
{
  public:
    SurfacesCylindrical(Mesh *pm, ParameterInput *pin, const int par_ix);
};

class SurfacesSpherical : public Surfaces
{
  public:
    SurfacesSpherical(Mesh *pm, ParameterInput *pin, const int par_ix);
};


// single surface class -------------------------------------------------------
class Surface
{
  // The collection class needs access to write_hdf5() for async writes
  friend Surfaces;

  public:
    Surface(Mesh *pm,
            ParameterInput *pin,
            Surfaces *psurfs,
            const int surf_ix);
    virtual ~Surface();

    // General interface for the reduction
    void Reduce(const int ncycle, const Real time);
    // Compute phase: prepare interpolators + interpolate (no MPI)
    void Reduce_Compute();
    // Communicate phase: MPI_Allreduce of u_vars
    void Reduce_Communicate();
    // Set up from scratch (cleaning up internally) surface interp etc
    // active=false: only tear down (AMR invalidated MeshBlock pointers)
    // active=true:  tear down + re-prepare interpolators
    void ReinitializeSurface(const bool active = true);

  public:

    // Number of field components that we wish to dump for each var
    AthenaArray<int> N_cpts;

    // collective array storing all interpolated data
    aliases::AA u_vars;

    // Number of field components to project onto scalar spherical harmonics,
    // one entry per variable in psurfs->variables_sh (empty if unused).
    AthenaArray<int> N_cpts_sh;

    // Channel index into the middle ("kind") dimension of a_lm below:
    //   sh_a0 - m=0 coefficients   (RealHarmonicTable::spec0, l = 0..lmax_sh)
    //   sh_ac - m>0 cosine coefficients (RealHarmonicTable::specc)
    //   sh_as - m>0 sine   coefficients (RealHarmonicTable::specs)
    // sh_a0 only populates the first (lmax_sh + 1) entries of the trailing
    // (lm) dimension; the remainder is unused padding shared with the
    // (larger) ac/as extent so all three channels fit in one array.
    enum sh_channel { sh_a0 = 0, sh_ac = 1, sh_as = 2, sh_nch = 3 };

    // Extent of the trailing (lm) dimension of a_lm, i.e. (lmax_sh+1)^2;
    // set alongside a_lm's allocation. 0 when SH projection is unused.
    int lmpoints_sh = 0;

    // Projected (l,m) scalar spherical-harmonic coefficients for all field
    // components listed (flattened) across psurfs->variables_sh, stored
    // together in a single array rather than one AthenaArray per kind:
    //   a_lm(c, sh_a0/sh_ac/sh_as, lm)
    // where c indexes field components and lm = RealHarmonicTable::lmindex.
    // Only allocated (geometry-specific) when psurfs->lmax_sh >= 0.
    aliases::AA a_lm;

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
    void write_hdf5(const Real time);
    void hdf5_get_next_filename(std::string & filename);

    // Geometry-specific: write coordinate scalars and arrays to HDF5 file
    virtual void write_hdf5_coordinates(hid_t & id_file,
                                        const std::string & six) = 0;

    // Geometry-specific: set up interpolators for the grid
    virtual void PrepareInterpolators() = 0;
    // Geometry-specific: interpolate all variables on the grid
    virtual void DoInterpolations() = 0;
    // Geometry-specific: project variables_sh onto scalar spherical
    // harmonics, filling a_lm_0/a_lm_c/a_lm_s. Default is a no-op so
    // geometries without a natural angular basis (Cartesian, Cylindrical)
    // need not implement it. Only SurfaceSpherical currently overrides this.
    virtual void DoProjections() {}

    // Tear down interpolators (clear pools, reset masks)
    void TearDownInterpolators();
    // Chunked MPI_Allreduce on u_vars, a_lm
    void MPI_Reduce();

  protected:
    Mesh * pm;
    Surfaces *psurfs;

    const int surf_ix;

    // For storage of interpolators / target point masks
    typedef LagrangeInterpND<2 * NGHOST - 1, 3> LagInterp;
    typedef LagrangeInterpND<1, 3> LagInterpLinear;

    // Contiguous interpolator pools (one entry per occupied grid point)
    std::vector<LagInterp> interp_pool_Lag_cc;
    std::vector<LagInterp> interp_pool_Lag_vc;
    std::vector<LagInterpLinear> interp_pool_LagLinear_cc;
    std::vector<LagInterpLinear> interp_pool_LagLinear_vc;

    // Index into interp_pool_*; -1 means no interpolator at this grid point
    AthenaArray<int> mask_interp_idx_cc;
    AthenaArray<int> mask_interp_idx_vc;

    // Pointer to MeshBlock that owns each grid point (geometry-specific shape)
    AthenaArray<MeshBlock *> mask_mb;

    // have we allocated interpolators for a given grid structure?
    bool prepared = false;
};

class SurfaceCylindrical : public Surface
{
  public:
    enum class variety_sampling { uniform };
    enum class variety_interpolator { Lagrange, LagrangeLinear };

    SurfaceCylindrical(Mesh *pm,
                       ParameterInput *pin,
                       Surfaces *psurfs,
                       const int surf_ix);

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
    virtual void PrepareInterpolators() override;
    virtual void DoInterpolations() override;

    Real InterpolateAtPoint(aliases::AA & raw_cpt,
                            Surfaces::variety_base_grid vs,
                            const int tar_i, const int tar_j);

  // output specific ----------------------------------------------------------
  protected:
    virtual void write_hdf5_coordinates(hid_t & id_file,
                                        const std::string & six) override;

};


class SurfaceCartesian : public Surface
{
  public:
    enum class variety_sampling { uniform, cgl };
    enum class variety_interpolator { Lagrange, LagrangeLinear };

    SurfaceCartesian(Mesh *pm,
                     ParameterInput *pin,
                     Surfaces *psurfs,
                     const int surf_ix);

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

  private:

    inline void uniform_gr_x(aliases::AA & x_in)
    {
      if (N_x == 1)
      {
        x_in(0) = 0.5 * (x_min + x_max);
        return;
      }
      const Real dx = (x_max-x_min) / static_cast<Real>(N_x - 1);
      for (int n=0; n<N_x; ++n)
      {
        x_in(n) = x_min + dx * n;
      }
    }

    inline void uniform_gr_y(aliases::AA & y_in)
    {
      if (N_y == 1)
      {
        y_in(0) = 0.5 * (y_min + y_max);
        return;
      }
      const Real dy = (y_max-y_min) / static_cast<Real>(N_y - 1);
      for (int n=0; n<N_y; ++n)
      {
        y_in(n) = y_min + dy * n;
      }
    }

    inline void uniform_gr_z(aliases::AA & z_in)
    {
      if (N_z == 1)
      {
        z_in(0) = 0.5 * (z_min + z_max);
        return;
      }
      const Real dz = (z_max-z_min) / static_cast<Real>(N_z - 1);
      for (int n=0; n<N_z; ++n)
      {
        z_in(n) = z_min + dz * n;
      }
    }

    inline void cgl_gr_x(aliases::AA & x_in)
    {
      if (N_x == 1)
      {
        x_in(0) = 0.5 * (x_min + x_max);
        return;
      }
      const Real mi = 0.5 * (x_min + x_max);
      const Real hr = 0.5 * (x_max - x_min);

      for (int n=0; n<N_x; ++n)
      {
        x_in(n) = mi + hr * std::cos(PI * n / (N_x - 1));
      }
    }

    inline void cgl_gr_y(aliases::AA & y_in)
    {
      if (N_y == 1)
      {
        y_in(0) = 0.5 * (y_min + y_max);
        return;
      }
      const Real mi = 0.5 * (y_min + y_max);
      const Real hr = 0.5 * (y_max - y_min);

      for (int n=0; n<N_y; ++n)
      {
        y_in(n) = mi + hr * std::cos(PI * n / (N_y - 1));
      }
    }

    inline void cgl_gr_z(aliases::AA & z_in)
    {
      if (N_z == 1)
      {
        z_in(0) = 0.5 * (z_min + z_max);
        return;
      }
      const Real mi = 0.5 * (z_min + z_max);
      const Real hr = 0.5 * (z_max - z_min);

      for (int n=0; n<N_z; ++n)
      {
        z_in(n) = mi + hr * std::cos(PI * n / (N_z - 1));
      }
    }

  // interpolator specific ----------------------------------------------------
  private:
    virtual void PrepareInterpolators() override;
    virtual void DoInterpolations() override;

    Real InterpolateAtPoint(aliases::AA & raw_cpt,
                            Surfaces::variety_base_grid vs,
                            const int tar_i, const int tar_j, const int tar_k);

  // output specific ----------------------------------------------------------
  private:
    virtual void write_hdf5_coordinates(hid_t & id_file,
                                        const std::string & six) override;

};

class SurfaceSpherical : public Surface
{
  public:
    // uniform:      midpoint sampling in theta, uniform in phi
    // gausslegendre: Gauss-Legendre nodes in cos(theta), uniform in phi;
    //                requires N_th == N_ph / 2 (as in grid_theta_phi::Grid)
    enum class variety_sampling { uniform, gausslegendre };
    enum class variety_interpolator { Lagrange, LagrangeLinear };

    SurfaceSpherical(Mesh *pm,
                     ParameterInput *pin,
                     Surfaces *psurfs,
                     const int surf_ix);

  private:

    Real rad;
    int N_th;
    int N_ph;
    int N_pts;

    // For storage of grids
    aliases::AA th;
    aliases::AA ph;
    aliases::AA x_o_th_ph;  // (x1(rad,th,ph), x2(rad,th,ph), x3(rad,th,ph))

    // Quadrature weights (solid-angle element) at each (th_i, ph_j); used
    // only for scalar spherical-harmonic projection (see DoProjections()).
    aliases::AA weights;

    // Real scalar spherical-harmonic table, built on (th, ph) when
    // psurfs->lmax_sh >= 0. See utils/spherical_harmonics.hpp.
    gra::sph_harm::RealHarmonicTable ylm;

    variety_sampling vs;
    variety_interpolator vi;

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

    // Midpoint quadrature weights: dOmega = sin(th) dth dph.
    // Assumes th (member array) has already been filled by gr_th().
    inline void wt_uniform(aliases::AA & w_in)
    {
      const Real dth = PI / static_cast<Real>(N_th);
      const Real dph = 2.0 * PI / static_cast<Real>(N_ph);
      for (int i=0; i<N_th; ++i)
      {
        const Real dcosth = std::sin(th(i)) * dth;
        for (int j=0; j<N_ph; ++j)
        {
          w_in(i, j) = dcosth * dph;
        }
      }
    }

    // Gauss-Legendre nodes in cos(theta), uniform in phi. Fills th and
    // weights together (weight depends on the same GL node used for th).
    // Requires N_th == N_ph / 2, matching grid_theta_phi::Grid.
    inline void gr_wt_gausslegendre(aliases::AA & th_in, aliases::AA & w_in)
    {
      if (N_th != N_ph / 2)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in SurfaceSpherical" << std::endl
            << "gausslegendre requires nth == nph/2, got nth=" << N_th
            << " nph=" << N_ph << std::endl;
        ATHENA_ERROR(msg);
      }

      const Real dph = 2.0 * PI / static_cast<Real>(N_ph);

      std::vector<Real> gl_nodes(N_th);
      std::vector<Real> gl_weights(N_th);
      gra::grids::theta_phi::GLQuadNodesWeights(
        -1.0, 1.0, gl_nodes.data(), gl_weights.data(), N_th);

      for (int i=0; i<N_th; ++i)
      {
        th_in(i) = std::acos(gl_nodes[i]);
        for (int j=0; j<N_ph; ++j)
        {
          w_in(i, j) = gl_weights[i] * dph;
        }
      }
    }

  // interpolator specific ----------------------------------------------------
  private:
    virtual void PrepareInterpolators() override;
    virtual void DoInterpolations() override;
    virtual void DoProjections() override;

    Real InterpolateAtPoint(aliases::AA & raw_cpt,
                            Surfaces::variety_base_grid vs,
                            const int tar_i, const int tar_j);

  // output specific ----------------------------------------------------------
  private:
    virtual void write_hdf5_coordinates(hid_t & id_file,
                                        const std::string & six) override;

};

// ============================================================================
} // namespace gra::mesh::surfaces
// ============================================================================

#endif  // MESH_SURFACES_HPP
