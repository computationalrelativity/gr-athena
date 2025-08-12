#ifndef MESH_RESCALING_HPP_
#define MESH_RESCALING_HPP_

// Athena++ headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"

// Forward declarations
class Mesh;

// ============================================================================
namespace gra::hydro::rescaling {
// ============================================================================

enum class variety_cs { conserved_hydro, conserved_scalar };

class Rescaling
{
  public:
    Rescaling(Mesh *pm, ParameterInput *pin);
    ~Rescaling() {
      if (opt.dump_status)
      {
        OutputFinalize();
      }
    };

  // storage ------------------------------------------------------------------
  private:
    Mesh * pm;
    ParameterInput * pin;

  public:
    // for storage of initial values
    struct {
      bool initialized; // have initial values been computed?

      Real m;               // conserved mass (integrated D_tilde)
      AthenaArray<Real> S;  // conserved (integrated) passive scalars
    } ini;

    // for storage of current values
    struct {
      Real min_D;       // minimum of _undensitized_ conserved density
      AthenaArray<Real> min_s;

      Real m;
      AthenaArray<Real> S;

      Real err_rel_D;
      AthenaArray<Real> err_rel_S;

      Real fac_mul_D;
      AthenaArray<Real> fac_mul_s;

      Real cut_D;
      Real rsc_D;

      AthenaArray<Real> cut_s;
      AthenaArray<Real> rsc_s;

    } cur;

    // for storage of options
    struct {
      bool verbose;
      bool use_cutoff;
      bool rescale_conserved_density;
      bool rescale_conserved_scalars;

      bool apply_on_substeps;
      bool disable_on_first_failure;

      bool dump_status;

      Real start_time;
      Real end_time;

      Real fac_mul_D;
      Real fac_mul_s;

      Real err_rel_hydro;
      Real err_rel_scalars;
    } opt;

  // methods ------------------------------------------------------------------
  public:
    void Initialize();
    void Apply();
    void FinalizePreOutput();

    // Compute volume-weighted sum of a variable on a given rank
    // Optionally use a lower (undensitized) cut.
    Real CompensatedSummation(const variety_cs v_cs,
                              const int n,
                              const Real s_cut);

    // Globally integrate phydro->u(n,:,:,:) or pscalars->s(n,:,:,:);
    // (undensitized) values of V > V_cut are considered
    Real IntegrateField(const variety_cs v_cs,
                        const int n,
                        const Real V_cut);

    // Get global minimum of a quantity
    Real GlobalMinimum(const variety_cs v_cs,
                       const int n,
                       const bool require_positive);

  // I/O methods --------------------------------------------------------------
  private:
    std::string filename;
    FILE * pofile;

    void OutputPrepare();
    void OutputFinalize();

  public:
    void OutputWrite(const int iter, const Real time, const int nstage);

};

// ============================================================================
} // namespace gra::hydro::rescaling
// ============================================================================


#endif // MESH_RESCALING_HPP_

//
// :D
//