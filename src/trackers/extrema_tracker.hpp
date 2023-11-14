#ifndef EXTREMA_TRACKER_HPP
#define EXTREMA_TRACKER_HPP

// c/c++

#include <string>

// Athena++
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/lagrange_interp.hpp"

// Forward declaration
// class Mesh;
// class MeshBlock;
// class MeshBlockTree;
class ParameterInput;
class Coordinates;
class ParameterInput;

class ExtremaTracker
{
  public:
    ExtremaTracker(Mesh * pmesh, ParameterInput * pin, int res_flag);
    ~ExtremaTracker();

  public:
    void ReduceTracker();
    void EvolveTracker();
    void WriteTracker(int iter, Real time) const;

  private:
    void InitializeFromParFile(ParameterInput * pin);
    void PrepareTrackerFiles();

  public:
    int N_tracker;

    AthenaArray<int> ref_level;
    AthenaArray<int> ref_type;
    AthenaArray<Real> ref_zone_radius;

    AthenaArray<bool> minima;  // T for minima, F for maxima
    AthenaArray<Real> c_x1;
    AthenaArray<Real> c_x2;
    AthenaArray<Real> c_x3;

    int root_level;

    Real update_max_step_factor;
    int update_strategy;

    enum class control_fields {
      // wave eqn.
      wave_auxiliary_ref,
      // Z4c.
      Z4c_alpha,
      Z4c_chi
    };

    control_fields control_field;

  private:
    Mesh const * pmesh;
    std::string output_filename;
    std::string control_field_name;
    bool is_io_process;

    AthenaArray<int> multiplicity_update;

#ifdef MPI_PARALLEL
    const int rank_root = 0;
    int rank;
#endif

  public:
    AthenaArray<Real> c_dx1;
    AthenaArray<Real> c_dx2;
    AthenaArray<Real> c_dx3;

};

class ExtremaTrackerLocal
{
  public:
    ExtremaTrackerLocal(MeshBlock * pmb, ParameterInput * pin);
     ~ExtremaTrackerLocal();

  public:
    void TreatCentreIfLocalMember();
    int LocateCentrePhysicalIndex(const int n, const int axis);
    bool IsOrdPhysical(const int axis, const Real x);

  private:
    void ZeroInternalTracker();

    int IxMinimaOffset(const Real a, const Real b, const Real c);
    int IxExtremaOffset(const int n, const Real a, const Real b, const Real c);

    void UpdateLocStepByControlFieldTimeStep(const int n);

    Real ExtremaStepQuadInterp(const Real ds,
                               const Real f_0,
                               const Real f_1,
                               const Real f_2);

    void UpdateLocStepByControlFieldQuadInterp(const int n);

    inline int sign(Real val)
    {
      return (val > 0) ? 1 : ((val < 0) ? -1 : 0);
    };

  public:
    int N_tracker;

    AthenaArray<int> to_update;

    AthenaArray<Real> sign_minima;  // T/F->premul by 1/-1 to get min/max search

    AthenaArray<Real> loc_c_dx1;
    AthenaArray<Real> loc_c_dx2;
    AthenaArray<Real> loc_c_dx3;

    // Field for which we interrogate extrema
    AthenaArray<Real> * control_field;
    AthenaArray<Real> control_field_slicer;  // Work-around

  private:
    MeshBlock * pmy_block;
    ExtremaTracker * ptracker_extrema;

    int ndim;

    // Switching of sampling
    MB_info * mbi;

    typedef LagrangeInterpND<2*(NGHOST-1)-1, 3> Interp_Lag3;
    typedef LagrangeInterpND<2*(NGHOST-1)-1, 2> Interp_Lag2;
    typedef LagrangeInterpND<2*(NGHOST-1)-1, 1> Interp_Lag1;

};

#endif // EXTREMA_TRACKER_HPP