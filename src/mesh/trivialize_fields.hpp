#ifndef MESH_TRIVIALIZE_HPP_
#define MESH_TRIVIALIZE_HPP_

// Athena++ headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"

// Forward declarations
class Mesh;

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

// ============================================================================
namespace gra::trivialize {
// ============================================================================


class TrivializeFields
{
  public:
    TrivializeFields(Mesh *pm, ParameterInput *pin);
    ~TrivializeFields() { };

  // storage ------------------------------------------------------------------
  private:
    Mesh * pm;
    ParameterInput * pin;

  public:
    // for storage of options
    struct {
      bool active;
      bool apply_on_substeps;
      bool verbose;
      bool retain_nn_for_cut;

      struct {
        bool active;
        bool set_vacuum;
        bool correct_layer;
        Real cut_D;
        Real flux_fac;
        int num_neighbors;
        int num_neighbors_layer_extend;
      } hydro;

    } opt;

  // methods ------------------------------------------------------------------
  public:
    void Update();
    void Update_();
    void Reweight();

    void AllocateLocal(MeshBlock * pmb);

    bool IsTrivialHydro(
      MeshBlock *pmb
    );

    bool IsTrivialMatter(
      MeshBlock *pmb
    );

    // true: {trivial block, non-trivial point}
    // false: {non-trivial block, a trivial point}
    bool MaskHydro(MeshBlock *pmb, const int k, const int j, const int i);

    // Get masks
    AA_B & GetMaskHydroPT(MeshBlock *pmb);
    AA_B & GetMaskHydroNN(MeshBlock *pmb);
    AA_B & GetMaskHydroLY(MeshBlock *pmb);

    // internal methods ---------------------------------------------------------
  public:
    void PrepareMask(MeshBlock *pmb);
    void PrepareMasks();

    bool CutMaskHydro(MeshBlock *pmb, const int k, const int j, const int i);
    void CutMask(MeshBlock *pmb);
    void CutMasks();

};

// ============================================================================
} // namespace gra::trivialize
// ============================================================================

#endif // MESH_TRIVIALIZE_HPP_

//
// :D
//