// C headers

// C++ headers
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "bvals_cc.hpp"

//-----------------------------------------------------------------------------

void CellCenteredBoundaryVariable::CalculateProlongationIndices(
  NeighborBlock &nb,
  int &si, int &ei,
  int &sj, int &ej,
  int &sk, int &ek)
{
  MeshBlock * pmb = pmy_block_;

  static const int pcng = pmb->cnghost - 1;

  CalculateProlongationIndices(pmb->loc.lx1, nb.ni.ox1, pcng,
                               pmb->cis, pmb->cie,
                               si, ei,
                               true);
  CalculateProlongationIndices(pmb->loc.lx2, nb.ni.ox2, pcng,
                               pmb->cjs, pmb->cje,
                               sj, ej,
                               pmb->block_size.nx2 > 1);
  CalculateProlongationIndices(pmb->loc.lx3, nb.ni.ox3, pcng,
                               pmb->cks, pmb->cke,
                               sk, ek,
                               pmb->block_size.nx3 > 1);

}

void CellCenteredBoundaryVariable::CalculateProlongationIndicesFine(
  NeighborBlock &nb,
  int &fsi, int &fei,
  int &fsj, int &fej,
  int &fsk, int &fek)
{
  MeshBlock * pmb = pmy_block_;

  int si, ei, sj, ej, sk, ek;
  CalculateProlongationIndices(nb, si, ei, sj, ej, sk, ek);

  // ghost-ghost zones are filled and prolongated,
  // calculate the loop limits for the finer grid
  fsi = (si - pmb->cis)*2 + pmb->is;
  fei = (ei - pmb->cis)*2 + pmb->is + 1;
  if (pmb->block_size.nx2 > 1) {
    fsj = (sj - pmb->cjs)*2 + pmb->js;
    fej = (ej - pmb->cjs)*2 + pmb->js + 1;
  } else {
    fsj = pmb->js;
    fej = pmb->je;
  }
  if (pmb->block_size.nx3 > 1) {
    fsk = (sk - pmb->cks)*2 + pmb->ks;
    fek = (ek - pmb->cks)*2 + pmb->ks + 1;
  } else {
    fsk = pmb->ks;
    fek = pmb->ke;
  }
}

//
// :D
//