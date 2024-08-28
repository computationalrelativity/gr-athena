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
#include "bvals_fc.hpp"

//-----------------------------------------------------------------------------

void FaceCenteredBoundaryVariable::CalculateProlongationIndices(
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

void FaceCenteredBoundaryVariable::CalculateProlongationSharedIndices(
  NeighborBlock &nb,
  const int si, const int ei,
  const int sj, const int ej,
  const int sk, const int ek,
  int &il, int &iu, int &jl, int &ju, int &kl, int &ku)
{
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;

  const int &mylevel = pmb->loc.level;

  il = si, iu = ei + 1;
  if ((nb.ni.ox1 >= 0) &&
      (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 1][nb.ni.ox1] >= mylevel))
    il++;

  if ((nb.ni.ox1 <= 0) &&
      (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 1][nb.ni.ox1 + 2] >= mylevel))
    iu--;

  if (pmb->block_size.nx2 > 1) {
    jl = sj, ju = ej + 1;
    if ((nb.ni.ox2 >= 0) &&
        (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2][nb.ni.ox1 + 1] >=
         mylevel))
      jl++;
    if ((nb.ni.ox2 <= 0) &&
        (pbval->nblevel[nb.ni.ox3 + 1][nb.ni.ox2 + 2][nb.ni.ox1 + 1] >=
         mylevel))
      ju--;
  } else {
    jl = sj;
    ju = ej;
  }

  if (pmb->block_size.nx3 > 1) {
    kl = sk, ku = ek + 1;
    if ((nb.ni.ox3 >= 0) &&
        (pbval->nblevel[nb.ni.ox3][nb.ni.ox2 + 1][nb.ni.ox1 + 1] >=
         mylevel))
      kl++;
    if ((nb.ni.ox3 <= 0) &&
        (pbval->nblevel[nb.ni.ox3 + 2][nb.ni.ox2 + 1][nb.ni.ox1 + 1] >=
         mylevel))
      ku--;
  } else {
    kl = sk;
    ku = ek;
  }

}

void FaceCenteredBoundaryVariable::CalculateProlongationIndicesFine(
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