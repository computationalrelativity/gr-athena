//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_standard_refinement.cpp
//  \brief Default AMR refinement callback, driven by ExtremaTracker (and,
//         when compiled, apparent-horizon finders).
//
//  This function is installed as the default `AMRFlag_` in the Mesh
//  constructor(s) whenever `adaptive == true`. Problem generators that want
//  a bespoke criterion can override it by calling
//    EnrollUserRefinementCondition(MyRefinementCondition);
//  in their `InitUserMeshData` hook.
//
//  Behaviour:
//    - If no ExtremaTracker is configured (or it has N_tracker == 0), return
//      0 (no-op) so that default enrollment is safe.
//    - Otherwise iterate over trackers. Each tracker has a `ref_type`:
//        0 : refine a MeshBlock if the tracker's centre is contained.
//        1 : refine if the MeshBlock intersects a sphere of fixed radius
//            centred on the tracker.
//        2 : refine if the MeshBlock intersects a sphere with radius drawn
//            from the smallest apparent-horizon minimum radius across all
//            located horizons (only useful when AHFs are configured).
//      The maximum matched `ref_level` is compared against the MeshBlock's
//      physical level:
//        mb_physical_level <  req_level -> return +1 (refine)
//        mb_physical_level == req_level -> return  0 (no change)
//        mb_physical_level >  req_level -> return -1 (derefine)

// C++ headers
#include <algorithm>
#include <limits>

// Athena++ headers
#include "../trackers/extrema_tracker.hpp"
#include "mesh.hpp"
#if Z4C_ENABLED
#include "../z4c/ahf.hpp"
#endif

//----------------------------------------------------------------------------------------
//! \fn int Mesh::StandardRefinementCondition(MeshBlock* pmb)
//  \brief Default tracker-based AMR flag function.

int Mesh::StandardRefinementCondition(MeshBlock* pmb)
{
  Mesh* pmesh                      = pmb->pmy_mesh;
  ExtremaTracker* ptracker_extrema = pmesh->ptracker_extrema;

  // Safe no-op when no trackers have been configured.
  if (ptracker_extrema == nullptr || ptracker_extrema->N_tracker <= 0)
  {
    return 0;
  }

  const int root_level        = ptracker_extrema->root_level;
  const int mb_physical_level = pmb->loc.level - root_level;

  // Iterate over refinement levels offered by trackers.
  //
  // By default if a point is not in any sphere, completely de-refine.
  int req_level = 0;

  for (int n = 1; n <= ptracker_extrema->N_tracker; ++n)
  {
    bool is_contained       = false;
    const int cur_req_level = ptracker_extrema->ref_level(n - 1);

    if (ptracker_extrema->ref_type(n - 1) == 0)
    {
      is_contained = pmb->PointContained(ptracker_extrema->c_x1(n - 1),
                                         ptracker_extrema->c_x2(n - 1),
                                         ptracker_extrema->c_x3(n - 1));
    }
    else if (ptracker_extrema->ref_type(n - 1) == 1)
    {
      is_contained =
        pmb->SphereIntersects(ptracker_extrema->c_x1(n - 1),
                              ptracker_extrema->c_x2(n - 1),
                              ptracker_extrema->c_x3(n - 1),
                              ptracker_extrema->ref_zone_radius(n - 1));
    }
    else if (ptracker_extrema->ref_type(n - 1) == 2)
    {
#if Z4C_ENABLED
      // If any excision; activate this refinement.
      bool use = false;

      // Get the minimal radius over all apparent horizons.
      Real horizon_radius = std::numeric_limits<Real>::infinity();

      for (auto pah_f : pmesh->pah_finder)
      {
        if (not pah_f->IsFound())
          continue;

        if (pah_f->GetHorizonMinRadius() < horizon_radius)
        {
          horizon_radius = pah_f->GetHorizonMinRadius();
        }
        else
        {
          continue;
        }

        // Populate the tracker with AHF-based information.
        ptracker_extrema->ref_zone_radius(n - 1) =
          (pah_f->GetHorizonMinRadius());

        use = true;
      }

      if (use)
      {
        is_contained =
          pmb->SphereIntersects(ptracker_extrema->c_x1(n - 1),
                                ptracker_extrema->c_x2(n - 1),
                                ptracker_extrema->c_x3(n - 1),
                                ptracker_extrema->ref_zone_radius(n - 1));
      }
#endif  // Z4C_ENABLED
    }

    if (is_contained)
    {
      req_level = std::max(cur_req_level, req_level);
    }
  }

  if (req_level > mb_physical_level)
  {
    return 1;  // currently too coarse, refine
  }
  else if (req_level == mb_physical_level)
  {
    return 0;  // level satisfied, do nothing
  }

  // otherwise de-refine
  return -1;
}
