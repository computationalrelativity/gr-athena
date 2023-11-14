
// c/c++
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Athena++
#include "extrema_tracker.hpp"

#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// for registration of control field..
#include "../wave/wave.hpp"
#include "../z4c/z4c.hpp"

ExtremaTracker::ExtremaTracker(Mesh * pmesh, ParameterInput * pin,
                               int res_flag):
  pmesh(pmesh)
{
  N_tracker = pin->GetOrAddInteger("trackers_extrema", "N_tracker", 0);

  if (N_tracker > 0)
  {

#ifdef MPI_PARALLEL
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    is_io_process = (rank == rank_root);
#else
    is_io_process = true;
#endif
    output_filename = pin->GetOrAddString("trackers_extrema",
                                          "filename",
                                          "trackers_extrema");

    control_field_name = pin->GetOrAddString("trackers_extrema",
                                             "control_field",
                                             "none");

    if (control_field_name == "wave.auxiliary_ref_field")
    {
      control_field = control_fields::wave_auxiliary_ref;
    }
    else if (control_field_name == "Z4c.alpha")
    {
      control_field = control_fields::Z4c_alpha;
    }
    else if (control_field_name == "Z4c.chi")
    {
      control_field = control_fields::Z4c_chi;
    }
    else
    {
      std::cout << "tracker_extrema unknown control_field" << std::endl;
      assert(false); // you shouldn't be here
      abort();
    }

    ref_level.NewAthenaArray(N_tracker);
    ref_type.NewAthenaArray(N_tracker);
    ref_zone_radius.NewAthenaArray(N_tracker);

    minima.NewAthenaArray(N_tracker);
    c_x1.NewAthenaArray(N_tracker);
    c_x2.NewAthenaArray(N_tracker);
    c_x3.NewAthenaArray(N_tracker);

    c_x1.Fill(0.);
    c_x2.Fill(0.);
    c_x3.Fill(0.);

    multiplicity_update.NewAthenaArray(N_tracker);
    c_dx1.NewAthenaArray(N_tracker);
    c_dx2.NewAthenaArray(N_tracker);
    c_dx3.NewAthenaArray(N_tracker);

    c_dx1.Fill(0.);
    c_dx2.Fill(0.);
    c_dx3.Fill(0.);

    root_level = pmesh->root_level;

    InitializeFromParFile(pin);

    if (!res_flag)
    {
      // initialize from scratch (not restart file)
      PrepareTrackerFiles();
    }

  }
}

ExtremaTracker::~ExtremaTracker()
{
  if (N_tracker > 0)
  {
    ref_level.DeleteAthenaArray();
    ref_type.DeleteAthenaArray();
    ref_zone_radius.DeleteAthenaArray();

    minima.DeleteAthenaArray();
    c_x1.DeleteAthenaArray();
    c_x2.DeleteAthenaArray();
    c_x3.DeleteAthenaArray();

    multiplicity_update.DeleteAthenaArray();
    c_dx1.DeleteAthenaArray();
    c_dx2.DeleteAthenaArray();
    c_dx3.DeleteAthenaArray();

  }
}

void ExtremaTracker::InitializeFromParFile(ParameterInput * pin)
{
  update_max_step_factor = pin->GetOrAddReal("trackers_extrema",
                                             "update_max_step_factor", 1.);

  update_strategy = pin->GetOrAddInteger("trackers_extrema",
                                         "update_strategy", 0);

  for(int n=1; n<=N_tracker; ++n)
  {

    std::string n_str = std::to_string(n);

    ref_level(n-1) = pin->GetOrAddInteger(
      "trackers_extrema", "ref_level_" + n_str, 1
    );

    ref_type(n-1) = pin->GetOrAddInteger(
      "trackers_extrema", "ref_type_" + n_str, 0
    );

    if (ref_type(n-1) == 1)
    {
      ref_zone_radius(n-1) = pin->GetOrAddReal(
        "trackers_extrema", "ref_zone_radius_" + n_str, 1
      );
    }

    // minima / maxima? by default min.
    minima(n-1) = pin->GetOrAddBoolean(
      "trackers_extrema", "minima_" + n_str, true);

    switch (pmesh->ndim)
    {
      case 3:
        c_x3(n-1) = pin->GetOrAddReal(
          "trackers_extrema", "ini_" + n_str + "_x3", 0
        );
      case 2:
        c_x2(n-1) = pin->GetOrAddReal(
          "trackers_extrema", "ini_" + n_str + "_x2", 0
        );
      case 1:
        c_x1(n-1) = pin->GetOrAddReal(
          "trackers_extrema", "ini_" + n_str + "_x1", 0
        );
        break;
      default:
        std::cout << "tracker_extrema requires 1<=pmesh->ndim<=3" << std::endl;
    }
  }
}

void ExtremaTracker::PrepareTrackerFiles()
{
  if (is_io_process)
  {
    FILE * pofile[N_tracker];

    for (int n=1; n<=N_tracker; ++n)
    {
      pofile[n-1] = NULL;
      std::string title = output_filename + std::to_string(n) + ".txt";
      pofile[n-1] = fopen(title.c_str(), "w");

      if (NULL == pofile[n-1]) {
        std::stringstream msg;
        msg << "### FATAL ERROR in Tracker constructor" << std::endl;
        msg << "Could not open file '" << title << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
      fprintf(pofile[n-1],
        "#%-12s%-13s%-13s%-13s%-13s\n",
        "1:iter", "2:time", "3:T-x", "4:T-y", "5:T-z");
      fprintf(pofile[n-1], "%-13d%-13.5e", 0, 0.);
      fprintf(pofile[n-1], "%-13.5e%-13.5e%-13.5e\n",
        c_x1(n-1),
        c_x2(n-1),
        c_x3(n-1));
      fclose(pofile[n-1]);
    }

  }
}

void ExtremaTracker::ReduceTracker()
{
  if (!(N_tracker > 0))  // ensure we have something to do
    return;

  MeshBlock const * pmb = pmesh->pblock;

  // we need to collect all contributions to dx
  c_dx1.Fill(0);
  c_dx2.Fill(0);
  c_dx3.Fill(0);
  multiplicity_update.Fill(0);

  // local to process
  while (pmb != NULL)
  {
    for (int n=1; n<=N_tracker; ++n)
    {
      if (pmb->ptracker_extrema_loc->to_update(n-1))
      {
        c_dx1(n-1) += pmb->ptracker_extrema_loc->loc_c_dx1(n-1);
        c_dx2(n-1) += pmb->ptracker_extrema_loc->loc_c_dx2(n-1);
        c_dx3(n-1) += pmb->ptracker_extrema_loc->loc_c_dx3(n-1);

        multiplicity_update(n-1)++;

      }
    }
    pmb = pmb->next;
  }


#ifdef MPI_PARALLEL
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == rank_root)
  {
    MPI_Reduce(MPI_IN_PLACE,
               multiplicity_update.data(), N_tracker, MPI_INT,
               MPI_SUM, rank_root, MPI_COMM_WORLD);

    MPI_Reduce(MPI_IN_PLACE,
               c_dx1.data(), N_tracker, MPI_ATHENA_REAL,
               MPI_SUM, rank_root, MPI_COMM_WORLD);

    MPI_Reduce(MPI_IN_PLACE,
               c_dx2.data(), N_tracker, MPI_ATHENA_REAL,
               MPI_SUM, rank_root, MPI_COMM_WORLD);

    MPI_Reduce(MPI_IN_PLACE,
               c_dx3.data(), N_tracker, MPI_ATHENA_REAL,
               MPI_SUM, rank_root, MPI_COMM_WORLD);

  }
  else
  {
    MPI_Reduce(multiplicity_update.data(),
               multiplicity_update.data(),
               N_tracker,
               MPI_INT, MPI_SUM, rank_root, MPI_COMM_WORLD);

    MPI_Reduce(c_dx1.data(),
               c_dx1.data(),
               N_tracker,
               MPI_ATHENA_REAL, MPI_SUM, rank_root, MPI_COMM_WORLD);

    MPI_Reduce(c_dx2.data(),
               c_dx2.data(),
               N_tracker,
               MPI_ATHENA_REAL, MPI_SUM, rank_root, MPI_COMM_WORLD);

    MPI_Reduce(c_dx3.data(),
               c_dx3.data(),
               N_tracker,
               MPI_ATHENA_REAL, MPI_SUM, rank_root, MPI_COMM_WORLD);
  }

#endif // MPI_PARALLEL

  // deal with multiplicity (is_io_process equiv to rank_root)
  if (is_io_process)
  {
    for (int n=1; n<=N_tracker; ++n)
    {
      if (multiplicity_update(n-1) > 0)
      {
        const Real recip = 1. / multiplicity_update(n-1);
        c_dx1(n-1) *= recip;
        c_dx2(n-1) *= recip;
        c_dx3(n-1) *= recip;
      }
    }
  }

}

void ExtremaTracker::EvolveTracker()
{
  if (!(N_tracker > 0))  // ensure we have something to do
    return;


  if (is_io_process)
  {
    for (int n=1; n<=N_tracker; ++n)
    {
      c_x1(n-1) += c_dx1(n-1);
      c_x2(n-1) += c_dx2(n-1);
      c_x3(n-1) += c_dx3(n-1);
    }
  }

#ifdef MPI_PARALLEL
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(c_x1.data(), N_tracker, MPI_ATHENA_REAL,
            rank_root, MPI_COMM_WORLD);
  MPI_Bcast(c_x2.data(), N_tracker, MPI_ATHENA_REAL,
            rank_root, MPI_COMM_WORLD);
  MPI_Bcast(c_x3.data(), N_tracker, MPI_ATHENA_REAL,
            rank_root, MPI_COMM_WORLD);

  MPI_Bcast(c_dx1.data(), N_tracker, MPI_ATHENA_REAL,
            rank_root, MPI_COMM_WORLD);
  MPI_Bcast(c_dx2.data(), N_tracker, MPI_ATHENA_REAL,
            rank_root, MPI_COMM_WORLD);
  MPI_Bcast(c_dx3.data(), N_tracker, MPI_ATHENA_REAL,
            rank_root, MPI_COMM_WORLD);
#endif // MPI_PARALLEL

}

void ExtremaTracker::WriteTracker(int iter, Real time) const
{
  if (!(N_tracker > 0))  // ensure we have something to do
    return;

  if (is_io_process && (iter > 0))
  {
    FILE *pfile[N_tracker];

    for (int n=1; n<=N_tracker; ++n)
    {
      std::string title = output_filename + std::to_string(n) + ".txt";
      pfile[n-1] = fopen(title.c_str(), "a");
      fprintf(pfile[n-1], "%-13d%-13.5e", iter, time);
      fprintf(pfile[n-1], "%-13.5e%-13.5e%-13.5e\n",
        c_x1(n-1),
        c_x2(n-1),
        c_x3(n-1));
      fclose(pfile[n-1]);
    }
  }
}

ExtremaTrackerLocal::ExtremaTrackerLocal(
  MeshBlock * pmb,
  ParameterInput * pin)
  :
  pmy_block(pmb),
  ptracker_extrema(
    pmb->pmy_mesh->ptracker_extrema
  ),
  ndim(pmy_block->pmy_mesh->ndim)
{
  N_tracker = pin->GetOrAddInteger("trackers_extrema", "N_tracker", 0);

  if (N_tracker > 0)
  {
    // BD: TODO control field sampling should be set based on variable type
    mbi = &pmb->pz4c->mbi;

    // determine control field sampling ---------------------------------------
    /*
    std::string par_control_field_sampling = pin->GetOrAddString(
      "trackers_extrema",
      "control_field_sampling",
      "VC");

    if (par_control_field_sampling == "VC")
    {
      mbi->il = pmb->ivs;            // il
      mbi->iu = pmb->ive;            // iu
      mbi->jl = pmb->jvs;            // jl
      mbi->ju = pmb->jve;            // ju
      mbi->kl = pmb->kvs;            // kl
      mbi->ku = pmb->kve;            // ku

      mbi->nn1 = pmb->nverts1;
      mbi->nn2 = pmb->nverts2;
      mbi->nn3 = pmb->nverts3;

      mbi->x1.InitWithShallowSlice(pmb->pcoord->x1f, 1, 0, mbi->nn1);
      mbi->x2.InitWithShallowSlice(pmb->pcoord->x2f, 1, 0, mbi->nn2);
      mbi->x3.InitWithShallowSlice(pmb->pcoord->x3f, 1, 0, mbi->nn3);

      mbi->dx1.InitWithShallowSlice(pmb->pcoord->dx1f, 1, 0, mbi->nn1);
      mbi->dx2.InitWithShallowSlice(pmb->pcoord->dx2f, 1, 0, mbi->nn2);
      mbi->dx3.InitWithShallowSlice(pmb->pcoord->dx3f, 1, 0, mbi->nn3);
    }
    else  // CC & CX have same logic
    {
      mbi->il = pmb->is;             // il
      mbi->iu = pmb->ie;             // iu
      mbi->jl = pmb->js;             // jl
      mbi->ju = pmb->je;             // ju
      mbi->kl = pmb->ks;             // kl
      mbi->ku = pmb->ke;             // ku

      mbi->nn1 = pmb->ncells1;
      mbi->nn2 = pmb->ncells2;
      mbi->nn3 = pmb->ncells3;

      mbi->x1.InitWithShallowSlice(pmb->pcoord->x1v, 1, 0, mbi->nn1);
      mbi->x2.InitWithShallowSlice(pmb->pcoord->x2v, 1, 0, mbi->nn2);
      mbi->x3.InitWithShallowSlice(pmb->pcoord->x3v, 1, 0, mbi->nn3);

      mbi->dx1.InitWithShallowSlice(pmb->pcoord->dx1v, 1, 0, mbi->nn1);
      mbi->dx2.InitWithShallowSlice(pmb->pcoord->dx2v, 1, 0, mbi->nn2);
      mbi->dx3.InitWithShallowSlice(pmb->pcoord->dx3v, 1, 0, mbi->nn3);
    }
    // ------------------------------------------------------------------------
    */

    to_update.NewAthenaArray(N_tracker);

    sign_minima.NewAthenaArray(N_tracker);

    for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
      sign_minima(n-1) = (ptracker_extrema->minima(n-1)) ? 1 : -1;

    loc_c_dx1.NewAthenaArray(N_tracker);
    loc_c_dx2.NewAthenaArray(N_tracker);
    loc_c_dx3.NewAthenaArray(N_tracker);

    // register control field
    switch (ptracker_extrema->control_field)
    {
      case ExtremaTracker::control_fields::wave_auxiliary_ref:
        // control_field = &(pmb->pwave->aux_ref.auxiliary_ref_field);
        break;
      case ExtremaTracker::control_fields::Z4c_alpha:
        // need to slice and point

        control_field_slicer.InitWithShallowSlice(
          pmb->pz4c->storage.u, Z4c::I_Z4c_alpha, 1
        );

        control_field = &(control_field_slicer);

        break;
      case ExtremaTracker::control_fields::Z4c_chi:
        // need to slice and point

        control_field_slicer.InitWithShallowSlice(
          pmb->pz4c->storage.u, Z4c::I_Z4c_chi, 1
        );

        control_field = &(control_field_slicer);

        break;
    }

  }

}

void ExtremaTrackerLocal::TreatCentreIfLocalMember()
{
  ZeroInternalTracker();

  // check if current MeshBlock contains a centre
  for (int n=1; n<=ptracker_extrema->N_tracker; ++n)
  {
    const bool is_contained = pmy_block->PointContained(
      ptracker_extrema->c_x1(n-1),
      ptracker_extrema->c_x2(n-1),
      ptracker_extrema->c_x3(n-1)
    );

    if (is_contained)
    {
      // record that current MeshBlock updates current centre
      to_update(n-1) = 1;

      // inspect the control field and update as required
      switch (ptracker_extrema->update_strategy)
      {
        case 0:
          UpdateLocStepByControlFieldTimeStep(n);
          break;
        case 1:
          UpdateLocStepByControlFieldQuadInterp(n);
          break;
        default:
          std::cout << "ExtremaTracker update_strategy unknown" << std::endl;

      }
    }

  }

}

int ExtremaTrackerLocal::IxMinimaOffset(const Real a,
                                        const Real b,
                                        const Real c)
{
  // returns -1, 0, 1 when a, b, c are minimal resp.

  Real cmp[] = {a, b, c};

  return (std::min_element(cmp, cmp+3) - cmp) - 1;
}

int ExtremaTrackerLocal::IxExtremaOffset(const int n,
                                         const Real a,
                                         const Real b,
                                         const Real c)
{
  // returns -1, 0, 1 when a, b, c are minimal (def) resp.

  Real cmp[] = {
    sign_minima(n-1) * a,
    sign_minima(n-1) * b,
    sign_minima(n-1) * c
  };

  return (std::min_element(cmp, cmp+3) - cmp) - 1;
}

void ExtremaTrackerLocal::UpdateLocStepByControlFieldTimeStep(const int n)
{
  // Take a step (of size dt) in direction of auxiliary field descent

  switch (ndim)
  {
    case 3:
    {
      int ix_c_x1, ix_c_x2, ix_c_x3;
      int offset_ix_x1, offset_ix_x2, offset_ix_x3;

      ix_c_x1 = LocateCentrePhysicalIndex(n, 0);
      ix_c_x2 = LocateCentrePhysicalIndex(n, 1);
      ix_c_x3 = LocateCentrePhysicalIndex(n, 2);

      offset_ix_x1 = IxExtremaOffset(
        n,
        (*control_field)(ix_c_x3, ix_c_x2, ix_c_x1-1),
        (*control_field)(ix_c_x3, ix_c_x2, ix_c_x1),
        (*control_field)(ix_c_x3, ix_c_x2, ix_c_x1+1)
      );

      offset_ix_x2 = IxExtremaOffset(
        n,
        (*control_field)(ix_c_x3, ix_c_x2-1, ix_c_x1),
        (*control_field)(ix_c_x3, ix_c_x2,   ix_c_x1),
        (*control_field)(ix_c_x3, ix_c_x2+1, ix_c_x1)
      );

      offset_ix_x3 = IxExtremaOffset(
        n,
        (*control_field)(ix_c_x3-1, ix_c_x2, ix_c_x1),
        (*control_field)(ix_c_x3,   ix_c_x2, ix_c_x1),
        (*control_field)(ix_c_x3+1, ix_c_x2, ix_c_x1)
      );

      loc_c_dx1(n-1) = offset_ix_x1 * (pmy_block->pmy_mesh->dt);
      loc_c_dx2(n-1) = offset_ix_x2 * (pmy_block->pmy_mesh->dt);
      loc_c_dx3(n-1) = offset_ix_x3 * (pmy_block->pmy_mesh->dt);
      break;
    }
    case 2:
    {
      int ix_c_x1, ix_c_x2;
      int offset_ix_x1, offset_ix_x2;

      ix_c_x1 = LocateCentrePhysicalIndex(n, 0);
      ix_c_x2 = LocateCentrePhysicalIndex(n, 1);

      offset_ix_x1 = IxExtremaOffset(
        n,
        (*control_field)(ix_c_x2, ix_c_x1-1),
        (*control_field)(ix_c_x2, ix_c_x1),
        (*control_field)(ix_c_x2, ix_c_x1+1)
      );

      offset_ix_x2 = IxExtremaOffset(
        n,
        (*control_field)(ix_c_x2-1, ix_c_x1),
        (*control_field)(ix_c_x2,   ix_c_x1),
        (*control_field)(ix_c_x2+1, ix_c_x1)
      );

      loc_c_dx1(n-1) = offset_ix_x1 * (pmy_block->pmy_mesh->dt);
      loc_c_dx2(n-1) = offset_ix_x2 * (pmy_block->pmy_mesh->dt);
      break;
    }
    case 1:
    {
      int ix_c_x1;
      int offset_ix_x1;

      ix_c_x1 = LocateCentrePhysicalIndex(n, 0);

      offset_ix_x1 = IxExtremaOffset(
        n,
        (*control_field)(ix_c_x1-1),
        (*control_field)(ix_c_x1),
        (*control_field)(ix_c_x1+1)
      );

      loc_c_dx1(n-1) = offset_ix_x1 * (pmy_block->pmy_mesh->dt);
      break;
    }
    default:
      std::cout << "ExtremaTrackerLocal requires ndim<=3" << std::endl;
  }

}

Real ExtremaTrackerLocal::ExtremaStepQuadInterp(const Real ds,
                                                const Real f_0,
                                                const Real f_1,
                                                const Real f_2)
{
  // Given function values:
  // {(x_0, f_0), (x_1, f_1), (x_2, f_2)}
  // With x_0 = x_1 - ds = x_2 - 2 * ds
  // Fit quadratic poly. and extract dx* such that x_1 + dx* is an extremum

  Real i_fac = (f_0 - f_2) / (2. * (f_0 - 2 * f_1 + f_2));

  // implement safety factor as specified in parameter file
  const Real umsf = ptracker_extrema->update_max_step_factor;
  i_fac = (std::abs(i_fac) > umsf) ? sign(i_fac) * umsf : i_fac;

  const Real dx_st = (std::isfinite(dx_st)) ? ds * i_fac : 0.0;
  return dx_st;

  /*
  // safety-check
  if (!std::isfinite(dx_st))
  {
    return 0.0;
  }

  // interpolated minima value
  Real imin_val = (
    (ds - dx_st) * (-dx_st * f_0 + 2 * (ds + dx_st) * f_1) +
    dx_st * (ds + dx_st) * f_2
  ) / (2 * SQR(ds));

  // min should be in [x_0, x_2] and bounded above by f_1
  // if (imin_val > f_1)
  if (f_0 - 2 * f_1 + f_2 <=0)
  {
    std::cout << imin_val << std::endl;
    std::cout << f_0 << std::endl;
    std::cout << f_1 << std::endl;
    std::cout << f_2 << std::endl;
  }

  return dx_st;
  */
}

void ExtremaTrackerLocal::UpdateLocStepByControlFieldQuadInterp(const int n)
{
  // Take a step (of size dx_i) in direction of auxiliary field descent
  // Uses quadratic poly. fit
  // Uniform grid spacing assumed

  Real origin[ndim];
  Real ds[ndim];
  int sz[ndim];
  Real coord[ndim];

  // for interpolation
  Real f_0, f_1, f_2;
  Real dx_st[ndim];

  // populate salient data in this block
  switch (ndim)
  {
    case 3:
    {
      origin[2] = mbi->x3(0);
      sz[2] = mbi->nn3;
      ds[2] = mbi->dx3(0);
      coord[2] = ptracker_extrema->c_x3(n-1);
    }
    case 2:
    {
      origin[1] = mbi->x2(0);
      sz[1] = mbi->nn2;
      ds[1] = mbi->dx2(0);
      coord[1] = ptracker_extrema->c_x2(n-1);
    }
    case 1:
    {
      origin[0] = mbi->x1(0);
      sz[0] = mbi->nn1;
      ds[0] = mbi->dx1(0);
      coord[0] = ptracker_extrema->c_x1(n-1);
      break;
    }
    default:
    {
      std::cout << "ExtremaTrackerLocal requires ndim<=3" << std::endl;
    }
  }

  // performed required interpolation
  switch (ndim)
  {
    case 3:
    {
      Interp_Lag3 * pinterp3 = nullptr;

      // axis 0 -----------------------------------------------------
      coord[0] -= ds[0];
      pinterp3 = new Interp_Lag3(origin, ds, sz, coord);

      f_0 = pinterp3->eval(&((*control_field)(0, 0, 0)));
      f_1 = pinterp3->eval(&((*control_field)(0, 0, 1)));
      f_2 = pinterp3->eval(&((*control_field)(0, 0, 2)));

      dx_st[0] = ExtremaStepQuadInterp(ds[0], f_0, f_1, f_2);
      coord[0] += ds[0];

      delete pinterp3;

      // axis 1 -----------------------------------------------------
      coord[1] -= ds[1];
      pinterp3 = new Interp_Lag3(origin, ds, sz, coord);

      f_0 = pinterp3->eval(&((*control_field)(0, 0, 0)));
      f_1 = pinterp3->eval(&((*control_field)(0, 1, 0)));
      f_2 = pinterp3->eval(&((*control_field)(0, 2, 0)));

      dx_st[1] = ExtremaStepQuadInterp(ds[1], f_0, f_1, f_2);
      coord[1] += ds[1];

      delete pinterp3;

      // axis 2 -----------------------------------------------------
      coord[2] -= ds[2];
      pinterp3 = new Interp_Lag3(origin, ds, sz, coord);

      f_0 = pinterp3->eval(&((*control_field)(0, 0, 0)));
      f_1 = pinterp3->eval(&((*control_field)(1, 0, 0)));
      f_2 = pinterp3->eval(&((*control_field)(2, 0, 0)));

      dx_st[2] = ExtremaStepQuadInterp(ds[2], f_0, f_1, f_2);
      coord[2] += ds[2];

      delete pinterp3;

      // update local -----------------------------------------------
      loc_c_dx1(n-1) = dx_st[0];
      loc_c_dx2(n-1) = dx_st[1];
      loc_c_dx3(n-1) = dx_st[2];

      break;
    }
    case 2:
    {
      Interp_Lag2 * pinterp2 = nullptr;

      // axis 0 -----------------------------------------------------
      coord[0] -= ds[0];
      pinterp2 = new Interp_Lag2(origin, ds, sz, coord);

      f_0 = pinterp2->eval(&((*control_field)(0, 0)));
      f_1 = pinterp2->eval(&((*control_field)(0, 1)));
      f_2 = pinterp2->eval(&((*control_field)(0, 2)));

      dx_st[0] = ExtremaStepQuadInterp(ds[0], f_0, f_1, f_2);
      coord[0] += ds[0];

      delete pinterp2;

      // axis 1 -----------------------------------------------------
      coord[1] -= ds[1];
      pinterp2 = new Interp_Lag2(origin, ds, sz, coord);

      f_0 = pinterp2->eval(&((*control_field)(0, 0)));
      f_1 = pinterp2->eval(&((*control_field)(1, 0)));
      f_2 = pinterp2->eval(&((*control_field)(2, 0)));

      dx_st[1] = ExtremaStepQuadInterp(ds[1], f_0, f_1, f_2);
      coord[1] += ds[1];

      delete pinterp2;

      // update local -----------------------------------------------
      loc_c_dx1(n-1) = dx_st[0];
      loc_c_dx2(n-1) = dx_st[1];

      break;
    }
    case 1:
    {
      Interp_Lag1 * pinterp1 = nullptr;

      // axis 0 -----------------------------------------------------
      coord[0] -= ds[0];
      pinterp1 = new Interp_Lag1(origin, ds, sz, coord);

      f_0 = pinterp1->eval(&((*control_field)(0)));
      f_1 = pinterp1->eval(&((*control_field)(1)));
      f_2 = pinterp1->eval(&((*control_field)(2)));

      dx_st[0] = ExtremaStepQuadInterp(ds[0], f_0, f_1, f_2);
      coord[0] += ds[0];

      delete pinterp1;

      // update local -----------------------------------------------
      loc_c_dx1(n-1) = dx_st[0];

      break;
    }
  }

  return;
}

int ExtremaTrackerLocal::LocateCentrePhysicalIndex(const int n,
                                                   const int axis)
{
  // n starts at 1
  // axis is the usual 0,1,2
  //
  // Search over only _physical_ nodes; data sorted
  //
  // std::lower_bound:
  // points to the first element that is not less than value,
  // or last if no such element is found.
  //
  // We therefore compare result and result at one index smaller

  Real src_val;
  int idx_l, idx_u;
  AthenaArray<Real> * x;

  switch(axis)
  {
    case 2:
      src_val = ptracker_extrema->c_x3(n-1);
      x = &(mbi->x3);
      idx_l = mbi->kl-1;
      idx_u = mbi->ku+1;
      break;
    case 1:
      src_val = ptracker_extrema->c_x2(n-1);
      x = &(mbi->x2);
      idx_l = mbi->jl-1;
      idx_u = mbi->ju+1;
      break;
    case 0:
      src_val = ptracker_extrema->c_x1(n-1);
      x = &(mbi->x1);
      idx_l = mbi->il-1;
      idx_u = mbi->iu+1;
      break;
    default:
      std::cout << "ExtremaTrackerLocal requires 0<=axis<3" << std::endl;
  }

  int idx = std::lower_bound(
    &((*x)(idx_l)),
    &((*x)(idx_u)),
    src_val) - &((*x)(idx_l));
  idx += idx_l;

  // TODO:
  // check upper range for lower_bound index does not need a +1
  // i.e. the last entry is also checked c.f. function body of IxMinimaOffset
  // std::cout << IxMinimaOffset(3, 4, 2) << std::endl;

  // check lower
  Real const low_val = (*x)(idx-1);
  Real const fnd_val = (*x)(idx);
  if (std::abs(src_val - low_val) < std::abs(src_val - fnd_val))
  {
    idx -= 1;
  }

  return idx;
}

bool ExtremaTrackerLocal::IsOrdPhysical(const int axis, const Real x)
{
  switch(axis)
  {
    case 2:
      return (
        (mbi->x3(mbi->kl-1) <= x) && (x <= mbi->x3(mbi->ku+1))
      );
    case 1:
      return (
        (mbi->x2(mbi->jl-1) <= x) && (x <= mbi->x2(mbi->ju+1))
      );
    case 0:
      return (
        (mbi->x1(mbi->il-1) <= x) && (x <= mbi->x1(mbi->iu+1))
      );
    default:
      std::cout << "ExtremaTrackerLocal requires 0<=axis<3" << std::endl;
  }

  return false;
}

void ExtremaTrackerLocal::ZeroInternalTracker()
{
  if (N_tracker > 0)
  {
    to_update.Fill(0);

    loc_c_dx1.Fill(0);
    loc_c_dx2.Fill(0);
    loc_c_dx3.Fill(0);
  }
}

ExtremaTrackerLocal::~ExtremaTrackerLocal()
{
  if (N_tracker > 0)
  {
    to_update.DeleteAthenaArray();

    sign_minima.DeleteAthenaArray();

    loc_c_dx1.DeleteAthenaArray();
    loc_c_dx2.DeleteAthenaArray();
    loc_c_dx3.DeleteAthenaArray();
  }
}
