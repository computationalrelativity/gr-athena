// C++ standard headers
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "hydro.hpp"
#include "../utils/linear_algebra.hpp"
#include "../z4c/z4c.hpp"

#include "rescaling.hpp"

// External libraries

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// ============================================================================
namespace gra::hydro::rescaling {
// ============================================================================

Rescaling::Rescaling(Mesh *pm, ParameterInput *pin) :
  pm (pm),
  pin (pin),
  ini {
    false, // initialized
  }
{
  // scrape settings ----------------------------------------------------------
  {
    opt.verbose = pin->GetOrAddBoolean("rescaling", "verbose", false);
    opt.use_cutoff = pin->GetOrAddBoolean("rescaling", "use_cutoff", false);
    opt.rescale_conserved_density =
        pin->GetOrAddBoolean("rescaling", "rescale_conserved_density", false);
    opt.rescale_conserved_scalars =
        pin->GetOrAddBoolean("rescaling", "rescale_conserved_scalars", false);

    opt.apply_on_substeps =
        pin->GetOrAddBoolean("rescaling", "apply_on_substeps", false);
    opt.disable_on_first_failure =
        pin->GetOrAddBoolean("rescaling", "disable_on_first_failure", false);

    opt.start_time =
        pin->GetOrAddReal("rescaling", "start_time", -1.0);
    opt.end_time =
        pin->GetOrAddReal("rescaling", "end_time", -1.0);

    opt.dump_status =
        pin->GetOrAddBoolean("rescaling", "dump_status", false);


    filename = pin->GetOrAddString("rescaling", "filename", "resc");
    filename += ".txt";
  }

  if (opt.rescale_conserved_density)
  {
    opt.fac_mul_D = pin->GetOrAddReal("rescaling", "fac_mul_D", 1.5);
    opt.err_rel_hydro = pin->GetOrAddReal("rescaling", "err_rel_hydro", 1e-8);

    cur.fac_mul_D = SQR(opt.fac_mul_D);
  }

  if (opt.rescale_conserved_scalars)
  {
    opt.fac_mul_s = pin->GetOrAddReal("rescaling", "fac_mul_s", 1.5);
    opt.err_rel_scalars = pin->GetOrAddReal("rescaling",
                                            "err_rel_scalars",
                                            1e-8);

    cur.fac_mul_s.NewAthenaArray(NSCALARS);
    for (int n=0; n<NSCALARS; ++n)
    {
      cur.fac_mul_s(n) = SQR(opt.fac_mul_s);
    }

    cur.min_s.NewAthenaArray(NSCALARS);
    cur.cut_s.NewAthenaArray(NSCALARS);
    cur.rsc_s.NewAthenaArray(NSCALARS);
  }

  if ((opt.rescale_conserved_density || opt.rescale_conserved_scalars) &&
      !pin->GetOrAddBoolean("z4c", "extended_aux_adm", false))
  {
    std::stringstream msg;
    msg << "### FATAL ERROR: Rescaling activation requires" << std::endl
        << "z4c/extended_aux_adm = true";

    ATHENA_ERROR(msg);
  }
}

void Rescaling::Initialize()
{
  if (ini.initialized)
  {
    return;
  }

  if (opt.rescale_conserved_density)
  {
    const Real D_cut = 0;
    ini.m = IntegrateField(variety_cs::conserved_hydro, IDN, D_cut);
    cur.m = ini.m;
  }

  if (opt.rescale_conserved_scalars)
  {
    const Real s_cut = 0;
    ini.S.NewAthenaArray(NSCALARS);
    cur.S.NewAthenaArray(NSCALARS);
    cur.err_rel_S.NewAthenaArray(NSCALARS);

    for (int n=0; n<NSCALARS; ++n)
    {
      ini.S(n) = IntegrateField(variety_cs::conserved_scalar, n, s_cut);
      cur.S(n) = ini.S(n);
    }
  }

  ini.initialized = true;
  OutputPrepare();
}

void Rescaling::Apply()
{
  if (!ini.initialized)
  {
    Initialize();
  }

  // check that we are in the correct time-range to apply the rescaling -------
  if (opt.start_time > 0)
  {
    if (pm->time < opt.start_time)
    {
      return;
    }
  }

  if (opt.end_time > 0)
  {
    if (pm->time >= opt.end_time)
    {
      return;
    }
  }
  // --------------------------------------------------------------------------

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  int nmb = -1;
  std::vector<MeshBlock*> pmb_array;

  if (opt.rescale_conserved_density ||
      ((NSCALARS > 0) && opt.rescale_conserved_scalars))
  {
    // initialize a vector of MeshBlock pointers
    pm->GetMeshBlocksMyRank(pmb_array);
    nmb = pmb_array.size();
  }
  else
  {
    // don't need to do anything
    return;
  }

  // check cur.X is finite - if not, disable ----------------------------------
  if (opt.rescale_conserved_density)
  {
    opt.rescale_conserved_density = (
      opt.rescale_conserved_density && std::isfinite(cur.m)
    );
  }

  if (opt.rescale_conserved_scalars)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      opt.rescale_conserved_scalars = (
        opt.rescale_conserved_scalars && std::isfinite(cur.S(n))
      );
    }
  }
  // --------------------------------------------------------------------------

  bool outside_threshold = false;

  if (opt.rescale_conserved_density)
  {
    // infer any required cut:
    if (opt.use_cutoff)
    {
      cur.fac_mul_D /= opt.fac_mul_D;

      const bool require_positive = true;
      cur.min_D = GlobalMinimum(variety_cs::conserved_hydro,
                                IDN, require_positive);
      cur.m = IntegrateField(variety_cs::conserved_hydro, IDN, 0);

      const Real dm = std::abs(ini.m - cur.m);

      // Find the amount of matter contained under a cut-off;
      // this needs to account for the above `dm` for rescaling
      Real cur_m_atm = 0;

      while (cur_m_atm < dm)
      {
        cur.fac_mul_D *= opt.fac_mul_D;
        cur_m_atm = cur.m - IntegrateField(variety_cs::conserved_hydro,
                                           IDN,
                                           cur.fac_mul_D * cur.min_D);
      }

      cur.cut_D = cur.fac_mul_D * cur.min_D;
      cur.rsc_D = (cur_m_atm != 0)
        ? std::abs(((ini.m - cur.m) + cur_m_atm) / cur_m_atm)
        : 1.0;
    }
    else
    {
      cur.cut_D = -std::numeric_limits<Real>::infinity();
      cur.m = IntegrateField(variety_cs::conserved_hydro, IDN, cur.cut_D);
      cur.rsc_D = std::abs(ini.m / cur.m);
    }

    // Now do rescaling: ------------------------------------------------------
    // cur.err_rel_D = std::abs(1-cur.rsc_D);
    cur.err_rel_D = std::abs(1-cur.m / ini.m);

    if (opt.err_rel_hydro > cur.err_rel_D)
    {
      #pragma omp parallel for num_threads(nthreads)
      for (int ix = 0; ix < nmb; ++ix)
      {
        MeshBlock *pmb = pmb_array[ix];
        Hydro * ph = pmb->phydro;
        Z4c * pz4c = pmb->pz4c;

        Z4c::Aux_extended_vars aux_extended;
        pz4c->SetAuxExtendedAliases(pz4c->storage.aux_extended,
                                    aux_extended);

        CC_GLOOP2(k, j)
        for (int n=NHYDRO-1; n>=0; --n)
        CC_GLOOP1(i)
        {
          const Real oo_sqrt_detgamma = OO(
            aux_extended.cc_sqrt_detgamma(k,j,i)
          );

          const Real fac = ((oo_sqrt_detgamma * ph->u(IDN,k,j,i)) <=
                            std::abs(cur.cut_D))
            ? cur.rsc_D
            : 1.0;

          ph->u(n,k,j,i) *= fac;
        }
      }
    }
    else
    {
      outside_threshold = true;
    }
  }

  if (opt.rescale_conserved_scalars)
  {
    // infer any required cut:
    if (opt.use_cutoff)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        cur.fac_mul_s(n) /= opt.fac_mul_s;

        const bool require_positive = true;
        cur.min_s(n) = GlobalMinimum(variety_cs::conserved_scalar,
                                     n, require_positive);
        cur.S(n) = IntegrateField(variety_cs::conserved_scalar, n, 0);

        const Real ds = std::abs(ini.S(n) - cur.S(n));

        // Find the amount of content contained under a cut-off;
        // this needs to account for the above `ds` for rescaling
        Real cur_S_atm = 0;

        while (cur_S_atm < ds)
        {
          cur.fac_mul_s(n) *= opt.fac_mul_s;
          cur_S_atm = cur.S(n) - IntegrateField(
            variety_cs::conserved_scalar,
            n,
            cur.fac_mul_s(n) * cur.min_s(n)
          );
        }

        cur.cut_s(n) = cur.fac_mul_s(n) * cur.min_s(n);
        cur.rsc_s(n) = (cur_S_atm != 0)
          ? std::abs(((ini.S(n) - cur.S(n)) + cur_S_atm) / cur_S_atm)
          : 1.0;
      }

    }
    else
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        cur.cut_s(n) = -std::numeric_limits<Real>::infinity();
        cur.S(n) = IntegrateField(variety_cs::conserved_scalar,
                                  n, cur.cut_s(n));
        cur.rsc_s(n) = std::abs(ini.S(n) / cur.S(n));
      }
    }

    // Now do rescaling: ------------------------------------------------------
    for (int n=0; n<NSCALARS; ++n)
    {
      // cur.err_rel_S(n) = std::abs(1-cur.rsc_s(n));
      cur.err_rel_S(n) = std::abs(1-cur.S(n) / ini.S(n));
    }

    #pragma omp parallel for num_threads(nthreads)
    for (int ix = 0; ix < nmb; ++ix)
    {
      MeshBlock *pmb = pmb_array[ix];
      PassiveScalars * ps = pmb->pscalars;
      Z4c * pz4c = pmb->pz4c;

      Z4c::Aux_extended_vars aux_extended;
      pz4c->SetAuxExtendedAliases(pz4c->storage.aux_extended,
                                  aux_extended);

      CC_GLOOP2(k, j)
      for (int n=0; n<NSCALARS; ++n)
      {
        if (opt.err_rel_scalars > cur.err_rel_S(n))
        {
          CC_GLOOP1(i)
          {
            const Real oo_sqrt_detgamma = OO(
              aux_extended.cc_sqrt_detgamma(k,j,i)
            );

            const Real fac = ((oo_sqrt_detgamma * ps->s(n,k,j,i)) <=
                              std::abs(cur.cut_s(n)))
              ? cur.rsc_s(n)
              : 1.0;

            ps->s(n,k,j,i) *= fac;
          }
        }
        else
        {
          outside_threshold = true;
        }
      }
    }
  }

  // optionally disable all future rescalings if any current one failed:
  if (outside_threshold && opt.disable_on_first_failure)
  {
    opt.rescale_conserved_density = false;
    opt.rescale_conserved_scalars = false;

    // and keep it disabled for future restarts
    pin->OverwriteParameter("rescaling",
                            "rescale_conserved_density",
                            false);

    pin->OverwriteParameter("rescaling",
                            "rescale_conserved_scalars",
                            false);
  }

  // debug info ---------------------------------------------------------------
  if (opt.verbose && (Globals::my_rank == 0))
  {
    std::cout << "Rescaling: \n";

    if (opt.rescale_conserved_density)
    {
      std::cout << "ini.m: " << ini.m << "\n";
      std::cout << "cur.m: " << cur.m << "\n";
      std::cout << "cur.err_rel_D: " << cur.err_rel_D << "\n";
      std::cout << "cur.rsc_D: " << cur.rsc_D << "\n";
      if (opt.use_cutoff)
      {
        std::cout << "cur.min_D: " << cur.min_D << "\n";
        std::cout << "cur.cut_D: " << cur.cut_D << "\n";
      }
    }

    if (opt.rescale_conserved_scalars)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        std::cout << "ini.S : " << n << " : " << ini.S(n) << "\n";
        std::cout << "cur.S : " << n << " : " << cur.S(n) << "\n";
        std::cout << "cur.err_rel_S : " << n << " : ";
        std::cout << cur.err_rel_S(n) << "\n";
        std::cout << "cur.rsc_s : " << n << " : ";
        std::cout << cur.rsc_s(n) << "\n";
        if (opt.use_cutoff)
        {
          std::cout << "cur.min_s : " << n << ":" << cur.min_s(n) << "\n";
          std::cout << "cur.cut_s : " << n << ":" << cur.cut_s(n) << "\n";
        }
      }
    }
  }

}

Real Rescaling::CompensatedSummation(const variety_cs v_cs,
                                     const int n,
                                     const Real v_cut)
{
  using namespace LinearAlgebra;

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  int nmb = -1;
  std::vector<MeshBlock*> pmb_array;

  pm->GetMeshBlocksMyRank(pmb_array);
  nmb = pmb_array.size();

  AA partial_KBN(nmb);

  #pragma omp parallel for num_threads(nthreads)
  for (int ix = 0; ix < nmb; ++ix)
  {
    MeshBlock *pmb = pmb_array[ix];
    Z4c * pz4c = pmb->pz4c;

    Z4c::Aux_extended_vars aux_extended;
    pz4c->SetAuxExtendedAliases(pz4c->storage.aux_extended,
                                aux_extended);

    AA arr;
    AA warr;
    warr.NewAthenaArray(pmb->ke+1, pmb->je+1, pmb->ie+1);

    switch (v_cs)
    {
      case variety_cs::conserved_hydro:
      {
        arr.InitWithShallowSlice(pmb->phydro->u, 4, n, 1);
        break;
      }
      case variety_cs::conserved_scalar:
      {
        arr.InitWithShallowSlice(pmb->pscalars->s, 4, n, 1);
        break;
      }
      default:
      {
        assert(false);
      }
    }

    CC_ILOOP3(k, j, i)
    {
      const Real oo_sqrt_detgamma = OO(
        aux_extended.cc_sqrt_detgamma(k,j,i)
      );
      const Real v = oo_sqrt_detgamma * arr(0,k,j,i);
      const Real vol = pmb->pcoord->GetCellVolume(k,j,i);

      warr(k,j,i) = (v > v_cut) ? vol * arr(0,k,j,i) : 0;
    }

    partial_KBN(ix) = FloatingPoint::KB_compensated(
      warr, 0, 0,
      pmb->ks, pmb->ke,
      pmb->js, pmb->je,
      pmb->is, pmb->ie
    );

    pmb = pmb->next;
  }

  return FloatingPoint::KB_compensated(partial_KBN, 0, nmb-1);
}

Real Rescaling::IntegrateField(const variety_cs v_cs,
                               const int n,
                               const Real V_cut)
{
  Real val = CompensatedSummation(v_cs, n, V_cut);

#ifdef MPI_PARALLEL
  const int rank_root = 0;
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  AthenaArray<Real> buf(Globals::nranks);

  MPI_Allgather(&val, 1, MPI_ATHENA_REAL, buf.data(), 1, MPI_ATHENA_REAL,
                MPI_COMM_WORLD);

  val = FloatingPoint::KB_compensated(buf, 0, Globals::nranks-1);
#endif // MPI_PARALLEL

  return val;
}

// Extract global minimum over physical points
Real Rescaling::GlobalMinimum(const variety_cs v_cs,
                              const int n,
                              const bool require_positive)
{
  using namespace LinearAlgebra;

  Real min_V = std::numeric_limits<Real>::infinity();

  int nthreads = pm->GetNumMeshThreads();
  (void)nthreads;

  int nmb = -1;
  std::vector<MeshBlock*> pmb_array;

  pm->GetMeshBlocksMyRank(pmb_array);
  nmb = pmb_array.size();

  #pragma omp parallel for num_threads(nthreads) reduction(min:min_V)
  for (int ix = 0; ix < nmb; ++ix)
  {
    MeshBlock *pmb = pmb_array[ix];
    Z4c * pz4c = pmb->pz4c;

    Z4c::Aux_extended_vars aux_extended;
    pz4c->SetAuxExtendedAliases(pz4c->storage.aux_extended,
                                aux_extended);

    AA arr;

    CC_ILOOP2(k, j)
    #pragma omp simd reduction(min:min_V)
    for (int i=pmb->is; i<=pmb->ie; ++i)
    {

      switch (v_cs)
      {
        case variety_cs::conserved_hydro:
        {
          arr.InitWithShallowSlice(pmb->phydro->u, 4, n, 1);
          break;
        }
        case variety_cs::conserved_scalar:
        {
          arr.InitWithShallowSlice(pmb->pscalars->s, 4, n, 1);
          break;
        }
        default:
        {
          assert(false);
        }
      }

      const Real oo_sqrt_detgamma = OO(
        aux_extended.cc_sqrt_detgamma(k,j,i)
      );

      const Real V = oo_sqrt_detgamma * arr(k,j,i);

      if (require_positive)
      {
        min_V = (V > 0) ? std::min(V, min_V) : min_V;
      }
      else
      {
        min_V = std::min(V, min_V);
      }
    }
  }

#ifdef MPI_PARALLEL
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (require_positive && min_V <= 0)
  {
    min_V = std::numeric_limits<Real>::infinity();
  }

  // if (Globals::my_rank == 0)
  // {
  //   MPI_Reduce(MPI_IN_PLACE, &min_V, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
  //              MPI_COMM_WORLD);
  // }
  // else
  // {
  //   MPI_Reduce(&min_V, &min_V, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
  //              MPI_COMM_WORLD);
  // }

  Real min_all_V;
  MPI_Allreduce(&min_V, &min_all_V, 1,
    MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  return min_V;
}

// I/O ------------------------------------------------------------------------
void Rescaling::OutputPrepare()
{
  if (!opt.dump_status)
    return;

  if (0 == Globals::my_rank)
  {

    // check if output file already exists
    if (access(filename.c_str(), F_OK) == 0) {
      pofile = fopen(filename.c_str(), "a");
    }
    else
    {
      pofile = fopen(filename.c_str(), "w");
      if (NULL == pofile)
      {
        std::stringstream msg;
        msg << "### FATAL ERROR in Rescaling" << std::endl;
        msg << "Could not open file '" << filename << "' for writing!";
        throw std::runtime_error(msg.str().c_str());
      }
    }

    int ix = 0;
    fprintf(pofile, "# ");
    fprintf(pofile, "[%d]=iter ", ix++);
    fprintf(pofile, "[%d]=time ", ix++);
    fprintf(pofile, "[%d]=stage ", ix++);

    if (opt.rescale_conserved_density)
    {
      // fprintf(pofile, "[%d]=cur.m ", ix++);
      fprintf(pofile, "[%d]=cur.err_rel_D ", ix++);
      fprintf(pofile, "[%d]=cur.rsc_D ", ix++);
      if (opt.use_cutoff)
      {
        fprintf(pofile, "[%d]=cur.min_D ", ix++);
        fprintf(pofile, "[%d]=cur.cut_D ", ix++);
      }
    }

    if (opt.rescale_conserved_scalars)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        // fprintf(pofile, "[%d]=cur.S_%d ", ix++, n);
        fprintf(pofile, "[%d]=cur.err_rel_S_%d ", ix++, n);
        fprintf(pofile, "[%d]=cur.rsc_s_%d ", ix++, n);
        if (opt.use_cutoff)
        {
          fprintf(pofile, "[%d]=cur.min_s_%d ", ix++, n);
          fprintf(pofile, "[%d]=cur.cut_s_%d ", ix++, n);
        }
      }
    }

    fprintf(pofile, "\n");
    fflush(pofile);
  }
}

void Rescaling::OutputWrite(const int iter, const Real time, const int nstage)
{
  if (!opt.dump_status)
    return;

  if (Globals::my_rank == 0)
  {
    fprintf(pofile, "%d ", iter);
    fprintf(pofile, "%.*g ", FPRINTF_PREC, time);
    fprintf(pofile, "%d ", nstage);

    if (opt.rescale_conserved_density)
    {
      // fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.m);
      fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.err_rel_D);
      fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.rsc_D);

      if (opt.use_cutoff)
      {
        fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.min_D);
        fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.cut_D);
      }
    }

    if (opt.rescale_conserved_scalars)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        // fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.S(n));
        fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.err_rel_S(n));
        fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.rsc_s(n));

        if (opt.use_cutoff)
        {
          fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.min_s(n));
          fprintf(pofile, "%.*g ", FPRINTF_PREC, cur.cut_s(n));
        }
      }
    }

    fprintf(pofile, "\n");
    fflush(pofile);
  }
}

void Rescaling::OutputFinalize()
{
  if (!opt.dump_status)
    return;

  if (Globals::my_rank == 0)
  {
    fclose(pofile);
  }
}

// ============================================================================
} // namespace gra::hydro::rescaling
// ============================================================================

//
// :D
//