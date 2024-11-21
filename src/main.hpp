#ifndef MAIN_HPP
#define MAIN_HPP


// C++ headers
#include "athena_aliases.hpp"
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <iomanip>
#include <iostream>   // cout, endl

// External libraries

// MPI/OpenMP headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Athena++ headers
#include "athena.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "main_triggers.hpp"

#include "outputs/io_wrapper.hpp"
#include "outputs/outputs.hpp"
#include "mesh/mesh.hpp"
#include "task_list/gr/task_list.hpp"
#include "task_list/m1/task_list.hpp"
#include "task_list/wave_equations/task_list.hpp"
#include "utils/utils.hpp"

#include "z4c/wave_extract.hpp"
#include "z4c/puncture_tracker.hpp"
#include "z4c/ahf.hpp"
#ifdef EJECTA_ENABLED
#include "z4c/ejecta.hpp"
#endif
#if CCE_ENABLED
#include "z4c/cce/cce.hpp"
#endif

#if M1_ENABLED
#include "m1/m1.hpp"
#endif

#include "trackers/extrema_tracker.hpp"

// Note:
// ENABLE_EXCEPTIONS is _always_ assumed

// MPI/OMP  -------------------------------------------------------------------
namespace gra::parallelism {

inline void Teardown()
{
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
}

inline void Init(int argc, char *argv[])
{
  #ifdef MPI_PARALLEL
  #ifdef OPENMP_PARALLEL
    int mpiprv;
    if (MPI_SUCCESS != MPI_Init_thread(&argc,
                                       &argv,
                                       MPI_THREAD_MULTIPLE, &mpiprv))
    {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "MPI Initialization failed." << std::endl;
      std::exit(0);
    }
    if (mpiprv != MPI_THREAD_MULTIPLE) {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
                << MPI_THREAD_MULTIPLE << " : " << mpiprv
                << std::endl;
      Teardown();
      std::exit(0);
    }
  #else  // no OpenMP
    if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "MPI Initialization failed." << std::endl;
      std::exit(0);
    }
  #endif  // OPENMP_PARALLEL
    // Get process id (rank) in MPI_COMM_WORLD
    if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank)))
    {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "MPI_Comm_rank failed." << std::endl;
      Teardown();
      std::exit(0);
    }

    // Get total number of MPI processes (ranks)
    if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks))
    {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "MPI_Comm_size failed." << std::endl;
      Teardown();
      std::exit(0);
    }

    // Get the maximum MPI tag
    void * value;
    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);
    Globals::mpi_tag_ub = *(int *)value;

  #else  // no MPI
    Globals::my_rank = 0;
    Globals::nranks  = 1;
    Globals::mpi_tag_ub = 0;
  #endif  // MPI_PARALLEL
}

inline void Barrier()
{
#ifdef MPI_PARALLEL
  MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_PARALLEL
}

}  // namespace gra::parallelism

// Runtime stuff --------------------------------------------------------------
namespace gra::timing {

class Clocks
{
  public:
    Clocks()
    {
      tstart = clock();
#ifdef OPENMP_PARALLEL
      omp_start_time = omp_get_wtime();
#endif
    };

    inline void Stop()
    {
      tstop = clock();
#ifdef OPENMP_PARALLEL
      omp_time = omp_get_wtime() - omp_start_time;
#endif
    }

  public:
    clock_t tstart, tstop;
    double omp_start_time;
    double omp_time;
};

}  // namespace gra::timing

// General procedures ---------------------------------------------------------
namespace gra {

// BD: TODO- would be an excellent move to inject git commit into ver. info
std::string athena_version = "version 19.0 - August 2019";

struct Pathing
{
  char *input_filename = nullptr;
  char *restart_filename = nullptr;
  char *prundir = nullptr;
};

struct Flags
{
  int res;  // set to 1 if -r        argument is on cmdline
  int narg; // set to 1 if -n        argument is on cmdline
  int iarg; // set to 1 if -i <file> argument is on cmdline
  int mesh; // set to <nproc> if -m <nproc> argument is on cmdline
  int wtlim;
};

// Command line parsing -------------------------------------------------------
inline void ParseCommandLine(int argc, char *argv[], Pathing *ppat, Flags *pfl)
{
  for (int i=1; i<argc; i++)
  {
    // If argv[i] is a 2 character string of the form "-?" then:
    if (*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0')
    {
      // check validity of command line options + arguments:
      char opt_letter = *(argv[i]+1);
      switch(opt_letter) {
        // options that do not take arguments:
        case 'n':
        case 'c':
        case 'h':
          break;
          // options that require arguments:
        default:
          if ((i+1 >= argc) // flag is at the end of the command line options
              || (*argv[i+1] == '-') ) { // flag is followed by another flag
            if (Globals::my_rank == 0)
            {
              std::cout << "### FATAL ERROR in main" << std::endl
                        << "-" << opt_letter << " must be followed by a valid argument\n";

              gra::parallelism::Teardown();
              std::exit(0);
            }
          }
      }
      switch(*(argv[i]+1))
      {
        case 'i':                      // -i <input_filename>
          ppat->input_filename = argv[++i];
          pfl->iarg = 1;
          break;
        case 'r':                      // -r <restart_file>
          pfl->res = 1;
          ppat->restart_filename = argv[++i];
          break;
        case 'd':                      // -d <run_directory>
          ppat->prundir = argv[++i];
          break;
        case 'n':
          pfl->narg = 1;
          break;
        case 'm':                      // -m <nproc>
          pfl->mesh = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
          break;
        case 't':                      // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          pfl->wtlim = wth*3600 + wtm*60 + wts;
          break;
        case 'c':
          if (Globals::my_rank == 0) ShowConfig();
          gra::parallelism::Teardown();
          std::exit(0);
          break;
        case 'h':
        default:
          if (Globals::my_rank == 0) {
            std::cout << "Athena++ " << athena_version << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m <nproc>      output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
          gra::parallelism::Teardown();
          std::exit(0);
          break;
      }
    } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
  }

  if (ppat->restart_filename == nullptr && ppat->input_filename == nullptr)
  {
    // no input file is given
    std::cout << "### FATAL ERROR in main" << std::endl
              << "No input file or restart file is specified." << std::endl;
    gra::parallelism::Teardown();
    std::exit(0);
  }
}

// Input parsing --------------------------------------------------------------
inline void ParseInputs(int argc, char *argv[],
                        Pathing *ppat, Flags *pfl, ParameterInput *pin,
                        IOWrapper &infile, IOWrapper &restartfile)
{

  try
  {
    if (pfl->res == 1)
    {
      restartfile.Open(ppat->restart_filename, IOWrapper::FileMode::read);
      pin->LoadFromFile(restartfile);
      // If both -r and -i are specified, make sure next_time gets corrected.
      // This needs to be corrected on the restart file because we need the old dt.
      if (pfl->iarg == 1) pin->RollbackNextTime();
      // leave the restart file open for later use
    }
    if (pfl->iarg == 1)
    {
      // if both -r and -i are specified, override the parameters using the input file
      infile.Open(ppat->input_filename, IOWrapper::FileMode::read);
      pin->LoadFromFile(infile);
      infile.Close();
    }
    pin->ModifyFromCmdline(argc ,argv);
  }
  catch(std::bad_alloc& ba)
  {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed initializing class ParameterInput: "
              << ba.what() << std::endl;
    if (pfl->res == 1) restartfile.Close();
    gra::parallelism::Teardown();
    std::exit(0);
  }
  catch(std::exception const& ex)
  {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    if (pfl->res == 1) restartfile.Close();
    gra::parallelism::Teardown();
    std::exit(0);
  }
}

// Mesh manipulation ----------------------------------------------------------
inline Mesh * InitMesh(Flags *pfl, ParameterInput *pin, IOWrapper &resfile)
{
  Mesh *pmesh;

  try
  {
    if (pfl->res == 0)
    {
      pmesh = new Mesh(pin, pfl->mesh);
    } else {
      pmesh = new Mesh(pin, resfile, pfl->mesh);
    }
  }
  catch(std::bad_alloc& ba)
  {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed initializing class Mesh: "
              << ba.what() << std::endl;
    if (pfl->res == 1) resfile.Close();
    gra::parallelism::Teardown();
    std::exit(0);
  }
  catch(std::exception const& ex)
  {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    if (pfl->res == 1) resfile.Close();
    gra::parallelism::Teardown();
    std::exit(0);
  }

  // With current mesh time possibly read from restart file,
  // correct next_time for outputs
  if (pfl->iarg == 1 && pfl->res == 1)
  {
    // if both -r and -i are specified,
    // ensure that next_time  >= mesh_time - dt
    pin->ForwardNextTime(pmesh->time);
  }

  // Dump input parameters and quit if code was run with -n option.
  if (pfl->narg)
  {
    if (Globals::my_rank == 0) pin->ParameterDump(std::cout);
    if (pfl->res == 1) resfile.Close();
    gra::parallelism::Teardown();
    std::exit(0);
  }

  if (pfl->res == 1) resfile.Close(); // close the restart file here

  // Quit if -m was on cmdline.  This option builds and outputs mesh structure.
  if (pfl->mesh > 0)
  {
    gra::parallelism::Teardown();
    std::exit(0);
  }

  return pmesh;
}

inline void InitMeshData(Flags *pfl, ParameterInput *pin, Mesh *pm)
{
  try
  {
    pm->Initialize(pfl->res, pin);
    pm->InitializePostFirstInitialize(pin);
    pm->DeleteTemporaryUserMeshData();
  }
  catch(std::bad_alloc& ba)
  {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed "
              << "in problem generator " << ba.what() << std::endl;
    gra::parallelism::Teardown();
    std::exit(0);
  }
  catch(std::exception const& ex)
  {
    std::cout << ex.what() << std::endl;  // prints diagnostic message
    gra::parallelism::Teardown();
    std::exit(0);
  }
}

// Output handlers ------------------------------------------------------------
inline Outputs * InitOutputs(Flags *pfl,
                             Pathing *ppat,
                             ParameterInput *pin,
                             Mesh *pm)
{
  try
  {
    ChangeRunDir(ppat->prundir);

    Outputs *pouts;

    pouts = new Outputs(pm, pin);

    if (pfl->res == 0)
    {
      pouts->MakeOutputs(pm, pin);
    }

    return pouts;
  }
  catch(std::bad_alloc& ba)
  {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "memory allocation failed setting initial conditions: "
              << ba.what() << std::endl;
    gra::parallelism::Teardown();
    return(0);
  }
  catch(std::exception const& ex)
  {
    std::cout << ex.what() << std::endl;
    gra::parallelism::Teardown();
    return(0);
  }
}

inline void MakeOutputs(const bool is_final,
                        ParameterInput *pin,
                        Mesh *pm,
                        Outputs *pouts)
{
    try
    {
      pouts->MakeOutputs(pm, pin, is_final);
    }
    catch(std::bad_alloc& ba)
    {
      std::cout << "### FATAL ERROR in main" << std::endl
                << "memory allocation failed during output: ";
      std::cout << ba.what() <<std::endl;
      gra::parallelism::Teardown();
      std::exit(0);
    }
    catch(std::exception const& ex)
    {
      std::cout << ex.what() << std::endl;
      gra::parallelism::Teardown();
      std::exit(0);
    }
}

// General info
inline void PrintRankZero(std::string msg)
{
  if (Globals::my_rank == 0)
  {
    std::cout << msg << std::endl;
  }
}

inline void PrintDiagnostics(std::uint64_t mbcnt,
                             gra::timing::Clocks * pclk,
                             Mesh *pm)
{
  if (Globals::my_rank == 0)
  {
    pm->OutputCycleDiagnostics();
    if (SignalHandler::GetSignalFlag(SIGTERM) != 0) {
      std::cout << std::endl << "Terminating on Terminate signal" << std::endl;
    } else if (SignalHandler::GetSignalFlag(SIGINT) != 0) {
      std::cout << std::endl << "Terminating on Interrupt signal" << std::endl;
    } else if (SignalHandler::GetSignalFlag(SIGALRM) != 0) {
      std::cout << std::endl << "Terminating on wall-time limit" << std::endl;
    } else if (pm->ncycle == pm->nlim) {
      std::cout << std::endl << "Terminating on cycle limit" << std::endl;
    } else {
      std::cout << std::endl << "Terminating on time limit" << std::endl;
    }

    std::cout << "time=" << pm->time << " cycle=" << pm->ncycle << std::endl;
    std::cout << "tlim=" << pm->tlim << " nlim=" << pm->nlim << std::endl;

    if (pm->adaptive)
    {
      std::cout << std::endl << "Number of MeshBlocks = " << pm->nbtotal
                << "; " << pm->nbnew << "  created, " << pm->nbdel
                << " destroyed during this simulation." << std::endl;
    }

    // Calculate and print the zone-cycles/cpu-second and wall-second
    pclk->Stop();
    double cpu_time = (pclk->tstop > pclk->tstart ?
                       static_cast<double> (pclk->tstop-pclk->tstart) :
                       1.0)/static_cast<double> (CLOCKS_PER_SEC);
    std::uint64_t zonecycles = mbcnt * static_cast<std::uint64_t> (
      pm->pblock->GetNumberOfMeshBlockCells()
    );
    double zc_cpus = static_cast<double> (zonecycles) / cpu_time;

    std::cout << std::endl << "zone-cycles = " << zonecycles << std::endl;
    std::cout << "cpu time used = " << cpu_time << std::endl;
    std::cout << "zone-cycles/cpu_second = " << zc_cpus << std::endl;
#ifdef OPENMP_PARALLEL
    double zc_omps = static_cast<double> (zonecycles) / pclk->omp_time;
    std::cout << std::endl << "omp wtime used = "
                           << pclk->omp_time << std::endl;
    std::cout << "zone-cycles/omp_wsecond = " << zc_omps << std::endl;
#endif
  }

}

}  // namespace gra

// Handle collections of tasklists --------------------------------------------
namespace gra::tasklist {

struct Collection
{
  gra::triggers::Triggers & trgs;
  TaskLists::GeneralRelativity::GR_Z4c      * gr_z4c       = nullptr;
  TaskLists::GeneralRelativity::GRMHD_Z4c   * grmhd_z4c    = nullptr;

  TaskLists::GeneralRelativity::Aux_Z4c     * aux_z4c      = nullptr;
  TaskLists::GeneralRelativity::PostAMR_Z4c * postamr_z4c  = nullptr;

  TaskLists::M1::M1N0                       * m1n0         = nullptr;
  TaskLists::M1::PostAMR_M1N0               * postamr_m1n0 = nullptr;

  TaskLists::WaveEquations::Wave_2O         * wave_2o      = nullptr;
};

inline void PopulateCollection(Collection &ptlc,
                               Mesh *pm,
                               ParameterInput *pin)
{
  try
  {
    if (Z4C_ENABLED)
    {
      using namespace TaskLists::GeneralRelativity;

      if (FLUID_ENABLED)
      {
        // GR(M)HD
        ptlc.grmhd_z4c = new GRMHD_Z4c(pin, pm, ptlc.trgs);
      }
      else
      {
        // GR: vacuum
        ptlc.gr_z4c = new GR_Z4c(pin, pm, ptlc.trgs);
      }

      ptlc.aux_z4c     = new Aux_Z4c(pin, pm, ptlc.trgs);
      ptlc.postamr_z4c = new PostAMR_Z4c(pin, pm, ptlc.trgs);
    }

    if (M1_ENABLED)
    {
      using namespace TaskLists::M1;
      ptlc.m1n0 = new M1N0(pin, pm, ptlc.trgs);
      ptlc.postamr_m1n0 = new PostAMR_M1N0(pin, pm, ptlc.trgs);
    }

    if (WAVE_ENABLED)
    {
      using namespace TaskLists::WaveEquations;
      ptlc.wave_2o = new Wave_2O(pin, pm, ptlc.trgs);
    }
  }
  catch(std::bad_alloc& ba)
  {
    std::cout << "### FATAL ERROR in main" << std::endl;
    std::cout << "memory allocation failed ";
    std::cout << "in creating task list " << ba.what() << std::endl;

    gra::parallelism::Teardown();
    std::exit(0);
  }
}

inline void TearDown(Collection &ptlc)
{
  if (Z4C_ENABLED)
  {
    if (FLUID_ENABLED)
    {
      // GR(M)HD
      delete ptlc.grmhd_z4c;
    }
    else
    {
      // GR: vacuum
      delete ptlc.gr_z4c;
    }

    delete ptlc.aux_z4c;
    delete ptlc.postamr_z4c;
  }

  if (M1_ENABLED)
  {
    delete ptlc.m1n0;
    delete ptlc.postamr_m1n0;
  }

  if (WAVE_ENABLED)
  {
    delete ptlc.wave_2o;
  }

}

}  // namespace gra::tasklist

// Integration loop handling / triggers, etc ----------------------------------
namespace gra::evolve {

using namespace gra::triggers;
typedef Triggers::TriggerVariant tvar;
typedef Triggers::OutputVariant ovar;

inline void Z4c_Vacuum(gra::tasklist::Collection &ptlc,
                       Mesh *pmesh)
{
  for (int stage=1; stage<=ptlc.gr_z4c->nstages; ++stage)
  {
    ptlc.gr_z4c->DoTaskListOneStage(pmesh, stage);
    // Iterate bnd comm. as required
    pmesh->CommunicateIteratedZ4c(Z4C_CX_NUM_RBC);
  }
}

inline void Z4c_GRMHD(gra::tasklist::Collection &ptlc,
                      Mesh *pmesh)
{
#ifdef DBG_SCATTER_MATTER_GRMHD
  std::vector<MeshBlock*> pmb_array;
  pmesh->GetMeshBlocksMyRank(pmb_array);
#endif

  for (int stage=1; stage<=ptlc.grmhd_z4c->nstages; ++stage)
  {
    ptlc.grmhd_z4c->DoTaskListOneStage(pmesh, stage);
    // Iterate bnd comm. as required
    pmesh->CommunicateIteratedZ4c(Z4C_CX_NUM_RBC);

    if (stage == ptlc.grmhd_z4c->nstages)
    {
      // Rescale as required
      pmesh->Rescale_Conserved();
    }

#ifdef DBG_SCATTER_MATTER_GRMHD
    pmesh->ScatterMatter(pmb_array);
#endif
  }

}

inline void Z4c_DerivedQuantities(gra::tasklist::Collection &ptlc,
                                  gra::triggers::Triggers &trgs,
                                  Mesh *pmesh)
{
  // After state vector propagated, derived diagnostics (i.e. GW, trackers)
  // are at the new time-step ...
  const Real time_end_stage   = pmesh->time+pmesh->dt;
  const int ncycle_end_stage  = pmesh->ncycle+1;

  // Derivatives of ADM metric and other auxiliary computations needed below
  if (trgs.IsSatisfied(tvar::Z4c_AHF))
  {
    pmesh->CalculateStoreMetricDerivatives();
  }

  // Auxiliary variable logic
  // Currently this handles Weyl communication & decomposition
  if (trgs.IsSatisfied(tvar::Z4c_Weyl))
  {
    // May be required to prevent task-list overlaps
    // gra::parallelism::Barrier();
    pmesh->CommunicateAuxZ4c();
    ptlc.aux_z4c->DoTaskListOneStage(pmesh, 1);  // only 1 stage
  }

  if (trgs.IsSatisfied(tvar::Z4c_Weyl, ovar::user))
  {
    for (auto pwextr : pmesh->pwave_extr)
    {
      pwextr->ReduceMultipole();
      pwextr->Write(ncycle_end_stage, time_end_stage);
    }
  }

// BD: TODO - CCE needs to be cleaned up & tested
#if CCE_ENABLED
  // only do a CCE dump if NextTime threshold cleared (updated below)
  bool debug_pr = true;
  for (auto cce : pmesh->pcce)
  {
    if (pmesh->ncycle % cce->freq != 0) continue;
    
    int cce_iter = pmesh->ncycle / cce->freq;

    if (Globals::my_rank == 0 && debug_pr == true)
    {
      printf("cce_iter = %d, cce_freq = %d, pmesh->ncycle = %d, pmesh->time = %0.15f\n",
              cce_iter, cce->freq, pmesh->ncycle, pmesh->time);
    }
    debug_pr = false;
    cce->ReduceInterpolation();
    cce->DecomposeAndWrite(cce_iter, time_end_stage);
  }
#endif

  // RWZ wave extraction
  //TODO

  // AHF
  if (trgs.IsSatisfied(tvar::Z4c_AHF))
  {

    for (auto pah_f : pmesh->pah_finder)
    {
      if (pah_f->CalculateMetricDerivatives(ncycle_end_stage,
                                            time_end_stage))
      {
        break;
      }
    }
    for (auto pah_f : pmesh->pah_finder)
    {
      pah_f->Find(ncycle_end_stage, time_end_stage);
      pah_f->Write(ncycle_end_stage, time_end_stage);
    }
    for (auto pah_f : pmesh->pah_finder)
    {
      if (pah_f->DeleteMetricDerivatives(ncycle_end_stage, time_end_stage))
      {
        break;
      }
    }

  }

#ifdef EJECTA_ENABLED
  // Ejecta analysis
  for (auto pej : pmesh->pej_extract) {
    pej->Calculate(pmesh->ncycle, pmesh->time);
  }
#endif

  // Puncture trackers
  for (auto ptracker : pmesh->pz4c_tracker)
  {
    ptracker->EvolveTracker();
  }

  if (trgs.IsSatisfied(tvar::Z4c_tracker_punctures, ovar::user))
  for (auto ptracker : pmesh->pz4c_tracker)
  {
    ptracker->WriteTracker(ncycle_end_stage, time_end_stage);
  }

}

inline void M1N0(gra::tasklist::Collection &ptlc,
                 Mesh *pmesh)
{
  std::vector<MeshBlock*> pmb_array;
  pmesh->GetMeshBlocksMyRank(pmb_array);

  for (int stage=1; stage<=ptlc.m1n0->nstages; ++stage)
  {
    ptlc.m1n0->DoTaskListOneStage(pmesh, stage);

    // Last stage performs Con2Prim, scatter this, call GetMatter
    if (stage == ptlc.m1n0->nstages)
    {
      pmesh->ScatterMatter(pmb_array);
    }
  }
}

inline void Z4c_GRMHD_M1N0(gra::tasklist::Collection &ptlc,
                           Mesh *pmesh)
{
#if M1_ENABLED

  if (pmesh->pblock->pm1->opt.use_split_step)
  {
    const Real t  = pmesh->time;
    const Real dt = pmesh->dt;

    // step dt/2 ------------------------------------------------------------
    pmesh->dt   = 0.5 * dt;
    pmesh->time = t;

    Z4c_GRMHD(ptlc, pmesh);

    pmesh->dt   = dt;
    pmesh->time = t;

    // step dt --------------------------------------------------------------
    // weight is 1.0; no need to change dt
    M1N0(ptlc, pmesh);

    // step dt/2 ------------------------------------------------------------
    pmesh->dt   = 0.5 * dt;
    pmesh->time = t + pmesh->dt;

    Z4c_GRMHD(ptlc, pmesh);

    // revert time
    pmesh->time -= pmesh->dt;
    pmesh->dt    = dt;
  }
  else
  {
    Z4c_GRMHD(ptlc, pmesh);
    M1N0(ptlc, pmesh);
  }

#endif // M1_ENABLED
}

inline void Wave_2O(gra::tasklist::Collection &ptlc,
                    Mesh *pmesh)
{
  for (int stage=1; stage<=ptlc.wave_2o->nstages; ++stage)
  {
    ptlc.wave_2o->DoTaskListOneStage(pmesh, stage);
  }
}

inline void TrackerExtrema(gra::tasklist::Collection &ptlc,
                           gra::triggers::Triggers &trgs,
                           Mesh *pmesh)
{
  // Extrema tracker is based on propagated state vector
  const Real time_end_stage   = pmesh->time+pmesh->dt;
  const Real ncycle_end_stage = pmesh->ncycle+1;

  // Trace AMR state
  pmesh->ptracker_extrema->ReduceTracker();
  pmesh->ptracker_extrema->EvolveTracker();

  if (trgs.IsSatisfied(tvar::tracker_extrema, ovar::user))
  {
    pmesh->ptracker_extrema->WriteTracker(ncycle_end_stage, time_end_stage);
  }
}

}  // namespace gra::evolve

#endif // MAIN_HPP

//
// :D
//
