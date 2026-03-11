//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their history.

// C headers

// C++ headers
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

// Athena++ headers
#include "../defs.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../z4c/z4c.hpp"
#include "../wave/wave.hpp"
#include "../mesh/mesh.hpp"
#include "../scalars/scalars.hpp"
#include "outputs.hpp"

// NEW_OUTPUT_TYPES:

// "3" for 1-KE, 2-KE, 3-KE additional columns (come before tot-E)
#define NHISTORY_VARS (((NHYDRO) + 3) * (FLUID_ENABLED) + \
                       (NFIELD) + (NSCALARS) + \
                       3 * (WAVE_ENABLED) + \
                       8 * (Z4C_ENABLED))

// Index of the WAVE "err-max-pw" slot (a max, not a sum).
// Only meaningful when WAVE_ENABLED=1.
#define NHISTORY_WAVE_ABSMA_IDX (((NHYDRO) + 3) * (FLUID_ENABLED) + \
                                 (NFIELD) + (NSCALARS) + 2)

//----------------------------------------------------------------------------------------
//! \fn void OutputType::HistoryFile()
//  \brief Writes a history file
void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag)
{
  MeshBlock *pmb = pm->pblock;
  Real real_max = std::numeric_limits<Real>::max();
  Real real_lowest = std::numeric_limits<Real>::lowest();
  AthenaArray<Real> vol(pmb->ncells1);

  const int nuser_history_output_ = pm->user_history_func_.size();
  const int nhistory_output = NHISTORY_VARS + nuser_history_output_;

  std::unique_ptr<Real[]> hst_data(new Real[nhistory_output]);
  // initialize built-in variable sums to 0.0
  for (int n=0; n<NHISTORY_VARS; ++n) hst_data[n] = 0.0;
  // initialize user-defined history outputs depending on the requested operation
  for (int n=0; n<nuser_history_output_; n++) {
    switch (pm->user_history_ops_[n]) {
      case UserHistoryOperation::sum:
        hst_data[NHISTORY_VARS+n] = 0.0;
        break;
      case UserHistoryOperation::max:
        hst_data[NHISTORY_VARS+n] = real_lowest;
        break;
      case UserHistoryOperation::min:
        hst_data[NHISTORY_VARS+n] = real_max;
        break;
    }
  }

  int ix_cons_dens = 0, ix_cons_scalar = 0;

  // Loop over MeshBlocks
  while (pmb != nullptr) {
    Hydro *phyd = pmb->phydro;
    Field *pfld = pmb->pfield;
    PassiveScalars *psclr = pmb->pscalars;
    Wave *pwave = pmb->pwave;
    Z4c *pz4c = pmb->pz4c;


    Real abs_ma = 0;
    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          // NEW_OUTPUT_TYPES:

          int isum = 0;
          if (FLUID_ENABLED) {
            // Hydro conserved variables:
            Real& u_d  = phyd->u(IDN,k,j,i);
            Real& u_mx = phyd->u(IM1,k,j,i);
            Real& u_my = phyd->u(IM2,k,j,i);
            Real& u_mz = phyd->u(IM3,k,j,i);

#if FLUID_ENABLED
            if (pm->presc->opt.rescale_conserved_density)
            {
              ix_cons_dens = isum;
            }
#endif

            hst_data[isum++] += vol(i)*u_d;
            hst_data[isum++] += vol(i)*u_mx;
            hst_data[isum++] += vol(i)*u_my;
            hst_data[isum++] += vol(i)*u_mz;
            // + partitioned KE by coordinate direction:
            hst_data[isum++] += vol(i)*0.5*SQR(u_mx)/u_d;
            hst_data[isum++] += vol(i)*0.5*SQR(u_my)/u_d;
            hst_data[isum++] += vol(i)*0.5*SQR(u_mz)/u_d;

            if (NON_BAROTROPIC_EOS) {
              Real& u_e = phyd->u(IEN,k,j,i);;
              hst_data[isum++] += vol(i)*u_e;
            }
            // Cell-centered magnetic energy, partitioned by coordinate direction:
            if (MAGNETIC_FIELDS_ENABLED) {
              Real& bcc1 = pfld->bcc(IB1,k,j,i);
              Real& bcc2 = pfld->bcc(IB2,k,j,i);
              Real& bcc3 = pfld->bcc(IB3,k,j,i);
              // constexpr int prev_out = NHYDRO + 3;
              hst_data[isum++] += vol(i)*0.5*bcc1*bcc1;
              hst_data[isum++] += vol(i)*0.5*bcc2*bcc2;
              hst_data[isum++] += vol(i)*0.5*bcc3*bcc3;
            }

            // (conserved variable) Passive scalars:
#if FLUID_ENABLED
            if (NSCALARS > 0)
            {
              if (pm->presc->opt.rescale_conserved_scalars)
              {
                ix_cons_scalar = isum;
              }
            }
#endif

            for (int n=0; n<NSCALARS; n++)
            {
              Real& s = psclr->s(n,k,j,i);
              // constexpr int prev_out = NHYDRO + 3 + NFIELD;
              hst_data[isum++] += vol(i)*s;
            }
          }

          if (WAVE_ENABLED)
          {
            Real& wave_error = pwave->error(k,j,i);
            abs_ma = (abs_ma < std::abs(wave_error)) ? std::abs(wave_error) : abs_ma;
            hst_data[isum++] += vol(i)*std::abs(wave_error);
            hst_data[isum++] += vol(i)*SQR(wave_error);
            isum++; // skip abs_ma slot (filled per-MeshBlock below)
          }

          // BD: TODO - compute the norm properly;
          // The numerical quadratures on a spatial hypersurface should be
          // det(\gamma) weighted - using conformal factor this is cheap to do
          if (Z4C_ENABLED)
          {
            const Real x1 = pmb->pcoord->x1v(i);
            const Real x2 = pmb->pcoord->x2v(j);
            const Real x3 = pmb->pcoord->x3v(k);
            const Real R2 = SQR(x1) + SQR(x2) + SQR(x3);
            const Real r2_max = SQR(pz4c->opt.r_max_con);

            if (R2 <= r2_max)
            {
              Real const H_err  = std::abs(pz4c->con.H(k,j,i));
              Real const M2_err = std::abs(pz4c->con.M(k,j,i));
              Real const Mx_err = std::abs(pz4c->con.M_d(0,k,j,i));
              Real const My_err = std::abs(pz4c->con.M_d(1,k,j,i));
              Real const Mz_err = std::abs(pz4c->con.M_d(2,k,j,i));
              Real const Z2_err = std::abs(pz4c->con.Z(k,j,i));
              Real const theta  = std::abs(pz4c->z4c.Theta(k,j,i));
              Real const C2_err = std::abs(pz4c->con.C(k,j,i));

              hst_data[isum++] += vol(i)*SQR(H_err);
              hst_data[isum++] += vol(i)*M2_err; //M is already squared
              hst_data[isum++] += vol(i)*SQR(Mx_err);
              hst_data[isum++] += vol(i)*SQR(My_err);
              hst_data[isum++] += vol(i)*SQR(Mz_err);
              hst_data[isum++] += vol(i)*Z2_err; //Z is already squared
              hst_data[isum++] += vol(i)*SQR(theta);
              hst_data[isum++] += vol(i)*C2_err; //C is already squared
            }
            else
            {
              isum += 8;
            }
          }

        }
      }
    }

    // Update the WAVE abs_ma slot with the running max across MeshBlocks
    // (this is a max-reduction quantity, not a sum)
    if (WAVE_ENABLED) {
      hst_data[NHISTORY_WAVE_ABSMA_IDX] =
          std::max(abs_ma, hst_data[NHISTORY_WAVE_ABSMA_IDX]);
    }

    for (int n=0; n<nuser_history_output_; ++n)
    {
      Real usr_val = pm->user_history_func_[n](pmb, n);
      switch (pm->user_history_ops_[n])
      {
        case UserHistoryOperation::sum:
          // TODO(felker): this should automatically volume-weight the sum, like the
          // built-in variables. But existing user-defined .hst fns are currently
          // weighting their returned values.
          hst_data[NHISTORY_VARS+n] += usr_val;
          break;
        case UserHistoryOperation::max:
          hst_data[NHISTORY_VARS+n] = std::max(usr_val, hst_data[NHISTORY_VARS+n]);
          break;
        case UserHistoryOperation::min:
          hst_data[NHISTORY_VARS+n] = std::min(usr_val, hst_data[NHISTORY_VARS+n]);
          break;
      }
    }
    pmb = pmb->next;
  }  // end loop over MeshBlocks

#ifdef MPI_PARALLEL
  // The WAVE abs_ma slot is a max-reduction, not a sum. Save it before
  // the MPI_SUM reduction so we can reduce it separately with MPI_MAX.
  Real wave_absma_local = 0.0;
  if (WAVE_ENABLED) {
    wave_absma_local = hst_data[NHISTORY_WAVE_ABSMA_IDX];
  }

  // sum built-in/predefined hst_data[] over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, hst_data.get(), NHISTORY_VARS, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(hst_data.get(), hst_data.get(), NHISTORY_VARS, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  }

  // Correct the WAVE abs_ma slot: reduce with MPI_MAX instead of MPI_SUM
  if (WAVE_ENABLED) {
    if (Globals::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &wave_absma_local, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      hst_data[NHISTORY_WAVE_ABSMA_IDX] = wave_absma_local;
    } else {
      MPI_Reduce(&wave_absma_local, &wave_absma_local, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
                 MPI_COMM_WORLD);
    }
  }
  // Batch user-defined history reductions: partition into SUM/MAX/MIN groups
  // and issue at most 3 MPI_Reduce calls instead of N individual ones.
  if (nuser_history_output_ > 0) {
    // Count entries per operation type
    int n_sum = 0, n_max = 0, n_min = 0;
    for (int n = 0; n < nuser_history_output_; ++n) {
      switch (pm->user_history_ops_[n]) {
        case UserHistoryOperation::sum: ++n_sum; break;
        case UserHistoryOperation::max: ++n_max; break;
        case UserHistoryOperation::min: ++n_min; break;
      }
    }
    // Stack buffers (user history outputs are typically few, <20)
    constexpr int kMaxStack = 256;
    Real buf_sum_s[kMaxStack], buf_max_s[kMaxStack], buf_min_s[kMaxStack];
    int  idx_sum_s[kMaxStack], idx_max_s[kMaxStack], idx_min_s[kMaxStack];
    // Use heap only if needed
    std::unique_ptr<Real[]> buf_sum_h, buf_max_h, buf_min_h;
    std::unique_ptr<int[]>  idx_sum_h, idx_max_h, idx_min_h;
    Real *buf_sum, *buf_max, *buf_min;
    int  *idx_sum, *idx_max, *idx_min;
    if (n_sum <= kMaxStack) { buf_sum = buf_sum_s; idx_sum = idx_sum_s; }
    else { buf_sum_h.reset(new Real[n_sum]); idx_sum_h.reset(new int[n_sum]);
           buf_sum = buf_sum_h.get(); idx_sum = idx_sum_h.get(); }
    if (n_max <= kMaxStack) { buf_max = buf_max_s; idx_max = idx_max_s; }
    else { buf_max_h.reset(new Real[n_max]); idx_max_h.reset(new int[n_max]);
           buf_max = buf_max_h.get(); idx_max = idx_max_h.get(); }
    if (n_min <= kMaxStack) { buf_min = buf_min_s; idx_min = idx_min_s; }
    else { buf_min_h.reset(new Real[n_min]); idx_min_h.reset(new int[n_min]);
           buf_min = buf_min_h.get(); idx_min = idx_min_h.get(); }
    // Gather values into per-op buffers and record original indices
    int is = 0, ix = 0, in = 0;
    for (int n = 0; n < nuser_history_output_; ++n) {
      Real val = hst_data[NHISTORY_VARS + n];
      switch (pm->user_history_ops_[n]) {
        case UserHistoryOperation::sum:
          idx_sum[is] = n; buf_sum[is] = val; ++is; break;
        case UserHistoryOperation::max:
          idx_max[ix] = n; buf_max[ix] = val; ++ix; break;
        case UserHistoryOperation::min:
          idx_min[in] = n; buf_min[in] = val; ++in; break;
      }
    }
    // Issue at most 3 MPI_Reduce calls
    auto batch_reduce = [](Real *buf, int count, MPI_Op op) {
      if (count <= 0) return;
      if (Globals::my_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, buf, count, MPI_ATHENA_REAL, op, 0,
                   MPI_COMM_WORLD);
      } else {
        MPI_Reduce(buf, buf, count, MPI_ATHENA_REAL, op, 0,
                   MPI_COMM_WORLD);
      }
    };
    batch_reduce(buf_sum, n_sum, MPI_SUM);
    batch_reduce(buf_max, n_max, MPI_MAX);
    batch_reduce(buf_min, n_min, MPI_MIN);
    // Scatter results back to hst_data (only rank 0 needs correct values,
    // but writing back on all ranks is harmless and keeps code simple)
    for (int i = 0; i < n_sum; ++i)
      hst_data[NHISTORY_VARS + idx_sum[i]] = buf_sum[i];
    for (int i = 0; i < n_max; ++i)
      hst_data[NHISTORY_VARS + idx_max[i]] = buf_max[i];
    for (int i = 0; i < n_min; ++i)
      hst_data[NHISTORY_VARS + idx_min[i]] = buf_min[i];
  }
#endif

#if FLUID_ENABLED
  // use also compensated summation when computing hst and adjusting cons. ----
  if (pm->presc->opt.rescale_conserved_density)
  {
    hst_data[ix_cons_dens] = pm->presc->IntegrateField(
      gra::hydro::rescaling::variety_cs::conserved_hydro, IDN, 0
    );
  }

  if (pm->presc->opt.rescale_conserved_scalars)
  {
    for (int n=0; n<NSCALARS; ++n)
    {

      hst_data[ix_cons_scalar+n] = pm->presc->IntegrateField(
        gra::hydro::rescaling::variety_cs::conserved_scalar, n, 0
      );
    }
  }
#endif
  // --------------------------------------------------------------------------

  // only the master rank writes the file
  // create filename: "file_basename" + ".hst".  There is no file number.
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign(output_params.file_basename);
    fname.append(".hst");

    // open file for output
    FILE *pfile;
    std::stringstream msg;

    // This bool allows to rewrite header below, and is useful
    // if the output folder changes from the restart.
    // This should be harmless otherwise.
    bool new_file = true;
    if (access(fname.c_str(), F_OK) == 0) {
      //printf("Found %s!\n", fname.c_str());
      new_file = false;
    }

    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      msg << "### FATAL ERROR in function [OutputType::HistoryFile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      ATHENA_ERROR(msg);
    }

    // If this is the first output, write header
    if (output_params.file_number == 0 || new_file) {
      // NEW_OUTPUT_TYPES:

      int iout = 1;
      // descriptor + hash is first line ---
      std::string ver("# GR-Athena++ (");
      ver.append(GIT_HASH);
      ver.append(") history data\n");
      std::fprintf(pfile,"%s", ver.c_str());
      // -----------------------------------

      std::fprintf(pfile,"# [%d]=time ", iout++);
      std::fprintf(pfile,"[%d]=dt ", iout++);
      std::fprintf(pfile,"[%d]=N_MeshBlock ", iout++);
      if (FLUID_ENABLED) {
        std::fprintf(pfile,"[%d]=mass ", iout++);
        std::fprintf(pfile,"[%d]=1-mom ", iout++);
        std::fprintf(pfile,"[%d]=2-mom ", iout++);
        std::fprintf(pfile,"[%d]=3-mom ", iout++);
        std::fprintf(pfile,"[%d]=1-KE ", iout++);
        std::fprintf(pfile,"[%d]=2-KE ", iout++);
        std::fprintf(pfile,"[%d]=3-KE ", iout++);
        if (NON_BAROTROPIC_EOS) std::fprintf(pfile,"[%d]=tot-E ", iout++);
        if (MAGNETIC_FIELDS_ENABLED) {
          std::fprintf(pfile,"[%d]=1-ME ", iout++);
          std::fprintf(pfile,"[%d]=2-ME ", iout++);
          std::fprintf(pfile,"[%d]=3-ME ", iout++);
        }
        for (int n=0; n<NSCALARS; n++) {
          std::fprintf(pfile,"[%d]=%d-scalar ", iout++, n);
        }
      }

      if (WAVE_ENABLED) {
        std::fprintf(pfile,"[%d]=err-norm1 ", iout++);
        std::fprintf(pfile,"[%d]=err-norm2 ", iout++);
        std::fprintf(pfile,"[%d]=err-max-pw ", iout++);
      }

      if (Z4C_ENABLED) {
        std::fprintf(pfile,"[%d]=H-norm2 ",     iout++);
        std::fprintf(pfile,"[%d]=M-norm2 ",     iout++);
        std::fprintf(pfile,"[%d]=Mx-norm2 ",    iout++);
        std::fprintf(pfile,"[%d]=My-norm2 ",    iout++);
        std::fprintf(pfile,"[%d]=Mz-norm2 ",    iout++);
        std::fprintf(pfile,"[%d]=Z-norm2 ",     iout++);
        std::fprintf(pfile,"[%d]=Theta-norm2 ", iout++);
        std::fprintf(pfile,"[%d]=C-norm2 ",     iout++);
      }

      for (int n=0; n<nuser_history_output_; n++)
        std::fprintf(pfile,"[%d]=%s ", iout++,
                     pm->user_history_output_names_[n].c_str());
      std::fprintf(pfile,"\n");                              // terminate line
    }

    // write history variables
    std::fprintf(pfile, output_params.data_format.c_str(), pm->time);
    std::fprintf(pfile, output_params.data_format.c_str(), pm->dt);
    std::fprintf(pfile, " %d", pm->nbtotal);
    for (int n=0; n<nhistory_output; ++n)
      std::fprintf(pfile, output_params.data_format.c_str(), hst_data[n]);
    std::fprintf(pfile,"\n"); // terminate line
    std::fclose(pfile);
  }

  // increment counters, clean up
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  return;
}
