#ifndef M1_TASK_NAMES_HPP_
#define M1_TASK_NAMES_HPP_

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../task_list.hpp"

namespace TaskNames::M1::M1N0 {

const TaskID NONE(0);
const TaskID CLEAR_ALLBND(1);

const TaskID UPDATE_BG(2);

const TaskID CALC_FIDU(3);
const TaskID CALC_CLOSURE(4);
const TaskID CALC_FIDU_FRAME(5);
const TaskID CALC_OPAC(6);

const TaskID CALC_FLUX(7);
const TaskID SEND_FLUX(8);
const TaskID RECV_FLUX(9);

const TaskID ADD_FLX_DIV(10);
const TaskID ADD_GRSRC(11);
const TaskID CALC_UPDATE(12);

const TaskID SEND(13);
const TaskID RECV(14);
const TaskID SETB(15);
const TaskID PROLONG(16);
const TaskID PHY_BVAL(17);

const TaskID ANALYSIS(18);

const TaskID USERWORK(19);

const TaskID NEW_DT(20);
const TaskID FLAG_AMR(21);

const TaskID UPDATE_COUPLING(22);

// MHD re-scatter tasks (active when Z4C_ENABLED && FLUID_ENABLED).
// Embedded in the M1N0 DAG so MHD ghost exchange overlaps M1 analysis.
// Only wired when the monolithic GRMHD path is selected (embed_mhd_rescatter).
#if Z4C_ENABLED && FLUID_ENABLED
const TaskID CONS2PRIMP_HYD(23);
const TaskID SEND_HYD(24);
const TaskID RECV_HYD(25);
const TaskID SETB_HYD(26);
const TaskID PROLONG_HYD(27);
const TaskID PHY_BVAL_HYD(28);
const TaskID CONS2PRIMG_HYD(29);
const TaskID UPDATE_SRC_HYD(30);
const TaskID CLEAR_MAININT(31);
#endif

}  // TaskNames::M1::M1N0

namespace TaskNames::M1::PostAMR_M1N0 {

const TaskID NONE(0);

const TaskID UPDATE_BG(2);

const TaskID CALC_FIDU(3);
const TaskID CALC_CLOSURE(4);

const TaskID CALC_FIDU_FRAME(5);

const TaskID CALC_OPAC(6);

const TaskID ANALYSIS(7);

}  // namespace TaskNames::M1::PostAMR_M1N0

#endif  // M1_TASK_NAMES_HPP_
