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

const TaskID USERWORK(18);

const TaskID NEW_DT(19);
const TaskID FLAG_AMR(20);

const TaskID UPDATE_COUPLING(21);

}  // TaskNames::M1::M1N0

#endif  // M1_TASK_NAMES_HPP_
