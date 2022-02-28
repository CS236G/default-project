#include <pybind11/pybind11.h>

#include "matchcost/matchcost.hpp"
#include "nmdistance/nmdistance.hpp"

PYBIND11_MODULE(_metrics_backend, m) {
  m.def("nmdistance_forward", &nmdistance_forward, "NmDistance forward (CUDA)");
  m.def("nmdistance_backward", &nmdistance_backward,
        "NmDistance backward (CUDA)");
  m.def("approxmatch_forward", &approxmatch_forward,
        "ApproxMatch forward (CUDA)");
  m.def("matchcost_forward", &matchcost_forward, "MatchCost forward (CUDA)");
  m.def("matchcost_backward", &matchcost_backward, "MatchCost backward (CUDA)");
}
