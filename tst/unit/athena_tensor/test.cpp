#include <cassert>
#include <cstdlib>
#include <iostream>

#include "athena_arrays.hpp"
#include "athena_tensor.hpp"

using namespace std;

int main(int argc, char ** argv) {
  // check definition of vectors
  AthenaTensor<float, TensorSymm::none, 3, 1> vel;
  assert(vel.ndof() == 3);

  // check definition of symmetric tensors
  AthenaTensor<float, TensorSymm::sym2, 3, 2> metric;
  assert(metric.ndof() == 6);

  // check definition of doubly-symmetric tensors
  AthenaTensor<float, TensorSymm::sym22, 3, 4> ddg;
  for(int a = 0; a < 3; ++a)
  for(int b = 0; b < 3; ++b) {
    for(int c = 0; c < 3; ++c) {
      for(int d = 0; d < 3; ++d) {
        assert(ddg.idxmap(a,b,c,d) == ddg.idxmap(b,a,c,d));
        assert(ddg.idxmap(a,b,c,d) == ddg.idxmap(a,b,d,c));
        assert(ddg.idxmap(a,b,c,d) == ddg.idxmap(b,a,d,c));
      }
    }
  }

  return EXIT_SUCCESS;
}
