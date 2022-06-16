#ifndef VECPAR_DATA_TYPES_HPP
#define VECPAR_DATA_TYPES_HPP

#include "vecpar/core/definitions/common.hpp"

struct X {
  int a;
  double b;

  TARGET double f() { return a * b; }

  TARGET int square_a() { return a * a; }
};

#endif // VECPAR_DATA_TYPES_HPP
