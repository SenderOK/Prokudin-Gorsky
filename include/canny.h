#pragma once

#include "filters.h"
#include "matrix.h"

#include <tuple>
using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;

PreciseImage canny(const PreciseImage &m, double threshold1, double threshold2);

inline bool eq(double a, double b)
{
    return fabs(a - b) < 0.0000000001;
}
