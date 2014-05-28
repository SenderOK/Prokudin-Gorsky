#pragma once

#include <tuple>
using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;

#include "matrix.h"
#include "io.h"

PreciseImage gray_world(const PreciseImage &m);
PreciseImage autocontrast(const PreciseImage &m, double f);

enum InterpType {
    NEIGHBOUR = 0,
    BILINEAR = 1,
    BICUBIC = 2,
};

PreciseImage resize(const PreciseImage &m, double scale, InterpType type);
