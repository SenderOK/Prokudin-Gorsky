#pragma once

#include <string>

#include "filters.h"
#include "matrix.h"
#include "improc.h"

using std::string;

PreciseImage align(const PreciseImage &m, InterpType interp_type, double subpixels, string postprocessing, double param = 0.0);
