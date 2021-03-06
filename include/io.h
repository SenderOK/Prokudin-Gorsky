#pragma once

#include "matrix.h"
#include "EasyBMP.h"

#include <tuple>

typedef std::tuple<uint, uint, uint> Pixel;
typedef Matrix<std::tuple<uint, uint, uint>> Image;

typedef std::tuple<double, double, double> PrecisePixel;
typedef Matrix<std::tuple<double, double, double>> PreciseImage;

std::ostream &operator<<(std::ostream &out, const std::tuple<uint, uint, uint> &t);

template <class T>
Matrix< std::tuple<T, T, T> > load_image(const char *path);

template <class T>
void save_image(const Matrix<std::tuple<T, T, T>> &im, const char *path);

#include "io.hpp"
