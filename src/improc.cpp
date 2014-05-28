#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "improc.h"
#include "filters.h"

using std::vector;

const double eps = 0.000001;

PreciseImage gray_world(const PreciseImage &m)
{
    uint n_rows = m.n_rows;
    uint n_cols = m.n_cols;
    PreciseImage result(n_rows, n_cols);
    
    double r, g, b;
    double sum_r = 0, sum_g = 0, sum_b = 0;
    for (uint i = 0; i < n_rows; ++i) {
        for (uint j = 0; j < n_cols; ++j) {
            tie(r, g, b) = m(i, j);
            sum_r = sum_r + r;
            sum_g = sum_g + g;
            sum_b = sum_b + b;
        }
    }
    
    double sum = (sum_r + sum_g + sum_b) / 3;
    
    for (uint i = 0; i < n_rows; ++i) {
        for (uint j = 0; j < n_cols; ++j) {
            tie(r, g, b) = m(i, j);
            double ans_r = (fabs(sum_r) < eps) ? 255 : (r * sum) / sum_r;
            double ans_g = (fabs(sum_g) < eps) ? 255 : (g * sum) / sum_g;
            double ans_b = (fabs(sum_b) < eps) ? 255 : (b * sum) / sum_b;
            result(i, j) = make_tuple(ans_r, ans_g, ans_b);
        }
    }  
    
    return result;
}

//______________________________________________________________________________

PreciseImage autocontrast(const PreciseImage &m, double f)
{
    Matrix<double> brightness = m.unary_map(WeightPixelFilter(0.2125, 0.7154, 0.0721));
    Matrix<uint> brightness_i(m.n_rows, m.n_cols);
    PreciseImage result(m.n_rows, m.n_cols);
    uint ignore = uint((m.n_rows * m.n_cols) * f);
    uint hist[256] = {0};
    
    for (uint i = 0; i < m.n_rows; ++i) {
        for (uint j = 0; j < m.n_cols; ++j) {
            uint tmp = uint(std::min(std::max(brightness(i, j), 0.0), 255.0));   
            brightness_i(i, j) = tmp; 
            ++hist[tmp];
        }
    }
         
    uint pos = 0, brightness_min = 0, brightness_max = 255;
    uint need2gnore = ignore;
    while (need2gnore) {
        if (hist[pos] <= need2gnore) {
            need2gnore -= hist[pos];
        } else {
            need2gnore = 0; 
            brightness_min = pos;
        }
        ++pos;
    }
    
    pos = 255;
    need2gnore = ignore;
    while (need2gnore) {
        if (hist[pos] <= need2gnore) {
            need2gnore -= hist[pos];
        } else {
            need2gnore = 0; 
            brightness_max = pos;
        }
        --pos;
    }
    
    if (brightness_max == brightness_min)
        throw std::string("autocontrast failed: there is only one type of brightness left");
    
    double r, g, b; 
    for (uint i = 0; i < m.n_rows; ++i) {
        for (uint j = 0; j < m.n_cols; ++j) {
            tie(r, g, b) = m(i, j);
            if (brightness_i(i, j) < brightness_min) {
                result(i, j) = make_tuple(0, 0, 0);
            } else if(brightness_i(i, j) > brightness_max) {
                result(i, j) = make_tuple(255, 255, 255);
            } else {
                double new_brightness = (brightness_i(i, j) - brightness_min) * (255.0 / (brightness_max - brightness_min));
                double coeff = new_brightness / brightness_i(i, j);
                result(i, j) = make_tuple(r * coeff, g * coeff, b * coeff);
            }
        }
    }
            
    return result;
}

//______________________________________________________________________________

typedef PrecisePixel (*fptr) (const PreciseImage &m, double virt_row, double virt_col);

PrecisePixel neighbour(const PreciseImage &m, double virt_row, double virt_col)
{
    uint row = uint(std::floor(virt_row));
    uint col = uint(std::floor(virt_col));
    return m(row, col);
}

double linear_val(double x1, double y1, double x2, double y2, double x)
{
    return y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
}

double bilinear_val(double row1, double row2, double col1, double col2, double v11, double v12, double v21, double v22, double r, double c)
{
    double a = linear_val(col1, v11, col2, v12, c);
    double b = linear_val(col1, v21, col2, v22, c);
    return linear_val(row1, a, row2, b, r);
}

PrecisePixel bilinear(const PreciseImage &m, double virt_row, double virt_col)
{
    virt_row += 0.5; 
    virt_col += 0.5;
    uint row = uint(std::floor(virt_row));
    uint col = uint(std::floor(virt_col));
    
    double r11, g11, b11, r12, g12, b12, r21, g21, b21, r22, g22, b22;
    tie(r11, g11, b11) = m(row, col);
    tie(r12, g12, b12) = m(row, col + 1);
    tie(r21, g21, b21) = m(row + 1, col);
    tie(r22, g22, b22) = m(row + 1, col + 1);
    
    double r, g, b;
    r = bilinear_val(row, row + 1, col, col + 1, r11, r12, r21, r22, virt_row, virt_col);
    g = bilinear_val(row, row + 1, col, col + 1, g11, g12, g21, g22, virt_row, virt_col);
    b = bilinear_val(row, row + 1, col, col + 1, b11, b12, b21, b22, virt_row, virt_col);
    
    return make_tuple(r, g, b);
}

double cubic_val(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4, double x)
{
    // Lagrange's polynoms
    double p1, p2, p3, p4;
    p1 = ((x - x2) / (x1 - x2)) * ((x - x3) / (x1 - x3)) * ((x - x4) / (x1 - x4));
    p2 = ((x - x1) / (x2 - x1)) * ((x - x3) / (x2 - x3)) * ((x - x4) / (x2 - x4));
    p3 = ((x - x1) / (x3 - x1)) * ((x - x2) / (x3 - x2)) * ((x - x4) / (x3 - x4));
    p4 = ((x - x1) / (x4 - x1)) * ((x - x2) / (x4 - x2)) * ((x - x3) / (x4 - x3));
    return y1 * p1 + y2 * p2 + y3 * p3 + y4 * p4;
}

double bicubic_val(double row, double col, const Matrix<double> &m, double r, double c)
{
    vector<double> v(4);
    for (uint i = 0; i < 4; ++i)
       v[i] = cubic_val(col, m(i, 0), col + 1, m(i, 1), col + 2, m(i, 2), col + 3, m(i, 3), c);

    return cubic_val(row, v[0], row + 1, v[1], row + 2, v[2], row + 3, v[4], r);
}

PrecisePixel bicubic(const PreciseImage &m, double virt_row, double virt_col)
{
    virt_row += 1.5; 
    virt_col += 1.5;
    uint row = uint(std::floor(virt_row));
    uint col = uint(std::floor(virt_col));
    
    PreciseImage im = m.submatrix(row - 1, col - 1, 4, 4);
    double r = bicubic_val(row - 1, col - 1, im.unary_map(WeightPixelFilter(1, 0, 0)), virt_row, virt_col);
    double g = bicubic_val(row - 1, col - 1, im.unary_map(WeightPixelFilter(0, 1, 0)), virt_row, virt_col);
    double b = bicubic_val(row - 1, col - 1, im.unary_map(WeightPixelFilter(0, 0, 1)), virt_row, virt_col);
       
    return make_tuple(r, g, b);
}

PreciseImage resize(const PreciseImage &m, double scale, InterpType type)
{
    uint res_nrows = uint(m.n_rows * scale);
    uint res_ncols = uint(m.n_cols * scale);
    
    PreciseImage im = m.MirrorExpand(type, type);    
    PreciseImage result(res_nrows, res_ncols);
    
    fptr funcs[3] = {neighbour, bilinear, bicubic};
    fptr f = funcs[type];
    
    for (uint i = 0; i < res_nrows; ++i)
        for (uint j = 0; j < res_ncols; ++j)
            result(i, j) = f(im, (i + (type == NEIGHBOUR ? 0.0 : 0.5))/scale, 
                                 (j + (type == NEIGHBOUR ? 0.0 : 0.5))/scale);                            
    
    return result;   
}
