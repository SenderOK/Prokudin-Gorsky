#include "alignment.h"
#include "canny.h"

#include <tuple>
#include <vector>

using std::tuple;
using std::get;
using std::vector;
using std::max;
using std::min;
using std::pair;

const double canny_threshold1 = 50;
const double canny_threshold2 = 100;
const double align_border_size = 0.06;
const uint align_vicinity_size = 3;

const int horz_shift = 25;
const int vert_shift = 25;
const uint min_dim = 800;

uint border(const PreciseImage &m, uint border_size, uint vicinity_size)
{
    PreciseImage bord_map = canny(m, canny_threshold1, canny_threshold2);
    uint rows = min(border_size, m.n_rows);
    vector<int> n_border_pixels(rows);
    
    for (uint i = 0; i < rows; ++i)
        for (uint j = 0; j < m.n_cols; ++j){
            double tmp = get<0>(bord_map(i, j));
            if (!eq(tmp, 0))
                ++n_border_pixels[i];
        }

    int first_max = -1, first_max_row = -1;
    for (uint i = 0; i < rows; ++i) {
        if (n_border_pixels[i] > first_max) {
            first_max = n_border_pixels[i];
            first_max_row = i;
        }
    }
    
    int second_max = -1, second_max_row = -1;
    for (uint i = 0; i < rows; ++i) {
        if (i < first_max_row - vicinity_size || i > first_max_row + vicinity_size)
            if (n_border_pixels[i] > second_max) {
                second_max = n_border_pixels[i];
                second_max_row = i;
            }
    }
    
    return max(first_max_row, second_max_row);
}

PreciseImage remove_borders(const PreciseImage &m, double border_size, uint vicinity_size)
{
    uint top_border = border(m, uint(border_size * m.n_rows), vicinity_size);
    PreciseImage turned = m.rotate_clockwise(1);
    uint left_border = border(turned, uint(border_size * m.n_cols), vicinity_size);
    turned = turned.rotate_clockwise(1);
    uint bottom_border = border(turned, uint(border_size * m.n_rows), vicinity_size);
    turned = turned.rotate_clockwise(1);
    uint right_border = border(turned, uint(border_size * m.n_cols), vicinity_size);
    
    uint prow = top_border + 1;
    uint pcol = left_border + 1;
    
    uint rows = (m.n_rows - bottom_border) - top_border - 1;
    uint cols = (m.n_cols - right_border) - left_border - 1;
    
    return m.submatrix(prow, pcol, rows, cols);
}

double MSE(const Matrix<double> & m1, const Matrix<double> & m2)
{
    double ans = 0;
    if(m1.n_rows != m2.n_rows || m1.n_cols != m2.n_cols)
        throw std::string("MSE: matrices must be of the same size");
        
    for (uint i = 0; i < m1.n_rows;++i) 
        for (uint j = 0; j < m1.n_cols; ++j) 
            ans += (m1(i, j) - m2(i, j)) * (m1(i, j) - m2(i, j));
            
    return ans / double(m1.n_rows * m1.n_cols);
}

double CrossCorr(const Matrix<double> & m1, const Matrix<double> & m2)
{
    double ans = 0;
    if(m1.n_rows != m2.n_rows || m1.n_cols != m2.n_cols)
        throw std::string("CrossCorr: matrices must be of the same size");
        
    for (uint i = 0; i < m1.n_rows;++i) 
        for (uint j = 0; j < m1.n_cols; ++j) 
            ans += m1(i, j)*m2(i, j);
            
    return ans;
}


pair<int, int> combine(const PreciseImage &base_im, const PreciseImage &added_im, InterpType interp_type)
{
    vector<Matrix<double>> base, added;
    PreciseImage a = added_im;
    PreciseImage b = base_im;
    
    base.push_back(b.unary_map(WeightPixelFilter(1, 0, 0)));
    added.push_back(a.unary_map(WeightPixelFilter(1, 0, 0)));

    uint n_times = 0;
    while (min(min(min(base[n_times].n_rows, base[n_times].n_cols), added[n_times].n_rows), added[n_times].n_cols) > min_dim) {
        a = resize(a, 0.5, interp_type);
        b = resize(b, 0.5, interp_type);
        base.push_back(b.unary_map(WeightPixelFilter(1, 0, 0)));
        added.push_back(a.unary_map(WeightPixelFilter(1, 0, 0)));
        ++n_times;
    }
    
  
    int vert_l = -vert_shift;
    int vert_r = vert_shift;
    int horz_l = -horz_shift;
    int horz_r = horz_shift;
    
    pair<int, int> ans;    
    for (int it = n_times; it >= 0; --it) {
        double metrics = 1.0e+40; 

        for (int i = vert_l; i < vert_r; ++i) {
            for (int j = horz_l; j < horz_r; ++j) {
                uint pin_base_row = max(i, 0);
                uint pin_base_col = max(j, 0);
                uint pin_added_row = max(-i, 0);
                uint pin_added_col = max(-j, 0);
                
                uint rows_base = base[it].n_rows - pin_base_row;
                uint cols_base = base[it].n_cols - pin_base_col;
                uint rows_added = added[it].n_rows - pin_added_row;
                uint cols_added = added[it].n_cols - pin_added_col;
                
                uint rows = min(rows_base, rows_added);
                uint cols = min(cols_base, cols_added);
                
                Matrix<double> part_base = base[it].submatrix(pin_base_row, pin_base_col, rows, cols);  
                Matrix<double> part_added = added[it].submatrix(pin_added_row, pin_added_col, rows, cols);
                
                double tmp = MSE(part_base, part_added);
   
                if (tmp < metrics) {
                    metrics = tmp;
                    ans = pair<int, int>(i, j);
                }
            }
        }
        vert_l = ans.first * 2;
        vert_r = ans.first * 2 + 1;
        horz_l = ans.second * 2;
        horz_r = ans.second * 2 + 1;
    }
    
    return ans;
}

PreciseImage align(const PreciseImage &m, InterpType interp_type, double subpixels, string postprocessing, double param)
{
    PreciseImage b_channel = m.submatrix(0, 0, m.n_rows / 3, m.n_cols);
    PreciseImage g_channel = m.submatrix(m.n_rows / 3, 0, m.n_rows / 3, m.n_cols);
    PreciseImage r_channel = m.submatrix(2 * (m.n_rows / 3), 0, m.n_rows -  2 * (m.n_rows / 3), m.n_cols);
    
    r_channel = remove_borders(r_channel, align_border_size, align_vicinity_size);
    g_channel = remove_borders(g_channel, align_border_size, align_vicinity_size);
    b_channel = remove_borders(b_channel, align_border_size, align_vicinity_size);
    
//    save_image(r_channel, "r_channel");    save_image(g_channel, "g_channel");    save_image(b_channel, "b_channel");
    
    bool subpix;
    if (eq(subpixels, 0)) {
        subpix = false;
        subpixels = 1.0;
    } else {
        subpix = true;
        r_channel = resize(r_channel, subpixels, interp_type);
        g_channel = resize(g_channel, subpixels, interp_type);
        b_channel = resize(b_channel, subpixels, interp_type);
    }
    
    // now we fix  one channel and count shifts
    pair<int, int> g_shift = combine(r_channel, g_channel, interp_type);
    pair<int, int> b_shift = combine(r_channel, b_channel, interp_type);
    
    uint pin_r_row = max(max(g_shift.first, 0), b_shift.first);
    uint pin_r_col = max(max(g_shift.second, 0), b_shift.second);
    uint pin_g_row = pin_r_row - g_shift.first;
    uint pin_g_col = pin_r_col - g_shift.second;
    uint pin_b_row = pin_r_row - b_shift.first;
    uint pin_b_col = pin_r_col - b_shift.second;
    
    uint r_rows = r_channel.n_rows - pin_r_row;
    uint r_cols = r_channel.n_cols - pin_r_col;
    
    uint g_rows = g_channel.n_rows - pin_g_row;
    uint g_cols = g_channel.n_cols - pin_g_col;
    
    uint b_rows = b_channel.n_rows - pin_b_row;
    uint b_cols = b_channel.n_cols - pin_b_col;

    uint rows = min(min(r_rows, g_rows), b_rows);
    uint cols = min(min(r_cols, g_cols), b_cols);

    Matrix<double> r_layer = r_channel.unary_map(WeightPixelFilter(1, 0, 0));
    Matrix<double> g_layer = g_channel.unary_map(WeightPixelFilter(1, 0, 0));
    Matrix<double> b_layer = b_channel.unary_map(WeightPixelFilter(1, 0, 0));
    r_layer = r_layer.submatrix(pin_r_row, pin_r_col, rows, cols);
    g_layer = g_layer.submatrix(pin_g_row, pin_g_col, rows, cols);
    b_layer = b_layer.submatrix(pin_b_row, pin_b_col, rows, cols);
      
    PreciseImage ans = ternary_map(UniteThreeFilter(), r_layer, g_layer, b_layer); 
    
    if (subpix)
        ans = resize(ans, 1 / subpixels, interp_type);
        
    if (postprocessing == "--gray-world") {
        ans = gray_world(ans);
    } else if (postprocessing == "--unsharp") {
        ans = ans.unary_map(CustomFilter({ {-1.0/6, -2.0/3, -1.0/6}, {-2.0/3, 13.0/3, -2.0/3}, {-1.0/6, -2.0/3, -1.0/6} } ));
    } else {
        ans = autocontrast(ans, param);
    }
        
    return ans;
}
