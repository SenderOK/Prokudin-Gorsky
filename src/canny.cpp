#include <cmath>
#include <vector>
#include <iostream>

#include "canny.h"

using std::vector;
using std::cout;
using std::endl;

const int shifts[8][2] = {
    { 0, -1},
    { 1, -1},
    { 1,  0},
    { 1,  1},
    { 0,  1},
    {-1,  1},
    {-1,  0},
    {-1, -1},
};

const double PI = 3.141592653589793238;
const double STRONG = 6000000000;
const double WEAK = 3000000000;

void build_final_map(Matrix<double> & m)
{
    vector<int> comps(1); 
    vector<bool> has_strong;
    int n_comps = 0;
    
    for (uint i = 0; i < m.n_rows; ++i) {
        for(uint j = 0; j < m.n_cols; ++j) {        
            if (!eq(m(i, j), 0)) {
                bool strong = eq(m(i, j), STRONG);
                bool neigh_lu = (i > 0) && (j > 0) && (!eq(m(i - 1, j - 1), 0));
                bool neigh_u = (i > 0) && (!eq(m(i - 1, j), 0));
                bool neigh_ru = (i > 0) && (j + 1 < m.n_cols) && (!eq(m(i - 1, j + 1), 0));
                bool neigh_l = (j > 0) && (!eq(m(i, j - 1), 0));
                                
                int curr_comp;
                
                if (!neigh_lu && !neigh_u && !neigh_ru && !neigh_l) {
                    comps.push_back(++n_comps);
                    has_strong.push_back(false);
                    curr_comp = n_comps;
                } else if(neigh_lu) {
                    int lu_comp_ind = round(m(i - 1, j - 1));
                    curr_comp = comps[lu_comp_ind];
                } else if (neigh_u) {
                    int u_comp_ind = round(m(i - 1, j));
                    curr_comp = comps[u_comp_ind];
                } else if (neigh_ru) {
                    int ru_comp_ind = round(m(i - 1, j + 1));
                    curr_comp = comps[ru_comp_ind];
                } else {
                    int l_comp_ind = round(m(i, j - 1));
                    curr_comp = comps[l_comp_ind];
                }
                
                m(i, j) = curr_comp;
                has_strong[curr_comp] = has_strong[curr_comp] || strong;
                
                if (neigh_lu) comps[int(round(m(i - 1, j - 1)))] = curr_comp;
                if (neigh_u) comps[int(round(m(i - 1, j)))] = curr_comp;
                if (neigh_ru) comps[int(round(m(i - 1, j + 1)))] = curr_comp;                
                if (neigh_l) comps[int(round(m(i, j - 1)))] = curr_comp;
            }
        }
    }
    
    for (uint i = 0; i < m.n_rows; ++i) {
        for(uint j = 0; j < m.n_cols; ++j) {        
            if (!eq(m(i, j), 0)) {
                int curr_comp = comps[int(round(m(i, j)))];
                if (has_strong[curr_comp]) {
                    m(i, j) = 255;
                } else {
                    m(i, j) = 0;
                }
            }
        }
    }
}

PreciseImage canny(const PreciseImage &m, double threshold1, double threshold2)
{
    uint radius = 2;
    double sigma = 1.4;
    
    PreciseImage tmp = m.unary_map(CustomFilter(GenGaussKernel(0, radius, sigma)));
    PreciseImage im_gauss = tmp.unary_map(CustomFilter(GenGaussKernel(radius, 0, sigma)));
    
    PreciseImage sobel_x_3layers = im_gauss.unary_map(CustomFilter({ {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} } ));
    Matrix<double> sobel_x = sobel_x_3layers.unary_map(WeightPixelFilter(1, 0, 0));
    
    PreciseImage sobel_y_3layers = im_gauss.unary_map(CustomFilter({ {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} } ));
    Matrix<double> sobel_y = sobel_y_3layers.unary_map(WeightPixelFilter(1, 0, 0));
    
    Matrix<double> grad_vals = binary_map(CalcGradValFilter(), sobel_y, sobel_x);
    Matrix<double> grad_dirs = binary_map(CalcGradDirFilter(), sobel_y, sobel_x);
    
    Matrix<double> grad_vals_expanded = grad_vals.MirrorExpand(1, 1);
    for (int i = 1; i < int(grad_vals_expanded.n_rows - 1); ++i) {
        for (int j = 1; j < int(grad_vals_expanded.n_cols - 1); ++j) {
            int usual_i = i - 1, usual_j = j - 1;
            uint shift_num = ((uint(trunc( 8 * grad_dirs(usual_i, usual_j) / PI + 8)) + 1 ) / 2) % 8;
            if (!(grad_vals_expanded(i, j) > grad_vals_expanded(i + shifts[shift_num][0], j + shifts[shift_num][1]) &&
                  grad_vals_expanded(i, j) > grad_vals_expanded(i - shifts[shift_num][0], j - shifts[shift_num][1]))) {
                grad_vals(usual_i, usual_j) = 0;
            }
            
            if (grad_vals(usual_i, usual_j) < threshold1) {
                grad_vals(usual_i, usual_j) = 0;
            } else if (grad_vals(usual_i, usual_j) > threshold2) {
                grad_vals(usual_i, usual_j) = STRONG;
            } else {
                grad_vals(usual_i, usual_j) = WEAK;
            }
        }
    }

    build_final_map(grad_vals);
        
    return ternary_map(UniteThreeFilter(), grad_vals, grad_vals, grad_vals);
}
