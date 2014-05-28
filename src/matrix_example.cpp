#include <iostream>

#include "matrix.h"
#include "io.h"
#include "filters.h"

using std::cout;
using std::endl;

using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;

// Matrix usage example
// Also see: matrix.h, matrix.hpp for comments on how filtering works

class BoxFilterOp
{
public:
    tuple<uint, uint, uint> operator () (const Image &m) const
    {
        uint size = 2 * r_vert + 1;
        uint r, g, b, sum_r = 0, sum_g = 0, sum_b = 0;
        for (uint i = 0; i < size; ++i) {
            for (uint j = 0; j < size; ++j) {
                // Tie is useful for taking elements from tuple
                tie(r, g, b) = m(i, j);
                sum_r += r;
                sum_g += g;
                sum_b += b;
            }
        }
        auto norm = size * size;
        sum_r /= norm;
        sum_g /= norm;
        sum_b /= norm;
        return make_tuple(sum_r, sum_g, sum_b);
    }
    static const uint r_vert = 1;
    static const uint r_horz = 1;
};

int main(int argc, char **argv)
{
    // Image = Matrix<tuple<uint, uint, uint>>
    // tuple is from c++ 11 standard
    
    //Image img = load_image<uint>(argv[1]);
 
    //Image img2 = img.unary_map(BoxFilterOp());
    //save_image(img2, argv[2]);
    //cout << img2;
    //const double c = 1/9.0;
    //Image img3 = img.unary_map(CustomFilter(Matrix<double>({1})));
    
    //Image img3 = img.unary_map(MedianFilter(1, 1));    
    //save_image(img3, "MedianOPPAN111");

    //Image img4 = ConstantMedianFilter(img, 1);
    //save_image(img4, "ConstantMedianOPPAN111");
    
    //img3 = load_image("Median");
    //img4 = load_image("LinearMedian");
    
    Image img3 = load_image<uint>("lena.bmp");
    Image img4 = load_image<uint>("lena_autocontrast.bmp");

    uint n_errors = 0;
    for (uint i = 0; i < img3.n_rows; ++i) {
        for (uint j = 0; j < img3.n_cols; ++j) {
            uint r1, r2, g1, g2, b1, b2;
            tie(r1, g1, b1) = img3(i, j);
            tie(r2, g2, b2) = img4(i, j);
            if (r1 != r2 || g1 != g2 || b1 != b2) {
                ++n_errors;//cout << i << " " << j << endl;
            }
        }
    }    
    cout << n_errors << endl;


    
    //Matrix<int> m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    //try{
    //cout << m.MirrorExpand(1, 0);
    //} catch(std::string &s) {
    //    cout << s << endl;
    //}
    
    //cout << GenGaussKernel(2, 2, 1);
    
    return 0;
}
