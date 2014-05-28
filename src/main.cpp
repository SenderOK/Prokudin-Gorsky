#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <vector>

using std::string;
using std::vector;
using std::stringstream;
using std::cout;
using std::cerr;
using std::endl;
using std::numeric_limits;

#include "io.h"
#include "matrix.h"
#include "filters.h"
#include "improc.h"
#include "canny.h"
#include "alignment.h"

void print_help(const char *argv0)
{
    const char *usage =
R"(where PARAMS are from list:

--align [--bicubic-interp] [--subpixel <subpixels>] [--gray-world | --unsharp | --autocontrast [<fraction>]]
    align images with one of postprocessing functions
    bicubic interpolation or subpixel alignment can be switched on for alignment

--gaussian <sigma> [<radius>=3 * sigma]
    gaussian blur of image, 0.1 < sigma < 100, radius = 1, 2, ...

--gaussian-separable <sigma> [<radius>=3 * sigma]
    same, but gaussian is separable

--sobel-x
    Sobel x derivative of image

--sobel-y
    Sobel y derivative of image

--unsharp
    sharpen image

--gray-world
    gray world color balancing

--autocontrast [<fraction>=0.0]
    autocontrast image. <fraction> of pixels must be croped for robustness

--resize <scale> [--bicubic-interp]
    resize image with factor scale. scale is real number > 0.
    

--canny <threshold1> <threshold2>
    apply Canny filter to grayscale image. threshold1 < threshold2,
    both are in 0..360

--custom <kernel_string>
    convolve image with custom kernel, which is given by kernel_string, example:
    kernel_string = '1,2,3;4,5,6;7,8,9' defines kernel of size 3
    
--median <radius> 
    apply median filter with O(r^2) complexity, raduis = 1, 2, ...

--median-linear <radius>
    apply median filter with O(r) complexity, raduis = 1, 2, ...
    
--median-const <radius>
    apply median filter with O(1) complexity, raduis = 1, 2, ...

[<param>=default_val] means that parameter is optional.
)";
    cout << "Usage: " << argv0 << " <input_image_path> <output_image_path> "
         << "PARAMS" << endl;
    cout << usage;
}

template<typename ValueType>
ValueType read_value(string s)
{
    stringstream ss(s);
    ValueType res;
    ss >> res;
    if (ss.fail() or not ss.eof())
        throw string("bad argument: ") + s;
    return res;
}

template<typename ValueT>
void check_number(string val_name, ValueT val, ValueT from,
                  ValueT to=numeric_limits<ValueT>::max())
{
    if (val < from)
        throw val_name + string(" is too small");
    if (val > to)
        throw val_name + string(" is too big");
}

void check_argc(int argc, int from, int to=numeric_limits<int>::max())
{
    if (argc < from)
        throw string("too few arguments for operation");

    if (argc > to)
        throw string("too many arguments for operation");
}

Matrix<double> parse_kernel(string kernel)
{
    
    stringstream ss(kernel);
    string msg = "invalid kernel string";
    
    vector<double> v;
    double tmp;
    int n_rows = 0;
    int n_cols = -1;
    int curr_cols = 0;
    
    while (1) {
        if (!(ss >> std::noskipws >> tmp))
            throw msg;

        v.push_back(tmp);
        ++curr_cols;
        char delim;
        if(ss >> delim) {
            if (delim != ';' && delim != ',')
                throw msg + ": delimiter must be ',' or ';'";
            if (delim == ';') {
                if (n_cols != -1 && curr_cols != n_cols)
                    throw msg + ": all the rows must have the same number of elements";
                n_cols = curr_cols;
                ++n_rows;
                curr_cols = 0;
            }
        } else {
            if (n_cols != -1 && curr_cols != n_cols)
                throw msg + ": all the rows must have the same number of elements";
            n_cols = curr_cols;
            ++n_rows;
            break;
        }
    }
    
    if (n_rows % 2 == 0 || n_cols % 2 == 0) {
        throw msg + ": kernel must have uneven number of rows and columns";
    }
    
    Matrix<double> result(n_rows, n_cols);
    uint pos = 0;
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            result(i, j) = v[pos++];
        }
    }

    return result;
}

int main(int argc, char **argv)
{
    try {
            
        check_argc(argc, 2);
        if (string(argv[1]) == "--help") {
            print_help(argv[0]);
            return 0;
        }

        check_argc(argc, 4);
        PreciseImage src_image = load_image<double>(argv[1]), dst_image;
        
        string action(argv[3]);

        if (action == "--sobel-x") {
            check_argc(argc, 4, 4);
            dst_image = src_image.unary_map(CustomFilter({ {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} } ));
        } else if (action == "--sobel-y") {
            check_argc(argc, 4, 4);
            dst_image = src_image.unary_map(CustomFilter({ {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} } ));
        } else if (action == "--unsharp") {
            check_argc(argc, 4, 4);
            dst_image = src_image.unary_map(CustomFilter({ {-1.0/6, -2.0/3, -1.0/6}, {-2.0/3, 13.0/3, -2.0/3}, {-1.0/6, -2.0/3, -1.0/6} } ));
        } else if (action == "--gray-world") {
            check_argc(argc, 4, 4);
            dst_image = gray_world(src_image);
        } else if (action == "--resize") {
            check_argc(argc, 5, 6);
            double scale = read_value<double>(argv[4]);
            check_number("scale", scale, 0.1, 10.0);
            if (argc == 6) {
                if (string(argv[5]) == "--bicubic-interp") {
                    dst_image = resize(src_image, scale, BICUBIC);
                } else {
                    throw string("unknown resize option ") + string(argv[5]);
                }
            } else {
                dst_image = resize(src_image, scale, BILINEAR);
            }
        }  else if (action == "--custom") {
            check_argc(argc, 5, 5);
            dst_image = src_image.unary_map(CustomFilter(parse_kernel(argv[4])));
        } else if (action == "--autocontrast") {
            check_argc(argc, 4, 5);
            double fraction = 0.0;
            if (argc == 5) {
                fraction = read_value<double>(argv[4]);
                check_number("fraction", fraction, 0.0, 0.4);
            }
            dst_image = autocontrast(src_image, fraction);
        } else if (action == "--gaussian" || action == "--gaussian-separable") {
            check_argc(argc, 5, 6);
            double sigma = read_value<double>(argv[4]);
            check_number("sigma", sigma, 0.1, 100.0);
            int radius = 3 * sigma;
            if (argc == 6) {
                radius = read_value<int>(argv[5]);
                check_number("radius", radius, 1);
            }
            if (action == "--gaussian") {
                dst_image = src_image.unary_map(CustomFilter(GenGaussKernel(radius, radius, sigma)));
            } else {
                PreciseImage tmp = src_image.unary_map(CustomFilter(GenGaussKernel(0, radius, sigma)));
                dst_image = tmp.unary_map(CustomFilter(GenGaussKernel(radius, 0, sigma)));
            }
        } else if (action == "--canny") {
            check_argc(argc, 6, 6);
            int threshold1 = read_value<int>(argv[4]);
            check_number("threshold1", threshold1, 0, 360);
            int threshold2 = read_value<int>(argv[5]);
            check_number("threshold2", threshold2, 0, 360);
            if (threshold1 >= threshold2)
                throw string("threshold1 must be less than threshold2");
            dst_image = canny(src_image, threshold1, threshold2);
        } else if (action == "--median" || action == "--median-linear" || action == "--median-const") {
            check_argc(argc, 5, 5);
            int radius = read_value<int>(argv[4]);
            check_number("radius", radius, 1);
            Image im = src_image.unary_map(ClipPixelFilter()), result;
            if (action == "--median") {
                result = im.unary_map(MedianFilter(radius, radius));
            } else if (action == "--median-linear") {
                result = LinearMedianFilter(im, radius);
            } else {
                result = ConstantMedianFilter(im, radius);
            }
            dst_image = result.unary_map(MakePreciseFilter());
        } else if (action == "--align") {
            check_argc(argc, 5, 9);
            InterpType interp_type = BILINEAR;
            double subpixels = 0.0;
            int arg_shift = 0;
            if (string(argv[4]) == "--bicubic-interp") {
                check_argc(argc, 6, 9);
                interp_type = BICUBIC;
                arg_shift = 1;
                if (string(argv[5]) == "--subpixel") {
                    check_argc(argc, 8, 9);
                    subpixels = read_value<double>(argv[6]);
                    check_number("subpixels", subpixels, 1.0, 10.0);
                    arg_shift = 3;
                }
            } else if (string(argv[4]) == "--subpixel") {
                check_argc(argc, 7, 8);
                subpixels = read_value<double>(argv[5]);
                check_number("subpixels", subpixels, 1.0, 10.0);
                arg_shift = 2;
            }
            check_argc(argc, 5 + arg_shift, 6 + arg_shift);
            string postprocessing(argv[4 + arg_shift]);
            if (postprocessing == "--gray-world" || postprocessing == "--unsharp") {
                check_argc(argc, 5 + arg_shift, 5 + arg_shift);
                dst_image = align(src_image, interp_type, subpixels, postprocessing);
            } else if (postprocessing == "--autocontrast") {
                double fraction = 0.0;
                if (argc == 6 + arg_shift) {
                    fraction = read_value<double>(argv[5 + arg_shift]);
                    check_number("fraction", fraction, 0.0, 0.4);
                }
                dst_image = align(src_image, interp_type, subpixels, postprocessing, fraction);
            } else {
                throw string("unknown align option ") + postprocessing;
            }
        } else {
            throw string("unknown action ") + action;
        }
        save_image(dst_image, argv[2]);
    } catch (const string &s) {
        cerr << "Error: " << s << endl;
        cerr << "For help type: " << endl << argv[0] << " --help" << endl;
        return 1;
    }
}
