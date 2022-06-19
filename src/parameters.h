#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

using namespace std;

typedef struct {
    double target_rms_val;
    uint64_t frame_len;
    uint32_t min_filter_width_one_side;
    uint32_t gaussian_filter_width_one_side;
    double gain_upper_limit;
    double gain_lower_limit;
    double max_allowable_val;
    uint32_t filter_kernel_block_size;
} user_args;

typedef struct {
    double max_allowable_val; // Audio samples are not allowed to go above this value
    double gain_upper_limit; // Gain applied to a frame will not be above this value
    double gain_lower_limit; // Gain applied to a frame will not be below this value
    double target_rms_val; // The RMS value that the algorithm tries to establish for all frames
    double epsilon; // The smallest nonzero value that can be represented using a double; used to avoid dividing by zero
} gain_parameters_t;

// Determine Gaussian filter weights
double* Get_Gaussian_Filter_Coef(uint32_t gaussian_filter_width_one_side) {
    // Choose sigma so that the width of the filter covers three standard deviations on either side
    double sigma;
    if (gaussian_filter_width_one_side == 0) {
        // If gaussian_filter_width_one_side is zero, then c1 is simply multiplied to every frame gain.
            // setting sigma to this makes c1 = 1, so the filter has no effect
        sigma = 1/sqrt(2.0 * M_PI);
    }
    else {
        sigma = gaussian_filter_width_one_side/3.0;
    }
    const double c1 = 1.0 / (sigma * sqrt(2.0 * M_PI));
    const double c2 = 2.0 * pow(sigma, 2.0);
    int32_t num_coef = 2*gaussian_filter_width_one_side + 1;
    double* gaussian_coef = (double*)malloc(sizeof(double)*num_coef);
    //cout << "Filter coefficients:" << endl;
    for (int32_t i = 0; i < num_coef; ++i) {
        uint32_t d = abs(i - (num_coef >> 1)); // Distance from the center point of the Gaussian kernel
        gaussian_coef[i] = c1 * exp(-(pow(d, 2.0) / c2));
        //cout << gaussian_coef[i] << " ";
    }
    //cout << endl;
    return gaussian_coef;
}

#endif