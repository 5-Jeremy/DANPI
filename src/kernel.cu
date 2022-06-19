// #include <cuda_runtime_api.h> 
// #include <cuda.h> 
// #include <cooperative_groups.h>
#include <inttypes.h>
#include "utility_macros.h"

using namespace std;

// For debugging
#define THREAD_0_PRINTF if (threadIdx.x == 0) printf

// Determine maximum possible gain and the gain that achieves the desired RMS value, and keep the lower one
__global__ void Analyze_Kernel(double* device_samples, uint64_t num_samples, double* frame_gain, uint64_t frame_len, const gain_parameters_t gain_params) {
    // Use coalesced memory access to bring samples into shared memory
        // Each thread performs its first computation before saving anything to shared memory
    // The dynamically allocated shared memory needs to be divided into two arrays with half the length
    extern __shared__ double frame_gain_data [];
    double* frame_max = frame_gain_data; // frame_max occupies the first half
    double* frame_rms = &frame_gain_data[frame_len/2]; // frame_rms occupies the second half

    uint64_t element_1_indx = 2*blockIdx.x*blockDim.x + threadIdx.x;
    uint64_t element_2_indx = element_1_indx + blockDim.x;
    // Perform the first computation and store in shared memory
    if (element_1_indx < num_samples) {
        double element_1 = device_samples[element_1_indx];
        if (element_2_indx < num_samples) {
            double element_2 = device_samples[element_2_indx];
            frame_max[threadIdx.x] = MAX(fabs(element_1), fabs(element_2));
            frame_rms[threadIdx.x] = pow(element_1,2) + pow(element_2,2);
        }
        else {
            // If there is no second element for a thread to access, just put the first element into shared memory
                // directly
            frame_max[threadIdx.x] = fabs(element_1);
            frame_rms[threadIdx.x] = pow(element_1,2);
        }
    }
    else {
        // Initialize to the identity value if out of bounds
        frame_max[threadIdx.x] = 0;
        frame_rms[threadIdx.x] = 0;
    }

    // Use reduction to determine the peak amplitude and RMS value of the frame
        // A potential optimization would be to assign each thread to either the max array or the rms array by
        // giving them a pointer and then have all the threads operate on just one array
    for (uint64_t i = blockDim.x/2; i >= 1; i = i >> 1) {
        __syncthreads();
        if (threadIdx.x < i) {
            frame_max[threadIdx.x] = MAX(frame_max[threadIdx.x], frame_max[threadIdx.x + i]);
            frame_rms[threadIdx.x] += frame_rms[threadIdx.x + i];
        }
    }
    __syncthreads();
    
    // Do the final processing, aggregate the results into global memory, and end this kernel
        // This processing is not done on the host since there is no other reason to copy the 
        // frame_gain array to the host
    if (threadIdx.x == 0) {
        double frame_max_gain = gain_params.max_allowable_val/frame_max[0];
        // This handles the case where a frame goes over the number of samples
        uint64_t samples_in_average;
        if ((num_samples % frame_len != 0) && (blockIdx.x == gridDim.x - 1)) {
            samples_in_average = num_samples % frame_len;
        }
        else {
            samples_in_average = frame_len;
        }
        double frame_total_rms = MAX(sqrt(frame_rms[0]/samples_in_average), gain_params.epsilon);
        double frame_rms_gain = MAX(gain_params.target_rms_val/frame_total_rms, gain_params.epsilon);
        // GAIN_LIMIT is used for both gain that increases the value and gain that decreases the value
            // This is a modification of the original algorithm
        if (frame_rms_gain >= 1) {
            frame_rms_gain = MIN(frame_rms_gain, gain_params.gain_upper_limit);
        }
        else {
            frame_rms_gain = MAX(frame_rms_gain, gain_params.gain_lower_limit);
        }
        frame_gain[blockIdx.x] = MIN(frame_rms_gain, frame_max_gain);
    }
}

__global__ void Min_Filter_Kernel(double* frame_gain, double* frame_gain_temp, uint32_t width_one_side, uint64_t num_frames) {
    extern __shared__ double local_frame_gain_data [];
    // Use coalesced memory access to bring frame gain data into shared memory
    // index is the index of the element that the current thread will be applying the filter to in the array frame_gain
    // local_index is the index of the element that the current thread will be applying the filter to in the local copy
    uint64_t index = blockIdx.x*blockDim.x + threadIdx.x;
    uint64_t local_index = width_one_side + threadIdx.x; // local_index - width_one_side == threadIdx.x
    if (index < num_frames) {
        local_frame_gain_data[local_index] = frame_gain[index];
    }
    // If possible, enough gain factors should be included such that the first and last threads are able to
        // apply the full window size of the filter. However, if there are not enough gain factors, do not include any more.
    uint32_t num_before_first = MIN(blockIdx.x*blockDim.x, width_one_side);
    int64_t possible_num_after = MAX((int64_t)num_frames - (int64_t)((blockIdx.x+1)*blockDim.x), (int64_t)0);
    int32_t num_after_last = MIN(possible_num_after, (int64_t)width_one_side);

    if (threadIdx.x < num_before_first) {
        local_frame_gain_data[threadIdx.x] = frame_gain[index - num_before_first];
    }
    
    if (threadIdx.x < num_after_last) {
        local_frame_gain_data[local_index + blockDim.x] = frame_gain[index + blockDim.x];
    }
    // Done collecting data from global memory
    __syncthreads();

    // All blocks dealing with data that is at least width_one_side away from the beginning or end of the
        // frame_gain array can avoid the extra logic seen in the else statement
    if ((blockIdx.x*blockDim.x >= width_one_side) && ((num_frames - blockIdx.x*blockDim.x) >= (width_one_side + blockDim.x))) {
        double local_min_gain = local_frame_gain_data[local_index];
        for (uint32_t j = local_index - width_one_side; j <= local_index + width_one_side; ++j) {
            local_min_gain = MIN(local_min_gain, local_frame_gain_data[j]);
        }
        // Save to global memory
        frame_gain_temp[index] = local_min_gain;
    }
    else {
        // The first index in local_frame_gain_data that contains a gain factor (after copying data)
        uint64_t first_val_indx = width_one_side - num_before_first;
        // The last index in local_frame_gain_data that contains a gain factor (after copying data)
        uint64_t last_val_indx = width_one_side + MIN(num_frames - blockIdx.x*blockDim.x, blockDim.x + width_one_side) - 1;

        // Apply minimum filter and store resulting gain factors in the temporary array
        // Number of elements preceding this thread's element to include in the filter window
        uint64_t first_index_in_window = MAX(first_val_indx, local_index - width_one_side);
        uint64_t last_index_in_window = MIN(last_val_indx, local_index + width_one_side);

        // If there are more threads than elements to apply the filter on, some threads do nothing
        if (index < num_frames) {
            double local_min_gain = local_frame_gain_data[local_index];
            for (uint32_t i = first_index_in_window; i <= last_index_in_window; ++i) {
                local_min_gain = MIN(local_min_gain, local_frame_gain_data[i]);
            }
            // Save to global memory
            frame_gain_temp[index] = local_min_gain;
        }
    }
}

__global__ void Gaussian_Filter_Kernel(double* frame_gain, double* frame_gain_temp, double* coefs, uint32_t width_one_side, uint64_t num_frames) {
    extern __shared__ double local_frame_gain_data [];
    // Use coalesced memory access to bring frame gain data into shared memory
    // Unlike in Min_Filter_Kernel, the data is taken from frame_gain_temp and stored in frame_gain
    uint64_t index = blockIdx.x*blockDim.x + threadIdx.x;
    uint64_t local_index = width_one_side + threadIdx.x; // local_index - width_one_side == threadIdx.x
    if (index < num_frames) {
        local_frame_gain_data[local_index] = frame_gain_temp[index];
    }
    // If possible, enough gain factors should be included such that the first and last threads are able to
        // apply the full window size of the filter. However, if there are not enough samples, do not include any more.
    uint32_t num_before_first = MIN(blockIdx.x*blockDim.x, width_one_side);
    int64_t possible_num_after = MAX((int64_t)num_frames - (int64_t)((blockIdx.x+1)*blockDim.x), (int64_t)0);
    int32_t num_after_last = MIN(possible_num_after, (int64_t)width_one_side);

    if (threadIdx.x < num_before_first) {
        local_frame_gain_data[threadIdx.x] = frame_gain_temp[index - num_before_first];
    }
    
    if (threadIdx.x < num_after_last) {
        local_frame_gain_data[local_index + blockDim.x] = frame_gain_temp[index + blockDim.x];
    }
    // Done collecting data from global memory
    __syncthreads();
    
    // All blocks dealing with data that is at least width_one_side away from the beginning or end of the
        // frame_gain array can avoid the extra logic seen in the else statement
    if ((blockIdx.x*blockDim.x >= width_one_side) && ((num_frames - blockIdx.x*blockDim.x) >= (width_one_side + blockDim.x))) {
        double filter_total = 0;
        for (uint32_t i = local_index - width_one_side; i <= local_index + width_one_side; ++i) {
            filter_total += local_frame_gain_data[i]*coefs[i - threadIdx.x];
        }
        // Save to global memory
        frame_gain[index] = filter_total;
    }
    else {
        // The first index in local_frame_gain_data that contains a gain factor (after copying data)
        uint64_t first_val_indx = width_one_side - num_before_first;
        // The last index in local_frame_gain_data that contains a gain factor (after copying data)
        uint64_t last_val_indx = width_one_side + MIN(num_frames - blockIdx.x*blockDim.x, blockDim.x + width_one_side) - 1;

        // Number of elements preceding this thread's element to include in the filter window
        uint64_t first_index_in_window = MAX(first_val_indx, local_index - width_one_side);
        uint64_t last_index_in_window = MIN(last_val_indx, local_index + width_one_side);

        // If there are more threads than elements to apply the filter on, some threads do nothing
        if (index < num_frames) {
            double filter_total = 0;
            for (uint32_t i = first_index_in_window; i <= last_index_in_window; ++i) {
                filter_total += local_frame_gain_data[i]*coefs[i - threadIdx.x];
            }
            // Save to global memory
            frame_gain[index] = filter_total;
        }
    }
}

// Apply final gain factors
__global__ void Gain_Kernel(double* device_samples, uint64_t num_samples, double* frame_gain) {
    // Apply gain element-by-element, each thread maps to one element
    double gain = frame_gain[blockIdx.x];
    uint64_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < num_samples) {
        device_samples[index] = device_samples[index] * gain;
    }
}