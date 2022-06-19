#include <cstdlib>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <limits> // For numeric_limits<double>::epsilon()
#include <filesystem> // For filesystem::directory_iterator
#include <chrono> // For measuring execution time
#include <AudioFile.h>
#include "utility_macros.h"
#include "parameters.h"
#include "valid_input_checks.h"
#include "csvfile.h"
#include "log_csv.h"
#include "kernel.cu"

using namespace std;

// Leave uncommented to measure the execution time of each part of the normalization process and log it to a file
#define TEST_RUNTIME

// Set one of these options to 1 to iterate over different values for the parameters defined below while processing
    // each audio file.
    // If multiple are set to 1, only the first will take effect
#define VARY_FRAME_LEN 0
#define VARY_FILTER_WIDTH 0
#define VARY_FILTER_BLOCK_SIZE 0

#define INPUT_FOLDER_NAME "AudioSamples"
#define OUTPUT_FOLDER_NAME "Normalized"

// Gain_Kernel only works for frame length up to 1024 (max # threads per block)
    // Could go higher if the kernels are modified to reuse shared memory, or frames are split into multiple blocks
#define DEFAULT_FRAME_LEN (uint64_t)1024 
#define DEFAULT_MAX_ALLOWABLE_VAL 0.95 // Audio samples are not allowed to go above this value
#define DEFAULT_GAIN_UPPER_LIMIT 10.0
#define DEFAULT_GAIN_LOWER_LIMIT 0.1
#define DEFAULT_TARGET_RMS_VAL 0.06
// The number of frames on either side of the current frame to consider when applying each filter 
    // The original algorithm by LoRd_MuldeR uses 15 by default
#define DEFAULT_MIN_FILTER_WIDTH_ONE_SIDE (uint32_t)15 
#define DEFAULT_GAUSSIAN_FILTER_WIDTH_ONE_SIDE (uint32_t)15
// The block size used for both the min filter kernel and the gaussian filter kernel (should not be larger than
    // the number of gain factors)
#define DEFAULT_FILTER_KERNEL_BLOCK_SIZE 30
#define FILTER_KERNEL_BLOCK_SIZE MIN(filter_kernel_block_size, num_frames)

void Save_File(AudioFile<double>* file, string filename);
AudioFile<double> Get_Test_Samples();

int main (int argc, char *argv[]) {
    cudaError_t cuda_ret;

    // Get locations to load files from and save files to
    const filesystem::path curr_path = filesystem::current_path();
    const filesystem::path input_folder = curr_path / INPUT_FOLDER_NAME;
    const filesystem::path output_folder = curr_path / OUTPUT_FOLDER_NAME;

    #ifdef TEST_RUNTIME
        INIT_FULL_TIMER
        INIT_SECTION_TIMER
        Init_CSV();
    #endif
    
    uint32_t min_filter_width_one_side, gaussian_filter_width_one_side, filter_kernel_block_size;
    uint64_t frame_len;
    double max_allowable_val, gain_upper_limit, gain_lower_limit, target_rms_val;

    user_args args = {DEFAULT_TARGET_RMS_VAL, DEFAULT_FRAME_LEN, DEFAULT_MIN_FILTER_WIDTH_ONE_SIDE, \
                    DEFAULT_GAUSSIAN_FILTER_WIDTH_ONE_SIDE, DEFAULT_GAIN_UPPER_LIMIT, \
                    DEFAULT_GAIN_LOWER_LIMIT, DEFAULT_MAX_ALLOWABLE_VAL, DEFAULT_FILTER_KERNEL_BLOCK_SIZE};

    // Read in arguments
    try {
        switch(argc) {
            // Each additional arguments sets another parameter
            case 9: if (argv[8][0] != 'd') { args.filter_kernel_block_size = atoi(argv[8]); }
            case 8: if (argv[7][0] != 'd') { args.max_allowable_val = atof(argv[7]); }
            case 7: if (argv[6][0] != 'd') { args.gain_lower_limit = atof(argv[6]); }
            case 6: if (argv[5][0] != 'd') { args.gain_upper_limit = atof(argv[5]); }
            case 5: if (argv[4][0] != 'd') { args.gaussian_filter_width_one_side = atoi(argv[4]); }
            case 4: if (argv[3][0] != 'd') { args.min_filter_width_one_side = atoi(argv[3]); }
            case 3: if (argv[2][0] != 'd') { args.frame_len = atoi(argv[2]); }
            case 2: if (argv[1][0] != 'd') { args.target_rms_val = atof(argv[1]); }
            case 1: // No user arguments
            break;
            default: throw invalid_argument("Invalid number of arguments"); break;
        }
        target_rms_val = args.target_rms_val;
        Target_RMS_Valid_Check(target_rms_val);
        frame_len = args.frame_len;
        // Frame length must be a power of 2 so that reduction will work, and the frame length must not
            // exceed 2048 samples or else the shared memory requirement for a block will be too high
        Frame_Length_Valid_Check(frame_len);
        min_filter_width_one_side = args.min_filter_width_one_side;
        Filter_Width_Valid_Check(min_filter_width_one_side);
        gaussian_filter_width_one_side = args.gaussian_filter_width_one_side;
        Filter_Width_Valid_Check(gaussian_filter_width_one_side);
        gain_upper_limit = args.gain_upper_limit;
        Gain_Limit_Valid_Check(gain_upper_limit);
        gain_lower_limit = args.gain_lower_limit;
        Gain_Limit_Valid_Check(gain_lower_limit);
        max_allowable_val = args.max_allowable_val;
        Max_Allowable_Value_Valid_Check(max_allowable_val);
        filter_kernel_block_size = args.filter_kernel_block_size;
        Filter_Block_Size_Valid_Check(filter_kernel_block_size);
    }
    catch(exception &e) {
        cout << e.what() << endl;
        cout << "Aborting" << endl;
        return 1;
    }

    const double epsilon = numeric_limits<double>::epsilon();
    const gain_parameters_t gain_params = {max_allowable_val, gain_upper_limit, gain_lower_limit, target_rms_val, epsilon};

    // Prepare to load audio files
    uint32_t num_audio_files = 0;
    AudioFile<double> audioFile;

    // Generate list of audio files and order them
    vector<filesystem::path> input_files;
    for (auto& entry : filesystem::directory_iterator(input_folder)) { 
        if (entry.path().extension() == ".wav") {
            // Make sure that the dummy file is processed first so that the real test files
                // are not affected by whatever causes the analyze kernel to take so long
                // the first time it is executed
            if (entry.path().filename().string() == "Dummy.wav") {
                input_files.insert(input_files.begin(),entry.path());
            }
            else {
                input_files.push_back(entry.path());
            }
        }
    }

    // Processing loop
    for (filesystem::path input_path : input_files) { 
        num_audio_files++;

        #if VARY_FRAME_LEN
            for (frame_len = 2; frame_len <= 1024; frame_len *= 2) {
        #elif VARY_FILTER_WIDTH
            for (uint32_t _width = 1; _width <= 1024; _width *= 2) {
                min_filter_width_one_side = _width;
                gaussian_filter_width_one_side = _width;
        #elif VARY_FILTER_BLOCK_SIZE
            for (uint32_t _size = 15; _size <= 1024; _size += 100) {
                filter_kernel_block_size = _size;
        #endif


        #ifdef TEST_RUNTIME
            START_FULL_TIMER();
        #endif

        audioFile.load(input_path);
        //audioFile = Get_Test_Samples();
        audioFile.printSummary(); 

        // Make sure the number of channels is correct
        try {
            Check_Input_File(&audioFile);
        }
        catch(exception &e) {
            cout << e.what() << endl;
            cout << "Skipping file " << input_path.filename() << endl;
            num_audio_files--;
            #ifdef TEST_RUNTIME
                STOP_FULL_TIMER();
            #endif
            continue;
        }

        /* Split the array of audio samples into frames so that a different gain can be applied to each one */
        uint64_t num_samples = audioFile.getNumSamplesPerChannel();
        uint64_t num_bytes_in_memory = sizeof(double)*num_samples;
        uint64_t num_bytes_per_frame = sizeof(double)*frame_len;
        uint64_t num_frames = (num_samples - 1)/frame_len + 1; // Use enough frames to fully cover the audio file

        #if !VARY_FILTER_BLOCK_SIZE
            filter_kernel_block_size = FILTER_KERNEL_BLOCK_SIZE;
        #endif

        #ifdef TEST_RUNTIME
            Write_To_CSV(num_audio_files); // File #
            Write_To_CSV(num_samples);
            Write_To_CSV(frame_len);
            Write_To_CSV(min_filter_width_one_side);
            START_SECTION_TIMER();
        #endif

        /* Analyze Kernel ----------------------------------------------- */
        // Allocate device memory for samples and frame gain factors
            // (no need to allocate new host memory, everything stays on the device)
        double *device_samples, *frame_gain;
        cuda_ret = cudaMalloc((void**)&device_samples, num_bytes_in_memory);
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to allocate device memory for audio samples" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cuda_ret = cudaMalloc((void**)&frame_gain, num_frames*sizeof(double));
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to allocate device memory for frame gain factors" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cudaDeviceSynchronize();
        // Copy samples to device
        cuda_ret = cudaMemcpy(device_samples, audioFile.samples[0].data(), num_bytes_in_memory, cudaMemcpyHostToDevice);
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to copy audio samples to device" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cudaDeviceSynchronize();

        // Execute kernel, using dynamically allocated shared memory
        // All blocks are 1D, i.e. their y and z dimensions are 1
            // We asssume that the length of a frame is even, and that it is no more than twice the length 
            // of the largest possible thread block
        dim3 AK_DimGrid(num_frames, 1, 1);
        dim3 AK_DimBlock(frame_len/2, 1, 1);
        // The amount of shared memory allocated is equal to the size of a frame because there are 
            // two reductions being performed in parallel, each of which requires half the size of a frame
        Analyze_Kernel<<<AK_DimGrid,AK_DimBlock,num_bytes_per_frame>>>(device_samples, num_samples, frame_gain, frame_len, gain_params);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) {
            cout << "Analyze_Kernel failed" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        
        // Stop measuring time to get initial frame gain
        #ifdef TEST_RUNTIME
            STOP_SECTION_TIMER();
            Write_To_CSV(SECTION_DURATION());
        #endif

        // Start measuring time to apply minimum filter
        #ifdef TEST_RUNTIME
            START_SECTION_TIMER();
        #endif

        /* Min Filter Kernel ----------------------------------------------- */
        // Make room in global memory for the filtered gain factors to be stored in
        double* frame_gain_temp;
        cuda_ret = cudaMalloc((void**)&frame_gain_temp, num_frames*sizeof(double));
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to allocate device memory for frame gain factors (temporary copy)" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cudaDeviceSynchronize();
        // The number of threads per block in the min filter kernel must be at least half the
            // width of the filter in order for the method of transferring gain factors to 
            // shared memory to work
        uint64_t MFK_block_size = MAX(filter_kernel_block_size, min_filter_width_one_side);
        // Shared memory is allocated for the number of gain factors that will be modified as well as enough
            // of the preceding and following gain factors to allow the filter to be applied
        uint32_t MFK_shared_bytes = (MFK_block_size + 2*min_filter_width_one_side)*sizeof(double);
        // Execute kernel
        dim3 MFK_DimGrid((num_frames-1)/MFK_block_size + 1, 1, 1);
        dim3 MFK_DimBlock(MFK_block_size, 1, 1);
        Min_Filter_Kernel<<<MFK_DimGrid,MFK_DimBlock,MFK_shared_bytes>>>(frame_gain, frame_gain_temp, min_filter_width_one_side, num_frames);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) {
            cout << "Min_Filter_Kernel failed" << endl;
            cout << "Aborting" << endl;
            return 1;
        }

        // Stop measuring time to apply minimum filter
        #ifdef TEST_RUNTIME
            STOP_SECTION_TIMER();
            Write_To_CSV(SECTION_DURATION());
        #endif

        // Start measuring time to apply gaussian filter
        #ifdef TEST_RUNTIME
            START_SECTION_TIMER();
        #endif

        /* Gaussian Filter Kernel ----------------------------------------------- */
        // Determine Gaussian filter weights
        double* gaussian_coef = Get_Gaussian_Filter_Coef(gaussian_filter_width_one_side);
        // Allocate memory on the device for the Gaussian filter coefficients
        double *device_coefs;
        uint32_t coef_bytes = (2*gaussian_filter_width_one_side + 1)*sizeof(double);
        cuda_ret = cudaMalloc((void**)&device_coefs, coef_bytes);
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to allocate device memory for filter coefficients" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cudaDeviceSynchronize();
        // Copy Gaussian filter coefficients to the device
        cuda_ret = cudaMemcpy(device_coefs, gaussian_coef, coef_bytes, cudaMemcpyHostToDevice);
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to copy filter coefficients to device" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cudaDeviceSynchronize();
        // The number of threads per block in the gaussian filter kernel must be at least half the
            // width of the filter in order for the method of transferring gain factors to 
            // shared memory to work
        uint64_t GFK_block_size = MAX(filter_kernel_block_size, gaussian_filter_width_one_side);
        // Shared memory is allocated for the number of gain factors that will be modified as well as enough
            // of the preceding and following gain factors to allow the filter to be applied
        uint32_t GFK_shared_bytes = (GFK_block_size + 2*gaussian_filter_width_one_side)*sizeof(double);
        // Execute kernel
        dim3 GFK_DimGrid((num_frames-1)/GFK_block_size + 1, 1, 1);
        dim3 GFK_DimBlock(GFK_block_size, 1, 1);
        Gaussian_Filter_Kernel<<<GFK_DimGrid,GFK_DimBlock,GFK_shared_bytes>>>(frame_gain, frame_gain_temp, device_coefs, gaussian_filter_width_one_side, num_frames);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) {
            cout << "Gaussian_Filter_Kernel failed" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        
        // Stop measuring time to apply gaussian filter
        #ifdef TEST_RUNTIME
            STOP_SECTION_TIMER();
            Write_To_CSV(SECTION_DURATION());
        #endif

        // Start measuring time to apply gain to each frame
        #ifdef TEST_RUNTIME
            START_SECTION_TIMER();
        #endif

        /* Gain Kernel ----------------------------------------------- */
        // Execute kernel
        dim3 GK_DimGrid(num_frames, 1, 1);
        dim3 GK_DimBlock(frame_len, 1, 1);
        Gain_Kernel<<<GK_DimGrid,GK_DimBlock>>>(device_samples, num_samples, frame_gain);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) {
            cout << "Gain_Kernel failed" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        // Copy processed samples to host memory
        double* host_samples = (double*)malloc(num_bytes_in_memory);
        cuda_ret = cudaMemcpy(audioFile.samples[0].data(), device_samples, num_bytes_in_memory, cudaMemcpyDeviceToHost);
        if (cuda_ret != cudaSuccess) {
            cout << "Unable to copy audio samples to host" << endl;
            cout << "Aborting" << endl;
            return 1;
        }
        cudaDeviceSynchronize();

        // Stop measuring time to apply gain to each frame
        #ifdef TEST_RUNTIME
            STOP_SECTION_TIMER();
            Write_To_CSV(SECTION_DURATION());
        #endif

        // Save processed samples to audio file
            // The filenames for the output files have a '_n' appended to them
        string output_file_name = input_path.filename().string();
        output_file_name.replace(output_file_name.find(".wav"), output_file_name.back(), "_n.wav");
        filesystem::path output_path = output_folder;
        output_path.append(output_file_name);
        cout << "Saving audio file" << endl;
        if (!audioFile.save(output_path.string())) {
            cout << "Saving audio file failed" << endl;
        }

        // Free all memory allocated for this audio file
        cudaFree(device_samples);
        cudaFree(frame_gain);
        cudaFree(frame_gain_temp);
        free(gaussian_coef);

        #ifdef TEST_RUNTIME
            STOP_FULL_TIMER();
            Write_To_CSV(FULL_DURATION());
            Write_To_CSV(MFK_block_size);
            CSV_Next_Line();
        #endif

        #if VARY_FRAME_LEN || VARY_FILTER_WIDTH || VARY_FILTER_BLOCK_SIZE
            } // Closing bracket to end the for loop
        #endif
    }

    // Print summary
    cout << "Processed " << num_audio_files << " audio files." << endl;

    return 0;
}

AudioFile<double> Get_Test_Samples() {
    srand(24);
    vector<double> test_samples;
    for (int i = 0; i < 10; i++) {
        double val = (rand() % 100)/1000.0;
        if (i % 2) val *= -1;
        cout << val << " ";
        test_samples.push_back(val);
    }
    cout << endl;
    vector<vector<double>> buf;
    buf.push_back(test_samples);
    AudioFile<double> test_file;
    test_file.setAudioBuffer(buf);
    return test_file;
}