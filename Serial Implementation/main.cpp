/*
This is a test of dynamic audio normalization running on just the CPU
    - This version attempts to smooth the gain between frames
*/

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits> // For numeric_limits<double>::epsilon()
#include <stdexcept>
#include <filesystem> // For filesystem::directory_iterator
#include <chrono> // For measuring execution time
#include <AudioFile.h>
#include "csvfile.h"
#include "log_csv.h"

using namespace std;

// Leave uncommented to measure the execution time of each part of the normalization process and log it to a file
#define TEST_RUNTIME

#define INPUT_FOLDER "/home/jcarl/DANPI_WSL/DANPI_WSL/AudioSamples/"
#define OUTPUT_FOLDER "/home/jcarl/DANPI_WSL/DANPI_WSL/AudioSamples/Normalized/"

#define FRAME_LEN (uint32_t)1024
#define MAX_ALLOWABLE_VAL 0.95 // Audio samples are not allowed to go above this value
#define GAIN_UPPER_LIMIT 10.0
#define GAIN_LOWER_LIMIT 0.1
#define TARGET_RMS_VAL 0.06
// The number of frames on either side of the current frame to consider when applying each filter 
    // The original algorithm by LoRd_MuldeR uses 15 by default
#define MIN_FILTER_WIDTH_ONE_SIDE (uint32_t)15
#define GAUSSIAN_FILTER_WIDTH_ONE_SIDE (uint32_t)15

void Normalize_File(AudioFile<double>* audioFile);
void Save_File(AudioFile<double>* file, string filename);
void Check_Input_File(AudioFile<double>* file);
AudioFile<double> Get_Test_Samples();

// The full timer records the total time taken to perform processing for a single audio file, from
    // before the file is loaded to after it is saved
// The section timer records the time taken for a single part of the process

using namespace std::chrono;
// Initialization of variables - values are irrelevant
#define INIT_FULL_TIMER auto full_timer_start_point = high_resolution_clock::now(); \
                        std::chrono::microseconds full_duration;
#define INIT_SECTION_TIMER auto section_timer_start_point = high_resolution_clock::now(); \
                           std::chrono::microseconds section_duration;
#define START_FULL_TIMER() full_timer_start_point = high_resolution_clock::now();
#define START_SECTION_TIMER() section_timer_start_point = high_resolution_clock::now();
#define STOP_FULL_TIMER() full_duration = duration_cast<microseconds>(high_resolution_clock::now() - full_timer_start_point)
#define STOP_SECTION_TIMER() section_duration = duration_cast<microseconds>(high_resolution_clock::now() - section_timer_start_point)
#define FULL_DURATION() full_duration.count()
#define SECTION_DURATION() section_duration.count()
using namespace std;

int main() {
    #ifdef TEST_RUNTIME
        INIT_FULL_TIMER
        Init_CSV();
    #endif

    AudioFile<double> audioFile;
    const filesystem::path input_folder = INPUT_FOLDER;
    const filesystem::path output_folder = OUTPUT_FOLDER;
    uint32_t num_audio_files = 0;
    
    // All audio files in the audio_folder directory are automatically processed
    for (auto& entry : filesystem::directory_iterator(input_folder)) { 
        if (entry.path().extension() == ".wav") {
            num_audio_files++;

            #ifdef TEST_RUNTIME
                Write_To_CSV(num_audio_files); // File #
                START_FULL_TIMER();
            #endif

            audioFile.load(entry.path());
            //audioFile = Get_Test_Samples();
            audioFile.printSummary(); 

            /* Make sure all of the parameters are within the valid range */
            try {
                Check_Input_File(&audioFile);
            }
            catch(exception &e) {
                cout << e.what() << endl;
                cout << "Skipping file " << entry.path().filename() << endl;
                continue;
            }
            
            Normalize_File(&audioFile);

            // The filenames for the output files have a '_n' appended to them
            string output_file_name = entry.path().filename().string();
            output_file_name.replace(output_file_name.find(".wav"), output_file_name.back(), "_n.wav");
            filesystem::path output_path = output_folder;
            output_path.append(output_file_name);
            Save_File(&audioFile, output_path.string());

            #ifdef TEST_RUNTIME
                STOP_FULL_TIMER();
                Write_To_CSV(FULL_DURATION());
                CSV_Next_Line();
            #endif
        }
    }


    cout << "Processed " << num_audio_files << " files" << endl;
    return 0;
}

void Normalize_File(AudioFile<double>* audioFile) {
    /* Split the array of audio samples into frames so that a different gain can be applied to each one */
    uint64_t num_samples = audioFile->getNumSamplesPerChannel();
    uint32_t num_frames = (num_samples - 1)/FRAME_LEN + 1; // Use enough frames to fully cover the audio file
    uint32_t last_frame_samples = num_samples % FRAME_LEN;

    // Start measuring time to get initial frame gain
    #ifdef TEST_RUNTIME
        Write_To_CSV(num_samples);
        Write_To_CSV(FRAME_LEN);
        Write_To_CSV(MIN_FILTER_WIDTH_ONE_SIDE);
        INIT_SECTION_TIMER
        START_SECTION_TIMER();
    #endif

    /* Choose the gain for each frame */
        // Determine maximum possible gain and the gain that achieves the desired RMS value, and keep the lower one
        // This is done one frame at a time until the values have been found for all frames
    double* frame_gain = (double*)malloc(sizeof(double)*num_frames);
    for (uint32_t i = 0; i < num_frames; ++i) {
        uint32_t frame_start = FRAME_LEN*i; // first value in frame
        double frame_max_val = fabs(audioFile->samples[0][frame_start]);
        double frame_rms = 0;
        for (uint32_t j = 0; j < FRAME_LEN; ++j) {
            double curr_sample = audioFile->samples[0][frame_start + j];
            frame_max_val = max(frame_max_val, fabs(curr_sample));
            frame_rms += pow(curr_sample,2);
        }
        double frame_max_gain = MAX_ALLOWABLE_VAL/frame_max_val;
        if ((i == num_frames - 1) && (num_samples % FRAME_LEN != 0)) {
            // If the frame at the end goes over the length of the samples, there are fewer samples to
                // consider in the average
            frame_rms = max(sqrt(frame_rms /(num_samples % FRAME_LEN)), numeric_limits<double>::epsilon());
        }
        else {
            frame_rms = max(sqrt(frame_rms / FRAME_LEN), numeric_limits<double>::epsilon());
        }
        
        double frame_rms_gain = max(TARGET_RMS_VAL/frame_rms, numeric_limits<double>::epsilon());
        // GAIN_LIMIT is used for both gain that increases the value and gain that decreases the value
            // This is a modification of the original algorithm
        if (frame_rms_gain >= 1) {
            frame_rms_gain = min(frame_rms_gain, GAIN_UPPER_LIMIT);
        }
        else {
            frame_rms_gain = max(frame_rms_gain, GAIN_LOWER_LIMIT);
        }
        frame_gain[i] = min(frame_rms_gain, frame_max_gain);
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

    /* Apply minimum filter */
    double* frame_gain_min = (double*)malloc(sizeof(double)*num_frames);
    for (uint32_t i = 0; i < num_frames; ++i) {
        double local_min_gain = frame_gain[i];
        for (uint32_t j = i - min(MIN_FILTER_WIDTH_ONE_SIDE, i); j <= min(i + MIN_FILTER_WIDTH_ONE_SIDE, num_frames - 1); ++j) {
            local_min_gain = min(local_min_gain, frame_gain[j]);
        }
        frame_gain_min[i] = local_min_gain;
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

    /* Apply gaussian filter */
    // Determine Gaussian filter weights
        // Choose sigma so that the width of the filter covers three standard deviations on either side
    double sigma;
    if (GAUSSIAN_FILTER_WIDTH_ONE_SIDE == 0) {
        // If GAUSSIAN_FILTER_WIDTH_ONE_SIDE is zero, then c1 is simply multiplied to every frame gain.
            // setting sigma to this makes c1 = 1, so the filter has no effect
        sigma = 1/sqrt(2.0 * M_PI);
    }
    else {
        sigma = GAUSSIAN_FILTER_WIDTH_ONE_SIDE/3.0;
    }
    const double c1 = 1.0 / (sigma * sqrt(2.0 * M_PI));
    const double c2 = 2.0 * pow(sigma, 2.0);
    int32_t num_coef = 2*GAUSSIAN_FILTER_WIDTH_ONE_SIDE + 1;
    double* gaussian_coef = (double*)malloc(sizeof(double)*num_coef);
    for (int32_t i = 0; i < num_coef; ++i) {
        uint32_t d = abs(i - (num_coef >> 1)); // Distance from the center point of the Gaussian kernel
        gaussian_coef[i] = c1 * exp(-(pow(d, 2.0) / c2));
    }
    for (uint32_t i = 0; i < num_frames; ++i) {
        double filter_total = 0;
        for (uint32_t j = i - min(GAUSSIAN_FILTER_WIDTH_ONE_SIDE, i); j <= min(i + GAUSSIAN_FILTER_WIDTH_ONE_SIDE, num_frames - 1); ++j) {
            filter_total += frame_gain_min[j] * gaussian_coef[j - i + GAUSSIAN_FILTER_WIDTH_ONE_SIDE];
        }
        // Reuse the frame_gain array to store the final gain values
        frame_gain[i] = filter_total;
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

    /* Apply final gain to each frame */
    for (uint32_t i = 0; i < num_frames; ++i) {
        for (uint32_t j = 0; j < FRAME_LEN; ++j) {
            audioFile->samples[0][FRAME_LEN*i + j] *= frame_gain[i];
        }
    }

    // Stop measuring time to apply gain to each frame
    #ifdef TEST_RUNTIME
        STOP_SECTION_TIMER();
        Write_To_CSV(SECTION_DURATION());
    #endif
    
    free(frame_gain);
    //free(frame_gain_min); // Was getting invalid pointer error because of this line
}

void Save_File(AudioFile<double>* audioFile, string filename) {
    cout << "Saving audio file" << endl;
    if (!audioFile->save(filename)) {
        cout << "Saving audio file failed" << endl;
    }
}

void Check_Input_File(AudioFile<double>* audioFile) {
    // Currently only operating on data with 1 channel
    if (audioFile->getNumChannels() > 1) {
        throw out_of_range("Audio file has more than one channel");
    }
}

AudioFile<double> Get_Test_Samples() {
    srand(69);
    vector<double> test_samples;
    for (int i = 0; i < 1000; i++) {
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