#ifndef __VALID_INPUT_CHECKS_H__
#define __VALID_INPUT_CHECKS_H__

#include <AudioFile.h>

using namespace std;

void Target_RMS_Valid_Check(double target_rms_val) {
    if (target_rms_val < 0 || target_rms_val > 1.0001) {
        throw invalid_argument("Invalid target RMS value, must be in the range [0, 1]");
    }
}

void Frame_Length_Valid_Check(uint64_t frame_len) {
    // Frame length must be a power of 2 so that reduction will work, and the frame length must not
        // exceed 2048 samples or else the shared memory requirement for a block will be too high
    if (!frame_len || (frame_len & (frame_len - 1))) {
        throw invalid_argument("Invalid frame length, must be a power of 2");
    }
    else if (frame_len > 1024) {
        throw invalid_argument("Invalid frame length, must be <= 1024");
    }
}

void Filter_Width_Valid_Check(uint32_t filter_width) {
    if (filter_width > 1024) {
        throw invalid_argument("Invalid filter half-width, must be <= 1024");
    }
}

void Gain_Limit_Valid_Check(double gain_limit) {
    if (gain_limit < 0) {
        throw invalid_argument("Invalid gain upper or lower limit, must be positive");
    }
}

void Max_Allowable_Value_Valid_Check(double max_allowable_val) {
    if (max_allowable_val <= 0 || max_allowable_val > 1.0) {
        throw invalid_argument("Invalid maximum allowable value, must be in the range (0, 1]");
    }
}

void Filter_Block_Size_Valid_Check(uint32_t block_size) {
    if (block_size == 0 || block_size > 1024) {
        throw invalid_argument("Invalid filter block size, must be be in the range (0, 1024]");
    }
}

void Check_Input_File(AudioFile<double>* audioFile) {
    // Currently only operating on data with 1 channel
    if (audioFile->getNumChannels() > 1) {
        throw out_of_range("Audio file has more than one channel");
    }
}

#endif