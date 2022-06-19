#ifndef __LOG_CSV_H__
#define __LOG_CSV_H__

#include "csvfile.h"

using namespace std;

#define LOG_FILE_NAME "Log.csv"

csvfile csv(filesystem::current_path() / LOG_FILE_NAME,",");

// Create the file and 
void Init_CSV() {
    csv << "File #" << "# Samples" << "Frame Length" << "Filter Width" << "Analyze Time" \
        << "Min Filter Time" << "Gauss Filter Time" << "Gain Time" << "Full Duration" \
        << "Filter Block Size" << endrow;
}

void Write_To_CSV(uint64_t val) {
    csv << val;
}

void CSV_Next_Line() {
    csv << endrow;
}

#endif