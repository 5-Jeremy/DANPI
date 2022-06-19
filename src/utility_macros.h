#ifndef __UTILITY_MACROS_H__
#define __UTILITY_MACROS_H__

#include <chrono> // For measuring execution time

using namespace std::chrono;
// Initialization of variables - values are irrelevant
#define INIT_FULL_TIMER auto full_timer_start_point = high_resolution_clock::now(); \
                        microseconds full_duration;
#define INIT_SECTION_TIMER auto section_timer_start_point = high_resolution_clock::now(); \
                           microseconds section_duration;
// The full timer records the total time taken to perform processing for a single audio file, from
    // before the file is loaded to after it is saved
// The section timer records the time taken for a single part of the process
#define START_FULL_TIMER() full_timer_start_point = high_resolution_clock::now();
#define START_SECTION_TIMER() section_timer_start_point = high_resolution_clock::now();
#define STOP_FULL_TIMER() full_duration = duration_cast<microseconds>(high_resolution_clock::now() - full_timer_start_point)
#define STOP_SECTION_TIMER() section_duration = duration_cast<microseconds>(high_resolution_clock::now() - section_timer_start_point)
#define FULL_DURATION() full_duration.count()
#define SECTION_DURATION() section_duration.count()
using namespace std;

#define MIN(X,Y) ((X < Y) ? X : Y)
#define MAX(X,Y) ((X > Y) ? X : Y)

#endif