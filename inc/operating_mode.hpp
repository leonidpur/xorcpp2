#pragma once

#include <cstddef>

class OperatingMode {
public:
    bool print_params = false;
    bool print_temp = false;
    bool print_epoch = false;
    size_t max_epoch = 1000;
    double loss_threshold = 0.01;
};
