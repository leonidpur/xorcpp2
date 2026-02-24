#pragma once

#include "tensor.hpp"

class Utils {
public:
    static shared_ptr<Tensor> tensor_from_samples(const vector<vector<double>> &input, vector<double> &output) {
        auto ret = TensorUtils::make_tensor(input.size(), input[0].size(), false, {}, "input");
        for (size_t r = 0; r < input.size(); r++)
            for (size_t c = 0; c < input[0].size(); c++)
                ret->set(r, c, input[r][c]);
        return ret;        
    }
};
