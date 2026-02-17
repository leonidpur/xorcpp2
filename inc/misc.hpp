#pragma once

#include "model.hpp"

class Misc {
public:
    shared_ptr<Tensor> mul(const Tensor&a, const Tensor&b) {
        auto ret = make_shared<Tensor>(a.rows, b.rows);
        for (size_t r_a = 0; r_a < a.rows; r_a++)
            for (size_t r_b = 0; r_b < b.rows; r_b++) {
                double sum(0.0);
                for(size_t k =0;k < a.cols; k++)
                    sum += a.get(r_a, k) * b.get(r_b, k);
                ret->set(r_a, r_b, sum);
            }
        return ret;
    }
};