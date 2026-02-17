#pragma once

#include "model.hpp"
#include "operating_mode.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using LossFn =
    std::function<std::shared_ptr<Tensor>(
        const std::shared_ptr<Tensor>&,
        const std::shared_ptr<Tensor>&)>;


struct Optimizer {
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual ~Optimizer() = default;
};

struct SGD : Optimizer {
    float lr;
    std::vector<std::shared_ptr<Tensor>> params;

    SGD(std::vector<std::shared_ptr<Tensor>> params_, float lr_)
        : lr(lr_), params(std::move(params_)) {}

    void step() override {
        for (auto& p : params) {
            if (!p || !p->requires_grad) continue;
            for (size_t i = 0; i < p->data.size(); ++i) {
                p->data[i] -= lr * p->grads[i];
            }
        }
    }

    void zero_grad() override {
        for (auto& p : params) {
            if (!p || !p->requires_grad) continue;
            std::fill(p->grads.begin(), p->grads.end(), 0.0f);
        }
    }
};


static inline std::shared_ptr<Tensor> bce_prob(
    const std::shared_ptr<Tensor>& p,
    const std::shared_ptr<Tensor>& y
) {
    if (p->rows != y->rows || p->cols != y->cols)
        throw std::runtime_error("BCE: shape mismatch");

    const double eps = 1e-7;
    const int N = p->rows * p->cols;

    auto loss = TensorUtils::make_tensor(1, 1, true, {p, y});
    loss->data.assign(1, 0.0);
    loss->grads.assign(1, 0.0);

    // forward: mean BCE
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double pi = std::clamp<double>(p->data[i], eps, 1.0 - eps);
        double yi = y->data[i]; // expect 0 or 1
        sum += -(yi * std::log(pi) + (1.0 - yi) * std::log(1.0 - pi));
    }
    loss->data[0] = sum / static_cast<double>(N);

    // backward: dL/dp = (-(y/p) + (1-y)/(1-p)) / N
    loss->backward_fn = [loss, p, y, N, eps]() {
        const double g = loss->grads[0] / static_cast<double>(N); // mean reduction

        if (p->requires_grad) {
            if (p->grads.size() != p->data.size())
                p->grads.assign(p->data.size(), 0.0);

            for (int i = 0; i < N; ++i) {
                double pi = std::clamp<double>(p->data[i], eps, 1.0 - eps);
                double yi = y->data[i];

                double dLdp = (-(yi / pi) + (1.0 - yi) / (1.0 - pi));
                p->grads[i] += dLdp * g;
            }
        }
        // usually target y has requires_grad=false, so we skip y.grad
    };

    return loss;
}


class Training {
public:
    void train(const std::shared_ptr<Tensor>& input,
                 const std::shared_ptr<Tensor>& target, Model &model, 
            Optimizer &optimizer,
            LossFn loss_fn,
            const OperatingMode& mode=OperatingMode{}) {
        const size_t max_epoch=1000;
        for(auto epoch=0;epoch<max_epoch;epoch++) {
            if (mode.print_epoch) {
                std::cout << "---------------\n";
            }
            const auto yhat = model.forward_fused(input, mode);
            auto loss = loss_fn(yhat, target);

            auto topo = TensorUtils::build_topo(loss);
            if (loss->grads.empty())
                loss->grads.assign(loss->data.size(), 0.0);
            if (!loss->grads.empty()) loss->grads[0] = 1.0;
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                if ((*it)->backward_fn) (*it)->backward_fn();
            }

            optimizer.step();
            optimizer.zero_grad();
        }
    }

};

void run_training(const OperatingMode& mode);
