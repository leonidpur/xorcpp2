#pragma once

#include "display_layout.hpp"
#include "model.hpp"
#include "objective.hpp"
#include "operating_mode.hpp"
#include <algorithm>
#include <iostream>


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


class Training {
public:
    void train(const std::shared_ptr<Tensor>& input,
                 const std::shared_ptr<Tensor>& target, Model &model, 
            Optimizer &optimizer,
            Objective &objective,
            const OperatingMode& mode=OperatingMode{}) {
        const size_t max_epoch = mode.max_epoch;
        for(auto epoch=0;epoch<max_epoch;epoch++) {
            DisplayLayout layout;
            layout.start_epoch(static_cast<size_t>(epoch));
            const auto p = model.forward_fused(input, mode, &layout, &objective);
            auto loss = objective.calculate_loss(p, target);
            if (mode.print_temp) layout.add_loss(loss);
            const double loss_val = loss->data.empty() ? 0.0 : loss->data[0];
            bool all_correct = true;
            for (size_t i = 0; i < target->data.size(); ++i) {
                const double pred = p->data[i] >= 0.5 ? 1.0 : 0.0;
                if (pred != target->data[i]) {
                    all_correct = false;
                    break;
                }
            }

            objective.backprop(loss);
            auto topo = TensorUtils::build_topo(loss);
            if (loss->grads.empty())
                loss->grads.assign(loss->data.size(), 0.0);
            if (!loss->grads.empty()) loss->grads[0] = 1.0;
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                if ((*it)->backward_fn) (*it)->backward_fn();
            }
            if (mode.print_temp) {
                for (const auto& node : topo) {
                    if (!node) continue;
                    if (node->requires_grad) {
                        layout.add_grad(node);
                    }
                }
            }
            optimizer.step();
            optimizer.zero_grad();
            layout.flush(mode);

            if (loss_val < mode.loss_threshold && all_correct) {
                std::cout << "satisfied\n";
                break;
            }
        }
    }

};

void run_training(const OperatingMode& mode);
