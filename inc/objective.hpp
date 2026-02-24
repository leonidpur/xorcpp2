#pragma once

#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

class Objective {
public:
    virtual std::shared_ptr<Tensor> activate(const std::shared_ptr<Tensor>& z) = 0;
    virtual std::shared_ptr<Tensor> calculate_loss(
        const std::shared_ptr<Tensor>& p,
        const std::shared_ptr<Tensor>& y
    ) = 0;
    virtual void backprop(const std::shared_ptr<Tensor>& loss) = 0;
    virtual ~Objective() = default;
};

class BinaryClassificationObjective : public Objective {
public:
    std::shared_ptr<Tensor> activate(const std::shared_ptr<Tensor>& z) override {
        return TensorUtils::sigmoid(z);
    }

    std::shared_ptr<Tensor> calculate_loss(
        const std::shared_ptr<Tensor>& p,
        const std::shared_ptr<Tensor>& y
    ) override {
        if (p->rows != y->rows || p->cols != y->cols)
            throw std::runtime_error("BCE: shape mismatch");

        const double eps = 1e-7;
        const int N = static_cast<int>(p->rows * p->cols);

        auto loss = TensorUtils::make_tensor(1, 1, true, {p, y}, "loss");
        loss->data.assign(1, 0.0);
        loss->grads.assign(1, 0.0);

        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            double pi = std::clamp<double>(p->data[i], eps, 1.0 - eps);
            double yi = y->data[i];
            sum += -(yi * std::log(pi) + (1.0 - yi) * std::log(1.0 - pi));
        }
        loss->data[0] = sum / static_cast<double>(N);

        return loss;
    }

    void backprop(const std::shared_ptr<Tensor>& loss) override {
        if (!loss || loss->parents.size() < 2) return;
        auto p = loss->parents[0];
        auto y = loss->parents[1];
        const double eps = 1e-7;
        const int N = static_cast<int>(p->rows * p->cols);
        loss->backward_fn = [loss, p, y, N, eps]() {
            const double g = loss->grads[0] / static_cast<double>(N);

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
        };
    }
};
