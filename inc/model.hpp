#pragma once

#include "display_layout.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "operating_mode.hpp"
#include <memory>
#include <random>
#include <string>

class Objective;

class Layer {
    shared_ptr<Tensor> w;
    shared_ptr<Tensor> b;
    size_t layer_index;
public:
    Layer(size_t _rows, size_t _cols, size_t _layer_index)
        : w(TensorUtils::make_tensor(_rows, _cols, true, {}, "w" + std::to_string(_layer_index))),
          b(TensorUtils::make_tensor(1, _cols, true, {}, "b" + std::to_string(_layer_index))),
          layer_index(_layer_index) {
        static thread_local std::mt19937 rng(42);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        for (size_t i = 0; i < w->data.size(); ++i) {
            w->data[i] = sign_dist(rng) ? 0.1 : -0.1;
        }
        // b is already zero-initialized.
    }


    shared_ptr<Tensor> forward(const shared_ptr<Tensor> &input) {
        shared_ptr<Tensor> iXw = TensorUtils::matmul(input, w);
        auto z = TensorUtils::add(iXw, b);
        return is_sigmoid?TensorUtils::sigmoid(z):TensorUtils::relu(z);
    }

    shared_ptr<Tensor> forward_fused(
        const shared_ptr<Tensor> &input,
        bool update_input,
        const OperatingMode& mode,
        DisplayLayout* layout,
        Objective* objective
    ) {
        if (layout && mode.print_params) {
            layout->add_weights(static_cast<int>(layer_index), w);
            layout->add_bias(static_cast<int>(layer_index), b);
        }
        auto z = TensorUtils::fused_linear_op(input, w, b, update_input, false);
        auto a = objective ? objective->activate(z) : TensorUtils::relu(z);
        if (layout && mode.print_temp) {
            layout->add_z(static_cast<int>(layer_index), z);
            layout->add_a(static_cast<int>(layer_index), a);
        }
        return a;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return {w, b};
    }
};

class Model {

    vector<Layer> layers;
public:
    Model(const size_t input_dim, const size_t hidden_size, const size_t output_size) {
        layers.emplace_back(Layer(input_dim, hidden_size, 0));
        layers.emplace_back(Layer(hidden_size, output_size, 1));
    }

    auto forward(const std::shared_ptr<Tensor>& input) {
        auto hidden_a = layers[0].forward(input);
        auto y_hat = layers[1].forward(hidden_a);
        return y_hat;
    }

    auto forward_fused(
        const std::shared_ptr<Tensor>& input,
        const OperatingMode& mode,
        DisplayLayout* layout,
        Objective* objective
    ) {
        auto hidden_a = layers[0].forward_fused(input, false, mode, layout, nullptr);
        auto y_hat = layers[1].forward_fused(hidden_a, true, mode, layout, objective);
        return y_hat;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        std::vector<std::shared_ptr<Tensor>> params;
        for (const auto& layer : layers) {
            auto layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

};
