#pragma once

#include "tensor.hpp"
#include "utils.hpp"
#include "operating_mode.hpp"
#include <memory>
#include <random>
#include <string>

class Layer {
    shared_ptr<Tensor> w;
    shared_ptr<Tensor> b;
    bool is_sigmoid;
    size_t layer_index;
public:
    Layer(size_t _rows, size_t _cols, size_t _layer_index, bool _is_sigmoid=true)
        : w(TensorUtils::make_tensor(_rows, _cols)),
          b(TensorUtils::make_tensor(1, _cols)),
          is_sigmoid(_is_sigmoid),
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
        const OperatingMode& mode
    ) {
        if (mode.print_params) {
            const std::string w_label = "{" + std::to_string(layer_index) + "} w";
            const std::string b_label = "{" + std::to_string(layer_index) + "} b";
            w->print_tensor(w_label.c_str());
            b->print_tensor(b_label.c_str());
        }
        auto z = TensorUtils::fused_linear_op(input, w, b, update_input, mode.print_temp);
        auto a = is_sigmoid ? TensorUtils::sigmoid(z) : TensorUtils::relu(z);
        if (mode.print_temp) {
            const std::string z_label = "{" + std::to_string(layer_index) + "} z";
            const std::string a_label = "{" + std::to_string(layer_index) + "} a";
            z->print_tensor(z_label.c_str());
            a->print_tensor(a_label.c_str());
        }
        return a;
    }
};

class Model {

    vector<Layer> layers;
public:
    Model(const size_t input_dim, const size_t hidden_size, const size_t output_size) {
        layers.emplace_back(Layer(input_dim, hidden_size, 0, false));
        layers.emplace_back(Layer(hidden_size, output_size, 1));
    }

    auto forward(const std::shared_ptr<Tensor>& input) {
        auto hidden_a = layers[0].forward(input);
        auto y_hat = layers[1].forward(hidden_a);
        return y_hat;
    }

    auto forward_fused(const std::shared_ptr<Tensor>& input, const OperatingMode& mode) {
        auto hidden_a = layers[0].forward_fused(input, false, mode);
        auto y_hat = layers[1].forward_fused(hidden_a, true, mode);
        return y_hat;
    }

};
