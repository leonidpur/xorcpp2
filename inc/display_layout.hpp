#pragma once

#include "operating_mode.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class DisplayLayout {
    struct MatrixData {
        size_t rows = 0;
        size_t cols = 0;
        std::vector<double> data;
        std::string name;
    };

    struct LayerBlock {
        MatrixData w;
        MatrixData b;
        MatrixData z;
        MatrixData a;
        MatrixData gw;
        MatrixData gb;
        MatrixData gz;
        MatrixData ga;
        bool has_w = false;
        bool has_b = false;
        bool has_z = false;
        bool has_a = false;
        bool has_gw = false;
        bool has_gb = false;
        bool has_gz = false;
        bool has_ga = false;
        const Tensor* w_id = nullptr;
        const Tensor* b_id = nullptr;
        const Tensor* z_id = nullptr;
        const Tensor* a_id = nullptr;
    };

    std::map<int, LayerBlock> layers;
    MatrixData loss;
    bool has_loss = false;
    std::vector<MatrixData> grads_extra;
    size_t epoch = 0;

    static MatrixData copy_tensor(const std::shared_ptr<Tensor>& t) {
        MatrixData m;
        if (!t) return m;
        m.rows = t->rows;
        m.cols = t->cols;
        m.data = t->data;
        m.name = t->name;
        return m;
    }

    static std::vector<std::string> format_rows(const MatrixData& m) {
        if (m.rows == 0 || m.cols == 0) return {};
        std::vector<std::string> cells;
        cells.reserve(m.rows * m.cols);
        size_t width = 0;
        for (size_t r = 0; r < m.rows; r++) {
            for (size_t c = 0; c < m.cols; c++) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << m.data[r * m.cols + c];
                const std::string s = oss.str();
                if (s.size() > width) width = s.size();
                cells.push_back(s);
            }
        }
        std::vector<std::string> rows_out;
        rows_out.reserve(m.rows);
        size_t idx = 0;
        for (size_t r = 0; r < m.rows; r++) {
            std::ostringstream line;
            for (size_t c = 0; c < m.cols; c++) {
                if (c > 0) line << ' ';
                line << std::setw(static_cast<int>(width)) << cells[idx++];
            }
            rows_out.push_back(line.str());
        }
        return rows_out;
    }

    static void print_side_by_side(
        const MatrixData& left,
        const std::string& left_label,
        const MatrixData& right,
        const std::string& right_label
    ) {
        const std::string left_header = left_label;
        const std::string right_header = right_label;

        const auto left_rows = format_rows(left);
        const auto right_rows = format_rows(right);

        size_t left_width = left_header.size();
        for (const auto& row : left_rows) {
            if (row.size() > left_width) left_width = row.size();
        }

        std::cout << std::setw(static_cast<int>(left_width)) << left_header
                  << " | " << right_header << "\n";

        const size_t max_rows = std::max(left_rows.size(), right_rows.size());
        for (size_t r = 0; r < max_rows; r++) {
            const std::string left_row = (r < left_rows.size()) ? left_rows[r] : "";
            const std::string right_row = (r < right_rows.size()) ? right_rows[r] : "";
            std::cout << std::setw(static_cast<int>(left_width)) << left_row
                      << " | " << right_row << "\n";
        }
    }

    static void print_four_side_by_side(
        const MatrixData& w,
        const MatrixData& b,
        const MatrixData& z,
        const MatrixData& a,
        const std::string& w_header,
        const std::string& b_header,
        const std::string& z_header,
        const std::string& a_header
    ) {

        const auto w_rows = format_rows(w);
        const auto b_rows = format_rows(b);
        const auto z_rows = format_rows(z);
        const auto a_rows = format_rows(a);

        size_t w_width = w_header.size();
        size_t b_width = b_header.size();
        size_t z_width = z_header.size();
        size_t a_width = a_header.size();
        for (const auto& row : w_rows) if (row.size() > w_width) w_width = row.size();
        for (const auto& row : b_rows) if (row.size() > b_width) b_width = row.size();
        for (const auto& row : z_rows) if (row.size() > z_width) z_width = row.size();
        for (const auto& row : a_rows) if (row.size() > a_width) a_width = row.size();

        std::cout << std::setw(static_cast<int>(w_width)) << w_header
                  << " | " << std::setw(static_cast<int>(b_width)) << b_header
                  << " | " << std::setw(static_cast<int>(z_width)) << z_header
                  << " | " << std::setw(static_cast<int>(a_width)) << a_header
                  << "\n";

        const size_t max_rows = std::max(std::max(w_rows.size(), b_rows.size()),
                                         std::max(z_rows.size(), a_rows.size()));
        for (size_t r = 0; r < max_rows; r++) {
            const std::string w_row = (r < w_rows.size()) ? w_rows[r] : "";
            const std::string b_row = (r < b_rows.size()) ? b_rows[r] : "";
            const std::string z_row = (r < z_rows.size()) ? z_rows[r] : "";
            const std::string a_row = (r < a_rows.size()) ? a_rows[r] : "";
            std::cout << std::setw(static_cast<int>(w_width)) << w_row
                      << " | " << std::setw(static_cast<int>(b_width)) << b_row
                      << " | " << std::setw(static_cast<int>(z_width)) << z_row
                      << " | " << std::setw(static_cast<int>(a_width)) << a_row
                      << "\n";
        }
    }

    static void print_matrix(const MatrixData& m, const std::string& label) {
        std::cout << label << "\n";
        const auto rows_out = format_rows(m);
        for (const auto& line : rows_out) {
            std::cout << line << "\n";
        }
    }

public:
    void start_epoch(size_t epoch_) {
        epoch = epoch_;
        layers.clear();
        grads_extra.clear();
        has_loss = false;
    }

    void add_weights(int layer, const std::shared_ptr<Tensor>& weights) {
        layers[layer].w = copy_tensor(weights);
        layers[layer].has_w = true;
        layers[layer].w_id = weights.get();
    }

    void add_bias(int layer, const std::shared_ptr<Tensor>& biases) {
        layers[layer].b = copy_tensor(biases);
        layers[layer].has_b = true;
        layers[layer].b_id = biases.get();
    }

    void add_z(int layer, const std::shared_ptr<Tensor>& z) {
        layers[layer].z = copy_tensor(z);
        layers[layer].has_z = true;
        layers[layer].z_id = z.get();
    }

    void add_a(int layer, const std::shared_ptr<Tensor>& a) {
        layers[layer].a = copy_tensor(a);
        layers[layer].has_a = true;
        layers[layer].a_id = a.get();
    }

    void add_loss(const std::shared_ptr<Tensor>& loss_) {
        loss = copy_tensor(loss_);
        has_loss = true;
    }

    void add_grad(const std::shared_ptr<Tensor>& grad_) {
        if (!grad_) return;
        const Tensor* id = grad_.get();
        for (auto& entry : layers) {
            LayerBlock& layer = entry.second;
            if (layer.w_id == id) {
                layer.gw = copy_tensor(grad_);
                layer.has_gw = true;
                return;
            }
            if (layer.b_id == id) {
                layer.gb = copy_tensor(grad_);
                layer.has_gb = true;
                return;
            }
            if (layer.z_id == id) {
                layer.gz = copy_tensor(grad_);
                layer.has_gz = true;
                return;
            }
            if (layer.a_id == id) {
                layer.ga = copy_tensor(grad_);
                layer.has_ga = true;
                return;
            }
        }
        grads_extra.push_back(copy_tensor(grad_));
    }

    void flush(const OperatingMode& mode) const {
        if (!mode.print_epoch && !mode.print_params && !mode.print_temp) return;

        if (mode.print_epoch) {
            std::cout << "--------------- epoch " << epoch << "\n";
        }

        if (mode.print_params || mode.print_temp) {
            auto it = layers.find(0);
            if (it != layers.end()) {
                std::cout << "hidden\n";
                if (mode.print_params && mode.print_temp &&
                    it->second.has_w && it->second.has_b &&
                    it->second.has_z && it->second.has_a) {
                    print_four_side_by_side(
                        it->second.w,
                        it->second.b,
                        it->second.z,
                        it->second.a,
                        it->second.w.name,
                        it->second.b.name,
                        it->second.z.name,
                        it->second.a.name
                    );
                } else {
                    if (mode.print_params && it->second.has_w && it->second.has_b)
                        print_side_by_side(it->second.w, it->second.w.name,
                                           it->second.b, it->second.b.name);
                    if (mode.print_temp && it->second.has_z && it->second.has_a)
                        print_side_by_side(it->second.z, it->second.z.name,
                                           it->second.a, it->second.a.name);
                }
            }

            it = layers.find(1);
            if (it != layers.end()) {
                std::cout << "output\n";
                if (mode.print_params && mode.print_temp &&
                    it->second.has_w && it->second.has_b &&
                    it->second.has_z && it->second.has_a) {
                    print_four_side_by_side(
                        it->second.w,
                        it->second.b,
                        it->second.z,
                        it->second.a,
                        it->second.w.name,
                        it->second.b.name,
                        it->second.z.name,
                        it->second.a.name
                    );
                } else {
                    if (mode.print_params && it->second.has_w && it->second.has_b)
                        print_side_by_side(it->second.w, it->second.w.name,
                                           it->second.b, it->second.b.name);
                    if (mode.print_temp && it->second.has_z && it->second.has_a)
                        print_side_by_side(it->second.z, it->second.z.name,
                                           it->second.a, it->second.a.name);
                }
            }
        }

        if (mode.print_temp) {
            if (has_loss) {
                print_matrix(loss, loss.name.empty() ? "loss" : loss.name);
            }
            if (!layers.empty() || !grads_extra.empty()) {
                std::cout << "grads\n";
                auto it = layers.find(1);
                if (it != layers.end()) {
                    if (it->second.has_gw && it->second.has_gb &&
                        it->second.has_gz && it->second.has_ga) {
                        print_four_side_by_side(
                            it->second.ga,
                            it->second.gz,
                            it->second.gb,
                            it->second.gw,
                            it->second.a.name + ".grad",
                            it->second.z.name + ".grad",
                            it->second.b.name + ".grad",
                            it->second.w.name + ".grad"
                        );
                    } else {
                        if (it->second.has_ga && it->second.has_gz)
                            print_side_by_side(it->second.ga, it->second.a.name + ".grad",
                                               it->second.gz, it->second.z.name + ".grad");
                        if (it->second.has_gb && it->second.has_gw)
                            print_side_by_side(it->second.gb, it->second.b.name + ".grad",
                                               it->second.gw, it->second.w.name + ".grad");
                    }
                }
                it = layers.find(0);
                if (it != layers.end()) {
                    if (it->second.has_gw && it->second.has_gb &&
                        it->second.has_gz && it->second.has_ga) {
                        print_four_side_by_side(
                            it->second.ga,
                            it->second.gz,
                            it->second.gb,
                            it->second.gw,
                            it->second.a.name + ".grad",
                            it->second.z.name + ".grad",
                            it->second.b.name + ".grad",
                            it->second.w.name + ".grad"
                        );
                    } else {
                        if (it->second.has_ga && it->second.has_gz)
                            print_side_by_side(it->second.ga, it->second.a.name + ".grad",
                                               it->second.gz, it->second.z.name + ".grad");
                        if (it->second.has_gb && it->second.has_gw)
                            print_side_by_side(it->second.gb, it->second.b.name + ".grad",
                                               it->second.gw, it->second.w.name + ".grad");
                    }
                }
                for (const auto& g : grads_extra) {
                    const std::string label = g.name.empty() ? "grad" : (g.name + ".grad");
                    print_matrix(g, label);
                }
            }
        }
    }
};
