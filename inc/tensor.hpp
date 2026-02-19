#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <sstream>
#include <vector>
#include <memory>
#include <functional>

using namespace std;

class Tensor {
public:
    vector<double> data;
    vector<double> grads;
    const size_t rows;
    const size_t cols;
    vector<shared_ptr<Tensor>> parents;
    bool requires_grad;
    function<void()> backward_fn;

public:
    Tensor(size_t _rows, size_t _cols, bool requires_grad=false):
        rows(_rows), cols(_cols), requires_grad(requires_grad) {
            data.assign(rows*cols, 0.0);
            grads.assign(rows*cols, 0.0);
        }

    double get(size_t r, size_t c) const {
        return data[r*cols + c];
    }
    void set(size_t r, size_t c, double val) {
        data[r*cols + c] = val;
    }
    double grad(size_t r, size_t c) const {
        return grads[r*cols + c];
    }
    void add_grad(size_t r, size_t c, double val) {
        grads[r*cols + c] += val;
    }
    void zero_grad() {
        fill(grads.begin(), grads.end(), 0.0);
    }

    void print_tensor(const char* name = nullptr) const {
        if (name && name[0] != '\0') {
            std::cout << name << " (" << rows << "x" << cols << ")\n";
        }
        std::vector<std::string> cells;
        cells.reserve(rows * cols);
        size_t width = 0;
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << get(r, c);
                const std::string s = oss.str();
                if (s.size() > width) width = s.size();
                cells.push_back(s);
            }
        }
        size_t idx = 0;
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                if (c > 0) std::cout << ' ';
                std::cout << std::setw(static_cast<int>(width)) << cells[idx++];
            }
            std::cout << '\n';
        }
    }
};


class TensorUtils {
public:
    static shared_ptr<Tensor> make_tensor(
        size_t rows,
        size_t cols,
        bool requires_grad=false,
        const vector<shared_ptr<Tensor>>& parents={}
    ) {
        auto ret = make_shared<Tensor>(rows, cols, requires_grad);
        ret->requires_grad = requires_grad;
        ret->parents = parents;
        return ret;
    }

    static shared_ptr<Tensor> matmul(const shared_ptr<Tensor> &a, 
        const shared_ptr<Tensor> &b) {
        if(a->cols != b->rows)
            return make_tensor(0, 0);

        shared_ptr<Tensor> ret = make_tensor(
            a->rows,
            b->cols,
            a->requires_grad || b->requires_grad,
            {a, b}
        );
        for(int r=0;r<a->rows;r++)
            for(int c=0;c<b->cols;c++) {
                double sum=0.0;
                for(int k=0;k<a->cols;k++)
                    sum+=a->get(r, k) * b->get(k,c);
                ret->set(r, c, sum);
            }

        if(ret->requires_grad && (a->requires_grad||b->requires_grad))
            ret->backward_fn = [ret, a, b]{
                if(a->requires_grad) {
                    for(size_t r=0; r<a->rows; r++)
                        for(size_t c=0; c<a->cols; c++) {
                            double sum = 0.0;
                            for(size_t k=0; k<b->cols; k++)
                                sum += ret->grad(r, k) * b->get(c, k);
                            a->add_grad(r, c, sum);
                        }
                }
                if(b->requires_grad) {
                    for(size_t r=0; r<b->rows; r++)
                        for(size_t c=0; c<b->cols; c++) {
                            double sum = 0.0;
                            for(size_t k=0; k<a->rows; k++)
                                sum += a->get(k, r) * ret->grad(k, c);
                            b->add_grad(r, c, sum);
                        }
                }
            };
        return ret;
    }

    static shared_ptr<Tensor> add(const shared_ptr<Tensor> &a, 
        const shared_ptr<Tensor> &b) {
        if(a->rows != b->rows || a->cols != b->cols)
            return make_tensor(0, 0);

        shared_ptr<Tensor> ret = make_tensor(
            a->rows,
            a->cols,
            a->requires_grad || b->requires_grad,
            {a, b}
        );
        for(int r=0;r < a->rows;r++)
            for(int c=0;c < a->cols;c++)
                ret->set(r, c, a->get(r, c) + b->get(r, c));
            

        if(ret->requires_grad && (a->requires_grad||b->requires_grad))
            ret->backward_fn = [ret, a, b]{
                if(a->requires_grad)
                    for(size_t r=0; r<a->rows; r++)
                        for(size_t c=0; c<a->cols; c++) 
                            a->add_grad(r, c, ret->grad(r, c));

                if(b->requires_grad) 
                    for(size_t r=0; r<b->rows; r++)
                        for(size_t c=0; c<b->cols; c++)
                            b->add_grad(r, c, ret->grad(r, c));
                
            };
        return ret;
    }

    static shared_ptr<Tensor> fused_linear_op(
        const shared_ptr<Tensor> &x,
        const shared_ptr<Tensor> &w,
        const shared_ptr<Tensor> &b,
        bool update_x,
        bool print_calc=false
    ) {
        if(x->cols != w->rows || w->cols != b->cols)
            return make_tensor(0, 0);

        shared_ptr<Tensor> ret = make_tensor(
            x->rows,
            w->cols,
            x->requires_grad || w->requires_grad || b->requires_grad,
            {x, w, b}
        );
        for(int r=0;r<x->rows;r++)
            for(int c=0;c<w->cols;c++) {
                double sum=b->get(0, c);
                if (print_calc) {
                    std::cout << "z[" << r << "," << c << "] = ";
                }
                for(int k=0;k<x->cols;k++) {
                    if (print_calc) {
                        if (k > 0) std::cout << " + ";
                        std::cout << x->get(r, k) << "*" << w->get(k, c);
                    }
                    sum+=x->get(r, k) * w->get(k,c);
                }
                if (print_calc) {
                    std::cout << " + " << b->get(0, c) << " = " << sum << "\n";
                }
                ret->set(r, c, sum);
            }
        const bool track_x = update_x && x->requires_grad;
        if(ret->requires_grad && (track_x||w->requires_grad||b->requires_grad))
            ret->backward_fn = [ret, x, w, b, track_x]{
                if(track_x) {
                    for(size_t r_x=0; r_x < x->rows; r_x++)
                        for(size_t c_x=0; c_x < x->cols; c_x++) {
                            double sum = 0.0;
                            for(size_t k=0; k < w->cols; k++)
                                sum += ret->grad(r_x, k) * w->get(c_x, k);
                            x->add_grad(r_x, c_x, sum);
                        }
                }
                
                if(w->requires_grad) {
                    for(size_t r_w=0; r_w < w->rows; r_w++)
                        for(size_t c_w=0; c_w < w->cols; c_w++) {
                            double sum = 0.0;
                            for(size_t r_x=0; r_x < ret->rows; r_x++)
                                sum += ret->grad(r_x, c_w) * x->get(r_x, r_w);
                            w->add_grad(r_w, c_w, sum);
                        }
                }
                
                if(b->requires_grad)
                    for(size_t c_b=0; c_b < b->cols; c_b++) {
                        double sum = 0.0;
                        for(size_t k=0; k< ret->rows; k++)
                            sum += ret->grad(k, c_b);
                        b->add_grad(0, c_b, sum);
                    }
            };

        return ret;
    }

    static shared_ptr<Tensor> relu(const shared_ptr<Tensor>& a) {
        shared_ptr<Tensor> ret = make_tensor(
            a->rows,
            a->cols,
            a->requires_grad,
            {a}
        );
        for(size_t r=0;r<a->rows;r++)
            for(size_t c=0;c<a->cols;c++)
                ret->set(r, c, max<double>(0, a->get(r,c)));
        if(ret->requires_grad && a->requires_grad)
            ret->backward_fn = [ret, a]{
                for(size_t r=0; r<a->rows; r++)
                    for(size_t c=0; c<a->cols; c++) {
                        double grad = (a->get(r, c) > 0.0) ? ret->grad(r, c) : 0.0;
                        a->add_grad(r, c, grad);
                    }
            };
        return ret;
    }

    static shared_ptr<Tensor> sigmoid(const shared_ptr<Tensor>& a) {
        shared_ptr<Tensor> ret = make_tensor(
            a->rows,
            a->cols,
            a->requires_grad,
            {a}
        );
        for (size_t r = 0; r < a->rows; r++)
            for (size_t c = 0; c < a->cols; c++) {
                double x = a->get(r, c);
                ret->set(r, c, 1.0 / (1.0 + std::exp(-x)));
            }
        if (ret->requires_grad && a->requires_grad)
            ret->backward_fn = [ret, a] {
                for (size_t r = 0; r < a->rows; r++)
                    for (size_t c = 0; c < a->cols; c++) {
                        double y = ret->get(r, c);
                        a->add_grad(r, c, ret->grad(r, c) * y * (1.0 - y));
                    }
            };
        return ret;
    }

    static std::vector<std::shared_ptr<Tensor>> build_topo(
        const std::shared_ptr<Tensor>& root
    ) {
        std::vector<std::shared_ptr<Tensor>> topo;
        std::unordered_set<const Tensor*> visited;
        std::function<void(const std::shared_ptr<Tensor>&)> dfs;
        dfs = [&](const std::shared_ptr<Tensor>& t) {
            if (!t) return;
            if (visited.count(t.get()) > 0) return;
            visited.insert(t.get());
            for (const auto& p : t->parents) dfs(p);
            topo.push_back(t);
        };
        dfs(root);
        return topo;
    }
};
