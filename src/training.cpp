#include "training.hpp"
#include <iostream>

void run_training(const OperatingMode& mode) {
    auto input = TensorUtils::make_tensor(4, 2);
    auto target = TensorUtils::make_tensor(4, 1);

    const double in_vals[8] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    const double tgt_vals[4] = {0.0, 1.0, 1.0, 0.0};
    for (size_t i = 0; i < 8; ++i) input->data[i] = in_vals[i];
    for (size_t i = 0; i < 4; ++i) target->data[i] = tgt_vals[i];

    Model model(2, 2, 1);
    std::vector<std::shared_ptr<Tensor>> params;
    SGD optimizer(params, 0.1f);

    Training trainer;
    trainer.train(input, target, model, optimizer, bce_prob, mode);

    std::cout << "training done\n";
}
