#include "training.hpp"
#include <iostream>

void run_training(const OperatingMode& mode) {
    auto input = TensorUtils::make_tensor(4, 2, false, {}, "input");
    auto target = TensorUtils::make_tensor(4, 1, false, {}, "target");

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
    auto params = model.parameters();
    SGD optimizer(params, 0.1f);
    BinaryClassificationObjective objective;

    Training trainer;
    trainer.train(input, target, model, optimizer, objective, mode);

    std::cout << "training done\n";
}
