#include "training.hpp"
#include <string>

int main(int argc, char** argv) {
    OperatingMode mode;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "print-params" || arg == "--print-params") {
            mode.print_params = true;
        } else if (arg == "print-temp" || arg == "--print-temp") {
            mode.print_temp = true;
        } else if (arg == "print-epoch" || arg == "--print-epoch") {
            mode.print_epoch = true;
        } else if (arg == "epochs" || arg == "--epochs") {
            if (i + 1 < argc) {
                mode.max_epoch = static_cast<size_t>(std::stoul(argv[++i]));
            }
        } else if (arg == "loss-threshold" || arg == "--loss-threshold") {
            if (i + 1 < argc) {
                mode.loss_threshold = std::stod(argv[++i]);
            }
        }
    }
    run_training(mode);
    return 0;
}
