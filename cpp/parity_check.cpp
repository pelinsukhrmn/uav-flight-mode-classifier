// cpp/verify_parity.py'nin ürettiği referans olasılıkları C++ ileri geçişiyle karşılaştırıp doğrular.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lstm_model.hpp"
#include "weights_current_fault.h"
#include "weights_next_fault.h"

namespace {

lstm_model::Weights make_weights_current() {
    using namespace current_fault;
    return {lstm_kernel, lstm_recurrent_kernel, lstm_bias, dense1_kernel, dense1_bias,
            dense2_kernel, dense2_bias, scaler_mean, scaler_scale, CLASS_NAMES,
            WINDOW_SIZE, N_FEATURES, LSTM_UNITS, DENSE1_UNITS, N_CLASSES, HORIZON};
}

lstm_model::Weights make_weights_next() {
    using namespace next_fault;
    return {lstm_kernel, lstm_recurrent_kernel, lstm_bias, dense1_kernel, dense1_bias,
            dense2_kernel, dense2_bias, scaler_mean, scaler_scale, CLASS_NAMES,
            WINDOW_SIZE, N_FEATURES, LSTM_UNITS, DENSE1_UNITS, N_CLASSES, HORIZON};
}

}  // namespace

int main() {
    std::ifstream in("cpp/parity_data.txt");
    if (!in) {
        std::fprintf(stderr, "Could not open cpp/parity_data.txt - run cpp/verify_parity.py first.\n");
        return 1;
    }

    int total_windows = 0;
    int total_mismatched_labels = 0;
    double max_prob_diff = 0.0;

    std::string line;
    std::string model_name;
    int n_windows = 0, n_features = 0, window_size = 0, n_classes = 0;
    lstm_model::Weights weights{};
    bool have_weights = false;

    while (std::getline(in, line)) {
        if (line.rfind("MODEL", 0) == 0) {
            std::istringstream header(line);
            std::string tag;
            header >> tag >> model_name >> n_windows >> n_features >> window_size >> n_classes;
            weights = (model_name == "fault") ? make_weights_current() : make_weights_next();
            have_weights = true;
            std::printf("\n=== %s: %d windows, %d features, window=%d, classes=%d ===\n",
                        model_name.c_str(), n_windows, n_features, window_size, n_classes);
            continue;
        }
        if (!have_weights || line.empty()) continue;

        auto sep = line.find(" | ");
        std::istringstream window_stream(line.substr(0, sep));
        std::istringstream prob_stream(line.substr(sep + 3));

        std::vector<float> window(window_size * n_features);
        for (float& v : window) window_stream >> v;
        std::vector<float> ref_probs(n_classes);
        for (float& v : ref_probs) prob_stream >> v;

        lstm_model::Prediction pred = lstm_model::predict(weights, window.data());

        int ref_best = 0;
        for (int k = 1; k < n_classes; ++k) {
            if (ref_probs[k] > ref_probs[ref_best]) ref_best = k;
        }
        if (pred.class_index != ref_best) total_mismatched_labels++;

        double diff = 0.0;
        for (int k = 0; k < n_classes; ++k) {
            double d = std::abs(pred.probabilities[k] - ref_probs[k]);
            if (d > diff) diff = d;
        }
        if (diff > max_prob_diff) max_prob_diff = diff;
        total_windows++;
    }

    std::printf("\n=== Parity summary ===\n");
    std::printf("Windows checked: %d\n", total_windows);
    std::printf("Label mismatches (C++ argmax vs Keras argmax): %d\n", total_mismatched_labels);
    std::printf("Max abs probability difference: %.6f\n", max_prob_diff);

    bool ok = total_mismatched_labels == 0 && max_prob_diff < 1e-3;
    std::printf("%s\n", ok ? "PASS - C++ forward pass matches the trained Keras model."
                            : "FAIL - see diffs above.");
    return ok ? 0 : 1;
}
