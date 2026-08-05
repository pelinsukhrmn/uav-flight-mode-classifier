// lstm_model::predict()'in izole gecikme ölçümünü yapar.
#include <chrono>
#include <cstdio>
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
    using namespace std::chrono;

    auto weights_current = make_weights_current();
    auto weights_next = make_weights_next();
    std::vector<float> window(weights_current.window_size * weights_current.n_features, 0.1f);

    const int warmup = 500;
    const int n = 2000;
    for (int i = 0; i < warmup; ++i) {
        lstm_model::predict(weights_current, window.data());
        lstm_model::predict(weights_next, window.data());
    }

    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        auto pc = lstm_model::predict(weights_current, window.data());
        auto pn = lstm_model::predict(weights_next, window.data());
        if (pc.class_index < 0 || pn.class_index < 0) std::printf("");
    }
    auto t1 = high_resolution_clock::now();

    double total_us = duration_cast<duration<double, std::micro>>(t1 - t0).count();
    std::printf("%d iterations, both models: %.1f us total\n", n, total_us);
    std::printf("Per iteration (both models): %.2f us\n", total_us / n);
    std::printf("Per single-model inference: %.2f us\n", total_us / (2 * n));
    return 0;
}
