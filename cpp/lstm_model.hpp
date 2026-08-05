// Keras LSTM+Dense+Dense modelinin bağımsız, elle yazılmış C++ ileri geçiş uygulaması.
#pragma once

#include <array>
#include <cmath>
#include <string>

namespace lstm_model {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Weights {
    const float* lstm_kernel;
    const float* lstm_recurrent_kernel;
    const float* lstm_bias;
    const float* dense1_kernel;
    const float* dense1_bias;
    const float* dense2_kernel;
    const float* dense2_bias;
    const float* scaler_mean;
    const float* scaler_scale;
    const char* const* class_names;
    int window_size;
    int n_features;
    int units;
    int dense1_units;
    int n_classes;
    int horizon;
};

struct Prediction {
    int class_index;
    std::string class_name;
    float confidence;
    std::array<float, 32> probabilities;
};

inline Prediction predict(const Weights& w, const float* window) {
    std::array<float, 128> h{};
    std::array<float, 128> c{};
    std::array<float, 512> gates{};
    std::array<float, 32> scaled_row{};

    for (int t = 0; t < w.window_size; ++t) {
        const float* raw_row = window + t * w.n_features;
        for (int f = 0; f < w.n_features; ++f) {
            scaled_row[f] = (raw_row[f] - w.scaler_mean[f]) / w.scaler_scale[f];
        }

        for (int g = 0; g < 4 * w.units; ++g) {
            float z = w.lstm_bias[g];
            for (int f = 0; f < w.n_features; ++f) {
                z += scaled_row[f] * w.lstm_kernel[f * 4 * w.units + g];
            }
            for (int u = 0; u < w.units; ++u) {
                z += h[u] * w.lstm_recurrent_kernel[u * 4 * w.units + g];
            }
            gates[g] = z;
        }

        for (int u = 0; u < w.units; ++u) {
            float i = sigmoid(gates[u]);
            float f_gate = sigmoid(gates[w.units + u]);
            float c_candidate = std::tanh(gates[2 * w.units + u]);
            float o = sigmoid(gates[3 * w.units + u]);
            c[u] = f_gate * c[u] + i * c_candidate;
            h[u] = o * std::tanh(c[u]);
        }
    }

    std::array<float, 64> dense1{};
    for (int j = 0; j < w.dense1_units; ++j) {
        float z = w.dense1_bias[j];
        for (int u = 0; u < w.units; ++u) {
            z += h[u] * w.dense1_kernel[u * w.dense1_units + j];
        }
        dense1[j] = z > 0.0f ? z : 0.0f;
    }

    std::array<float, 32> logits{};
    float max_logit = -1e30f;
    for (int k = 0; k < w.n_classes; ++k) {
        float z = w.dense2_bias[k];
        for (int j = 0; j < w.dense1_units; ++j) {
            z += dense1[j] * w.dense2_kernel[j * w.n_classes + k];
        }
        logits[k] = z;
        if (z > max_logit) max_logit = z;
    }

    Prediction pred{};
    float sum_exp = 0.0f;
    for (int k = 0; k < w.n_classes; ++k) {
        float e = std::exp(logits[k] - max_logit);
        pred.probabilities[k] = e;
        sum_exp += e;
    }
    int best_k = 0;
    for (int k = 0; k < w.n_classes; ++k) {
        pred.probabilities[k] /= sum_exp;
        if (pred.probabilities[k] > pred.probabilities[best_k]) best_k = k;
    }
    pred.class_index = best_k;
    pred.class_name = w.class_names[best_k];
    pred.confidence = pred.probabilities[best_k];
    return pred;
}

}  // namespace lstm_model
