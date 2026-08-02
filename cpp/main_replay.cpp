// C++ port of live_inference.py's replay-mode decision-support loop: streams
// raw feature rows from a CSV (see export_replay_csv.py), maintains a
// rolling window buffer, and scores both the current-mode and next-mode
// models as each window completes. Advisory only - same as the Python
// version, this never sends anything back to a vehicle.
//
// MAVLink live ingestion is NOT implemented here (unlike the Python
// prototype's --mode mavlink): vendoring the MAVLink C headers and standing
// up a SITL connection just to leave it untested is not worth it. This CSV
// replay path proves the numerics and the streaming/windowing logic; wiring
// a real MAVLink source in later just means writing one more row-producer
// with the same interface, once there's a connection to actually test it against.
#include <cstdio>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lstm_model.hpp"
#include "weights_current_mode.h"
#include "weights_next_mode.h"

namespace {

struct RawRow {
    double timestamp_us;
    float vertical_speed, horizontal_speed, roll_angle, pitch_angle;
};

lstm_model::Weights make_weights_current() {
    using namespace current_mode;
    return {lstm_kernel, lstm_recurrent_kernel, lstm_bias, dense1_kernel, dense1_bias,
            dense2_kernel, dense2_bias, scaler_mean, scaler_scale, CLASS_NAMES,
            WINDOW_SIZE, N_FEATURES, LSTM_UNITS, DENSE1_UNITS, N_CLASSES, HORIZON};
}

lstm_model::Weights make_weights_next() {
    using namespace next_mode;
    return {lstm_kernel, lstm_recurrent_kernel, lstm_bias, dense1_kernel, dense1_bias,
            dense2_kernel, dense2_bias, scaler_mean, scaler_scale, CLASS_NAMES,
            WINDOW_SIZE, N_FEATURES, LSTM_UNITS, DENSE1_UNITS, N_CLASSES, HORIZON};
}

// Builds the 8-feature (4 raw + 4 delta) window from window_size+1 buffered
// raw rows, matching flight_mode_inference.build_windows' ALL_FEATURES order
// and its diff().fillna(0) delta convention.
std::vector<float> build_window(const std::deque<RawRow>& buf, int window_size) {
    std::vector<float> window(window_size * 8);
    for (int t = 0; t < window_size; ++t) {
        const RawRow& cur = buf[t + 1];
        const RawRow& prev = buf[t];
        float* row = &window[t * 8];
        row[0] = cur.vertical_speed;
        row[1] = cur.horizontal_speed;
        row[2] = cur.roll_angle;
        row[3] = cur.pitch_angle;
        row[4] = cur.vertical_speed - prev.vertical_speed;
        row[5] = cur.horizontal_speed - prev.horizontal_speed;
        row[6] = cur.roll_angle - prev.roll_angle;
        row[7] = cur.pitch_angle - prev.pitch_angle;
    }
    return window;
}

std::string advisory(const std::string& current_label, float current_conf,
                      const std::string& next_label, float next_conf) {
    if (next_label == current_label) return "";
    bool urgent = (next_label == "anomaly" || next_label == "land" || next_label == "rtl") && next_conf >= 0.6f;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "   [%s] su an: %s (%.2f) -> tahmin: %s (%.2f)",
                  urgent ? "UYARI" : "bilgi", current_label.c_str(), current_conf, next_label.c_str(), next_conf);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string csv_path = argc > 1 ? argv[1] : "cpp/replay_data.csv";
    double min_interval_s = argc > 2 ? std::stod(argv[2]) : 1.0;

    std::ifstream in(csv_path);
    if (!in) {
        std::fprintf(stderr, "Could not open %s - run cpp/export_replay_csv.py first.\n", csv_path.c_str());
        return 1;
    }
    std::string header;
    std::getline(in, header);  // skip CSV header

    auto weights_current = make_weights_current();
    auto weights_next = make_weights_next();
    const int window_size = weights_current.window_size;
    const int horizon = weights_next.horizon;

    std::deque<RawRow> buffer;
    double start_ts = -1.0, last_scored_ts = -1.0;
    long n_rows = 0;

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string field;
        RawRow row{};
        std::getline(ss, field, ','); row.timestamp_us = std::stod(field);
        std::getline(ss, field, ','); row.vertical_speed = std::stof(field);
        std::getline(ss, field, ','); row.horizontal_speed = std::stof(field);
        std::getline(ss, field, ','); row.roll_angle = std::stof(field);
        std::getline(ss, field, ','); row.pitch_angle = std::stof(field);

        buffer.push_back(row);
        if (static_cast<int>(buffer.size()) > window_size + 1) buffer.pop_front();
        n_rows++;
        if (start_ts < 0) start_ts = row.timestamp_us;

        if (static_cast<int>(buffer.size()) < window_size + 1) continue;
        if (last_scored_ts >= 0 && (row.timestamp_us - last_scored_ts) / 1e6 < min_interval_s) continue;
        last_scored_ts = row.timestamp_us;

        std::vector<float> window = build_window(buffer, window_size);
        lstm_model::Prediction current = lstm_model::predict(weights_current, window.data());
        lstm_model::Prediction next = lstm_model::predict(weights_next, window.data());

        double elapsed = (row.timestamp_us - start_ts) / 1e6;
        double avg_dt = elapsed / (n_rows > 1 ? (n_rows - 1) : 1);
        double horizon_seconds = horizon * avg_dt;

        std::printf("t=%6.1fs  mode=%-10s (%.2f)  next~%4.1fs=%-10s (%.2f)%s\n",
                    elapsed, current.class_name.c_str(), current.confidence,
                    horizon_seconds, next.class_name.c_str(), next.confidence,
                    advisory(current.class_name, current.confidence, next.class_name, next.confidence).c_str());
    }
    return 0;
}
