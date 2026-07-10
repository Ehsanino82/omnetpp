#ifndef __DQN_POLICY_H
#define __DQN_POLICY_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

class DqnPolicy {
public:
    void load(const std::string& dir) {
        loadMatrix(dir + "/w1.csv", W1, rows1, cols1);
        loadVector(dir + "/b1.csv", b1);
        loadMatrix(dir + "/w2.csv", W2, rows2, cols2);
        loadVector(dir + "/b2.csv", b2);
        loadMatrix(dir + "/w3.csv", W3, rows3, cols3);
        loadVector(dir + "/b3.csv", b3);
        loadScaler(dir + "/scaler.csv");
        loaded = true;
    }

    int act(const std::vector<double>& rawState) const {
        if (!loaded)
            throw std::runtime_error("DqnPolicy: weights not loaded");

        // Normalize
        std::vector<double> x(rawState.size());
        for (size_t i = 0; i < rawState.size(); i++) {
            double s = (statStd.size() > i && statStd[i] > 1e-8) ? statStd[i] : 1.0;
            x[i] = (rawState[i] - statMean[i]) / s;
        }

        // Layer 1: ReLU(W1 * x + b1)
        std::vector<double> h1(rows1);
        matvec(W1, rows1, cols1, x, b1, h1);
        relu(h1);

        // Layer 2: ReLU(W2 * h1 + b2)
        std::vector<double> h2(rows2);
        matvec(W2, rows2, cols2, h1, b2, h2);
        relu(h2);

        // Layer 3: W3 * h2 + b3 (Q-values)
        std::vector<double> q(rows3);
        matvec(W3, rows3, cols3, h2, b3, q);

        // Argmax
        return (int)(std::max_element(q.begin(), q.end()) - q.begin());
    }

    bool isLoaded() const { return loaded; }

private:
    bool loaded = false;
    std::vector<double> W1, W2, W3;
    std::vector<double> b1, b2, b3;
    int rows1=0, cols1=0, rows2=0, cols2=0, rows3=0, cols3=0;
    std::vector<double> statMean, statStd;

    static void matvec(const std::vector<double>& M, int rows, int cols,
                       const std::vector<double>& x,
                       const std::vector<double>& bias,
                       std::vector<double>& out) {
        for (int r = 0; r < rows; r++) {
            double sum = bias[r];
            for (int c = 0; c < cols; c++)
                sum += M[r * cols + c] * x[c];
            out[r] = sum;
        }
    }

    static void relu(std::vector<double>& v) {
        for (auto& x : v)
            if (x < 0) x = 0;
    }

    void loadMatrix(const std::string& path, std::vector<double>& M, int& rows, int& cols) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open " + path);
        M.clear();
        rows = 0; cols = 0;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            int c = 0;
            while (std::getline(ss, cell, ',')) {
                M.push_back(std::stod(cell));
                c++;
            }
            if (rows == 0) cols = c;
            rows++;
        }
    }

    void loadVector(const std::string& path, std::vector<double>& v) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open " + path);
        v.clear();
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            v.push_back(std::stod(line));
        }
    }

    void loadScaler(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open " + path);
        statMean.clear();
        statStd.clear();
        std::string line;
        std::getline(f, line); // header: "mean,std"
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::getline(ss, cell, ',');
            statMean.push_back(std::stod(cell));
            std::getline(ss, cell, ',');
            statStd.push_back(std::stod(cell));
        }
    }
};

#endif
