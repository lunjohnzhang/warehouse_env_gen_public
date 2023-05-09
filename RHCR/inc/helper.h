#pragma once
#include <vector>
// #include<bits/stdc++.h>

namespace helper {
    std::tuple<double, double> mean_std(std::vector<double> v);
    void divide(std::vector<double> &v, double factor);
    double sum(std::vector<double> v);
}