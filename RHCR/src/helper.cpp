#include "helper.h"
#include <numeric>
#include <algorithm>
#include <math.h>
#include <tuple>

namespace helper
{
    std::tuple<double, double> mean_std(std::vector<double> v)
    {
        double sum = helper::sum(v);
        double mean = sum / v.size();

        std::vector<double> diff(v.size());
        std::transform(v.begin(), v.end(), diff.begin(),
                       [mean](double x)
                       { return x - mean; });

        double sq_sum = std::inner_product(diff.begin(), diff.end(),
                                           diff.begin(), 0.0);
        double sigma = sqrt(sq_sum / v.size());
        return std::make_tuple(mean, sigma);
    }

    void divide(std::vector<double> &v, double factor)
    {
        for(int i = 0; i < v.size(); i+=1)
        {
            v[i] /= factor;
        }
    }

    double sum(std::vector<double> v)
    {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        return sum;
    }
}
