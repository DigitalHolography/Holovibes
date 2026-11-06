#pragma once

#include <deque>
#include <stdexcept>
#include <opencv2/core.hpp>

namespace holovibes::compute
{
// Average signed, unscaled maps. Startup uses the maps available so far.
class RollingImageAverage
{
  public:
    explicit RollingImageAverage(unsigned level) : level_(level)
    {
        if (!level)
            throw std::invalid_argument("Image accumulation must be at least 1");
    }

    cv::Mat apply(const cv::Mat& map)
    {
        if (map.empty() || map.type() != CV_32FC1)
            throw std::invalid_argument("Image accumulation requires a single-channel float32 map");
        if (level_ == 1)
            return map;
        if (sum_.empty())
            sum_ = cv::Mat::zeros(map.size(), CV_64F);
        if (sum_.size() != map.size())
            throw std::invalid_argument("Image accumulation dimensions changed");
        if (history_.size() == level_)
        {
            cv::subtract(sum_, history_.front(), sum_, cv::noArray(), CV_64F);
            history_.pop_front();
        }
        cv::add(sum_, map, sum_, cv::noArray(), CV_64F);
        history_.push_back(map.clone());
        cv::Mat average;
        sum_.convertTo(average, CV_32F, 1. / history_.size());
        return average;
    }

  private:
    unsigned level_;
    std::deque<cv::Mat> history_;
    cv::Mat sum_;
};
}
