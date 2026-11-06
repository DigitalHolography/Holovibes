#include "gtest/gtest.h"
#include "rolling_image_average.hh"
#include <algorithm>
#include <array>

namespace
{
TEST(RollingImageAverageTest, AveragesAvailableSignedMapsAndDropsTheOldest)
{
    holovibes::compute::RollingImageAverage average(3);
    const std::array<std::array<float, 4>, 5> values = {{
        {{-8, -2, 4, 12}}, {{-4, 2, 8, 16}}, {{0, 6, 12, 20}},
        {{4, 10, 16, 24}}, {{8, 14, 20, 28}}
    }};
    for (size_t t = 0; t < values.size(); ++t)
    {
        cv::Mat input(2, 2, CV_32F);
        std::copy(values[t].begin(), values[t].end(), input.ptr<float>());
        const cv::Mat actual = average.apply(input);
        input.setTo(10000); // History must own its pixels, independently of the caller.
        const size_t first = t >= 2 ? t-2 : 0;
        for (size_t p = 0; p < 4; ++p)
        {
            double expected = 0.;
            for (size_t i = first; i <= t; ++i)
                expected += values[i][p];
            expected /= t-first+1;
            EXPECT_FLOAT_EQ(actual.ptr<float>()[p], static_cast<float>(expected));
        }
    }
}

TEST(RollingImageAverageTest, LevelOnePreservesEveryValue)
{
    holovibes::compute::RollingImageAverage average(1);
    cv::Mat input = (cv::Mat_<float>(2, 2) << -.5f, 0.f, .123f, 1.5f);
    const auto output = average.apply(input);
    EXPECT_EQ(cv::norm(output, input, cv::NORM_INF), 0.);
}

TEST(RollingImageAverageTest, RejectsInvalidLevelsTypesAndDimensions)
{
    EXPECT_THROW((holovibes::compute::RollingImageAverage(0)), std::invalid_argument);
    holovibes::compute::RollingImageAverage average(2);
    EXPECT_THROW(average.apply(cv::Mat()), std::invalid_argument);
    EXPECT_THROW(average.apply(cv::Mat::zeros(2, 2, CV_8U)), std::invalid_argument);
    average.apply(cv::Mat::ones(2, 2, CV_32F));
    EXPECT_THROW(average.apply(cv::Mat::ones(3, 2, CV_32F)), std::invalid_argument);
}
}
