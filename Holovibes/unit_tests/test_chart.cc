#include "chart.cuh"
#include "cuda_memory.cuh"
#include "gtest/gtest.h"

using holovibes::units::FDPixel;
using holovibes::units::Point;
using holovibes::units::RectFd;

static void chart_test(const ushort image_width,
                       const ushort image_height,
                       const ushort zone_width,
                       const ushort zone_height,
                       ushort x_zone_offset,
                       ushort y_zone_offset,
                       const float cell_value)
{
    ushort total_image_size = image_width * image_height;

    const ushort total_zone_size = zone_width * zone_height;

    float* input;
    cudaMallocManaged(&input, total_image_size * sizeof(float));
    for (int i = 0; i < total_image_size; ++i)
        input[i] = cell_value;

    double* output;
    cudaMallocManaged(&output, sizeof(double));
    *output = 0;

    RectFd zone;
    Point<FDPixel> dst;
    dst.x().set(zone_width);
    dst.y().set(zone_height);
    zone.setBottomRight(dst);
    Point<FDPixel> src;
    src.x().set(x_zone_offset);
    src.y().set(y_zone_offset);
    zone.setTopLeft(src);
    zone.setWidth(zone_width);
    zone.setHeight(zone_height);

    apply_zone_sum(input, image_height, image_width, output, zone, 0);
    cudaXStreamSynchronize(0);

    ASSERT_EQ(static_cast<ushort>(*output), total_zone_size * cell_value);
}

TEST(ChartTest, SmallCroppedExample)
{
    ushort image_width = 69;
    ushort image_height = image_width;

    ushort zone_width = image_width - 2;
    ushort zone_height = image_height - 2;

    ushort x_zone_offset = 1;
    ushort y_zone_offset = 1;

    float cell_value = 1.0f;

    chart_test(image_width, image_height, zone_width, zone_height, x_zone_offset, y_zone_offset, cell_value);
}

TEST(ChartTest, SmallSimple)
{
    ushort image_width = 64;
    ushort image_height = image_width;

    ushort zone_width = image_width;
    ushort zone_height = image_height;

    ushort x_zone_offset = 0;
    ushort y_zone_offset = 0;

    float cell_value = 1.0f;

    chart_test(image_width, image_height, zone_width, zone_height, x_zone_offset, y_zone_offset, cell_value);
}

TEST(ChartTest, SuperTinyZone)
{
    ushort image_width = 69;
    ushort image_height = image_width;

    ushort zone_width = 1;
    ushort zone_height = 1;

    ushort x_zone_offset = 37;
    ushort y_zone_offset = 43;

    float cell_value = 1.0f;

    chart_test(image_width, image_height, zone_width, zone_height, x_zone_offset, y_zone_offset, cell_value);
}

TEST(ChartTest, NonSquareImageAndZone)
{
    ushort image_width = 69;
    ushort image_height = image_width + 5;

    ushort zone_width = image_width - 5;
    ushort zone_height = image_height - 13;

    ushort x_zone_offset = 2;
    ushort y_zone_offset = 3;

    float cell_value = 1.0f;

    chart_test(image_width, image_height, zone_width, zone_height, x_zone_offset, y_zone_offset, cell_value);
}

TEST(ChartTest, SmallDifferentValuesImage)
{
    ushort image_width = 33;
    ushort image_height = image_width;

    ushort zone_width = 3;
    ushort zone_height = 4;

    ushort x_zone_offset = image_width - zone_width;
    ushort y_zone_offset = image_height - zone_height;

    ushort total_image_size = image_width * image_height;

    float* input;
    cudaMallocManaged(&input, total_image_size * sizeof(float));
    /*
     * 7 31 6
     * 1 3 84
     * 0 9 48
     * 4 15 40
     * = 248
     */
    input[x_zone_offset + y_zone_offset * image_width + 0] = 7;
    input[x_zone_offset + y_zone_offset * image_width + 1] = 31;
    input[x_zone_offset + y_zone_offset * image_width + 2] = 6;
    input[x_zone_offset + (y_zone_offset + 1) * image_width + 0] = 1;
    input[x_zone_offset + (y_zone_offset + 1) * image_width + 1] = 3;
    input[x_zone_offset + (y_zone_offset + 1) * image_width + 2] = 84;
    input[x_zone_offset + (y_zone_offset + 2) * image_width + 0] = 0;
    input[x_zone_offset + (y_zone_offset + 2) * image_width + 1] = 9;
    input[x_zone_offset + (y_zone_offset + 2) * image_width + 2] = 48;
    input[x_zone_offset + (y_zone_offset + 3) * image_width + 0] = 4;
    input[x_zone_offset + (y_zone_offset + 3) * image_width + 1] = 15;
    input[x_zone_offset + (y_zone_offset + 3) * image_width + 2] = 40;

    double* output;
    cudaMallocManaged(&output, sizeof(double));
    *output = 0;

    RectFd zone;
    Point<FDPixel> dst;
    dst.x().set(33);
    dst.y().set(33);
    zone.setBottomRight(dst);
    Point<FDPixel> src;
    src.x().set(x_zone_offset);
    src.y().set(y_zone_offset);
    zone.setTopLeft(src);
    zone.setWidth(zone_width);
    zone.setHeight(zone_height);

    apply_zone_sum(input, image_height, image_width, output, zone, 0);
    cudaXStreamSynchronize(0);

    ASSERT_EQ(static_cast<ushort>(*output), 248);
}

TEST(ChartTest, DifferentValuesImage)
{
    ushort image_width = 33;
    ushort image_height = 33;

    ushort zone_width = image_width;
    ushort zone_height = image_height;

    ushort x_zone_offset = 0;
    ushort y_zone_offset = 0;

    ushort total_image_size = image_width * image_height;

    double expected_value = 0.f;

    float* input;
    cudaMallocManaged(&input, total_image_size * sizeof(float));
    for (int i = 0; i < total_image_size; ++i)
    {
        input[i] = i;
        expected_value += i;
    }

    double* output;
    cudaMallocManaged(&output, sizeof(double));
    *output = 0;

    RectFd zone;
    Point<FDPixel> dst;
    dst.x().set(zone_width);
    dst.y().set(zone_height);
    zone.setBottomRight(dst);
    Point<FDPixel> src;
    src.x().set(x_zone_offset);
    src.y().set(y_zone_offset);
    zone.setTopLeft(src);
    zone.setWidth(zone_width);
    zone.setHeight(zone_height);

    apply_zone_sum(input, image_height, image_width, output, zone, 0);
    cudaXStreamSynchronize(0);

    ASSERT_EQ(*output, expected_value);
}
