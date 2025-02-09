/*! \file rgb.cuh
 *
 * \brief Declaration of rgb function and RGBPixel struct.
 */
#pragma once

#include "common.cuh"
#include "composite_struct.hh"

/*!
 * \brief A struct to represent a RGB pixel.
 *
 * Defines three operations: addition, division and multiplication.
 */
struct RGBPixel
{
    float r;
    float g;
    float b;

    RGBPixel __host__ __device__ operator+(const RGBPixel& other) const
    {
        RGBPixel result;
        result.r = this->r + other.r;
        result.g = this->g + other.g;
        result.b = this->b + other.b;
        return result;
    }

    RGBPixel __host__ __device__ operator/(float scalar) const
    {
        RGBPixel result;
        result.r = this->r / scalar;
        result.g = this->g / scalar;
        result.b = this->b / scalar;
        return result;
    }

    RGBPixel __host__ __device__ operator*(float scalar) const
    {
        RGBPixel result;
        result.r = this->r * scalar;
        result.g = this->g * scalar;
        result.b = this->b * scalar;
        return result;
    }
};

/*!
 * \brief Computes the RGB image from the input data by computing three different p slices and putting them in each
 * color channel.
 *
 * \param output The output RGB image.
 * \param input The input data.
 * \param frame_res The frame resolution.
 * \param auto_weights Whether to use auto weights.
 * \param min The minimum value.
 * \param max The maximum value.
 * \param weights The RGB weights.
 * \param stream The CUDA stream.
 */
void rgb(float* output,
         cuComplex* input,
         const size_t frame_res,
         bool auto_weights,
         const ushort min,
         const ushort max,
         holovibes::RGBWeights weights,
         const cudaStream_t stream);
