/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include "composite_struct.hh"

/**
 * @brief A struct to represent a RGB pixel.
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

// Computes 3 different p slices and put them in each color
void rgb(float* output,
         cuComplex* input,
         const size_t frame_res,
         bool auto_weights,
         const ushort min,
         const ushort max,
         holovibes::RGBWeights weights,
         const cudaStream_t stream);
