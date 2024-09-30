/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

void unpack_12_to_16bit(short* output,
                        const size_t output_size,
                        const unsigned char* input,
                        const size_t input_size,
                        const cudaStream_t stream);

void unpack_10_to_16bit(short* output,
                        const size_t output_size,
                        const unsigned char* input,
                        const size_t input_size,
                        const cudaStream_t stream);
