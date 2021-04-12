/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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