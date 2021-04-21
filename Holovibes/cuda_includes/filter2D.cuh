/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "common.cuh"

void filter2D(cuComplex* input,
              cuComplex* tmp_buffer,
              const uint batch_size,
              const cufftHandle plan2d,
              const holovibes::units::RectFd& zone,
              const holovibes::units::RectFd& subzone,
              const camera::FrameDescriptor& desc,
              const cudaStream_t stream);