/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "cuda_stream_handler.hh"
#include "common.cuh"

namespace holovibes::cuda_tools
{
CudaStreamHandler::CudaStreamHandler()
{
    cudaSafeCall(cudaStreamCreate(&stream_));
}

CudaStreamHandler::~CudaStreamHandler()
{
    cudaSafeCall(cudaStreamDestroy(stream_));
}
} // namespace holovibes::cuda_tools

#include "cuda_stream_handler.hxx"