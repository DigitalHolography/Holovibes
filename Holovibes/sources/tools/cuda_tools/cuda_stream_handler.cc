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
    std::cout << "Create stream : " << stream_ << std::endl;
}

CudaStreamHandler::~CudaStreamHandler()
{
    // cudaSafeCall(cudaStreamDestroy(stream_));
    std::cout << "Destroy stream : " << stream_ << std::endl;
}
} // namespace holovibes::cuda_tools

#include "cuda_stream_handler.hxx"