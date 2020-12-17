/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "cuda_stream_handler.hh"

namespace holovibes::cuda_tools
{

inline const cudaStream_t& CudaStreamHandler::get_stream() const
{
    return stream_;
}

} // namespace holovibes::cuda_tools

#include "cuda_stream_handler.hxx"