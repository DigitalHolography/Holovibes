/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file Handler for cudaStream_t used by the threads */

#pragma once

#include <cuda_runtime_api.h>

namespace holovibes::cuda_tools
{
class CudaStreamHandler
{
  public:
    /*! \brief Create the stream on the construction */
    CudaStreamHandler();

    /*! \brief Destroy the stream on the destruction */
    ~CudaStreamHandler();

    /*! \brief Getter for the cuda stream */
    inline const cudaStream_t& get() const;

  private:
    /*!
     * The actual cudaStream_t. Only the handler must have the ownership of the
     * stream. Others can only get a const reference. The stream must not be
     * modified.
     */
    cudaStream_t stream_;
};
} // namespace holovibes::cuda_tools

#include "cuda_stream_handler.hxx"