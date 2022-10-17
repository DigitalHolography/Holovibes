/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

//! \brief Max file buffer size
using FileBufferSize = UIntParameter<512, "file_buffer_size">;

using FileReadCache = MicroCache<FileBufferSize>;

} // namespace holovibes
