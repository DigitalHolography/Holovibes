/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

// clang-format off

//! \brief Max file buffer size
class FileBufferSize : public UIntParameter<512, "file_buffer_size">{};

// clang-format on

class FileReadCache : public MicroCache<FileBufferSize>
{
};

} // namespace holovibes
