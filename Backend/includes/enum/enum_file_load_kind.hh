/*! \file
 *
 * \brief Enum for the different ways to read from a file
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum Device
 *
 * \brief Input processes
 */
enum class FileLoadKind
{
    REGULAR = 0, /*!< Frames are read 'batch' by batch as the computation goes along */
    GPU,         /*!< The whole file is loaded at once in the GPU VRAM */
    CPU,         /*!< The whole file is loaded at once in the CPU RAM */
};

} // namespace holovibes
