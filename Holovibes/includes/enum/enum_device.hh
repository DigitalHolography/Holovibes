/*! \file
 *
 * \brief Enum for the different device mode
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum Device
 *
 * \brief Input processes
 */
enum class Device
{
    GPU = 0, /*!< Buffer of the queue allocated on the GPU */
    CPU      /*!< Buffer of the queue allocated on the CPU */
};

// clang-format off
SERIALIZE_JSON_ENUM(Device, {
    {Device::GPU, "GPU"},
    {Device::CPU, "CPU"},
})
// clang-format on

} // namespace holovibes