/*! \file
 *
 * \brief Enum for the different type of record
 */
#pragma once

namespace holovibes
{
/*! \enum RecordMode
 *
 * \brief Represents the type of data stored in a .holo file.
 * This exists because loading moments is a completely different task\
 * compared to regular images.
 */
enum class RecordedDataType
{
    RAW,     // Raw data, such as an interferogram or even processed images.
    MOMENTS, // The 3 recorded moments (0, 1 and 2)
};
} // namespace holovibes
