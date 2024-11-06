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
enum RecordedDataType
{
    RAW,       /*!< Raw data, an interferogram*/
    PROCESSED, /*!< A generated hologram */
    MOMENTS,   /*!< The 3 recorded moments (0, 1 and 2).
                    They are contiguous : moment 0, then 1, then 2, and so for each image. */
};
} // namespace holovibes
