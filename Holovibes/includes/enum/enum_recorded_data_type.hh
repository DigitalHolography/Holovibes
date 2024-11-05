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
    MOMENTS,   /*!< The 4 recorded moments (0, 1, 2 and 0 with flat field correction).
                    They are contiguous : moment 0, then 1, then 2, then moment 0 flat field, for each image. */
};
} // namespace holovibes
