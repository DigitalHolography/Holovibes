/*! \file
 *
 * \brief Enum for the different type of record
 */
#pragma once

#include <map>
#include <string>
#include "all_struct.hh"

namespace holovibes
{
/*! \enum RecordMode
 *
 * \brief Enum for the different record modes.
 * The record mode determines what data is stored when hitting the record button.
 *
 * \note The order of the enum is IMPORTANT as it reflects the same order in the GUI.
 * It is particularly relevant as both use static casts to convert between the two.
 * So if you want to add another mode, make sure to leave NONE in last as it is not in the GUI.
 */
enum class RecordMode
{
    RAW,      /*!< The raw input */
    HOLOGRAM, /*!< The image after all pipe computation */
    CHART,    /*!< The various chart data */
    MOMENTS,  /*!< The three moments of the image: m0, m1 and m2 */
    CUTS_XZ,  /*!< The 3D cuts in the XZ plane */
    CUTS_YZ,  /*!< The 3D cuts in the YZ plane */
    NONE,     /*!< No record mode; should only be used for default value and error purposes */
};

// clang-format off
SERIALIZE_JSON_ENUM(RecordMode, {
    {RecordMode::RAW, "RAW"},
    {RecordMode::HOLOGRAM, "HOLOGRAM"},
    {RecordMode::CHART, "CHART"},
    {RecordMode::MOMENTS, "MOMENTS"},
    {RecordMode::CUTS_XZ, "CUTS_XZ"},
    {RecordMode::CUTS_YZ, "CUTS_YZ"},
    {RecordMode::NONE, "NONE"}
})

/*!
 * \brief Describes better the possible file extensions of outputted files
 * 
 * \note Similarly to RecordMode,
 * the order of the enum is IMPORTANT as it reflects the same order in the GUI.
 */
enum OutputFormat
{
    HOLO,   /*!< .holo file */
    AVI,    /*!< .avi file */
    MP4,    /*!< .mp4 file */
    CSV,    /*!< .csv file */
    TXT,    /*!< .txt file */
};

// clang-format on
} // namespace holovibes
