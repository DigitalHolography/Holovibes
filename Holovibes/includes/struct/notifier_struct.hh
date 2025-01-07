/*! \file
 *
 * \brief File created to hold structures used by the Notifier system,
 * notably in the GUI.
 *
 */

#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*
 * \brief Structure used to transmit data regarding the progress of
 * a recording.
 */
struct RecordProgressData
{
    int value;
    int max;
};
} // namespace holovibes
