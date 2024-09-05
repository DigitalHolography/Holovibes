/*! \file
 *
 * \brief Api Struct
 *
 */

#pragma once

#include "all_struct.hh"

namespace holovibes
{
    struct RecordProgressData
    {
        int value;
        int max;
    };

    struct RecordBarColorData
    {
        QColor color;
        QString text;
    };
}