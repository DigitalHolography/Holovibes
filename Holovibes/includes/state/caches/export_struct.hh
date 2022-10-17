/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"
#include "enum_record_mode.hh"

namespace holovibes
{
struct ExportChartStruct
{
  public:
    bool is_enabled = false;
    uint nb_points_to_record_ = 0;

  public:
    bool get_is_enabled() { return is_enabled; }
    uint get_nb_points_to_record() { return nb_points_to_record_; }

    ExportChartStruct& set_is_enabled(bool value)
    {
        is_enabled = value;
        if (value == false)
            nb_points_to_record_ = 0;
        return *this;
    }
    ExportChartStruct& set_nb_points_to_record(uint value)
    {
        nb_points_to_record_ = value;
        return *this;
    }
};

// FIXME simple : rm is_enabled_
struct FrameRecordStruct
{
  public:
    bool is_enabled = false;
    RecordMode record_mode_ = RecordMode::NONE;

  public:
    bool get_is_enabled() { return is_enabled; }
    RecordMode get_record_mode() { return record_mode_; }

    FrameRecordStruct& set_is_enabled(bool value)
    {
        is_enabled = value;
        if (value == false)
            record_mode_ = RecordMode::NONE;
        return *this;
    }
    FrameRecordStruct& set_record_mode(RecordMode value)
    {
        if (value != RecordMode::NONE)
            is_enabled = true;
        else
            is_enabled = false;

        record_mode_ = value;
        return *this;
    }
};
} // namespace holovibes
