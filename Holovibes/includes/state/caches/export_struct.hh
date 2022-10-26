/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"
#include "enum_record_mode.hh"

namespace holovibes
{
struct ChartRecordStruct
{
  public:
    uint nb_points_to_record_ = 0;

  public:
    bool is_enable() const { return nb_points_to_record_ != 0; }
    uint get_nb_points_to_record() const { return nb_points_to_record_; }
    void set_nb_points_to_record(uint value) { nb_points_to_record_ = value; }
};

struct FrameRecordStruct
{
  public:
    RecordMode record_mode_ = RecordMode::NONE;

  public:
    bool is_enable() const { return record_mode_ != RecordMode::NONE; }
    RecordMode get_record_mode() const { return record_mode_; }
    void set_record_mode(RecordMode value) { record_mode_ = value; }
};
} // namespace holovibes
