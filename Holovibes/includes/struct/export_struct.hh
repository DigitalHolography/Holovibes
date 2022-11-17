/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"
#include "enum_record_mode.hh"
#include "all_struct.hh"

namespace holovibes
{
struct ChartRecordStruct
{
  public:
    uint nb_points_to_record_ = 0;

  public:
    void disable() { nb_points_to_record_ = 0; }
    bool is_enable() const { return nb_points_to_record_ != 0; }
    uint get_nb_points_to_record() const { return nb_points_to_record_; }
    void set_nb_points_to_record(uint value) { nb_points_to_record_ = value; }

    bool operator!=(const ChartRecordStruct& rhs) { return nb_points_to_record_ != rhs.nb_points_to_record_; }
};

struct FrameRecordStruct
{
  public:
    RecordMode record_mode = RecordMode::NONE;
    bool enabled = false;

  public:
    bool operator!=(const FrameRecordStruct& rhs) { return record_mode != rhs.record_mode || enabled != rhs.enabled; }
};

inline std::ostream& operator<<(std::ostream& os, const ChartRecordStruct& value)
{
    return os << value.nb_points_to_record_;
}

inline std::ostream& operator<<(std::ostream& os, const FrameRecordStruct& value)
{
    return os << value.record_mode << ", enabled : " << value.enabled;
}

} // namespace holovibes
