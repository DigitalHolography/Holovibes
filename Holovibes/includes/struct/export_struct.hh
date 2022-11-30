/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"

#include "all_struct.hh"

namespace holovibes
{
struct ChartRecordStruct
{
  public:
    std::string chart_file_path;
    uint nb_points_to_record = 0;
    bool is_running = false;
    bool is_selected = false;

    bool get_is_selected_if_is_running() const { return is_running && is_selected; }

  public:
    bool operator!=(const ChartRecordStruct& rhs) const
    {
        return nb_points_to_record != rhs.nb_points_to_record || is_running != rhs.is_running;
    }
};

struct FrameRecordStruct
{
  public:
    enum class RecordType
    {
        NONE,
        RAW,
        HOLOGRAM,
        CUTS_XZ,
        CUTS_YZ
    };

  public:
    std::string frames_file_path;
    uint nb_frames_to_record = 0;
    uint nb_frames_to_skip = 0;
    bool is_running = false;
    RecordType record_type;

  public:
    RecordType get_record_type_if_is_running() const
    {
        if (is_running == false)
            return RecordType::NONE;
        return record_type;
    }

  public:
    bool operator!=(const FrameRecordStruct& rhs) const
    {
        return nb_frames_to_record != rhs.nb_frames_to_record || nb_frames_to_skip != rhs.nb_frames_to_skip ||
               is_running != rhs.is_running || record_type != rhs.record_type;
    }
};

inline std::ostream& operator<<(std::ostream& os, const ChartRecordStruct& value)
{
    return os << value.nb_points_to_record << ", is_running : " << value.is_running;
    ;
}

inline std::ostream& operator<<(std::ostream& os, const FrameRecordStruct& value)
{
    return os << value.nb_frames_to_record << ", enabled : " << value.is_running;
}

// clang-format off
SERIALIZE_JSON_ENUM(FrameRecordStruct::RecordType, {
    {FrameRecordStruct::RecordType::NONE, "NONE"},
    {FrameRecordStruct::RecordType::RAW, "RAW"},
    {FrameRecordStruct::RecordType::CUTS_XZ, "CUTS_XZ"},
    {FrameRecordStruct::RecordType::CUTS_YZ, "CUTS_YZ"},
    {FrameRecordStruct::RecordType::HOLOGRAM, "HOLOGRAM"}
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const FrameRecordStruct::RecordType& value)
{
    return os << json{value};
}

} // namespace holovibes
