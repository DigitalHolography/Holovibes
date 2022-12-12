/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "types.hh"

#include "all_struct.hh"

namespace holovibes
{
struct RecordStruct
{
  public:
    enum class RecordType
    {
        NONE,
        CHART,
        RAW,
        HOLOGRAM,
        CUTS_XZ,
        CUTS_YZ
    };

  public:
    std::string file_path;
    uint nb_to_record = 0;
    uint nb_to_skip = 0;
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
    bool operator!=(const RecordStruct& rhs) const
    {
        return nb_to_record != rhs.nb_to_record || nb_to_skip != rhs.nb_to_skip || is_running != rhs.is_running ||
               record_type != rhs.record_type;
    }
};

inline std::ostream& operator<<(std::ostream& os, const RecordStruct& value)
{
    return os << value.nb_to_record << ", enabled : " << value.is_running;
}

// clang-format off
SERIALIZE_JSON_ENUM(RecordStruct::RecordType, {
    {RecordStruct::RecordType::NONE, "NONE"},
    {RecordStruct::RecordType::RAW, "RAW"},
    {RecordStruct::RecordType::CUTS_XZ, "CUTS_XZ"},
    {RecordStruct::RecordType::CUTS_YZ, "CUTS_YZ"},
    {RecordStruct::RecordType::HOLOGRAM, "HOLOGRAM"}
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const RecordStruct::RecordType& value) { return os << json{value}; }

} // namespace holovibes
