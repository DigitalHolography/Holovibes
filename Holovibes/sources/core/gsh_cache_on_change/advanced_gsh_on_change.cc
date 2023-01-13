#include "API.hh"

namespace holovibes
{
template <>
void AdvancedGSHOnChange::operator()<InputBufferSize>(uint& new_value)
{
    LOG_UPDATE_ON_CHANGE(InputBufferSize);
}

template <>
void AdvancedGSHOnChange::operator()<OutputBufferSize>(uint& new_value)
{
    LOG_UPDATE_ON_CHANGE(OutputBufferSize);
}

template <>
void AdvancedGSHOnChange::operator()<FileBufferSize>(uint& new_value)
{
    LOG_UPDATE_ON_CHANGE(FileBufferSize);
}

template <>
bool AdvancedGSHOnChange::change_accepted<RecordBufferSize>(uint new_value)
{
    LOG_UPDATE_ON_CHANGE(RecordBufferSize);
    return !api::detail::get_value<Record>().is_running;
}
} // namespace holovibes
