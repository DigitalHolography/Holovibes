#include "API.hh"

namespace holovibes
{
template <>
void AdvancedPipeRequestOnSync::operator()<InputBufferSize>(uint new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(InputBufferSize);
    pipe.get_gpu_input_queue().set_new_total_nb_frames(new_value);
    request_pipe_refresh();
}

template <>
void AdvancedPipeRequestOnSync::operator()<OutputBufferSize>(uint new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(OutputBufferSize);
    pipe.get_gpu_output_queue().resize(new_value);
    request_pipe_refresh();
}

template <>
void AdvancedPipeRequestOnSync::operator()<TimeTransformationCutsBufferSize>(uint new_value, Pipe& pipe)
{
    pipe.get_compute_cache().virtual_synchronize<TimeTransformationCutsEnable>(false, false, pipe);
}

} // namespace holovibes
