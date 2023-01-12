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
void operator()<FileBufferSize>(uint new_value, Pipe& pipe)
{
    LOG_UPDATE_ON_SYNC(FileBufferSize);
    // if (pipe.get_gpu_output_queue())
    //     pipe.get_gpu_output_queue().set_new_total_nb_frames(new_value);
    // request_pipe_refresh();
}