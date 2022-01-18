#include "jobs/job_list.hh"

namespace holovibes
{
void JobList::run(Job::BuffersEnv buffers) override
{
    // If the buffers have not been created, or if the input/output buffer differs from the expected input/output
    // buffer, create the buffers. This happens at the start of the programm, or after a refresh.
    if (buffers_.size() == 0 || buffers_.front().input_desc != buffers.input_desc ||
        buffers_.back().output_desc != buffers.output_desc)
        create_buffers(buffers);

    buffers_.front().input = buffers.input;
    buffers_.back().output = buffers.output;

    for (size_t i = 0; i < jobs_.size(); i++)
    {
        // TODO: rerun the loop when we need to accumulate images
        jobs_.back().run(buffers_.at(i));
    }
}

void JobList::create_buffers(const Job::BuffersEnv& buffers)
{
    gpu_memory_.clear();
    buffers_.clear();

    // For the first element we cant manage any image accumulation
    // because we dont own the input buffer

    // The expected dimension of the input buffer must be equal to the dimension of the first buffer
    jobs_.front().check_input_dimensions(buffers.input_desc);
    BufferDescriptor desc = jobs_.front().get_output_dimensions(buffers.input_desc);

    // TODO: create the 'inplace' system

    // Allocate the gpu_memory for the output buffer of the first job
    gpu_memory_.emplace_back(desc.get_buffer_size());
    buffers_.emplace_back(buffers.input, buffers.input_desc, gpu_memory_.back().get(), desc);

    for (size_t i = 1; i < jobs_.size(); i++)
    {
        // Get the number of input frames of the next job, using the number of output frames of the previous job
        unsigned short nb_frames = jobs_.at(i).get_nb_frames_input(desc.nb_frames);
        BufferDescriptor next_desc = desc;
        next_desc.nb_frames = nb_frames;

        jobs_.at(i).check_input_dimensions(next_desc);
        next_desc = jobs_.at(i).get_output_dimensions(next_desc);

        // If true, the next buffer needs to accumulate frames. Therefore, since the output buffer of the previous job
        // is the input buffer of the next one, we need to resize the output buffer of the previous job
        if (nb_frames != desc.nb_frames)
        {
            // Resize of the previous job's output buffer
            gpu_memory_.back().resize(std::max(desc.get_buffer_size(), next_desc.get_buffer_size()));
        }

        desc = next_desc;

        // TODO: create the 'inplace' system
        // For the last element we directly want to get the memory allocated by our parent
        auto ptr = (i != jobs.size() - 1) ? gpu_memory_.emplace_back(desc.get_buffer_size()).get() : buffers.output;
        buffers_.emplace_back(buffers_.back().output, buffers_.back().output_desc, ptr, desc);
    }
}

} // namespace holovibes