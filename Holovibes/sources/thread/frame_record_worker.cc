#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "global_state_holder.hh"
#include "API.hh"
#include "logger.hh"

namespace holovibes::worker
{
FrameRecordWorker::FrameRecordWorker()
    : Worker()
    , env_(api::get_compute_pipe().get_frame_record_env())
    , stream_(Holovibes::instance().get_cuda_streams().recorder_stream)
{
    auto& entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::RECORD);
    entry.recorded = &env_.current_nb_frames_recorded;
    entry.to_record = export_cache_.get_value<Record>().nb_to_record;

    GSH::fast_updates_map<FpsType>.create_entry(FpsType::SAVING_FPS);
}

FrameRecordWorker::~FrameRecordWorker()
{
    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::RECORD);
    GSH::fast_updates_map<FpsType>.remove_entry(FpsType::SAVING_FPS);
}

void FrameRecordWorker::run()
{
    env_.current_nb_frames_recorded = 0;
    env_.nb_frame_skip = export_cache_.get_value<Record>().nb_to_skip;

    const size_t output_frame_size = env_.gpu_frame_record_queue_->get_fd().get_frame_size();

    io_files::OutputFrameFile* output_frame_file =
        io_files::OutputFrameFileFactory::create(export_cache_.get_value<Record>().file_path,
                                                 env_.gpu_frame_record_queue_->get_fd(),
                                                 export_cache_.get_value<Record>().nb_to_record);

    output_frame_file->write_header();

    std::optional<int> contiguous_frames = std::nullopt;
    char* frame_buffer = new char[output_frame_size];

    Chrono chrono;

    while (export_cache_.get_value<Record>().nb_to_record == 0 ||
           env_.current_nb_frames_recorded < export_cache_.get_value<Record>().nb_to_record)
    {
        if (stop_requested_)
            break;

        if (env_.gpu_frame_record_queue_->has_overridden() ||
            Holovibes::instance().get_gpu_input_queue()->has_overridden())
        {
            // Due to frames being overwritten when the queue/batchInputQueue is full, the contiguity is lost.
            if (!contiguous_frames.has_value())
                contiguous_frames = std::make_optional(env_.current_nb_frames_recorded);
        }

        wait_for_frames(*env_.gpu_frame_record_queue_);

        if (env_.nb_frame_skip > 0)
        {
            env_.gpu_frame_record_queue_->dequeue();
            env_.nb_frame_skip--;
            continue;
        }

        env_.gpu_frame_record_queue_->dequeue(frame_buffer, stream_, cudaMemcpyDeviceToHost);
        output_frame_file->write_frame(frame_buffer, output_frame_size);
        GSH::fast_updates_map<FpsType>.get_entry(FpsType::OUTPUT_FPS).image_processed += 1;
        env_.current_nb_frames_recorded++;
    }

    LOG_INFO("Recording stopped, written frames : {}", env_.current_nb_frames_recorded);
    output_frame_file->correct_number_of_frames(env_.current_nb_frames_recorded);

    if (contiguous_frames.has_value())
    {
        LOG_WARN("Record lost its contiguousity at frame {}.", contiguous_frames.value());
        LOG_WARN("To prevent this lost, you might need to increase Input AND/OR Record buffer size.");
    }
    else
    {
        LOG_INFO("Record is contiguous!");
    }

    auto contiguous = contiguous_frames.value_or(env_.current_nb_frames_recorded);
    int export_fps = (int)((float)env_.current_nb_frames_recorded / ((float)chrono.get_milliseconds() * 1000.0f));
    output_frame_file->export_compute_settings(export_fps, contiguous);
    output_frame_file->write_footer();

    delete output_frame_file;
    delete[] frame_buffer;
}

void FrameRecordWorker::wait_for_frames(Queue& record_queue)
{
    while (!stop_requested_)
    {
        if (env_.gpu_frame_record_queue_->get_size() != 0)
            break;
    }
}

} // namespace holovibes::worker
