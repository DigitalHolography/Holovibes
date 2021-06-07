#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"

namespace holovibes::worker
{
FrameRecordWorker::FrameRecordWorker(
    const std::string& file_path,
    std::optional<unsigned int> nb_frames_to_record,
    bool raw_record,
    bool square_output,
    unsigned int nb_frames_skip)
    : Worker()
    , file_path_(get_record_filename(file_path))
    , nb_frames_to_record_(nb_frames_to_record)
    , nb_frames_skip_(nb_frames_skip)
    , processed_fps_(0)
    , raw_record_(raw_record)
    , square_output_(square_output)
    , stream_(Holovibes::instance().get_cuda_streams().recorder_stream)
{
}

void FrameRecordWorker::run()
{
    ComputeDescriptor& cd = Holovibes::instance().get_cd();

    if (cd.batch_size > global::global_config.frame_record_queue_max_size)
    {
        LOG_ERROR("[RECORDER] Batch size must be lower than record queue size");
        return;
    }

    std::atomic<unsigned int> nb_frames_recorded = 0;

    InformationContainer& info = Holovibes::instance().get_info_container();
    info.add_processed_fps(InformationContainer::FpsType::SAVING_FPS,
                           processed_fps_);

    if (nb_frames_to_record_.has_value())
    {
        info.add_progress_index(
            InformationContainer::ProgressType::FRAME_RECORD,
            nb_frames_recorded,
            nb_frames_to_record_.value());
    }
    else
    {
        info.add_progress_index(
            InformationContainer::ProgressType::FRAME_RECORD,
            nb_frames_recorded,
            nb_frames_recorded);
    }

    auto pipe = Holovibes::instance().get_compute_pipe();
    Queue& record_queue = init_gpu_record_queue(pipe);

    io_files::OutputFrameFile* output_frame_file = nullptr;
    char* frame_buffer = nullptr;

    try
    {
        camera::FrameDescriptor file_fd = record_queue.get_fd();
        output_frame_file = io_files::OutputFrameFileFactory::create(
            file_path_,
            file_fd,
            nb_frames_to_record_.has_value() ? nb_frames_to_record_.value()
                                             : 0);

        output_frame_file->export_compute_settings(cd, raw_record_);
        output_frame_file->set_make_square_output(square_output_);

        output_frame_file->write_header();

        const size_t output_frame_size = record_queue.get_frame_size();
        frame_buffer = new char[output_frame_size];

        while (nb_frames_to_record_ == std::nullopt ||
               nb_frames_recorded < nb_frames_to_record_.value() &&
                   !stop_requested_)
        {
            wait_for_frames(record_queue, pipe);

            if (stop_requested_)
                break;

            if (nb_frames_skip_ > 0)
            {
                record_queue.dequeue();
                --nb_frames_skip_;
                continue;
            }

            record_queue.dequeue(frame_buffer, stream_, cudaMemcpyDeviceToHost);
            output_frame_file->write_frame(frame_buffer, output_frame_size);
            ++processed_fps_;
            ++nb_frames_recorded;
        }

        if (stop_requested_)
        {
            LOG_INFO("[RECORDER] Recording stopped, written frames: " +
                     std::to_string(nb_frames_recorded));

            output_frame_file->correct_number_of_frames(nb_frames_recorded);
        }

        output_frame_file->write_footer();

        delete output_frame_file;
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("[RECORDER] " + std::string(e.what()));
        delete output_frame_file;
    }

    delete[] frame_buffer;

    if (record_queue.has_overridden())
    {
        LOG_ERROR("[RECORDER] Record queue overloaded, data has been lost! "
                  "Try to resize record buffer");
    }

    reset_gpu_record_queue(pipe);

    info.remove_processed_fps(InformationContainer::FpsType::SAVING_FPS);
    info.remove_progress_index(
        InformationContainer::ProgressType::FRAME_RECORD);
}

Queue& FrameRecordWorker::init_gpu_record_queue(std::shared_ptr<ICompute> pipe)
{
    std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();
    if (raw_view_queue)
        raw_view_queue->resize(4, stream_);

    std::shared_ptr<Queue> output_queue =
        Holovibes::instance().get_gpu_output_queue();
    if (output_queue)
        output_queue->resize(4, stream_);

    if (raw_record_)
    {
        pipe->request_raw_record(nb_frames_to_record_);
        while (pipe->get_raw_record_requested() != std::nullopt &&
               !stop_requested_)
            continue;
    }
    else
    {
        pipe->request_hologram_record(nb_frames_to_record_);
        while (pipe->get_hologram_record_requested() != std::nullopt &&
               !stop_requested_)
            continue;
    }

    return *pipe->get_frame_record_queue();
}

void FrameRecordWorker::wait_for_frames(Queue& record_queue,
                                        std::shared_ptr<ICompute> pipe)
{
    while (!stop_requested_)
    {
        if (record_queue.get_size() == 0)
        {
            if (record_queue.has_overridden())
                stop();
        }
        else
        {
            break;
        }
    }
}

void FrameRecordWorker::reset_gpu_record_queue(std::shared_ptr<ICompute> pipe)
{
    pipe->request_disable_frame_record();

    while (pipe->get_disable_frame_record_requested() && !stop_requested_)
        continue;

    std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();
    if (raw_view_queue)
        raw_view_queue->resize(global::global_config.output_queue_max_size,
                               stream_);

    std::shared_ptr<Queue> output_queue =
        Holovibes::instance().get_gpu_output_queue();

    if (output_queue)
        output_queue->resize(global::global_config.output_queue_max_size,
                             stream_);
}
} // namespace holovibes::worker
