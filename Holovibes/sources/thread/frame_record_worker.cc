/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "frame_record_worker.hh"
#include "output_frame_file_factory.hh"
#include "tools.hh"
#include "holovibes.hh"
#include "icompute.hh"

namespace holovibes::worker
{
    FrameRecordWorker::FrameRecordWorker(const std::string& file_path,
                                        unsigned int nb_frames_to_record,
                                        bool raw_record) :
        Worker(),
        file_path_(get_record_filename(file_path)),
        nb_frames_to_record_(nb_frames_to_record),
        processed_fps_(0),
        raw_record_(raw_record)
    {}

    void FrameRecordWorker::run()
    {
        std::atomic<size_t> nb_frames_recorded = 0;

        InformationContainer& info = Holovibes::instance().get_info_container();
        info.add_processed_fps(InformationContainer::FpsType::SAVING_FPS, processed_fps_);
        info.add_progress_index(InformationContainer::ProgressType::FRAME_RECORD, nb_frames_recorded, nb_frames_to_record_);

        auto pipe = Holovibes::instance().get_compute_pipe();
        Queue& record_queue = init_gpu_record_queue(pipe);

        io_files::OutputFrameFile* output_frame_file = nullptr;
        char* frame_buffer = nullptr;

        try
        {
            camera::FrameDescriptor file_fd = record_queue.get_fd();
            output_frame_file = io_files::OutputFrameFileFactory::create(file_path_, file_fd, nb_frames_to_record_);

            auto& cd = Holovibes::instance().get_cd();
            output_frame_file->export_compute_settings(cd, raw_record_);

            output_frame_file->write_header();

            const size_t output_frame_size = record_queue.get_frame_size();
            frame_buffer = new char[output_frame_size];

            for (; nb_frames_recorded < nb_frames_to_record_ && !stop_requested_; ++nb_frames_recorded)
            {
                wait_for_frames(record_queue, pipe);

                if (stop_requested_)
                    break;

                record_queue.dequeue(frame_buffer, cudaMemcpyDeviceToHost);
                output_frame_file->write_frame(frame_buffer, output_frame_size);
                processed_fps_++;
            }

            if (stop_requested_)
            {
                LOG_WARN("[RECORDER] Written frames: " + std::to_string(nb_frames_recorded)
                            + ", expected: " + std::to_string(nb_frames_to_record_)
                            + ". Correcting file integrity...");

                output_frame_file->correct_number_of_frames(nb_frames_recorded);

                LOG_INFO("[RECORDER] File integrity corrected!");
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

        reset_gpu_record_queue(pipe);

        info.remove_processed_fps(InformationContainer::FpsType::SAVING_FPS);
        info.remove_progress_index(InformationContainer::ProgressType::FRAME_RECORD);
    }

    Queue& FrameRecordWorker::init_gpu_record_queue(std::shared_ptr<ICompute> pipe)
    {
        std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();

        if (raw_view_queue)
            raw_view_queue->resize(4);

        Holovibes::instance().get_gpu_output_queue()->resize(4);

        if (raw_record_)
        {
            pipe->request_raw_record(nb_frames_to_record_);
            while (pipe->get_raw_record_requested() != std::nullopt && !stop_requested_);
        }
        else
        {
            pipe->request_hologram_record(nb_frames_to_record_);
            while (pipe->get_hologram_record_requested() != std::nullopt && !stop_requested_);
        }

        return *pipe->get_frame_record_queue();
    }

    void FrameRecordWorker::wait_for_frames(Queue& record_queue, std::shared_ptr<ICompute> pipe)
    {
        while (!stop_requested_)
        {
            if (record_queue.get_size() == 0)
            {
                if (pipe->get_remaining_frames_to_record() == 0)
                {
                    LOG_ERROR("[RECORDER] Record queue has been full, data has been lost! "
                            "Try to resize record buffer size");
                    stop();
                }
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

        while (pipe->get_disable_frame_record_requested() && !stop_requested_);

        std::unique_ptr<Queue>& raw_view_queue = pipe->get_raw_view_queue();

        if (raw_view_queue)
            raw_view_queue->resize(global::global_config.output_queue_max_size);

        Holovibes::instance().get_gpu_output_queue()->resize(global::global_config.output_queue_max_size);
    }
} // namespace holovibes::worker