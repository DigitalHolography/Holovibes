#include "file_frame_read_worker.hh"
#include "queue.hh"
#include "cuda_memory.cuh"
#include "unpack.cuh"
#include "input_frame_file_factory.hh"

#include "holovibes.hh"
#include "global_state_holder.hh"
#include "API.hh"

namespace holovibes::worker
{
FileFrameReadWorker::FileFrameReadWorker()
    : FrameReadWorker()
    , current_nb_frames_read_(0)
    , fps_handler_(api::get_input_fps())
    , cpu_frame_buffer_(nullptr)
    , gpu_frame_buffer_(nullptr)
    , gpu_packed_buffer_(nullptr)
{
    input_file_.reset(io_files::InputFrameFileFactory::open(api::detail::get_value<ImportFilePath>()));
    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::IMG_SOURCE) = "File";
    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT) = "FIXME File Format";

    to_record_ = api::get_nb_frame_to_read();
    auto& entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::READ);
    entry.recorded = &current_nb_frames_read_;
    entry.to_record = &to_record_;
}

FileFrameReadWorker::~FileFrameReadWorker()
{
    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::IMG_SOURCE);
    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::INPUT_FORMAT);
    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::READ);
}

void FileFrameReadWorker::run()
{
    if (!init_frame_buffers())
        return;

    // FIXME FASTUPDATEMAP
    auto fd = api::detail::get_value<ImportFrameDescriptor>();
    std::string input_descriptor_info = std::to_string(fd.width) + std::string("x") + std::to_string(fd.height) +
                                        std::string(" - ") + std::to_string(fd.depth * 8) + std::string("bit");

    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT, true) = input_descriptor_info;

    try
    {

        input_file_->set_pos_to_frame(api::get_start_frame() - 1);

        if (api::detail::get_value<LoadFileInGpu>())
            read_file_in_gpu();
        else
            read_file_batch();
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("{}", e.what());
    }

    // No more enqueue, thus release the producer ressources
    api::get_gpu_input_queue().stop_producer();

    cudaXFree(gpu_packed_buffer_);
    cudaXFree(gpu_frame_buffer_);
    cudaXFreeHost(cpu_frame_buffer_);
}

bool FileFrameReadWorker::init_frame_buffers()
{
    size_t buffer_nb_frames;

    if (api::detail::get_value<LoadFileInGpu>())
        buffer_nb_frames = api::get_nb_frame_to_read();
    else
        buffer_nb_frames = api::detail::get_value<FileBufferSize>();

    size_t buffer_size = api::get_import_frame_descriptor().get_frame_size() * buffer_nb_frames;

    cudaError_t error_code = cudaXRMallocHost(&cpu_frame_buffer_, buffer_size);

    std::string error_message = "";
    if (api::detail::get_value<LoadFileInGpu>())
        error_message += " (consider disabling \"Load file in GPU\" option)";

    if (error_code != cudaSuccess)
    {
        LOG_ERROR("Not enough CPU RAM to read file {}", error_message);
        return false;
    }

    error_code = cudaXRMalloc(&gpu_frame_buffer_, buffer_size);

    if (error_code != cudaSuccess)
    {
        LOG_ERROR("Not enough GPU VRAM to read file {}", error_message);

        cudaXFreeHost(cpu_frame_buffer_);
        return false;
    }

    error_code = cudaXRMalloc(&gpu_packed_buffer_, api::get_import_frame_descriptor().get_frame_size());

    if (error_code != cudaSuccess)
    {
        LOG_ERROR("Not enough GPU VRAM to read file {}", error_message);

        cudaXFreeHost(cpu_frame_buffer_);
        cudaXFree(gpu_frame_buffer_);
        return false;
    }

    return true;
}

void FileFrameReadWorker::read_file_in_gpu()
{
    fps_handler_.begin();

    // Read and copy the entire file
    size_t frames_read = read_copy_file(api::get_nb_frame_to_read());

    while (!stop_requested_)
    {
        enqueue_loop(frames_read);

        if (api::detail::get_value<LoopFile>())
            current_nb_frames_read_ = 0;
        else
        {
            LOG_DEBUG("End to read the file, stop because LoopFile == false");
            stop_requested_ = true;
        }
    }
}

void FileFrameReadWorker::read_file_batch()
{
    const unsigned int batch_size = api::detail::get_value<FileBufferSize>();

    fps_handler_.begin();

    // Read the entire file by batch
    while (!stop_requested_)
    {
        size_t frames_to_read = std::min(batch_size, api::get_nb_frame_to_read() - current_nb_frames_read_);
        // Read batch in cpu and copy it to gpu
        size_t frames_read = read_copy_file(frames_to_read);

        // Enqueue the batch frames one by one into the destination queue
        enqueue_loop(frames_read);

        // Reset to the first frame if needed
        if (current_nb_frames_read_ == api::get_nb_frame_to_read())
        {
            if (api::detail::get_value<LoopFile>() == false)
            {
                LOG_DEBUG("End to read the file, stop because LoopFile == false");
                stop_requested_ = true;
            }

            input_file_->set_pos_to_frame(api::get_start_frame() - 1);
            current_nb_frames_read_ = 0;
        }
    }
}

size_t FileFrameReadWorker::read_copy_file(size_t frames_to_read)
{
    // Read
    size_t frames_read = 0;
    int flag_packed;

    try
    {
        uint frame_size = api::get_import_frame_descriptor().get_frame_size();
        frames_read = input_file_->read_frames(cpu_frame_buffer_, frames_to_read, &flag_packed);
        size_t frames_total_size = frames_read * frame_size;

        if (flag_packed != 8 && flag_packed != 16)
        {
            const FrameDescriptor& fd = api::detail::get_value<ImportFrameDescriptor>();
            size_t packed_frame_size = fd.width * fd.height * (flag_packed / 8.f);
            for (size_t i = 0; i < frames_read; ++i)
            {
                // Memcopy in the gpu buffer
                cudaXMemcpyAsync(gpu_packed_buffer_,
                                 cpu_frame_buffer_ + i * packed_frame_size,
                                 packed_frame_size,
                                 cudaMemcpyHostToDevice,
                                 stream_);

                // Convert 12bit frame to 16bit
                if (flag_packed == 12)
                    unpack_12_to_16bit((short*)(gpu_frame_buffer_ + i * frame_size),
                                       frame_size / 2,
                                       (unsigned char*)gpu_packed_buffer_,
                                       packed_frame_size,
                                       stream_);
                else if (flag_packed == 10)
                    unpack_10_to_16bit((short*)(gpu_frame_buffer_ + i * frame_size),
                                       frame_size / 2,
                                       (unsigned char*)gpu_packed_buffer_,
                                       packed_frame_size,
                                       stream_);
            }
        }
        else
        {
            // Memcopy in the gpu buffer
            cudaXMemcpyAsync(gpu_frame_buffer_, cpu_frame_buffer_, frames_total_size, cudaMemcpyHostToDevice, stream_);
            // cudaXMemcpy(gpu_frame_buffer_, cpu_frame_buffer_, frames_total_size, cudaMemcpyHostToDevice);
        }

        cudaStreamSynchronize(stream_);
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("{}", e.what());
    }

    return frames_read;
}

void FileFrameReadWorker::enqueue_loop(size_t nb_frames_to_enqueue)
{
    size_t frames_enqueued = 0;

    while (frames_enqueued < nb_frames_to_enqueue && !stop_requested_)
    {
        fps_handler_.wait();

        if (api::detail::get_value<ExportRecordDontLoseFrame>())
        {
            while (Holovibes::instance().get_gpu_input_queue()->get_size() ==
                       Holovibes::instance().get_gpu_input_queue()->get_total_nb_frames() &&
                   !stop_requested_)
            {
            }
        }

        if (stop_requested_)
            break;

        api::get_gpu_input_queue().enqueue(gpu_frame_buffer_ +
                                               frames_enqueued * api::get_import_frame_descriptor().get_frame_size(),
                                           cudaMemcpyDeviceToDevice);

        processed_frames_++;
        frames_enqueued++;
        current_nb_frames_read_++;

        compute_fps();
    }

    // Synchronize forced, because of the cudaMemcpyAsync we have to finish to
    // enqueue the gpu_frame_buffer_ before storing next read frames in it.
    //
    // With load_file_in_gpu_ == true, all the file in in the buffer,
    // so we don't have to sync
    if (api::detail::get_value<LoadFileInGpu>() == false)
        api::get_gpu_input_queue().sync_current_batch();
}
} // namespace holovibes::worker
