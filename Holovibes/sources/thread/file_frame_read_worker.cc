#include "file_frame_read_worker.hh"
#include "queue.hh"
#include "cuda_memory.cuh"
#include "unpack.cuh"
#include "input_frame_file_factory.hh"
#include "logger.hh"

#include "holovibes.hh"
#include "API.hh"

namespace api = ::holovibes::api;

// sqrt for filter_2d
#include <cmath>
#include "spdlog/spdlog.h"
namespace holovibes::worker
{

void FileFrameReadWorker::open_file()
{
    auto file_path = setting<settings::InputFilePath>();
    input_file_.reset(io_files::InputFrameFileFactory::open(file_path));
    fd_ = input_file_->get_frame_descriptor();
    frame_size_ = fd_.value().get_frame_size();
}

void FileFrameReadWorker::read_file()
{
    if (setting<settings::FileLoadKind>() == FileLoadKind::REGULAR)
        read_file_batch();
    else
        read_file_in_memory();
}

void FileFrameReadWorker::run()
{
    LOG_TRACE("[FileFrameReadWorker] [run]");

    onrestart_settings_.apply_updates();
    total_nb_frames_to_read_ =
        static_cast<unsigned int>(setting<settings::InputFileEndIndex>() - setting<settings::InputFileStartIndex>());

    // Open file.

    try
    {
        open_file();
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("{}", e.what());
        return;
    }
    // const camera::FrameDescriptor& fd = input_file_->get_frame_descriptor();
    // frame_size_ = fd.get_frame_size();

    if (!init_frame_buffers())
        return;

    insert_fast_update_map_entries();

    try
    {
        input_file_->set_pos_to_frame(setting<settings::InputFileStartIndex>());

        read_file();
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR("{}", e.what());
    }

    // No more enqueue, thus release the producer ressources
    input_queue_.load()->stop_producer();

    remove_fast_update_map_entries();
    free_frame_buffers();
}

size_t FileFrameReadWorker::get_buffer_nb_frames()
{
    if (setting<settings::FileLoadKind>() != FileLoadKind::REGULAR)
        return total_nb_frames_to_read_;
    return setting<settings::FileBufferSize>();
}

bool FileFrameReadWorker::init_frame_buffers()
{
    // Function used within this method to handle any error that may occur during initialization.
    auto handleError = [&](const std::string& error_message_base, bool cleanupCpu = false, bool cleanupGpu = false)
    {
        std::string error_message = error_message_base;
        if (setting<settings::FileLoadKind>() == FileLoadKind::GPU)
            error_message += " (consider disabling \"Load file in GPU VRAM\" option)";
        else if (setting<settings::FileLoadKind>() == FileLoadKind::CPU)
            error_message += " (consider disabling \"Load file in CPU RAM\" option)";
        LOG_ERROR("{}", error_message);
        if (cleanupCpu)
            cudaXFreeHost(cpu_frame_buffer_);
        if (cleanupGpu)
            cudaXFree(gpu_file_frame_buffer_);

        return false; // Explicitly return false for clarity.
    };

    size_t buffer_size = frame_size_ * get_buffer_nb_frames();

    cudaError_t error_code = cudaXRMallocHost(&cpu_frame_buffer_, buffer_size);
    if (error_code != cudaSuccess)
    {
        return handleError("Not enough CPU RAM to read file");
    }

    error_code = cudaXRMalloc(&gpu_file_frame_buffer_, buffer_size);
    if (error_code != cudaSuccess)
    {
        return handleError("Not enough GPU DRAM to read file", true);
    }

    error_code = cudaXRMalloc(&gpu_packed_buffer_, frame_size_);
    if (error_code != cudaSuccess)
    {
        return handleError("Not enough GPU DRAM to read file", true, true);
    }

    return true;
}

void FileFrameReadWorker::free_frame_buffers()
{
    cudaXFree(gpu_packed_buffer_);
    if (setting<settings::FileLoadKind>() != FileLoadKind::CPU) // Already freed, must not free twice
        cudaXFree(gpu_file_frame_buffer_);
    if (setting<settings::FileLoadKind>() != FileLoadKind::GPU) // Already freed, must not free twice
        cudaXFreeHost(cpu_frame_buffer_);
}

void FileFrameReadWorker::insert_fast_update_map_entries()
{
    std::string input_descriptor_info = std::to_string(fd_.value().width) + std::string("x") +
                                        std::to_string(fd_.value().height) + std::string(" - ") +
                                        std::to_string(fd_.value().depth * 8) + std::string("bit");

    auto entry1 = FastUpdatesMap::map<IndicationType>.create_entry(IndicationType::IMG_SOURCE, true);
    auto entry2 = FastUpdatesMap::map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT, true);
    *entry1 = "File";
    *entry2 = input_descriptor_info;

    current_fps_ = FastUpdatesMap::map<IntType>.create_entry(IntType::INPUT_FPS);
}

void FileFrameReadWorker::remove_fast_update_map_entries()
{
    FastUpdatesMap::map<IndicationType>.remove_entry(IndicationType::IMG_SOURCE);
    FastUpdatesMap::map<IndicationType>.remove_entry(IndicationType::INPUT_FORMAT);
    FastUpdatesMap::map<IntType>.remove_entry(IntType::INPUT_FPS);
    FastUpdatesMap::map<ProgressType>.remove_entry(ProgressType::FILE_READ);
}

void FileFrameReadWorker::read_file_in_memory()
{
    // Read and copy the entire file
    size_t frames_read = read_copy_file(total_nb_frames_to_read_);
    if (setting<settings::FileLoadKind>() == FileLoadKind::GPU)
        cudaXFreeHost(cpu_frame_buffer_);
    if (setting<settings::FileLoadKind>() == FileLoadKind::CPU)
        cudaXFree(gpu_file_frame_buffer_);

    while (!stop_requested_)
    {
        enqueue_loop(frames_read);
        current_nb_frames_read_ = 0;
    }
}

void FileFrameReadWorker::read_file_batch()
{
    const unsigned int file_buffer_size = static_cast<unsigned int>(setting<settings::FileBufferSize>());

    // Read the entire file by batch
    while (!stop_requested_)
    {
        size_t frames_to_read = std::min(file_buffer_size, total_nb_frames_to_read_ - current_nb_frames_read_);

        // Read batch in cpu and copy it to gpu
        size_t frames_read = read_copy_file(frames_to_read);

        // Enqueue the batch frames one by one into the destination queue
        enqueue_loop(frames_read);

        // Reset to the first frame if needed
        if (current_nb_frames_read_ == total_nb_frames_to_read_)
        {
            size_t frame_id = setting<settings::InputFileStartIndex>();
            input_file_->set_pos_to_frame(frame_id);
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
        frames_read = input_file_->read_frames(cpu_frame_buffer_, frames_to_read, &flag_packed);
        size_t frames_total_size = frames_read * frame_size_;

        if (flag_packed % 8 != 0) // Irregular encoding (not aligned on bytes)
        {
            const camera::FrameDescriptor& fd = input_file_->get_frame_descriptor();
            size_t packed_frame_size = fd.width * fd.height * (flag_packed / 8.f);
            for (size_t i = 0; i < frames_read; ++i)
            {
                char* gpu_output = gpu_file_frame_buffer_ + i * frame_size_;
                // Memcopy in the gpu buffer
                cudaXMemcpyAsync(gpu_packed_buffer_,
                                 cpu_frame_buffer_ + i * packed_frame_size,
                                 packed_frame_size,
                                 cudaMemcpyHostToDevice,
                                 stream_);

                // Convert 12bit frame to 16bit
                if (flag_packed == 12)
                    unpack_12_to_16bit((short*)(gpu_output),
                                       frame_size_ / 2,
                                       (unsigned char*)gpu_packed_buffer_,
                                       packed_frame_size,
                                       stream_);
                else if (flag_packed == 10)
                    unpack_10_to_16bit((short*)(gpu_output),
                                       frame_size_ / 2,
                                       (unsigned char*)gpu_packed_buffer_,
                                       packed_frame_size,
                                       stream_);
                else
                    throw io_files::FileException("Image encoding format not supported: {} bits", flag_packed);
            }
            if (setting<settings::FileLoadKind>() == FileLoadKind::CPU)
            {
                // Copy back to host
                cudaXMemcpyAsync(cpu_frame_buffer_,
                                 gpu_file_frame_buffer_,
                                 frames_total_size,
                                 cudaMemcpyHostToDevice,
                                 stream_);
            }
        }
        else if (setting<settings::FileLoadKind>() != FileLoadKind::CPU)
        {
            // Memcopy in the gpu buffer
            cudaXMemcpyAsync(gpu_file_frame_buffer_,
                             cpu_frame_buffer_,
                             frames_total_size,
                             cudaMemcpyHostToDevice,
                             stream_);
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
    // Read in either cpu or gpu depending on the settings
    bool load_in_cpu = setting<settings::FileLoadKind>() == FileLoadKind::CPU;
    char* frame_buffer = load_in_cpu ? cpu_frame_buffer_ : gpu_file_frame_buffer_;
    auto enqueue_kind = load_in_cpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    while (frames_enqueued < nb_frames_to_enqueue && !stop_requested_)
    {
        uint real_frames_enqueued = static_cast<uint>(
            std::min(static_cast<size_t>(setting<settings::BatchSize>()), nb_frames_to_enqueue - frames_enqueued));
        fps_limiter_.wait(setting<settings::InputFPS>() / real_frames_enqueued);

        if (Holovibes::instance().is_cli)
        {
            const auto input_queue = API.compute.get_input_queue();
            // Wait for the queue to have enough space before enqueuing
            while (input_queue->get_size() >= input_queue->get_max_size() && !stop_requested_)
            {
            }
        }
        input_queue_.load()->enqueue(frame_buffer + frames_enqueued * frame_size_, enqueue_kind, real_frames_enqueued);

        current_nb_frames_read_ += real_frames_enqueued;
        frames_enqueued += real_frames_enqueued;

        *current_fps_ += real_frames_enqueued;

        if (stop_requested_)
            break;
    }

    // Synchronize forced, because of the cudaMemcpyAsync we have to finish to
    // enqueue the gpu_file_frame_buffer_ before storing next read frames in it.
    //
    // If all the file is in the buffer, we don't have to sync
    if (setting<settings::FileLoadKind>() != FileLoadKind::REGULAR)
        return;

    input_queue_.load()->sync_current_batch();
}
} // namespace holovibes::worker
