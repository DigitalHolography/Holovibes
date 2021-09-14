#include "file_frame_read_worker.hh"
#include "queue.hh"
#include "cuda_memory.cuh"
#include "unpack.cuh"
#include "input_frame_file_factory.hh"
#include "config.hh"
#include "holovibes.hh"

namespace holovibes::worker
{
FileFrameReadWorker::FileFrameReadWorker(
    const std::string& file_path,
    bool loop,
    unsigned int fps,
    unsigned int first_frame_id,
    unsigned int total_nb_frames_to_read,
    bool load_file_in_gpu,
    std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue)
    : FrameReadWorker(gpu_input_queue)
    , file_path_(file_path)
    , loop_(loop)
    , fps_handler_(FileFrameReadWorker::FpsHandler(fps))
    , first_frame_id_(first_frame_id)
    , current_nb_frames_read_(0)
    , total_nb_frames_to_read_(total_nb_frames_to_read)
    , load_file_in_gpu_(load_file_in_gpu)
    , input_file_(nullptr)
    , frame_size_(0)
    , cpu_frame_buffer_(nullptr)
    , gpu_frame_buffer_(nullptr)
    , gpu_packed_buffer_(nullptr)
{
}

void FileFrameReadWorker::run()
{
    try
    {
        input_file_.reset(io_files::InputFrameFileFactory::open(file_path_));
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR << "[READER]" << e.what();
        return;
    }

    const camera::FrameDescriptor& fd = input_file_->get_frame_descriptor();
    frame_size_ = fd.frame_size();

    if (!init_frame_buffers())
        return;

    std::string input_descriptor_info =
        std::to_string(fd.width) + std::string("x") +
        std::to_string(fd.height) + std::string(" - ") +
        std::to_string(fd.depth * 8) + std::string("bit");

    InformationContainer& info = Holovibes::instance().get_info_container();
    info.add_indication(InformationContainer::IndicationType::IMG_SOURCE,
                        "File");
    info.add_indication(InformationContainer::IndicationType::INPUT_FORMAT,
                        std::ref(input_descriptor_info));
    info.add_processed_fps(InformationContainer::FpsType::INPUT_FPS,
                           std::ref(processed_fps_));
    info.add_progress_index(InformationContainer::ProgressType::FILE_READ,
                            std::ref(current_nb_frames_read_),
                            std::ref(total_nb_frames_to_read_));

    try
    {
        input_file_->set_pos_to_frame(first_frame_id_);

        if (load_file_in_gpu_)
            read_file_in_gpu();
        else
            read_file_batch();
    }
    catch (const io_files::FileException& e)
    {
        LOG_ERROR << "[READER]" << e.what();
    }

    // No more enqueue, thus release the producer ressources
    gpu_input_queue_.load()->stop_producer();

    info.remove_indication(InformationContainer::IndicationType::IMG_SOURCE);
    info.remove_indication(InformationContainer::IndicationType::INPUT_FORMAT);
    info.remove_processed_fps(InformationContainer::FpsType::INPUT_FPS);
    info.remove_progress_index(InformationContainer::ProgressType::FILE_READ);

    cudaXFree(gpu_packed_buffer_);
    cudaXFree(gpu_frame_buffer_);
    cudaXFreeHost(cpu_frame_buffer_);
}

bool FileFrameReadWorker::init_frame_buffers()
{
    size_t buffer_nb_frames;

    if (load_file_in_gpu_)
        buffer_nb_frames = total_nb_frames_to_read_;
    else
        buffer_nb_frames = global::global_config.file_buffer_size;

    size_t buffer_size = frame_size_ * buffer_nb_frames;

    cudaError_t error_code = cudaMallocHost(&cpu_frame_buffer_, buffer_size);

    if (error_code != cudaSuccess)
    {
        std::string error_message = "[READER] Not enough CPU RAM to read file";

        if (load_file_in_gpu_)
            error_message +=
                " (consider disabling \"Load file in GPU\" option)";

        LOG_ERROR << error_message;

        return false;
    }

    error_code = cudaMalloc(&gpu_frame_buffer_, buffer_size);

    if (error_code != cudaSuccess)
    {
        std::string error_message = "[READER] Not enough GPU DRAM to read file";

        if (load_file_in_gpu_)
            error_message +=
                " (consider disabling \"Load file in GPU\" option)";

        LOG_ERROR << error_message;

        cudaXFreeHost(cpu_frame_buffer_);
        return false;
    }

    error_code = cudaMalloc(&gpu_packed_buffer_, frame_size_);

    if (error_code != cudaSuccess)
    {
        std::string error_message = "[READER] Not enough GPU DRAM to read file";

        if (load_file_in_gpu_)
            error_message +=
                " (consider disabling \"Load file in GPU\" option)";

        LOG_ERROR << error_message;

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
    size_t frames_read = read_copy_file(total_nb_frames_to_read_);

    while (!stop_requested_)
    {
        enqueue_loop(frames_read);

        if (loop_)
            current_nb_frames_read_ = 0;
        else
            stop_requested_ = true;
    }
}

void FileFrameReadWorker::read_file_batch()
{
    const unsigned int batch_size = global::global_config.file_buffer_size;

    fps_handler_.begin();

    // Read the entire file by batch
    while (!stop_requested_)
    {
        size_t frames_to_read =
            std::min(batch_size,
                     total_nb_frames_to_read_ - current_nb_frames_read_);

        // Read batch in cpu and copy it to gpu
        size_t frames_read = read_copy_file(frames_to_read);

        // Enqueue the batch frames one by one into the destination queue
        enqueue_loop(frames_read);

        // Reset to the first frame if needed
        if (current_nb_frames_read_ == total_nb_frames_to_read_)
        {
            if (loop_)
            {
                input_file_->set_pos_to_frame(first_frame_id_);
                current_nb_frames_read_ = 0;
            }
            else
            {
                stop_requested_ = true; // break
            }
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
        frames_read = input_file_->read_frames(cpu_frame_buffer_,
                                               frames_to_read,
                                               &flag_packed);
        size_t frames_total_size = frames_read * frame_size_;

        if (flag_packed != 8 && flag_packed != 16)
        {
            const camera::FrameDescriptor& fd =
                input_file_->get_frame_descriptor();
            size_t packed_frame_size =
                fd.width * fd.height * (flag_packed / 8.f);
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
                    unpack_12_to_16bit(
                        (short*)(gpu_frame_buffer_ + i * frame_size_),
                        frame_size_ / 2,
                        (unsigned char*)gpu_packed_buffer_,
                        packed_frame_size,
                        stream_);
                else if (flag_packed == 10)
                    unpack_10_to_16bit(
                        (short*)(gpu_frame_buffer_ + i * frame_size_),
                        frame_size_ / 2,
                        (unsigned char*)gpu_packed_buffer_,
                        packed_frame_size,
                        stream_);
            }
        }
        else
        {
            // Memcopy in the gpu buffer
            cudaXMemcpyAsync(gpu_frame_buffer_,
                             cpu_frame_buffer_,
                             frames_total_size,
                             cudaMemcpyHostToDevice,
                             stream_);
        }

        cudaStreamSynchronize(stream_);
    }
    catch (const io_files::FileException& e)
    {
        LOG_WARN << "[READER] " << e.what();
    }

    return frames_read;
}

void FileFrameReadWorker::enqueue_loop(size_t nb_frames_to_enqueue)
{
    size_t frames_enqueued = 0;

    while (frames_enqueued < nb_frames_to_enqueue && !stop_requested_)
    {
        fps_handler_.wait();
        gpu_input_queue_.load()->enqueue(gpu_frame_buffer_ +
                                             frames_enqueued * frame_size_,
                                         cudaMemcpyDeviceToDevice);

        current_nb_frames_read_++;
        processed_fps_++;
        frames_enqueued++;
    }

    // Synchronize forced, because of the cudaMemcpyAsync we have to finish to
    // enqueue the gpu_frame_buffer_ before storing next read frames in it.
    //
    // With load_file_in_gpu_ == true, all the file in in the buffer,
    // so we don't have to sync
    if (load_file_in_gpu_ == false)
        gpu_input_queue_.load()->sync_current_batch();
}
} // namespace holovibes::worker
