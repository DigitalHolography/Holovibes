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

static void memory_error(bool in_gpu)
{
    std::string error_message = "";
    if (api::detail::get_value<LoadFileInGpu>())
        error_message = " (consider disabling \"Load file in GPU\" option)";
    LOG_ERROR("Not enough {} RAM to read file {}", in_gpu ? "GPU" : "CPU", error_message);
}

template <>
void AdvancedFileRequestOnSync::operator()<FileBufferSize>(uint, FileFrameReadWorker&)
{
    if (api::detail::get_value<LoadFileInGpu>() == false)
        need_refresh();
}

template <>
void ImportFileRequestOnSync::operator()<InputFps>(uint new_value, FileFrameReadWorker& file_worker)
{
    file_worker.get_fps_handler().set_new_fps_target(new_value);
}

template <>
void ImportFileRequestOnSync::operator()<ImportFrameDescriptor>(const FrameDescriptor&,
                                                                FileFrameReadWorker& file_worker)
{
    cudaXFree(file_worker.get_gpu_packed_buffer());
    char* buffer = nullptr;
    cudaError_t error_code = cudaXRMalloc(&buffer, api::get_import_frame_descriptor().get_frame_size());
    file_worker.set_gpu_packed_buffer(buffer);

    if (error_code != cudaSuccess)
    {
        LOG_ERROR("Not enough GPU VRAM to read file");
        file_worker.set_gpu_packed_buffer(nullptr);
        request_fail();
    }

    need_refresh();
}

template <>
void ImportFileRequestOnSync::operator()<StartFrame>(uint, FileFrameReadWorker&)
{
    if (api::detail::get_value<LoadFileInGpu>())
        need_refresh();
}

template <>
void ImportFileRequestOnSync::operator()<EndFrame>(uint, FileFrameReadWorker&)
{
    if (api::detail::get_value<LoadFileInGpu>())
        need_refresh();
}

template <>
void ImportFileRequestOnSync::operator()<LoadFileInGpu>(bool, FileFrameReadWorker&)
{
    need_refresh();
}

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

    GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::READ).recorded = &current_nb_frames_read_;
}

FileFrameReadWorker::~FileFrameReadWorker()
{
    cudaXFree(gpu_packed_buffer_);
    cudaXFree(gpu_frame_buffer_);
    cudaXFreeHost(cpu_frame_buffer_);

    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::IMG_SOURCE);
    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::INPUT_FORMAT);
    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::READ);
}

void FileFrameReadWorker::run()
{
    FileRequestOnSync::begin_requests();
    import_cache_.synchronize_force(*this);
    advanced_cache_.synchronize_force(*this);

    auto fd = api::detail::get_value<ImportFrameDescriptor>();
    std::string input_descriptor_info = std::to_string(fd.width) + std::string("x") + std::to_string(fd.height) +
                                        std::string(" - ") + std::to_string(fd.depth * 8) + std::string("bit");
    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT, true) = input_descriptor_info;

    bool first_time = true;
    size_t frames_read = 0;

    fps_handler_.begin();

    while (!stop_requested_)
    {
        // refresh
        import_cache_.synchronize(*this);
        advanced_cache_.synchronize(*this);
        if (FileRequestOnSync::has_requests_fail() || FileRequestOnSync::do_need_refresh() || first_time)
        {
            if (FileRequestOnSync::has_requests_fail() || init_frame_buffers() == false)
            {
                LOG_ERROR("Error while allocating the buffers, exitting...");
                stop_requested_ = true;
                api::set_import_type(ImportTypeEnum::None);
                break;
            }

            input_file_->set_pos_to_frame(api::get_start_frame() - 1);

            if (api::detail::get_value<LoadFileInGpu>())
                frames_read = read_copy_file(api::get_nb_frame_to_read());

            first_time = false;
            FileRequestOnSync::begin_requests();

            current_nb_frames_read_ = 0;
            GSH::fast_updates_map<ProgressType>.get_entry(ProgressType::READ).to_record = api::get_nb_frame_to_read();
        }

        // Get Frame with LoadFileInGpu == false
        if (api::detail::get_value<LoadFileInGpu>() == false)
        {
            size_t frames_to_read = std::min(api::detail::get_value<FileBufferSize>(),
                                             api::get_nb_frame_to_read() - current_nb_frames_read_);
            frames_read = read_copy_file(frames_to_read);
        }

        // Compute
        enqueue_loop(frames_read);

        // Loop
        if (api::detail::get_value<LoadFileInGpu>() == true || current_nb_frames_read_ == api::get_nb_frame_to_read())
        {
            if (api::detail::get_value<LoopFile>())
            {
                current_nb_frames_read_ = 0;
                if (api::detail::get_value<LoadFileInGpu>() == false)
                    input_file_->set_pos_to_frame(api::get_start_frame() - 1);
            }
            else
            {
                LOG_DEBUG("End to read the file, stop because LoopFile == false");
                stop_requested_ = true;
            }
        }
    }

    api::get_gpu_input_queue().stop_producer();
}

void FileFrameReadWorker::refresh() {}

bool FileFrameReadWorker::init_frame_buffers()
{
    size_t buffer_nb_frames;

    if (api::detail::get_value<LoadFileInGpu>())
        buffer_nb_frames = api::get_nb_frame_to_read();
    else
        buffer_nb_frames = api::detail::get_value<FileBufferSize>();

    size_t buffer_size = api::get_import_frame_descriptor().get_frame_size() * buffer_nb_frames;

    cudaXFreeHost(cpu_frame_buffer_);
    cudaError_t error_code = cudaXRMallocHost(&cpu_frame_buffer_, buffer_size);

    if (error_code != cudaSuccess)
    {
        memory_error(false);
        return false;
    }

    cudaXFree(gpu_frame_buffer_);
    error_code = cudaXRMalloc(&gpu_frame_buffer_, buffer_size);

    if (error_code != cudaSuccess)
    {
        memory_error(true);
        return false;
    }

    return true;
}

size_t FileFrameReadWorker::read_copy_file(size_t frames_to_read)
{
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

        GSH::fast_updates_map<FpsType>.get_entry(FpsType::INPUT_FPS).image_processed += 1;
        frames_enqueued++;
        current_nb_frames_read_++;
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
