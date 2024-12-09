#include "worker_api.hh"

#include "API.hh"

namespace holovibes::api
{

void WorkerApi::start_frame_record(const std::function<void()>& callback)
{
    api_->compute.pipe_refresh();
    if (API.transform.get_batch_size() > get_record_buffer_size())
    {
        LOG_ERROR("[RECORDER] Batch size must be lower than record queue size");
        return;
    }

    set_record_frame_count(get_record_frame_count());

    if (!record_queue_.load())
        init_record_queue();

    frame_record_worker_controller_.set_callback(callback);
    // frame_record_worker_controller_.set_error_callback(error_callback_);
    frame_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(Holovibes::instance().realtime_settings_.settings_);
    frame_record_worker_controller_.start(all_settings,
                                          Holovibes::instance().get_cuda_streams().recorder_stream,
                                          record_queue_);
}

void WorkerApi::stop_frame_record() { frame_record_worker_controller_.stop(); }

void WorkerApi::start_chart_record(const std::function<void()>& callback)
{
    chart_record_worker_controller_.set_callback(callback);
    // chart_record_worker_controller_.set_error_callback(error_callback_);
    chart_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(Holovibes::instance().realtime_settings_.settings_);
    chart_record_worker_controller_.start(all_settings);
}

void WorkerApi::stop_chart_record() { chart_record_worker_controller_.stop(); }

} // namespace holovibes::api