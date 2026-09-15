#include "async_record_writer.hh"

#include <limits>
#include <algorithm>
#include <stdexcept>

#include "logger.hh"

namespace holovibes::worker
{
namespace
{
constexpr size_t GIB = size_t{1} << 30;

std::string describe_error(const AsyncRecordWriter::Job& job, const std::string& error)
{
    return job.file ? job.file->get_file_path() + ": " + error : error;
}
}

AsyncRecordWriter::AsyncRecordWriter() : thread_(&AsyncRecordWriter::run, this) {}

AsyncRecordWriter::~AsyncRecordWriter()
{
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
    if (thread_.joinable())
        thread_.join();
}

void AsyncRecordWriter::set_limit_gib(size_t gib)
{
    if (gib > std::numeric_limits<size_t>::max() / GIB)
        throw std::invalid_argument("Queued save RAM limit is too large");
    std::lock_guard lock(mutex_);
    limit_bytes_ = gib * GIB;
}

size_t AsyncRecordWriter::get_limit_gib() const
{
    std::lock_guard lock(mutex_);
    return limit_bytes_ / GIB;
}

bool AsyncRecordWriter::can_reserve(size_t bytes) const
{
    std::lock_guard lock(mutex_);
    return limit_bytes_ != 0 && reserved_bytes_ <= limit_bytes_ && bytes <= limit_bytes_ - reserved_bytes_;
}

bool AsyncRecordWriter::can_reserve_unbounded(size_t minimum_frame_bytes) const
{
    std::lock_guard lock(mutex_);
    const size_t share = std::max(minimum_frame_bytes, limit_bytes_ / 2);
    return limit_bytes_ != 0 && reserved_bytes_ <= limit_bytes_ && share <= limit_bytes_ - reserved_bytes_;
}

std::shared_ptr<AsyncRecordWriter::Job> AsyncRecordWriter::reserve(size_t bytes)
{
    std::lock_guard lock(mutex_);
    if (limit_bytes_ == 0 || reserved_bytes_ > limit_bytes_ || bytes > limit_bytes_ - reserved_bytes_ || stopping_)
        return nullptr;
    auto job = std::make_shared<Job>(bytes);
    reserved_bytes_ += bytes;
    jobs_.push_back(job);
    changed_.notify_all();
    return job;
}

std::shared_ptr<AsyncRecordWriter::Job> AsyncRecordWriter::reserve_unbounded(size_t minimum_frame_bytes)
{
    std::lock_guard lock(mutex_);
    const size_t share = std::max(minimum_frame_bytes, limit_bytes_ / 2);
    if (limit_bytes_ == 0 || reserved_bytes_ > limit_bytes_ || share > limit_bytes_ - reserved_bytes_ || stopping_)
        return nullptr;
    auto job = std::make_shared<Job>(share);
    reserved_bytes_ += share;
    jobs_.push_back(job);
    changed_.notify_all();
    return job;
}

void AsyncRecordWriter::attach_file(const std::shared_ptr<Job>& job,
                                    std::unique_ptr<io_files::OutputFrameFile> file)
{
    std::lock_guard lock(mutex_);
    job->file = std::move(file);
    changed_.notify_all();
}

void AsyncRecordWriter::enqueue(const std::shared_ptr<Job>& job, std::unique_ptr<char[]> frame, size_t size)
{
    std::unique_lock lock(mutex_);
    if (size > job->reserved_bytes)
        throw std::runtime_error("One recording frame exceeds its RAM staging reservation");
    changed_.wait(lock, [&] {
        return job->failed || job->capture_finished || size <= job->reserved_bytes - job->staged_bytes;
    });
    if (job->failed)
        throw std::runtime_error(job->error);
    if (job->capture_finished)
        throw std::runtime_error("Recording capture has already finished");
    job->staged_bytes += size;
    job->frames.push_back({std::move(frame), size});
    changed_.notify_all();
}

bool AsyncRecordWriter::has_failed(const std::shared_ptr<Job>& job) const
{
    std::lock_guard lock(mutex_);
    return job->failed;
}

void AsyncRecordWriter::finish(const std::shared_ptr<Job>& job, size_t recorded_frames)
{
    std::lock_guard lock(mutex_);
    job->recorded_frames = recorded_frames;
    job->capture_finished = true;
    changed_.notify_all();
}

void AsyncRecordWriter::abort(const std::shared_ptr<Job>& job, const std::string& error)
{
    std::lock_guard lock(mutex_);
    if (!job->failed)
    {
        job->failed = true;
        job->error = describe_error(*job, error);
        last_error_ = job->error;
        ++failed_count_;
        LOG_ERROR("Background recording save aborted: {}", last_error_);
    }
    job->capture_finished = true;
    changed_.notify_all();
}

size_t AsyncRecordWriter::pending_count() const
{
    std::lock_guard lock(mutex_);
    return jobs_.size();
}

size_t AsyncRecordWriter::failed_count() const
{
    std::lock_guard lock(mutex_);
    return failed_count_;
}

std::string AsyncRecordWriter::last_error() const
{
    std::lock_guard lock(mutex_);
    return last_error_;
}

void AsyncRecordWriter::run()
{
    std::unique_lock lock(mutex_);
    while (true)
    {
        changed_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty())
            return;

        auto job = jobs_.front();
        changed_.wait(lock, [&] { return job->failed || !job->frames.empty() || job->capture_finished; });

        while (!job->frames.empty() && !job->failed)
        {
            Frame frame = std::move(job->frames.front());
            job->frames.pop_front();
            auto* file = job->file.get();
            lock.unlock();
            try
            {
                if (!file)
                    throw std::runtime_error("Recording output file was not opened");
                file->write_frame(frame.data.get(), frame.size);
                frame.data.reset();
            }
            catch (const std::exception& e)
            {
                lock.lock();
                job->failed = true;
                job->error = describe_error(*job, e.what());
                last_error_ = job->error;
                ++failed_count_;
                LOG_ERROR("Background recording save failed: {}", last_error_);
                changed_.notify_all();
                break;
            }
            lock.lock();
            job->staged_bytes -= frame.size;
            changed_.notify_all();
        }

        if (job->failed && !job->capture_finished)
            changed_.wait(lock, [&] { return job->capture_finished; });
        if (!job->capture_finished)
            continue;

        if (!job->failed)
        {
            auto* file = job->file.get();
            lock.unlock();
            try
            {
                if (!file)
                    throw std::runtime_error("Recording output file was not opened");
                file->correct_number_of_frames(job->recorded_frames);
                file->write_footer();
                LOG_INFO("Background recording save finished: {}", file->get_file_path());
            }
            catch (const std::exception& e)
            {
                lock.lock();
                job->failed = true;
                job->error = describe_error(*job, e.what());
                last_error_ = job->error;
                ++failed_count_;
                LOG_ERROR("Background recording save failed: {}", last_error_);
                changed_.notify_all();
                lock.unlock();
            }
            lock.lock();
        }

        job->frames.clear();
        // Close the file before releasing the reservation or allowing the next
        // queued file to use the disk.
        auto file = std::move(job->file);
        lock.unlock();
        file.reset();
        lock.lock();
        reserved_bytes_ -= job->reserved_bytes;
        jobs_.pop_front();
        changed_.notify_all();
    }
}
} // namespace holovibes::worker
