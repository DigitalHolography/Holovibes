#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "output_frame_file.hh"

namespace holovibes::worker
{
// One disk writer serves all recordings. Finite captures reserve their entire
// expected size; manually stopped captures reserve half the configured budget
// and wait for the disk if that share fills.
class AsyncRecordWriter
{
  public:
    struct Frame
    {
        std::unique_ptr<char[]> data;
        size_t size = 0;
    };

    struct Job
    {
        explicit Job(size_t reserved) : reserved_bytes(reserved) {}

        const size_t reserved_bytes;
        std::unique_ptr<io_files::OutputFrameFile> file;
        std::deque<Frame> frames;
        size_t staged_bytes = 0;
        size_t recorded_frames = 0;
        bool capture_finished = false;
        bool failed = false;
        std::string error;
    };

    AsyncRecordWriter();
    ~AsyncRecordWriter();
    AsyncRecordWriter(const AsyncRecordWriter&) = delete;
    AsyncRecordWriter& operator=(const AsyncRecordWriter&) = delete;

    void set_limit_gib(size_t gib);
    size_t get_limit_gib() const;
    bool can_reserve(size_t bytes) const;
    bool can_reserve_unbounded(size_t minimum_frame_bytes) const;
    std::shared_ptr<Job> reserve(size_t bytes);
    std::shared_ptr<Job> reserve_unbounded(size_t minimum_frame_bytes);
    void attach_file(const std::shared_ptr<Job>& job, std::unique_ptr<io_files::OutputFrameFile> file);
    void enqueue(const std::shared_ptr<Job>& job, std::unique_ptr<char[]> frame, size_t size);
    bool has_failed(const std::shared_ptr<Job>& job) const;
    void finish(const std::shared_ptr<Job>& job, size_t recorded_frames);
    void abort(const std::shared_ptr<Job>& job, const std::string& error);
    size_t pending_count() const;
    size_t failed_count() const;
    std::string last_error() const;

  private:
    void run();

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::thread thread_;
    size_t limit_bytes_ = 0;
    size_t reserved_bytes_ = 0;
    size_t failed_count_ = 0;
    std::string last_error_;
    bool stopping_ = false;
};
} // namespace holovibes::worker
