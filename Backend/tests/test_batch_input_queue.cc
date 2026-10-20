#include "gtest/gtest.h"

#include "cuda_memory.cuh"
#include "batch_input_queue.hh"
#include "stamp_queue.hh"
#include "test_disable_log.hh"

#include <array>
#include <chrono>
#include <future>
#include <thread>

using camera::PixelDepth;

static void ASSERT_QUEUE_ELT_EQ(holovibes::BatchInputQueue& q, size_t pos, std::string expected)
{
    // TODO: getter max size
    // if (pos >= q.get_max_size())
    //    return;

    size_t frame_size = q.get_fd().get_frame_size();

    const char* d_buffer = static_cast<const char*>(q.get_data()); // device buffer
    char* h_buffer = new char[frame_size];                         // host buffer
    // Copy one frame from device buffer to host buffer
    cudaXMemcpy(h_buffer, d_buffer + pos * frame_size, frame_size, cudaMemcpyDeviceToHost);

    ASSERT_EQ(std::string(h_buffer), expected);
}

static char* dequeue_helper(holovibes::BatchInputQueue& q, uint batch_size)
{
    const size_t frame_size = q.get_fd().get_frame_size();
    static const holovibes::BatchInputQueue::dequeue_func_t lambda = [](const void* const src,
                                                                        void* const dest,
                                                                        const uint batch_size,
                                                                        const size_t frame_res,
                                                                        const camera::PixelDepth depth,
                                                                        const cudaStream_t stream)
    {
        const size_t size = static_cast<size_t>(batch_size) * frame_res * depth;
        cudaSafeCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, stream));
    };

    char* d_buff;
    cudaXMallocHost((void**)&d_buff, frame_size * batch_size);
    q.dequeue(d_buff, PixelDepth::Bits8, lambda);

    return d_buff;
}

TEST(BatchInputQueueTest, SimpleInstantiation)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);

    ASSERT_EQ(queue.get_size(), 0);
    // TODO: getter max size
    // ASSERT_EQ(queue.get_max_size(), 3);

    ASSERT_EQ(queue.get_fd().get_frame_size(), 4);
}

TEST(BatchInputQueueTest, SimpleEnqueueOfThreeElements)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr camera::FrameDescriptor fd = {2, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "a\0b\0c\0d\0e\0";

    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    ASSERT_EQ(queue.get_size(), 3);
    ASSERT_QUEUE_ELT_EQ(queue, 0, "a");
    ASSERT_QUEUE_ELT_EQ(queue, 1, "b");
    ASSERT_QUEUE_ELT_EQ(queue, 2, "c");
}

TEST(BatchInputQueueTest, SimpleEnqueueAndDequeueOfThreeElements)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr camera::FrameDescriptor fd = {2, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "a\0b\0c\0d\0e\0";

    // Enqueue
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice);

    char* elt1 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(elt1, std::string("a"));
    ASSERT_EQ(queue.get_size(), 2);

    char* elt2 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(elt2, std::string("b"));
    ASSERT_EQ(queue.get_size(), 1);

    char* elt3 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(elt3, std::string("c"));
    ASSERT_EQ(queue.get_size(), 0);
}

TEST(BatchInputQueueTest, CopiesTimestampsToTheSelectedRecordQueue)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue input(total_nb_frames, batch_size, fd);
    holovibes::Queue first_record(fd, total_nb_frames, holovibes::QueueType::RECORD_QUEUE_2, holovibes::Device::CPU);
    holovibes::Queue second_record(fd, total_nb_frames, holovibes::QueueType::RECORD_QUEUE_3, holovibes::Device::CPU);
    StampQueue first_stamps(8);
    StampQueue second_stamps(8);
    const std::array<unsigned char, 2> frames = {1, 2};

    input.enqueue_with_ids(frames.data(), cudaMemcpyHostToDevice, 2, 10, 1'000, 25, 2'000, 7);
    input.copy_multiple(first_record, 2, cudaMemcpyDeviceToHost, &first_stamps);

    const FrameStamp first = first_stamps.pop_one_blocking();
    const FrameStamp second = first_stamps.pop_one_blocking();
    EXPECT_EQ(first.id, 10);
    EXPECT_EQ(first.synced_us, 1'000);
    EXPECT_EQ(first.camera_us, 2'000);
    EXPECT_EQ(first.offset_us, 7);
    EXPECT_EQ(second.id, 11);
    EXPECT_EQ(second.synced_us, 1'025);
    EXPECT_EQ(second.camera_us, 2'025);

    cudaFreeHost(dequeue_helper(input, batch_size));
    input.enqueue_with_ids(frames.data(), cudaMemcpyHostToDevice, 2, 20, 3'000, 50, 4'000, 9);
    input.copy_multiple(second_record, 2, cudaMemcpyDeviceToHost, &second_stamps);

    const FrameStamp third = second_stamps.pop_one_blocking();
    const FrameStamp fourth = second_stamps.pop_one_blocking();
    EXPECT_EQ(third.id, 20);
    EXPECT_EQ(third.synced_us, 3'000);
    EXPECT_EQ(third.camera_us, 4'000);
    EXPECT_EQ(third.offset_us, 9);
    EXPECT_EQ(fourth.id, 21);
    EXPECT_EQ(fourth.synced_us, 3'050);
    EXPECT_EQ(fourth.camera_us, 4'050);
}

TEST(BatchInputQueueTest, DefersRecordingResetUntilTheConsumerFinishesItsBatch)
{
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue input(8, 2, fd);
    const std::array<unsigned char, 2> old_frames = {1, 2};
    input.enqueue_with_ids(old_frames.data(), cudaMemcpyHostToDevice, 2, 10, 1'000, 25, 2'000, 7);

    std::promise<void> consumer_entered;
    auto entered = consumer_entered.get_future();
    std::promise<void> release_consumer;
    auto release = release_consumer.get_future().share();
    std::thread consumer(
        [&]()
        {
            input.dequeue(nullptr,
                          PixelDepth::Bits8,
                          [&](const void*, void*, uint, size_t, PixelDepth, cudaStream_t)
                          {
                              consumer_entered.set_value();
                              release.wait();
                          });
        });

    EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // The consumer owns batch 0 while the producer's next batch is 1.
    input.request_reset();
    EXPECT_TRUE(input.has_pending_reset());
    EXPECT_EQ(input.get_size(), 1u);
    release_consumer.set_value();
    consumer.join();
    EXPECT_EQ(input.get_size(), 0u);

    input.enqueue_with_ids(old_frames.data(), cudaMemcpyHostToDevice, 2, 20, 3'000, 25, 4'000, 8);
    input.apply_pending_reset();
    EXPECT_FALSE(input.has_pending_reset());
    EXPECT_TRUE(input.is_empty());
    EXPECT_LE(input.get_size(), input.get_max_size());
    EXPECT_FALSE(input.has_overwritten());

    const std::array<unsigned char, 2> new_frames = {3, 4};
    input.enqueue_with_ids(new_frames.data(), cudaMemcpyHostToDevice, 2, 100, 5'000, 50, 6'000, 9);
    holovibes::Queue recorded(fd, 4, holovibes::record_queue_type(7), holovibes::Device::CPU);
    StampQueue stamps(8);
    input.copy_multiple(recorded, 2, cudaMemcpyDeviceToHost, &stamps);
    for (size_t index = 0; index < new_frames.size(); ++index)
    {
        unsigned char frame = 0;
        recorded.dequeue(&frame, 0, cudaMemcpyHostToHost);
        const auto stamp = stamps.pop_one_blocking();
        EXPECT_EQ(frame, new_frames[index]);
        EXPECT_EQ(stamp.id, 100 + index);
        EXPECT_EQ(stamp.synced_us, 5'000 + 50 * index);
        EXPECT_EQ(stamp.camera_us, 6'000 + 50 * index);
        EXPECT_EQ(stamp.offset_us, 9u);
    }
}

TEST(BatchInputQueueTest, ResetRequestDoesNotWaitForAPartialProducerBatch)
{
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue input(4, 2, fd);
    const unsigned char frame = 1;
    input.enqueue_with_ids(&frame, cudaMemcpyHostToDevice, 1, 10, 1'000, 25, 2'000, 7);

    // The producer keeps its mutex until the second frame completes this batch.
    auto requester = std::async(std::launch::async, [&]() { input.request_reset(); });
    EXPECT_EQ(requester.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(input.get_size(), 0u);

    // A different consumer thread must also return without waiting for the camera.
    auto resetter = std::async(std::launch::async, [&]() { input.apply_pending_reset(); });
    EXPECT_EQ(resetter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(input.has_pending_reset());
    EXPECT_EQ(input.get_size(), 0u);

    input.enqueue_with_ids(&frame, cudaMemcpyHostToDevice, 1, 11, 1'025, 25, 2'025, 7);
    requester.get();
    resetter.get();
    input.apply_pending_reset();
    EXPECT_FALSE(input.has_pending_reset());
    EXPECT_TRUE(input.is_empty());
}

TEST(BatchInputQueueTest, ExactRecordBufferFillPreservesFramesAndTimestampsWithoutOverflow)
{
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue input(4, 2, fd);
    holovibes::Queue recorded(fd, 2, holovibes::record_queue_type(7), holovibes::Device::CPU);
    StampQueue stamps(8);
    const std::array<unsigned char, 2> frames = {3, 4};
    input.enqueue_with_ids(frames.data(), cudaMemcpyHostToDevice, 2, 10, 1'000, 25, 2'000, 7);
    input.copy_multiple(recorded, 2, cudaMemcpyDeviceToHost, &stamps);
    EXPECT_EQ(recorded.get_size(), 2u);
    EXPECT_FALSE(recorded.has_overwritten());

    for (size_t index = 0; index < frames.size(); ++index)
    {
        unsigned char frame = 0;
        recorded.dequeue(&frame, 0, cudaMemcpyHostToHost);
        const auto stamp = stamps.pop_one_blocking();
        EXPECT_EQ(frame, frames[index]);
        EXPECT_EQ(stamp.id, 10 + index);
        EXPECT_EQ(stamp.synced_us, 1'000 + 25 * index);
        EXPECT_EQ(stamp.camera_us, 2'000 + 25 * index);
    }

    input.copy_multiple(recorded, 2, cudaMemcpyDeviceToHost);
    EXPECT_FALSE(recorded.has_overwritten());
    input.copy_multiple(recorded, 2, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(recorded.has_overwritten());
    EXPECT_EQ(recorded.get_size(), 2u);
}

TEST(BatchInputQueueTest, LastImageUsesFrameIndicesAfterDequeuingABatch)
{
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue input(8, 2, fd);
    const std::array<unsigned char, 4> frames = {1, 2, 3, 4};
    input.enqueue_with_ids(frames.data(), cudaMemcpyHostToDevice, 4, 10, 1'000, 25, 2'000, 7);
    input.dequeue();

    unsigned char latest = 0;
    cudaXMemcpy(&latest, input.get_last_image(), sizeof(latest), cudaMemcpyDeviceToHost);
    EXPECT_EQ(latest, 4u);
}

TEST(BatchInputQueueTest, ProducerSynchronizationWaitsForEveryTouchedBatchStream)
{
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue input(4, 1, fd);
    const unsigned char frame = 1;
    holovibes::cuda_tools::CudaUniquePtr<unsigned char> source(1);
    cudaXMemcpy(source.get(), &frame, sizeof(frame), cudaMemcpyHostToDevice);
    input.enqueue_with_ids(source.get(), cudaMemcpyDeviceToDevice, 1, 10, 1'000, 25, 2'000, 7);

    struct StreamBarrier
    {
        std::promise<void> entered;
        std::shared_future<void> release;
    } barrier;
    std::promise<void> release_stream;
    barrier.release = release_stream.get_future().share();
    auto entered = barrier.entered.get_future();
    std::thread consumer(
        [&]()
        {
            input.dequeue(nullptr,
                          PixelDepth::Bits8,
                          [&](const void*, void*, uint, size_t, PixelDepth, cudaStream_t stream)
                          {
                              cudaSafeCall(cudaLaunchHostFunc(
                                  stream,
                                  [](void* context)
                                  {
                                      auto& state = *static_cast<StreamBarrier*>(context);
                                      state.entered.set_value();
                                      state.release.wait();
                                  },
                                  &barrier));
                          });
        });

    EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // Batch 0 remains blocked, while the final producer batch uses stream 1.
    input.enqueue_with_ids(source.get(), cudaMemcpyDeviceToDevice, 1, 11, 1'025, 25, 2'025, 7);
    auto final_stream = std::async(std::launch::async, [&]() { input.sync_current_batch(); });
    if (final_stream.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
    {
        release_stream.set_value();
        consumer.join();
        final_stream.get();
        GTEST_SKIP() << "CUDA serialized the blocked host callback with the other batch stream";
    }
    final_stream.get();

    std::promise<void> synchronization_started;
    auto started = synchronization_started.get_future();
    auto synchronized = std::async(std::launch::async,
                                   [&]()
                                   {
                                       synchronization_started.set_value();
                                       input.sync_pending_copies();
                                   });
    EXPECT_EQ(started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(synchronized.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    release_stream.set_value();
    consumer.join();
    synchronized.get();
}

TEST(BatchInputQueueTest, SimpleOverwriteElements)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr camera::FrameDescriptor fd = {2, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "a\0b\0c\0d\0e\0";

    // Enqueue
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice); // A
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice); // B
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice); // C
    queue.enqueue(data + 3 * frame_size, cudaMemcpyHostToDevice); // D
    queue.enqueue(data + 4 * frame_size, cudaMemcpyHostToDevice); // E
    ASSERT_EQ(queue.get_size(), 3);

    char* elt1 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(elt1, std::string("c"));
    ASSERT_EQ(queue.get_size(), 2);

    char* elt2 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(elt2, std::string("d"));
    ASSERT_EQ(queue.get_size(), 1);

    char* elt3 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(elt3, std::string("e"));
    ASSERT_EQ(queue.get_size(), 0);
}

TEST(BatchInputQueueTest, SimpleOverwriteMoreElements)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr camera::FrameDescriptor fd = {4, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "abc\0ABC\0def\0DEF\0ghi\0GHI\0";

    // Enqueue ABC
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice); // abc
    ASSERT_EQ(queue.get_size(), 0);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice); // ABC
    ASSERT_EQ(queue.get_size(), 1);

    // Enqueue DEF
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice); // def
    ASSERT_EQ(queue.get_size(), 1);
    queue.enqueue(data + 3 * frame_size, cudaMemcpyHostToDevice); // DEF
    ASSERT_EQ(queue.get_size(), 2);

    // Dequeue ABC
    char* elt1 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);
    ASSERT_EQ(elt1, std::string("abc"));
    ASSERT_EQ(elt1 + 4, std::string("ABC"));

    // Enqueue GHI
    queue.enqueue(data + 4 * frame_size, cudaMemcpyHostToDevice); // ghi
    ASSERT_EQ(queue.get_size(), 1);
    queue.enqueue(data + 5 * frame_size, cudaMemcpyHostToDevice); // GHI
    ASSERT_EQ(queue.get_size(), 2);

    // Enqueue ABC
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice); // abc
    ASSERT_EQ(queue.get_size(), 2);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice); // ABC
    ASSERT_EQ(queue.get_size(), 2);

    // Dequeue GHI
    char* elt2 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);
    ASSERT_EQ(elt2, std::string("ghi"));
    ASSERT_EQ(elt2 + 4, std::string("GHI"));

    // Dequeue ABC
    char* elt3 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 0);
    ASSERT_EQ(elt3, std::string("abc"));
    ASSERT_EQ(elt3 + 4, std::string("ABC"));
}

TEST(BatchInputQueueTest, SimpleResizeSame)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr camera::FrameDescriptor fd = {5, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "ilan\0nico\0anto\0kaci\0theo\0";

    // Enqueue "ilan\0nico\0" and "anto\0kaci\0"
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_size, cudaMemcpyHostToDevice);
    dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);

    // Resize
    const uint new_batch_size = 2;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);

    // Enqueue "theo\0ilan\0"
    queue.enqueue(data + 4 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    ASSERT_EQ(queue.get_size(), 1);
    dequeue_helper(queue, new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);
}

TEST(BatchInputQueueTest, SimpleResizeGreater)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr camera::FrameDescriptor fd = {5, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "ilan\0nico\0anto\0kaci\0theo\0";

    // Enqueue "ilan\0nico\0" and "anto\0kaci\0"
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_size, cudaMemcpyHostToDevice);
    dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);

    // Resize
    const uint new_batch_size = 4;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);

    // Enqueue "theo\0ilan\0" and "anto\0kaci\0"
    queue.enqueue(data + 4 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_size, cudaMemcpyHostToDevice);
    ASSERT_EQ(queue.get_size(), 1);
    dequeue_helper(queue, new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);
}

TEST(BatchInputQueueTest, SimpleResizeLower)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr camera::FrameDescriptor fd = {5, 1, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_size = queue.get_fd().get_frame_size();

    const char* data = "ilan\0nico\0anto\0kaci\0theo\0";

    // Enqueue "ilan\0nico\0" and "anto\0kaci\0"
    queue.enqueue(data + 0 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_size, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_size, cudaMemcpyHostToDevice);
    dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);

    // Resize
    const uint new_batch_size = 1;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);

    // Enqueue "theo\0"
    queue.enqueue(data + 4 * frame_size, cudaMemcpyHostToDevice);
    ASSERT_EQ(queue.get_size(), 1);
    dequeue_helper(queue, new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);
}

void consumer(holovibes::BatchInputQueue& queue,
              const uint nb_actions,
              std::atomic<bool>& stop_requested,
              uint batch_size,
              bool resize_during_exec,
              const uint max_batch_size)
{
    for (uint i = 0; i < nb_actions && !stop_requested; i++)
    {
        while (!stop_requested && queue.get_size() == 0)
            continue;

        if (stop_requested)
            return;

        if (stop_requested)
            return;

        if (resize_during_exec && i % 4 == 0)
        {
            camera::FrameDescriptor fd = queue.get_fd();
            holovibes::Queue copy_queue(fd, queue.get_total_nb_frames());
            batch_size = std::min(batch_size * 2, max_batch_size);
            queue.copy_multiple(copy_queue);
            queue.resize(batch_size);
        }
        else
            dequeue_helper(queue, batch_size);
    }
}

void consumer_gpu(holovibes::BatchInputQueue& queue,
                  const uint nb_actions,
                  std::atomic<bool>& stop_requested,
                  uint batch_size,
                  float* const d_buff,
                  holovibes::BatchInputQueue::dequeue_func_t dequeue_func)
{
    for (uint i = 0; i < nb_actions && !stop_requested; i++)
    {
        while (!stop_requested && queue.get_size() == 0)
            continue;

        if (stop_requested)
            return;

        queue.dequeue(d_buff, PixelDepth::Bits32, dequeue_func);

        if (stop_requested)
            return;
    }
}

template <typename T>
void producer(holovibes::BatchInputQueue& queue, const uint nb_actions, const size_t frame_res)
{
    const T* frame = new T[frame_res];

    for (size_t i = 0; i < nb_actions; i++)
        queue.enqueue(frame, cudaMemcpyHostToDevice);

    queue.stop_producer();

    delete[] frame;
}

void producer_gpu(holovibes::BatchInputQueue& queue,
                  const uint nb_actions,
                  const size_t frame_res,
                  const float* const d_buff)
{

    for (size_t i = 0; i < nb_actions; i++)
    {
        queue.enqueue(d_buff, cudaMemcpyDeviceToDevice);
    }

    queue.stop_producer();
}

TEST(BatchInputQueueTest, ProducerConsumerSituationNoDeadlock)
{
    constexpr uint nb_tests = 50;
    for (uint i = 0; i < nb_tests; i++)
    {
        constexpr uint total_nb_frames = 1024;
        constexpr uint batch_size = 1;
        constexpr uint max_batch_size = total_nb_frames;
        constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits32, camera::Endianness::LittleEndian};
        holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
        size_t frame_res = queue.get_fd().get_frame_res();

        // Consumer will do less actions. It is maximum in case of batch size ==
        // 1
        constexpr uint consumer_actions = total_nb_frames;
        constexpr uint producer_actions = total_nb_frames;
        std::atomic<bool> stop_requested{false};

        std::thread consumer_thread(&(consumer),
                                    std::ref(queue),
                                    consumer_actions,
                                    std::ref(stop_requested),
                                    batch_size,
                                    true,
                                    max_batch_size);

        std::thread producer_thread(&(producer<float>), std::ref(queue), producer_actions, frame_res);

        producer_thread.join();
        stop_requested = true;
        consumer_thread.join();
    }

    ASSERT_EQ(0, 0);
}

TEST(BatchInputQueueTest, ProducerConsumerSituationNoDeadlockSmallSize)
{
    // Test with float
    constexpr camera::FrameDescriptor fd = {1, 1, PixelDepth::Bits32, camera::Endianness::LittleEndian};
    constexpr uint total_nb_frames = 2;
    constexpr uint batch_size = 1;
    const size_t frame_size = fd.get_frame_size();
    static const holovibes::BatchInputQueue::dequeue_func_t dequeue_func = [](const void* const src,
                                                                              void* const dest,
                                                                              const uint batch_size,
                                                                              const size_t frame_res,
                                                                              const camera::PixelDepth depth,
                                                                              const cudaStream_t stream)
    {
        const size_t size = static_cast<size_t>(batch_size) * frame_res * depth;
        cudaSafeCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice, stream));
    };

    float* d_producer;
    float* d_consumer;
    cudaXMalloc((void**)&d_producer, frame_size);
    cudaXMalloc((void**)&d_consumer, frame_size * batch_size);

    constexpr uint nb_tests = 50;
    for (uint i = 0; i < nb_tests; i++)
    {
        holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
        size_t frame_res = queue.get_fd().get_frame_res();

        // Consumer will do less actions. It is maximum in case of batch size ==
        // 1
        constexpr uint consumer_actions = 1000;
        constexpr uint producer_actions = 1000;
        std::atomic<bool> stop_requested{false};

        std::thread consumer_thread(&(consumer_gpu),
                                    std::ref(queue),
                                    consumer_actions,
                                    std::ref(stop_requested),
                                    batch_size,
                                    d_consumer,
                                    dequeue_func);

        std::thread producer_thread(&(producer_gpu), std::ref(queue), producer_actions, frame_res, d_producer);

        producer_thread.join();
        stop_requested = true;
        consumer_thread.join();
    }

    cudaSafeCall(cudaFree(d_producer));
    cudaSafeCall(cudaFree(d_consumer));
    ASSERT_EQ(0, 0);
}

TEST(BatchInputQueueTest, FullProducerConsumerSituationFloat)
{
    constexpr uint total_nb_frames = 4096;
    constexpr uint batch_size = 1;
    constexpr uint max_batch_size = total_nb_frames;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits32, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_res = queue.get_fd().get_frame_res();

    // Consumer will do less actions. It is maximum in case of batch size ==
    // 1
    constexpr uint actions = total_nb_frames;

    std::atomic<bool> stop_requested{false};

    std::thread consumer_thread(&(consumer),
                                std::ref(queue),
                                actions,
                                std::ref(stop_requested),
                                batch_size,
                                false,
                                max_batch_size);
    std::thread producer_thread(&(producer<float>), std::ref(queue), actions, frame_res);

    producer_thread.join();

    // consume all frames before stopping the thread
    while (queue.get_size() != 0)
        continue;

    stop_requested = true;

    consumer_thread.join();

    ASSERT_EQ(0, 0);
}

TEST(BatchInputQueueTest, FullProducerConsumerSituationChar)
{
    constexpr uint total_nb_frames = 4096;
    constexpr uint batch_size = 1;
    constexpr uint max_batch_size = total_nb_frames;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_res = queue.get_fd().get_frame_res();

    // Consumer will do less actions. It is maximum in case of batch size ==
    // 1
    constexpr uint actions = total_nb_frames;

    std::atomic<bool> stop_requested{false};

    std::thread consumer_thread(&(consumer),
                                std::ref(queue),
                                actions,
                                std::ref(stop_requested),
                                batch_size,
                                false,
                                max_batch_size);
    std::thread producer_thread(&(producer<char>), std::ref(queue), actions, frame_res);

    producer_thread.join();

    while (queue.get_size() != 0)
        continue;

    stop_requested = true;

    // consume all frames before stopping the thread
    consumer_thread.join();

    ASSERT_EQ(0, 0);
}

TEST(BatchInputQueueTest, FullProducerConsumerSituationShort)
{
    constexpr uint total_nb_frames = 4096;
    constexpr uint batch_size = 1;
    constexpr uint max_batch_size = total_nb_frames;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits16, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_res = queue.get_fd().get_frame_res();

    // Consumer will do less actions. It is maximum in case of batch size ==
    // 1
    constexpr uint actions = total_nb_frames;

    std::atomic<bool> stop_requested{false};

    std::thread consumer_thread(&(consumer),
                                std::ref(queue),
                                actions,
                                std::ref(stop_requested),
                                batch_size,
                                false,
                                max_batch_size);
    std::thread producer_thread(&(producer<short>), std::ref(queue), actions, frame_res);

    producer_thread.join();

    while (queue.get_size() != 0)
        continue;

    stop_requested = true;

    // consume all frames before stopping the thread
    consumer_thread.join();

    ASSERT_EQ(0, 0);
}

TEST(BatchInputQueueTest, PartialProducerConsumerSituationShort)
{
    constexpr uint total_nb_frames = 4096;
    constexpr uint batch_size = 1;
    constexpr uint max_batch_size = total_nb_frames;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits16, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);
    size_t frame_res = queue.get_fd().get_frame_res();

    // Consumer will do less actions. It is maximum in case of batch size ==
    // 1
    constexpr uint consumer_actions = 2;
    constexpr uint producer_actions = 10;

    std::atomic<bool> stop_requested{false};

    std::thread consumer_thread(&(consumer),
                                std::ref(queue),
                                consumer_actions,
                                std::ref(stop_requested),
                                batch_size,
                                false,
                                max_batch_size);
    std::thread producer_thread(&(producer<short>), std::ref(queue), producer_actions, frame_res);

    producer_thread.join();

    // consume all frames before stopping the thread
    consumer_thread.join();

    ASSERT_EQ(queue.get_size(), 8);
}

TEST(BatchInputQueueTest, CreateQueueSizeNotMatcingBatchSize)
{
    constexpr uint total_nb_frames = 11;
    constexpr uint batch_size = 5;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits16, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);

    ASSERT_EQ(queue.get_total_nb_frames(), total_nb_frames - total_nb_frames % batch_size);
}

TEST(BatchInputQueueTest, ResizeQueueSizeNotMatcingBatchSize)
{
    constexpr uint total_nb_frames = 10;
    constexpr uint batch_size = 5;
    constexpr camera::FrameDescriptor fd = {2, 2, PixelDepth::Bits16, camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);

    ASSERT_EQ(queue.get_total_nb_frames(), total_nb_frames);

    constexpr uint new_batch_size = 6;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_total_nb_frames(), total_nb_frames - (total_nb_frames % new_batch_size));
}
