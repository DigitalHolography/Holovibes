#include "gtest/gtest.h"

#include "cuda_memory.cuh"
#include "input_queue.hh"

#include <thread>

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
                                                                        const uint depth,
                                                                        const cudaStream_t stream)
    {
        const size_t size = static_cast<size_t>(batch_size) * frame_res * depth;
        cudaSafeCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, stream));
    };

    char* d_buff;
    cudaXMallocHost((void**)&d_buff, frame_size * batch_size);
    q.dequeue(d_buff, sizeof(char), lambda);

    return d_buff;
}

TEST(BatchInputQueueTest, SimpleInstantiation)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {2, 1, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {2, 1, sizeof(char), camera::Endianness::LittleEndian};
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

TEST(BatchInputQueueTest, SimpleOverwriteElements)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr camera::FrameDescriptor fd = {2, 1, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {4, 1, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {5, 1, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {5, 1, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {5, 1, sizeof(char), camera::Endianness::LittleEndian};
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

        queue.dequeue(d_buff, sizeof(float), dequeue_func);

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
        constexpr camera::FrameDescriptor fd = {2, 2, sizeof(float), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {1, 1, sizeof(float), camera::Endianness::LittleEndian};
    constexpr uint total_nb_frames = 2;
    constexpr uint batch_size = 1;
    const size_t frame_size = fd.get_frame_size();
    static const holovibes::BatchInputQueue::dequeue_func_t dequeue_func = [](const void* const src,
                                                                              void* const dest,
                                                                              const uint batch_size,
                                                                              const size_t frame_res,
                                                                              const uint depth,
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
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(float), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(char), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(short), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(short), camera::Endianness::LittleEndian};
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
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(short), camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);

    ASSERT_EQ(queue.get_total_nb_frames(), total_nb_frames - total_nb_frames % batch_size);
}

TEST(BatchInputQueueTest, ResizeQueueSizeNotMatcingBatchSize)
{
    constexpr uint total_nb_frames = 10;
    constexpr uint batch_size = 5;
    constexpr camera::FrameDescriptor fd = {2, 2, sizeof(short), camera::Endianness::LittleEndian};
    holovibes::BatchInputQueue queue(total_nb_frames, batch_size, fd);

    ASSERT_EQ(queue.get_total_nb_frames(), total_nb_frames);

    constexpr uint new_batch_size = 6;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_total_nb_frames(), total_nb_frames - (total_nb_frames % new_batch_size));
}
