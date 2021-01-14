/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "gtest/gtest.h"

#include "cuda_memory.cuh"
#include "cuda_tools.cuh"
#include "batch_input_queue.hh"

#include <thread>

template <typename T>
static void
ASSERT_QUEUE_ELT_EQ(BatchInputQueue<T>& q, size_t pos, std::string expected)
{
    if (pos >= q.get_max_size())
        return;

    size_t frame_size = q.get_frame_res() * sizeof(T);

    const T* d_buffer = q.get_data(); // device buffer
    T* h_buffer = new T[frame_size];  // host buffer
    // Copy one frame from device buffer to host buffer
    cudaXMemcpy(h_buffer,
                d_buffer + pos * frame_size,
                frame_size,
                cudaMemcpyDeviceToHost);

    ASSERT_EQ(std::string(h_buffer), expected);
}

template <typename T>
static T* dequeue_helper(BatchInputQueue<T>& q, uint batch_size)
{
    const uint frame_res = q.get_frame_res();
    const auto lambda = [](const T* const src,
                           T* const dest,
                           const uint batch_size,
                           const uint frame_res,
                           const cudaStream_t stream) {
        const size_t size =
            static_cast<size_t>(batch_size) * frame_res * sizeof(T);
        cuda_safe_call(
            cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, stream));
    };

    T* d_buff;
    cuda_safe_call(
        cudaMallocHost((void**)&d_buff, sizeof(T) * frame_res * batch_size));
    q.dequeue(d_buff, lambda);

    return d_buff;
}

TEST(BatchInputQueueTest, SimpleInstantiation)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr uint frame_res = 2;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    ASSERT_EQ(queue.get_size(), 0);
    ASSERT_EQ(queue.get_max_size(), 3);
    ASSERT_EQ(queue.get_frame_res(), 2);
}

TEST(BatchInputQueueTest, SimpleEnqueueOfThreeElements)
{
    constexpr uint total_nb_frames = 3;
    constexpr uint batch_size = 1;
    constexpr uint frame_res = 2;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "a\0b\0c\0d\0e\0";

    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice);

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
    constexpr uint frame_res = 2;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "a\0b\0c\0d\0e\0";

    // Enqueue
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice);

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
    constexpr uint frame_res = 2;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "a\0b\0c\0d\0e\0";

    // Enqueue
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice); // A
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice); // B
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice); // C
    queue.enqueue(data + 3 * frame_res, cudaMemcpyHostToDevice); // D
    queue.enqueue(data + 4 * frame_res, cudaMemcpyHostToDevice); // E
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
    constexpr uint frame_res = 4;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "abc\0ABC\0def\0DEF\0ghi\0GHI\0";

    // Enqueue ABC
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice); // abc
    ASSERT_EQ(queue.get_size(), 0);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice); // ABC
    ASSERT_EQ(queue.get_size(), 1);

    // Enqueue DEF
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice); // def
    ASSERT_EQ(queue.get_size(), 1);
    queue.enqueue(data + 3 * frame_res, cudaMemcpyHostToDevice); // DEF
    ASSERT_EQ(queue.get_size(), 2);

    // Dequeue ABC
    char* elt1 = dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);
    ASSERT_EQ(elt1, std::string("abc"));
    ASSERT_EQ(elt1 + 4, std::string("ABC"));

    // Enqueue GHI
    queue.enqueue(data + 4 * frame_res, cudaMemcpyHostToDevice); // ghi
    ASSERT_EQ(queue.get_size(), 1);
    queue.enqueue(data + 5 * frame_res, cudaMemcpyHostToDevice); // GHI
    ASSERT_EQ(queue.get_size(), 2);

    // Enqueue ABC
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice); // abc
    ASSERT_EQ(queue.get_size(), 2);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice); // ABC
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
    constexpr uint frame_res = 5;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "ilan\0nico\0anto\0kaci\0theo\0";

    // Enqueue "ilan\0nico\0" and "anto\0kaci\0"
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_res, cudaMemcpyHostToDevice);
    dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);

    // Resize
    const uint new_batch_size = 2;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);

    // Enqueue "theo\0ilan\0"
    queue.enqueue(data + 4 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    ASSERT_EQ(queue.get_size(), 1);
    dequeue_helper(queue, new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);
}

TEST(BatchInputQueueTest, SimpleResizeGreater)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr uint frame_res = 5;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "ilan\0nico\0anto\0kaci\0theo\0";

    // Enqueue "ilan\0nico\0" and "anto\0kaci\0"
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_res, cudaMemcpyHostToDevice);
    dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);

    // Resize
    const uint new_batch_size = 4;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);

    // Enqueue "theo\0ilan\0" and "anto\0kaci\0"
    queue.enqueue(data + 4 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_res, cudaMemcpyHostToDevice);
    ASSERT_EQ(queue.get_size(), 1);
    dequeue_helper(queue, new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);
}

TEST(BatchInputQueueTest, SimpleResizeLower)
{
    constexpr uint total_nb_frames = 4;
    constexpr uint batch_size = 2;
    constexpr uint frame_res = 5;
    BatchInputQueue<char> queue(total_nb_frames, batch_size, frame_res);

    char* data = "ilan\0nico\0anto\0kaci\0theo\0";

    // Enqueue "ilan\0nico\0" and "anto\0kaci\0"
    queue.enqueue(data + 0 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 1 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 2 * frame_res, cudaMemcpyHostToDevice);
    queue.enqueue(data + 3 * frame_res, cudaMemcpyHostToDevice);
    dequeue_helper(queue, batch_size);
    ASSERT_EQ(queue.get_size(), 1);

    // Resize
    const uint new_batch_size = 1;
    queue.resize(new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);

    // Enqueue "theo\0"
    queue.enqueue(data + 4 * frame_res, cudaMemcpyHostToDevice);
    ASSERT_EQ(queue.get_size(), 1);
    dequeue_helper(queue, new_batch_size);
    ASSERT_EQ(queue.get_size(), 0);
}

template <typename T>
void consumer(BatchInputQueue<T>& queue,
              const uint nb_actions,
              std::atomic<bool>& stop_requested,
              uint batch_size,
              const uint max_batch_size)
{
    for (uint i = 0; i < nb_actions && !stop_requested; i++)
    {
        while (!stop_requested && queue.get_size() == 0)
            continue;

        if (stop_requested)
            return;

        dequeue_helper(queue, batch_size);

        if (stop_requested)
            return;

        if (i % 4 == 0)
        {
            // batch_size = std::min(batch_size * 2, max_batch_size);
            queue.resize(batch_size);
        }
    }
}

template <typename T>
void producer(BatchInputQueue<T>& queue,
              const uint nb_actions,
              const uint frame_res)
{
    const T* frame = new T[frame_res];

    for (size_t i = 0; i < nb_actions; i++)
        queue.enqueue(frame, cudaMemcpyHostToDevice);

    queue.stop_producer();

    delete[] frame;
}

TEST(BatchInputQueueTest, SimpleProducerConsumerSituation)
{
    for (size_t i = 0; i < 1000; i++)
    {
        constexpr uint total_nb_frames = 4096;
        constexpr uint batch_size = 1;
        constexpr uint max_batch_size = total_nb_frames;
        constexpr uint frame_res = 4;
        BatchInputQueue<float> queue(total_nb_frames, batch_size, frame_res);

        // Consumer will do less actions. It is maximum in case of batch size ==
        // 1
        constexpr uint consumer_actions = total_nb_frames;
        constexpr uint producer_actions = total_nb_frames;
        std::atomic<bool> stop_requested{false};

        std::thread consumer_thread(&(consumer<float>),
                                    std::ref(queue),
                                    consumer_actions,
                                    std::ref(stop_requested),
                                    batch_size,
                                    max_batch_size);
        std::thread producer_thread(&(producer<float>),
                                    std::ref(queue),
                                    producer_actions,
                                    frame_res);

        producer_thread.join();
        stop_requested = true;
        consumer_thread.join();

        std::cout << "OK: " << i << std::endl;
    }

    ASSERT_EQ(0, 0);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}