#include "gtest/gtest.h"

#include <cuda.h>

#include "holovibes.hh"
#include "queue.hh"
#include "frame_desc.hh"
#include "cuda_memory.cuh"

using namespace holovibes;

static constexpr cudaStream_t stream = 0;

/*! \brief Get the element at a specific position in the queue */
static char* get_element_from_queue(holovibes::Queue& q, size_t pos)
{
    if (pos >= q.get_max_size())
        return nullptr;

    size_t frame_size = q.get_fd().get_frame_size();

    char* d_buffer = static_cast<char*>(q.get_data()); // device buffer
    char* h_buffer = new char[frame_size];             // host buffer
    // Copy one frame from device buffer to host buffer
    cudaXMemcpy(h_buffer, d_buffer + pos * frame_size, frame_size, cudaMemcpyDeviceToHost);
    return h_buffer;
}

#if 0
/*! \brief Print a queue (for debug purpose) */
static std::ostream& operator<<(std::ostream& os, holovibes::Queue& q)
{
    size_t pos = q.get_start_index();
    for (size_t i = 0; i != q.get_size(); ++i)
    {
        os << std::string(get_element_from_queue(q, pos)) << std::endl;
        pos = (pos + 1) % q.get_max_size();
    }
    return os;
}
#endif

TEST(QueueTest, SimpleInstantiatingTest)
{
    FrameDescriptor fd = {64, 64, 1, Endianness::BigEndian};
    holovibes::Queue q(fd, 5, holovibes::QueueType::UNDEFINED, 64, 64, 1);
    // WARNING: Set false because the queue is used in a CLI mode
    ASSERT_EQ(0.0, 0.0);
}

TEST(ZeroQueueInstantiation, ZeroQueue)
{
    FrameDescriptor fd = {4, 4, sizeof(char), Endianness::BigEndian};
    ASSERT_THROW(holovibes::Queue q(fd, 0, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth),
                 std::logic_error);
}

TEST(QueueEmpty, QueueIsFullTest)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 5, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    ASSERT_FALSE(q.is_full());
}

TEST(QueueNotFull, QueueIsFullTest)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    char* new_elt = new char[fd.get_frame_res()];

    // Enqueue
    // Warning: enqueue HostToDevice
    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_FALSE(q.is_full());

    delete[] new_elt;
}

TEST(QueueFull, QueueIsFull)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* new_elt = new char[fd.get_frame_res()];

    // Enqueue
    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(q.is_full());

    delete[] new_elt;
}

TEST(SimpleQueueResize, QueueResize)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_max_size(), 2);

    unsigned int new_size = 10;
    q.resize(new_size); // Resize here, empty the queue
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_max_size(), new_size);
}

TEST(EnqueueCheckValues, QueueEnqueue)
{
    FrameDescriptor fd = {1, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char elt1 = 'a';
    char elt2 = 'b';
    char elt3 = 'c';
    char* buffer = nullptr;

    q.enqueue(&elt1, stream, cudaMemcpyHostToDevice);
    buffer = get_element_from_queue(q, 0);
    ASSERT_EQ(q.get_size(), 1);
    ASSERT_EQ(*buffer, elt1);

    q.enqueue(&elt2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(*get_element_from_queue(q, 0), elt1);
    ASSERT_EQ(*get_element_from_queue(q, 1), elt2);

    q.enqueue(&elt3, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(*get_element_from_queue(q, 0), elt3);
    ASSERT_EQ(*get_element_from_queue(q, 1), elt2);
}

TEST(SimpleEnqueues, QueueEnqueue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    ASSERT_EQ(q.get_size(), 0);

    char* new_elt = new char[fd.get_frame_res()];

    bool res = q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 1);
    ASSERT_EQ(q.get_start_index(), 0);
    // just test onces the return value
    // We can't easily make the enqueue fail in the test
    ASSERT_TRUE(res);

    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    // Queue full, circular queue, update start index
    // Reminder: the size of the queue is 2
    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 1);

    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 1);

    delete[] new_elt;
}

TEST(EnqueueNotSquare, QueueEnqueue)
{
    FrameDescriptor fd = {56, 17, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    ASSERT_EQ(q.get_size(), 0);

    char* new_elt = new char[fd.get_frame_res()];

    bool res = q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 1);

    delete[] new_elt;
}

TEST(MultipleEnqueueCheckValues, QueueMultipleEnqueue)
{
    FrameDescriptor fd = {1, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char elts[] = {'a', 'b', 'c'};

    q.enqueue_multiple(elts, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(*get_element_from_queue(q, 0), elts[0]);
    ASSERT_EQ(*get_element_from_queue(q, 1), elts[1]);

    // enqueue 3rd element
    q.enqueue_multiple(elts + 2, 1, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(*get_element_from_queue(q, 0), elts[2]);
    ASSERT_EQ(*get_element_from_queue(q, 1), elts[1]);
    ASSERT_EQ(q.get_start_index(), 1);

    q.resize(2); // reset, same size
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 0);

    // Directly enqueue three elements. The first element is going to be skipped
    q.enqueue_multiple(elts, 3, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_start_index(), 1);
    ASSERT_EQ(*get_element_from_queue(q, 0), elts[2]);
    ASSERT_EQ(*get_element_from_queue(q, 1), elts[1]);

    q.enqueue_multiple(elts, 3, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_start_index(), 0);
    ASSERT_EQ(*get_element_from_queue(q, 0), elts[1]);
    ASSERT_EQ(*get_element_from_queue(q, 1), elts[2]);
}

TEST(MultipleEnqueueOddSize, QueueMultipleEnqueue)
{
    FrameDescriptor fd = {1, 1, sizeof(char), Endianness::BigEndian};
    constexpr uint queue_size = 3;
    holovibes::Queue q(fd, queue_size, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char elts[] = {'a', 'b', 'c', 'd'};

    q.enqueue_multiple(elts, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_start_index(), 0);
    ASSERT_EQ(q.get_size(), 2);

    q.enqueue_multiple(elts, 4, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_start_index(), 0);
    ASSERT_EQ(q.get_size(), queue_size);
    ASSERT_EQ(*get_element_from_queue(q, 0), elts[1]);
    ASSERT_EQ(*get_element_from_queue(q, 1), elts[2]);
    ASSERT_EQ(*get_element_from_queue(q, 2), elts[3]);
}

TEST(SimpleMultipleEnqueue, QueueMultipleEnqueue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    unsigned int nb_elts = 2;
    char* new_elt = new char[fd.get_frame_res() * nb_elts];

    bool res = q.enqueue_multiple(new_elt, nb_elts, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    delete[] new_elt;
}

TEST(CircularMultipleEnqueue, QueueMultipleEnqueue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    unsigned int nb_elts = 2;
    char* new_elt = new char[fd.get_frame_res() * nb_elts];

    bool res = q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 1);
    ASSERT_EQ(q.get_start_index(), 0);

    res = q.enqueue_multiple(new_elt, nb_elts, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 1);

    res = q.enqueue_multiple(new_elt, nb_elts, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 1);

    // Enqueue multiple of 1 element
    res = q.enqueue_multiple(new_elt, 1, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    delete[] new_elt;
}

TEST(OversizedMultipleEnqueue, QueueMultipleEnqueue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    unsigned int nb_elts = 3;
    char* new_elt = new char[fd.get_frame_res() * nb_elts];

    // Enqueue 3 elements at once but the maximum size of the queue is 2
    bool res = q.enqueue_multiple(new_elt, nb_elts, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 1);
}

TEST(FullMultipleEnqueue, QueueMultipleEnqueue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    unsigned int nb_elts = 2;
    char* new_elt = new char[fd.get_frame_res() * nb_elts];

    bool res = q.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    res = q.enqueue_multiple(new_elt, 1, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(res);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 1);
}

TEST(MultipleEnqueueNonSquare, QueueMultipleEnqueue)
{
    // 3 * 1 = 3 is the length of a string of two character + null character
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    // 2 frames of a resolution of 3
    char new_elt[] = "ab\0cd\0";

    q.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_start_index(), 0);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(std::string(get_element_from_queue(q, 0)), std::string(new_elt));
    ASSERT_EQ(std::string(get_element_from_queue(q, 1)), std::string(new_elt + fd.get_frame_size()));
}

TEST(EmptyDequeue, QueueDequeue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    // empty queue
    ASSERT_DEATH(q.dequeue(), "");
}

TEST(DequeueOneFrame, QueueDequeue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* new_elt = new char[fd.get_frame_res()];

    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 1);
    ASSERT_EQ(q.get_start_index(), 0);

    // dequeue
    q.dequeue();
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 1);

    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    q.dequeue();
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 0);

    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    q.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    ASSERT_TRUE(q.is_full());
    ASSERT_EQ(q.get_start_index(), 0);
    q.dequeue();
    ASSERT_EQ(q.get_start_index(), 1);
    q.dequeue();
    ASSERT_EQ(q.get_start_index(), 0);

    delete[] new_elt;
}

TEST(DequeueMultipleFrames, QueueDequeue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* new_elt = new char[fd.get_frame_res() * 2];

    q.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    q.dequeue(2);
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 2);

    q.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 2);

    q.dequeue(2);
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 1);
}

TEST(DequeueTooManyFrames, QueueDequeue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* new_elt = new char[fd.get_frame_res() * 2];

    q.enqueue_multiple(new_elt, q.get_max_size() - 1, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q.get_size(), 2);
    ASSERT_EQ(q.get_start_index(), 0);

    // Dequeue too many elements
    ASSERT_DEATH(q.dequeue(q.get_max_size()), "");
}

TEST(SimpleDequeueValueEmpty, QueueDequeueValue)
{
    FrameDescriptor fd = {64, 64, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* buff = new char[fd.get_frame_res()];

    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 0);
    ASSERT_DEATH(q.dequeue(buff, stream, cudaMemcpyDeviceToHost), "");

    delete[] buff;
}

TEST(SimpleDequeueValue, QueueDequeueValue)
{
    // 3 * 1 = 3 is the length of a string of two character + null character
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* res = new char[fd.get_frame_res()];

    char* buff[] = {strdup("ab\0"), strdup("cd\0"), strdup("ef\0")};
    q.enqueue(buff[0], stream, cudaMemcpyHostToDevice);

    // Make one enqueue and dequeue
    q.dequeue(res, stream, cudaMemcpyDeviceToHost);
    ASSERT_EQ(std::string(res), std::string(buff[0]));
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 1);
}

TEST(ComplexDequeueValue, QueueDequeueValue)
{
    // 3 * 1 = 3 is the length of a string of two character + null character
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char* res = new char[fd.get_frame_res()];

    char* buff[] = {strdup("ab\0"), strdup("cd\0"), strdup("ef\0")};

    // Change indexes
    q.enqueue(buff[0], stream, cudaMemcpyHostToDevice);
    q.dequeue(res, stream, cudaMemcpyDeviceToHost);

    // Make two enqueues followed by two dequeues
    q.enqueue(buff[1], stream, cudaMemcpyHostToDevice);
    q.enqueue(buff[2], stream, cudaMemcpyHostToDevice);

    q.dequeue(res, stream, cudaMemcpyDeviceToHost);
    ASSERT_EQ(std::string(res), std::string(buff[1]));
    ASSERT_EQ(q.get_size(), 1);
    ASSERT_EQ(q.get_start_index(), 0);

    q.dequeue(res, stream, cudaMemcpyDeviceToHost);
    ASSERT_EQ(std::string(res), std::string(buff[2]));
    ASSERT_EQ(q.get_size(), 0);
    ASSERT_EQ(q.get_start_index(), 1);

    delete[] res;
}

TEST(EmptyCopyMultiple, QueueCopyMultiple)
{
    FrameDescriptor fd = {4, 4, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 4, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    q_src.copy_multiple(q_dst, 1, stream);
}

TEST(SimpleCopyMultiple, QueueCopyMultiple)
{
    // 3 * 1 = 3 is the length of a string of two character + null character
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 4, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    // 2 frames of a resolution of 3
    char new_elt[] = "ab\0cd\0";

    q_src.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q_src.get_start_index(), 0);
    ASSERT_EQ(q_src.get_size(), 2);
    ASSERT_EQ(std::string(get_element_from_queue(q_src, 0)), std::string(new_elt));
    ASSERT_EQ(std::string(get_element_from_queue(q_src, 1)), std::string(new_elt + fd.get_frame_size()));

    q_src.copy_multiple(q_dst, 2, stream);
    ASSERT_EQ(q_dst.get_start_index(), 0);
    ASSERT_EQ(q_dst.get_size(), 2);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + fd.get_frame_size()));

    q_src.copy_multiple(q_dst, 2, stream);
    ASSERT_EQ(q_dst.get_start_index(), 0);
    ASSERT_EQ(q_dst.get_size(), 4);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 3)), std::string(new_elt + fd.get_frame_size()));
}

TEST(MoreElementCopyMultiple, QueueCopyMultiple)
{
    // 3 * 1 = 3 is the length of a string of two character + null character
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 2, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char new_elt[] = "ab\0cd\0ef\0gh\0ij\0";

    // Fill the source queue
    q_src.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    q_src.enqueue(new_elt + fd.get_frame_size(), stream, cudaMemcpyHostToDevice);

    // Fill the destination queue
    q_dst.enqueue_multiple(new_elt + 2 * fd.get_frame_size(), 3, stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q_dst.get_start_index(), 0);
    ASSERT_EQ(q_dst.get_size(), 3);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + 2 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + 3 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt + 4 * fd.get_frame_size()));

    // Copy more elements than the source queue size
    q_src.copy_multiple(q_dst, 3, stream);
    // Only two elments are supposed to be copied
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + fd.get_frame_size()));
    // So the last element remains unchanged
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt + 4 * fd.get_frame_size()));
}

TEST(DstOverflowCopyMultiple, DISABLED_QueueCopyMultiple)
{
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 4, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char new_elt[] = "ab\0cd\0ef\0gh\0";

    // Make the source queue full
    q_src.enqueue_multiple(new_elt, 4, stream, cudaMemcpyHostToDevice);

    // Copy all elements from q_src to q_dst
    // But the size of q_dst is lower than q_dst
    // This case needs to be correctly handled
    q_src.copy_multiple(q_dst, q_src.get_size(), stream);

    ASSERT_EQ(q_dst.get_size(), 3); // destination queue max size
    ASSERT_EQ(q_dst.get_start_index(), 0);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + 1 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + 2 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt + 3 * fd.get_frame_size()));
}

TEST(CircularSrcCopyMultiple, QueueCopyMultiple)
{
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char new_elt[] = "ab\0cd\0ef\0gh\0";

    // Create setup for the source queue
    q_src.enqueue(new_elt, stream, cudaMemcpyHostToDevice);
    q_src.dequeue();
    q_src.enqueue(new_elt + fd.get_frame_size(), stream, cudaMemcpyHostToDevice);
    q_src.dequeue();
    q_src.enqueue(new_elt + 2 * fd.get_frame_size(), stream, cudaMemcpyHostToDevice);
    q_src.enqueue(new_elt + 3 * fd.get_frame_size(), stream, cudaMemcpyHostToDevice);
    ASSERT_EQ(q_src.get_size(), 2);
    ASSERT_EQ(q_src.get_start_index(), 2);
    ASSERT_EQ(q_src.get_end_index(), 1);

    // copy the number of elements in the queue
    q_src.copy_multiple(q_dst, 2, stream);
    ASSERT_EQ(q_dst.get_start_index(), 0);
    ASSERT_EQ(q_dst.get_size(), 2);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + 2 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + 3 * fd.get_frame_size()));

    // copy more elements than the current size
    q_src.copy_multiple(q_dst, 3, stream);
    ASSERT_EQ(q_dst.get_start_index(), 1);
    ASSERT_EQ(q_dst.get_size(), 3);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + 3 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + 3 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt + 2 * fd.get_frame_size()));
}

TEST(CircularDstCopyMultiple, QueueCopyMultiple)
{
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char new_elt[] = "ab\0cd\0ef\0gh\0";

    // Make the source queue full
    q_src.enqueue_multiple(new_elt, 3, stream, cudaMemcpyHostToDevice);

    // Change index of the destination queue
    q_dst.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    q_dst.dequeue();
    q_dst.dequeue();
    ASSERT_EQ(q_dst.get_size(), 0);
    ASSERT_EQ(q_dst.get_start_index(), 2);

    q_src.copy_multiple(q_dst, 2, stream);
    ASSERT_EQ(q_dst.get_size(), 2);
    ASSERT_EQ(q_dst.get_start_index(), 2);
    ASSERT_EQ(q_dst.get_end_index(), 1);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + fd.get_frame_size()));
    // at position 1, there is not any element
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt));
}

TEST(CircularDstSrcCopyMultiple, QueueCopyMultiple)
{
    FrameDescriptor fd = {3, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 4, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    char new_elt[] = "ab\0cd\0ef\0gh\0ij\0";

    // Change index of the source queue
    q_src.enqueue_multiple(new_elt, 2, stream, cudaMemcpyHostToDevice);
    q_src.dequeue();
    q_src.dequeue();
    q_src.enqueue_multiple(new_elt, 3, stream, cudaMemcpyHostToDevice);

    // Change index of the destination queue
    q_dst.enqueue_multiple(new_elt + 3 * fd.get_frame_size(), 2, stream, cudaMemcpyHostToDevice);
    q_dst.dequeue();
    q_dst.dequeue();
    q_dst.enqueue_multiple(new_elt + 3 * fd.get_frame_size(), 2, stream, cudaMemcpyHostToDevice);

    // Copy all elements
    q_src.copy_multiple(q_dst, q_src.get_size(), stream);
    ASSERT_EQ(q_dst.get_start_index(), 1);
    ASSERT_EQ(q_dst.get_size(), 3);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt + fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + 2 * fd.get_frame_size()));
}

TEST(ManyDstOverflow, DISABLED_QueueCopyMultiple)
{
    FrameDescriptor fd = {2, 1, sizeof(char), Endianness::BigEndian};
    holovibes::Queue q_src(fd, 11, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);
    holovibes::Queue q_dst(fd, 3, holovibes::QueueType::UNDEFINED, fd.width, fd.height, fd.depth);

    // 11 + 3 = 14 characters
    char* new_elt = strdup("a\0b\0c\0d\0e\0f\0g\0h\0i\0j\0k\0l\0m\0n\0");

    // Make the queue full
    q_src.enqueue_multiple(new_elt, 11, stream, cudaMemcpyHostToDevice);

    // Change indices
    q_dst.enqueue_multiple(new_elt + 11 * fd.get_frame_size(), 2, stream, cudaMemcpyHostToDevice);
    q_dst.dequeue();
    q_dst.dequeue();
    ASSERT_EQ(q_dst.get_start_index(), 2);
    ASSERT_EQ(q_dst.get_size(), 0);
    ASSERT_TRUE(q_dst.get_start_index() == q_dst.get_end_index());

    // Copy multiple (10 elements).
    // The size of the destination queue is only 3
    q_src.copy_multiple(q_dst, 10, stream);

    // Expected. Not handle but should be fixed
    ASSERT_EQ(q_dst.get_start_index(), 2);
    ASSERT_EQ(q_dst.get_size(), 3);
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 0)), std::string(new_elt + 8 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 1)), std::string(new_elt + 9 * fd.get_frame_size()));
    ASSERT_EQ(std::string(get_element_from_queue(q_dst, 2)), std::string(new_elt + 7 * fd.get_frame_size()));

    // Source should be equal to:
    // | a (start index) | b | c | d | e | f | g | h | i | j | k |
    // Destination should be equal to:
    // | i | j | h (start index) |
}
