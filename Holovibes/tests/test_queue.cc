#include "gtest/gtest.h"

#include <cuda.h>

#include "queue.hh"
#include "frame_desc.hh"

TEST(QueueTest, SimpleInstantiatingTest)
{
    camera::FrameDescriptor fd = { 64, 64, 1, camera::Endianness::BigEndian };
    holovibes::Queue q(fd, 5, "TestQueue", 64, 64, 1);
    // WARNING: Set false because the queue is used in a CLI mode
    q.set_display(false);
    ASSERT_EQ(0.0, 0.0);
}

TEST(QueueEmpty, QueueIsFullTest)
{
     camera::FrameDescriptor fd = { 64, 64, sizeof(char), camera::Endianness::BigEndian };
    holovibes::Queue q(fd, 5, "TestQueue", fd.width, fd.height, fd.depth);
    q.set_display(false);
    ASSERT_FALSE(q.is_full());
}

TEST(QueueNotFull, QueueIsFullTest)
{
    camera::FrameDescriptor fd = { 64, 64, sizeof(char), camera::Endianness::BigEndian };
    holovibes::Queue q(fd, 2, "QueueIsFullTest", fd.width, fd.height, fd.depth);
    q.set_display(false);
    // Enqueue
    char* new_elt = new char[fd.width * fd.height];
    q.enqueue(new_elt, cudaMemcpyHostToDevice);
    ASSERT_FALSE(q.is_full());
    delete new_elt;
}

TEST(QueueFull, QueueIsFull)
{
    camera::FrameDescriptor fd = { 64, 64, sizeof(char), camera::Endianness::BigEndian };
    holovibes::Queue q(fd, 2, "QueueIsFullTest", fd.width, fd.height, fd.depth);
    q.set_display(false);
    // Enqueue
    char* new_elt = new char[fd.width * fd.height];
    q.enqueue(new_elt, cudaMemcpyHostToDevice);
    q.enqueue(new_elt, cudaMemcpyHostToDevice);
    ASSERT_TRUE(q.is_full());
    delete new_elt;
}

TEST(SimpleQueueResize, QueueResize)
{
    camera::FrameDescriptor fd = { 64, 64, sizeof(char), camera::Endianness::BigEndian };
    holovibes::Queue q(fd, 2, "QueueIsFullTest", fd.width, fd.height, fd.depth);
    q.set_display(false);
    ASSERT_EQ(q.get_current_elts(), 0);
    ASSERT_EQ(q.get_max_elts(), 2);

    unsigned int new_size = 10;
    q.resize(new_size); // Resize here, empty the queue
    ASSERT_EQ(q.get_current_elts(), 0);
    ASSERT_EQ(q.get_max_elts(), new_size);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}