#include "gtest/gtest.h"

#include "queue.hh"
#include "frame_desc.hh"

TEST(QueueTest, SimpleInstantiatingTest)
{
    camera::FrameDescriptor fd = { 64, 64, 1, camera::Endianness::BigEndian };
    holovibes::Queue q(fd, 5, "TestQueue", 64, 64, 1);
    q.set_display(false);
    ASSERT_EQ(0.0, 0.0);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}