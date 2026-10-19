#include "gtest/gtest.h"

#include "buffer_memory.hh"

#include <limits>

namespace memory = holovibes::buffer_memory;

TEST(BufferMemoryTest, SplitsQueueStorageAcrossCpuAndGpu)
{
    memory::Request request;
    request.width = 100;
    request.height = 50;
    request.input_bytes_per_pixel = 2;
    request.output_bytes_per_frame = 10000;
    request.record_bytes_per_frame = 30000;
    request.file_frames = 4;
    request.input_frames = 10;
    request.batch_size = 4;
    request.record_frames = 5;
    request.output_frames = 3;
    request.file_enabled = true;

    const auto estimate = memory::estimate(request);

    EXPECT_EQ(estimate.file.cpu, 40000);
    EXPECT_EQ(estimate.file.gpu, 50000);
    EXPECT_EQ(estimate.input.gpu, 80000); // Two complete batches, not ten frames.
    EXPECT_EQ(estimate.input.cpu, 8 * 4 * sizeof(uint64_t));
    EXPECT_EQ(estimate.record.cpu, 150000);
    EXPECT_EQ(estimate.record.gpu, 0);
    EXPECT_EQ(estimate.output.gpu, 30000);
    EXPECT_EQ(estimate.total.cpu, 190256);
    EXPECT_EQ(estimate.total.gpu, 160000);
    EXPECT_FALSE(estimate.overflow);
}

TEST(BufferMemoryTest, CountsAllRecordBuffersOnSelectedDevice)
{
    memory::Request request;
    request.record_bytes_per_frame = 4096;
    request.record_frames = 100;
    request.record_buffer_count = 3;

    auto estimate = memory::estimate(request);
    EXPECT_EQ(estimate.record.cpu, 1228800);
    EXPECT_EQ(estimate.record.gpu, 0);

    request.record_on_gpu = true;
    estimate = memory::estimate(request);
    EXPECT_EQ(estimate.record.cpu, 0);
    EXPECT_EQ(estimate.record.gpu, 1228800);
    EXPECT_EQ(estimate.total.gpu, estimate.record.gpu);
}

TEST(BufferMemoryTest, IncludesBothCutsAndOptionalDisplayQueues)
{
    memory::Request request;
    request.width = 100;
    request.height = 50;
    request.input_bytes_per_pixel = 1;
    request.output_bytes_per_frame = 30000; // RGB output differs from raw input.
    request.output_frames = 2;
    request.raw_view_enabled = true;
    request.filter2d_view_enabled = true;
    request.cuts_frames = 4;
    request.time_transformation_size = 10;
    request.cuts_enabled = true;

    const auto estimate = memory::estimate(request);

    EXPECT_EQ(estimate.output.gpu, (30000 + 5000 + 30000) * 2);
    EXPECT_EQ(estimate.cuts.gpu, (100 * 10 + 50 * 10) * 2 * 4);
    EXPECT_EQ(estimate.total.gpu, 142000);
    EXPECT_EQ(estimate.total.cpu, 0);
}

TEST(BufferMemoryTest, DisabledFileAndCutsDoNotConsumeMemory)
{
    memory::Request request;
    request.width = 1920;
    request.height = 1080;
    request.input_bytes_per_pixel = 2;
    request.file_frames = 10000;
    request.cuts_frames = 5000;
    request.time_transformation_size = 1024;

    const auto estimate = memory::estimate(request);

    EXPECT_EQ(estimate.total.cpu, 0);
    EXPECT_EQ(estimate.total.gpu, 0);
    EXPECT_FALSE(estimate.overflow);
}

TEST(BufferMemoryTest, MissingBatchSizeDoesNotDivideByZero)
{
    memory::Request request;
    request.width = 10;
    request.height = 10;
    request.input_bytes_per_pixel = 2;
    request.input_frames = 3;
    request.batch_size = 0;

    const auto estimate = memory::estimate(request);

    EXPECT_EQ(estimate.input.gpu, 600);
    EXPECT_EQ(estimate.input.cpu, 96);
}

TEST(BufferMemoryTest, MultiplicationSaturatesInsteadOfWrappingToSmallAllocation)
{
    memory::Request request;
    request.record_bytes_per_frame = (std::numeric_limits<uint64_t>::max)();
    request.record_frames = 2;
    request.record_buffer_count = 3;

    const auto estimate = memory::estimate(request);

    EXPECT_TRUE(estimate.overflow);
    EXPECT_EQ(estimate.record.cpu, (std::numeric_limits<uint64_t>::max)());
    EXPECT_EQ(estimate.total.cpu, (std::numeric_limits<uint64_t>::max)());
}

TEST(BufferMemoryTest, TotalAdditionSaturatesInsteadOfWrapping)
{
    memory::Request request;
    request.output_bytes_per_frame = (std::numeric_limits<uint64_t>::max)() - 10;
    request.output_frames = 1;
    request.record_bytes_per_frame = 20;
    request.record_frames = 1;
    request.record_on_gpu = true;

    const auto estimate = memory::estimate(request);

    EXPECT_TRUE(estimate.overflow);
    EXPECT_EQ(estimate.total.gpu, (std::numeric_limits<uint64_t>::max)());
}

TEST(BufferMemoryTest, IncreasingBuffersMustLeaveReserve)
{
    EXPECT_TRUE(memory::budget_allows(300, 200, 150, 50));
    EXPECT_FALSE(memory::budget_allows(301, 200, 150, 50));
    EXPECT_FALSE(memory::budget_allows(201, 200, 49, 50));
    EXPECT_EQ(memory::required_increase(300, 200), 100);
}

TEST(BufferMemoryTest, ExistingAllocationsAreNotCountedTwice)
{
    EXPECT_TRUE(memory::budget_allows(200, 200, 0, 50));
    EXPECT_TRUE(memory::budget_allows(100, 200, 0, 50));
    EXPECT_EQ(memory::required_increase(100, 200), 0);
}

TEST(BufferMemoryTest, MovingRecordBuffersMustFitDestinationMemory)
{
    const memory::Usage current{1000, 0};
    const memory::Usage proposed{0, 1000};
    const memory::Usage reserve{100, 100};
    const auto increase = memory::required_increase(proposed, current);

    EXPECT_EQ(increase.cpu, 0);
    EXPECT_EQ(increase.gpu, 1000);
    EXPECT_FALSE(memory::budget_allows(proposed, current, {100000, 1000}, reserve));
    EXPECT_TRUE(memory::budget_allows(proposed, current, {0, 1100}, reserve));
}
