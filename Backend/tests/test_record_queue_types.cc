#include "gtest/gtest.h"

#include "fast_updates_holder.hh"
#include "test_disable_log.hh"

namespace holovibes
{
static_assert(record_queue_type(0) == QueueType::RECORD_QUEUE);
static_assert(record_queue_type(1) == QueueType::RECORD_QUEUE_2);
static_assert(record_queue_type(2) == QueueType::RECORD_QUEUE_3);
static_assert(record_queue_index(record_queue_type(7)) == 7);

TEST(RecordQueueTypes, DistinguishesRecordQueuesBeyondTheOriginalThree)
{
    EXPECT_FALSE(is_record_queue(QueueType::UNDEFINED));
    EXPECT_FALSE(is_record_queue(QueueType::INPUT_QUEUE));
    EXPECT_FALSE(is_record_queue(QueueType::OUTPUT_QUEUE));

    for (std::size_t index = 0; index < 8; ++index)
    {
        const auto type = record_queue_type(index);
        EXPECT_TRUE(is_record_queue(type));
        EXPECT_EQ(record_queue_index(type), index);
        if (index > 0)
            EXPECT_NE(type, record_queue_type(index - 1));
    }
}

TEST(RecordQueueTypes, TracksAndRemovesEachRecordBufferIndependently)
{
    FastUpdatesHolder<QueueType> telemetry;
    for (std::size_t index = 0; index < 8; ++index)
    {
        auto entry = telemetry.create_entry(record_queue_type(index));
        std::get<0>(*entry) = static_cast<uint>(index);
        std::get<1>(*entry) = 16;
        std::get<2>(*entry) = Device::CPU;
    }

    EXPECT_TRUE(telemetry.remove_entry(record_queue_type(4)));
    EXPECT_FALSE(telemetry.contains(record_queue_type(4)));

    for (std::size_t index = 0; index < 8; ++index)
    {
        if (index == 4)
            continue;
        const auto entry = telemetry.get_entry(record_queue_type(index));
        EXPECT_EQ(std::get<0>(*entry).load(), index);
        EXPECT_EQ(std::get<1>(*entry).load(), 16);
        EXPECT_EQ(std::get<2>(*entry).load(), Device::CPU);
    }
}
} // namespace holovibes
