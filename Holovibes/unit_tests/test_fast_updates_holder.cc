#include "gtest/gtest.h"

#include "fast_updates_holder.hh"

TEST(FastUpdatesHolder, checkKeyTypeTrue) { ASSERT_TRUE(holovibes::is_fast_update_key_type<holovibes::ProgressType>); }

TEST(FastUpdatesHolder, checkKeyTypeFalse) { ASSERT_FALSE(holovibes::is_fast_update_key_type<int>); }

TEST(FastUpdatesHolder, testMapInsert)
{
    auto map = holovibes::FastUpdatesHolder<holovibes::ProgressType>();
    auto entry1 = map.create_entry(holovibes::ProgressType::FRAME_RECORD);
    entry1->first = 2;
    entry1->second = 2;

    auto entry2 = map.get_entry(holovibes::ProgressType::FRAME_RECORD);
    ASSERT_EQ(entry2->first, 2);
    ASSERT_EQ(entry2->second, 2);
}

TEST(FastUpdatesHolder, testMapRemove)
{
    auto map = holovibes::FastUpdatesHolder<holovibes::ProgressType>();
    map.create_entry(holovibes::ProgressType::FRAME_RECORD);

    ASSERT_TRUE(map.remove_entry(holovibes::ProgressType::FRAME_RECORD));
    ASSERT_THROW(map.get_entry(holovibes::ProgressType::FRAME_RECORD), std::out_of_range);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
