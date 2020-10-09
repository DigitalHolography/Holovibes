#include "gtest/gtest.h"

TEST(BasicTest, MoreThanSimpleTestExample)
{
    ASSERT_EQ(0.0, 0.0);
    ASSERT_NE(0, 1);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}