#include "gtest/gtest.h"
#include "test_disable_log.hh"

TEST(BasicTest, MoreThanSimpleTestExample)
{
    ASSERT_EQ(0.0, 0.0);
    ASSERT_NE(0, 1);
}
/*
int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}*/
