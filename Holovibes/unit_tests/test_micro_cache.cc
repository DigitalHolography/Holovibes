#include "gtest/gtest.h"

#include "micro_cache.hh"
#include "global_state_holder.hh"

namespace holovibes
{

struct TestCache : public MicroCache
{
    MONITORED_MEMBER(int, a)
    MONITORED_MEMBER(float, b)
    MONITORED_MEMBER(long long, c)

    void synchronize() override { MicroCache::synchronize(a, b, c); }

    friend struct TestMicroCache;
};

struct TestMicroCache
{
    TestCache x;
    TestCache y;

    TestMicroCache()
    {
        y.a.obj = 0;
        y.b.obj = 0.0;
        y.c.obj = 0;

        x.set_a(1);
        x.set_b(2.0);
        x.set_c(3);
    }
};

TEST(TestMicroCache, simple)
{
    TestMicroCache test;

    ASSERT_EQ(test.x.get_a(), 1);
    ASSERT_EQ(test.x.get_b(), 2.0);
    ASSERT_EQ(test.x.get_c(), 3);
}

TEST(TestMicroCache, before_synchronize)
{
    TestMicroCache test;

    ASSERT_EQ(test.y.get_a(), 0);
    ASSERT_EQ(test.y.get_b(), 0.0);
    ASSERT_EQ(test.y.get_c(), 0);
}

TEST(TestMicroCache, after_synchronize)
{
    TestMicroCache test;

    test.y.synchronize();
    ASSERT_EQ(test.y.get_a(), 1);
    ASSERT_EQ(test.y.get_b(), 2.0);
    ASSERT_EQ(test.y.get_c(), 3);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

} // namespace holovibes
