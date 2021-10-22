#include "gtest/gtest.h"

#include "micro_cache.hh"
#include "global_state_holder.hh"

namespace holovibes
{

struct TestCache1 : public MicroCache
{
    MONITORED_MEMBER(int, a)
    MONITORED_MEMBER(float, b)
    MONITORED_MEMBER(long long, c)

    void synchronize() override { MicroCache::synchronize(a, b, c); }

    friend struct TestMicroCache1;
    friend struct TestMicroCache3;
};

struct TestCache2 : public MicroCache
{
    using b_wrapper = std::vector<std::pair<float, double>>;
    using c_wrapper = std::map<std::string, std::string>;
    MONITORED_MEMBER(std::string, a)
    MONITORED_MEMBER(b_wrapper, b)
    MONITORED_MEMBER(c_wrapper, c)

    void synchronize() override { MicroCache::synchronize(a, b, c); }

    friend struct TestMicroCache2;
    friend struct TestMicroCache3;
};

struct TestMicroCache1
{
    TestCache1 x;
    TestCache1 y;

    TestMicroCache1()
    {
        y.a.obj = 0;
        y.b.obj = 0.0;
        y.c.obj = 0;

        x.set_a(1);
        x.set_b(2.0);
        x.set_c(3);
    }
};

struct TestMicroCache2
{
    TestCache2 x;
    TestCache2 y;

    TestMicroCache2()
    {
        x.get_a_ref().append("a");
        x.get_b_ref().emplace_back(1.0, 2.0);
        x.get_c_ref().emplace("key", "value");

        x.trigger_a();
        x.trigger_b();
        x.trigger_c();
    }
};

struct TestMicroCache3
{
    TestCache1 x;
    TestCache2 y;

    TestMicroCache3()
    {
        x.set_a(1);
        x.set_b(2.0);
        x.set_c(3);
    }
};

TEST(TestMicroCache, basic_types_simple)
{
    TestMicroCache1 test;

    ASSERT_EQ(test.x.get_a(), 1);
    ASSERT_EQ(test.x.get_b(), 2.0);
    ASSERT_EQ(test.x.get_c(), 3);
}

TEST(TestMicroCache, basic_types_before_synchronize)
{
    TestMicroCache1 test;

    ASSERT_EQ(test.y.get_a(), 0);
    ASSERT_EQ(test.y.get_b(), 0.0);
    ASSERT_EQ(test.y.get_c(), 0);
}

TEST(TestMicroCache, basic_types_after_synchronize)
{
    TestMicroCache1 test;

    test.y.synchronize();
    ASSERT_EQ(test.y.get_a(), 1);
    ASSERT_EQ(test.y.get_b(), 2.0);
    ASSERT_EQ(test.y.get_c(), 3);
}

TEST(TestMicroCache, stl_types_simple)
{
    TestMicroCache2 test;

    ASSERT_EQ(test.x.get_a(), "a");
    ASSERT_EQ(test.x.get_b().size(), 1);
    ASSERT_EQ(test.x.get_b()[0].first, 1.0);
    ASSERT_EQ(test.x.get_b()[0].second, 2.0);
    ASSERT_EQ(test.x.get_c().size(), 1);
    ASSERT_EQ(test.x.get_c().at("key"), "value");
}

TEST(TestMicroCache, stl_types_before_synchronize)
{
    TestMicroCache2 test;

    ASSERT_EQ(test.y.get_a(), "");
    ASSERT_EQ(test.y.get_b().size(), 0);
    ASSERT_EQ(test.y.get_c().size(), 0);
}

TEST(TestMicroCache, stl_types_after_synchronize)
{
    TestMicroCache2 test;

    test.y.synchronize();
    ASSERT_EQ(test.y.get_a(), "a");
    ASSERT_EQ(test.y.get_b().size(), 1);
    ASSERT_EQ(test.y.get_b()[0].first, 1.0);
    ASSERT_EQ(test.y.get_b()[0].second, 2.0);
    ASSERT_EQ(test.y.get_c().size(), 1);
    ASSERT_EQ(test.y.get_c().at("key"), "value");
}

TEST(TestMicroCache, dont_sync_different_types)
{
    TestMicroCache3 test;

    test.y.synchronize();
    ASSERT_EQ(test.y.get_a(), "");
    ASSERT_EQ(test.y.get_b().size(), 0);
    ASSERT_EQ(test.y.get_c().size(), 0);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

} // namespace holovibes
