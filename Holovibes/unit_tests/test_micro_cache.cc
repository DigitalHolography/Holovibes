#include "gtest/gtest.h"

#define MICRO_CACHE_DEBUG

#include "micro_cache.hh"

#undef MICRO_CACHE_DEBUG

#include "global_state_holder.hh"

namespace holovibes
{

NEW_MICRO_CACHE(TestCache1,
    (int, a),
    (float, b),
    (long long, c)
)

using b_wrapper = std::vector<std::pair<float, double>>;
using c_wrapper = std::map<std::string, std::string>;

NEW_MICRO_CACHE(TestCache2,
    (std::string, a),
    (b_wrapper, b),
    (c_wrapper, c)
)

struct TestMicroCache1
{
    TestCache1 x = true;
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
    TestCache2 x = true;
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
    TestCache1 x = true;
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

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

} // namespace holovibes
