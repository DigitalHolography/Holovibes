#include "gtest/gtest.h"

#define MICRO_CACHE_DEBUG

#include "micro_cache.hh"

#undef MICRO_CACHE_DEBUG

#include "global_state_holder.hh"

#include <chrono>

namespace holovibes
{

NEW_MICRO_CACHE(TestCache1, (unsigned, a), (float, b), (long, c))
// needed when typing contains commas (which are supposed to divide args of the macro)
using b_wrapper = std::vector<std::pair<float, double>>;
using c_wrapper = std::map<std::string, std::string>;

NEW_MICRO_CACHE(TestCache2, (std::string, a), (b_wrapper, b), (c_wrapper, c))

// Trying lots of macro recursion
NEW_MICRO_CACHE(TestCache3,
                (int, a2),
                (int, b2),
                (int, c2),
                (int, d),
                (int, e),
                (int, f),
                (int, g),
                (int, h),
                (int, i),
                (int, j),
                (int, k),
                (int, l),
                (int, m),
                (int, a),
                (int, n),
                (int, o),
                (int, p),
                (int, q),
                (int, r),
                (int, s),
                (int, t),
                (int, u),
                (int, v),
                (int, w),
                (int, x),
                (int, y),
                (int, z))

NEW_MICRO_CACHE(TestCache4, (uint8_t, a), (uint16_t, b), (uint32_t, c), (uint64_t, d), (float, e), (double, f))

struct TestMicroCache1
{
    TestCache1 x{true};
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
    TestCache2 x{true};
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
TEST(TestMicroCache, register_truth_works) { TestCache1 x{true}; }

TEST(TestMicroCache, assert_not_truth_found)
{
    ASSERT_DEATH({ TestCache1 x; }, "You must register a truth cache for class: TestCache1");
}

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

TEST(TestMicroCache, basic_types_sync_constructor)
{
    TestCache1 x{true};

    x.set_a(1);
    x.set_b(2.0);
    x.set_c(3);

    TestCache1 y;

    ASSERT_EQ(y.get_a(), 1);
    ASSERT_EQ(y.get_b(), 2.0);
    ASSERT_EQ(y.get_c(), 3);
}

TEST(TestMicroCache, stl_sync_constructor)
{
    TestCache2 x{true};

    x.get_a_ref().append("a");
    x.get_b_ref().emplace_back(1.0, 2.0);
    x.get_c_ref().emplace("key", "value");

    x.trigger_a();
    x.trigger_b();
    x.trigger_c();

    TestCache2 y;

    ASSERT_EQ(y.get_a(), "a");
    ASSERT_EQ(y.get_b().size(), 1);
    ASSERT_EQ(y.get_b()[0].first, 1.0);
    ASSERT_EQ(y.get_b()[0].second, 2.0);
    ASSERT_EQ(y.get_c().size(), 1);
    ASSERT_EQ(y.get_c().at("key"), "value");
}

void write_thread(TestCache4& x, bool& stop)
{
    unsigned long long count = 0;
    std::ofstream fout("write.txt");
    while (!stop)
    {
        uint8_t a = x.get_a();
        x.set_a((uint8_t)(a * 3) % 3 == 0 ? a * 3 : 3);
        // std::cerr << "Write: " << (int)x.get_a() << std::endl;
        uint16_t b = x.get_b();
        x.set_b((uint16_t)(b * 3) % 3 == 0 ? b * 3 : 3);
        // std::cerr << "Write: " << (int)x.get_b() << std::endl;
        uint32_t c = x.get_c();
        x.set_c((uint32_t)(c * 3) % 3 == 0 ? c * 3 : 3);
        // std::cerr << "Write: " << (int)x.get_c() << std::endl;
        uint64_t d = x.get_d();
        x.set_d((uint64_t)(d * 3) % 3 == 0 ? d * 3 : 3);
        // std::cerr << "Write: " << (int)x.get_d() << std::endl;
        /*float e = x.get_e();
        x.set_e((unsigned long long)(e * 3.f) % 3 == 0 ? e * 3.f : 3.f);
        // std::cerr << "Write: " << (int)x.get_e() << std::endl;
        double f = x.get_f();
        x.set_f((unsigned long long)(f * 3.f) % 3 == 0 ? f * 3.f : 3.f);
        // std::cerr << "Write: " << (int)x.get_f() << std::endl;*/
        count++;
    }

    std::cerr << "Values written: " << count << std::endl;
}

void read_thread(TestCache4& y, bool& stop)
{
    unsigned long long count = 0;
    while (!stop)
    {
        y.synchronize();
        uint8_t a = y.get_a();
        // std::cerr << "Read a: " << (int)a << std::endl;
        ASSERT_EQ(a % 3, 0) << "Value a is " << (int)a;
        uint16_t b = y.get_b();
        // std::cerr << "Read b: " << (int)b << std::endl;
        ASSERT_EQ(b % 3, 0) << "Value b is " << (int)b;
        uint32_t c = y.get_c();
        // std::cerr << "Read c: " << (int)c << std::endl;
        ASSERT_EQ(c % 3, 0) << "Value c is " << (int)c;
        uint64_t d = y.get_d();
        // std::cerr << "Read d: " << (int)d << std::endl;
        ASSERT_EQ(d % 3, 0) << "Value d is " << (int)d;
        /*float e = y.get_e();
        // std::cerr << "Read e: " << (int)e << std::endl;
        ASSERT_EQ(e % 3, 0) << "Value e is " << (int)e;
        double f = y.get_f();
        // std::cerr << "Read f: " << (int)f << std::endl;
        ASSERT_EQ(f % 3, 0) << "Value f is " << (int)f;*/
        count++;
    }

    std::cerr << "Values read: " << count << std::endl;
}

TEST(TestMicroCacheConcurrency, concurrency_1)
{
    TestCache4 x{true};
    TestCache4 y;

    bool stop = false;

    x.set_a(3);
    x.set_b(3);
    x.set_c(3);
    x.set_d(3);
    x.set_e(3);
    x.set_f(3);

    y.synchronize();

    auto write_thr = std::thread::thread(write_thread, std::ref(x), std::ref(stop));

    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    for (unsigned i = 0; i < 1 << 18; i++)
    {
        y.synchronize();
        uint8_t a = y.get_a();
        // std::cerr << "Read a: " << (int)a << std::endl;
        ASSERT_EQ(a % 3, 0) << "Value a is " << (int)a;
        uint16_t b = y.get_b();
        // std::cerr << "Read b: " << (int)b << std::endl;
        ASSERT_EQ(b % 3, 0) << "Value b is " << (int)b;
        uint32_t c = y.get_c();
        // std::cerr << "Read c: " << (int)c << std::endl;
        ASSERT_EQ(c % 3, 0) << "Value c is " << (int)c;
        uint64_t d = y.get_d();
        // std::cerr << "Read d: " << (int)d << std::endl;
        ASSERT_EQ(d % 3, 0) << "Value d is " << (int)d;
        /*float e = y.get_e();
        // std::cerr << "Read e: " << (int)e << std::endl;
        ASSERT_EQ(e % 3, 0) << "Value e is " << (int)e;
        double f = y.get_f();
        // std::cerr << "Read f: " << (int)f << std::endl;
        ASSERT_EQ(f % 3, 0) << "Value f is " << (int)f;*/
    }

    stop = true;
    write_thr.join();
}

TEST(TestMicroCacheConcurrency, concurrency_2)
{
    TestCache4 x{true};
    TestCache4 y;

    bool stop = false;

    x.set_a(3);
    x.set_b(3);
    x.set_c(3);
    x.set_d(3);
    x.set_e(3);
    x.set_f(3);

    y.synchronize();

    auto write_thr = std::thread::thread(write_thread, std::ref(x), std::ref(stop));
    auto read1_thr = std::thread::thread(read_thread, std::ref(y), std::ref(stop));

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(5s);

    stop = true;
    write_thr.join();
    read1_thr.join();
}

TEST(TestMicroCacheConcurrency, concurrency_3)
{
    TestCache4 x{true};
    TestCache4 y;

    bool stop = false;

    x.set_a(3);
    x.set_b(3);
    x.set_c(3);
    x.set_d(3);
    x.set_e(3);
    x.set_f(3);

    y.synchronize();

    auto write_thr = std::thread::thread(write_thread, std::ref(x), std::ref(stop));
    auto read1_thr = std::thread::thread(read_thread, std::ref(y), std::ref(stop));
    auto read2_thr = std::thread::thread(read_thread, std::ref(y), std::ref(stop));
    auto read3_thr = std::thread::thread(read_thread, std::ref(y), std::ref(stop));
    auto read4_thr = std::thread::thread(read_thread, std::ref(y), std::ref(stop));

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(5s);

    stop = true;
    write_thr.join();
    read1_thr.join();
    read2_thr.join();
    read3_thr.join();
    read4_thr.join();
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

} // namespace holovibes
