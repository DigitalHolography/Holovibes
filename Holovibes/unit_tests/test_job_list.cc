#include "gtest/gtest.h"
#include "jobs/vector_job.hh"
#include "jobs/buffer_output_job.hh"
#include "jobs/copy_job.hh"
#include "unique_ptr.hh"

#include <cstring>

using namespace holovibes;

static constexpr cudaStream_t stream = 0;

static int myCudaMemcmp(void* lhs, void* rhs, size_t bytes)
{
    std::vector<std::byte> left;
    std::vector<std::byte> right;
    left.reserve(bytes);
    right.reserve(bytes);

    cudaXMemcpy(left.data(), lhs, bytes, cudaMemcpyDeviceToHost);
    cudaXMemcpy(right.data(), rhs, bytes, cudaMemcpyDeviceToHost);

    return std::memcmp(left.data(), right.data(), bytes);
}

TEST(TestJobList, basic)
{
    size_t nb_frames = 0;
    Job::BufferDescriptor input_desc{1, 1, 1, 1};
    Job::BufferDescriptor output_desc{1, 1, 1, 1};
    cuda_tools::UniquePtr<std::byte> input{input_desc.get_buffer_size()};
    cuda_tools::UniquePtr<std::byte> output{output_desc.get_buffer_size()};

    VectorJob jobs;
    auto copy_job = jobs.emplace_back<CopyJob>();
    auto output_job = jobs.emplace_back<BufferOutputJob>(output.get(), nb_frames, 1);

    jobs.prepare(input_desc);
    jobs.run(Job::RunEnv{input.get(), input_desc, stream});
    cudaXStreamSynchronize(stream);
    ASSERT_EQ(jobs.size(), 2);
    EXPECT_NE(copy_job->get_output_buffer(), nullptr);
    EXPECT_EQ(nb_frames, 1);
    EXPECT_EQ(myCudaMemcmp(input.get(), output.get(), input_desc.get_buffer_size()), 0);
}

// TEST(TestJobList, inplace)
// {
//     Job::BufferDescriptor input_desc{1, 1, 1, 1};
//     Job::BufferDescriptor output_desc{1, 1, 1, 1};
//     cuda_tools::UniquePtr<std::byte> input{input_desc.get_buffer_size()};
//     cuda_tools::UniquePtr<std::byte> output{output_desc.get_buffer_size()};

//     auto runner = std::make_shared<JobList>(std::vector<shared_job>{
//         std::make_shared<CopyJob>(true),
//     });

//     runner->run(Job::RunEnv{input.get(), input_desc, output.get(), output_desc, stream});
//     cudaXStreamSynchronize(stream);
//     EXPECT_EQ(runner->get_gpu_memory().size(), 0);
//     EXPECT_EQ(myCudaMemcmp(input.get(), output.get(), input_desc.get_buffer_size()), 0);
// }

// TEST(TestJobList, multiple)
// {
//     Job::BufferDescriptor input_desc{1, 1, 1, 1};
//     Job::BufferDescriptor output_desc{1, 1, 1, 1};
//     cuda_tools::UniquePtr<std::byte> input{input_desc.get_buffer_size()};
//     cuda_tools::UniquePtr<std::byte> output{output_desc.get_buffer_size()};

//     auto runner = std::make_shared<JobList>(std::vector<shared_job>{
//         std::make_shared<CopyJob>(),
//         std::make_shared<CopyJob>(),
//         std::make_shared<CopyJob>(),
//     });

//     runner->run(Job::RunEnv{input.get(), input_desc, output.get(), output_desc, stream});
//     cudaXStreamSynchronize(stream);
//     EXPECT_EQ(runner->get_gpu_memory().size(), 2);
//     EXPECT_EQ(myCudaMemcmp(input.get(), output.get(), input_desc.get_buffer_size()), 0);
// }

// TEST(TestJobList, one_inplace)
// {
//     Job::BufferDescriptor input_desc{1, 1, 1, 1};
//     Job::BufferDescriptor output_desc{1, 1, 1, 1};
//     cuda_tools::UniquePtr<std::byte> input{input_desc.get_buffer_size()};
//     cuda_tools::UniquePtr<std::byte> output{output_desc.get_buffer_size()};

//     auto runner = std::make_shared<JobList>(std::vector<shared_job>{
//         std::make_shared<CopyJob>(),
//         std::make_shared<CopyJob>(true),
//         std::make_shared<CopyJob>(),
//     });

//     runner->run(Job::RunEnv{input.get(), input_desc, output.get(), output_desc, stream});
//     cudaXStreamSynchronize(stream);
//     EXPECT_EQ(runner->get_gpu_memory().size(), 1);
//     EXPECT_EQ(myCudaMemcmp(input.get(), output.get(), input_desc.get_buffer_size()), 0);
// }

// TEST(TestJobList, jobListCeption)
// {
//     Job::BufferDescriptor input_desc{1, 1, 1, 1};
//     Job::BufferDescriptor output_desc{1, 1, 1, 1};
//     cuda_tools::UniquePtr<std::byte> input{input_desc.get_buffer_size()};
//     cuda_tools::UniquePtr<std::byte> output{output_desc.get_buffer_size()};

//     auto runner = std::make_shared<JobList>(std::vector<shared_job>{
//         std::make_shared<CopyJob>(),
//         std::make_shared<JobList>(std::vector<shared_job>{std::make_shared<CopyJob>(), std::make_shared<CopyJob>()}),
//     });

//     runner->run(Job::RunEnv{input.get(), input_desc, output.get(), output_desc, stream});
//     cudaXStreamSynchronize(stream);
//     EXPECT_EQ(runner->get_gpu_memory().size(), 2);
//     EXPECT_EQ(myCudaMemcmp(input.get(), output.get(), input_desc.get_buffer_size()), 0);
// }