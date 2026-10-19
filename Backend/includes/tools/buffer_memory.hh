#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace holovibes::buffer_memory
{
/*! \brief Pixel storage and per-frame metadata, in bytes, split by memory device. */
struct Usage
{
    uint64_t cpu = 0;
    uint64_t gpu = 0;
};

/*! \brief Inputs for the configurable queues; no CUDA or application state is required.
 *
 * The caller supplies frame sizes after applying the selected display/record mode. For an in-memory file,
 * file_frames must be the number of selected file frames rather than the regular file-buffer setting.
 * Inactive queues can be excluded by passing zero frames.
 */
struct Request
{
    uint64_t width = 0;
    uint64_t height = 0;
    uint64_t input_bytes_per_pixel = 0;
    uint64_t output_bytes_per_frame = 0;
    uint64_t record_bytes_per_frame = 0;

    uint64_t file_frames = 0;
    uint64_t input_frames = 0;
    uint64_t record_frames = 0;
    uint64_t output_frames = 0;
    uint64_t cuts_frames = 0;
    uint64_t time_transformation_size = 0;
    uint64_t batch_size = 1;
    uint64_t record_buffer_count = 1;

    bool record_on_gpu = false;
    bool file_enabled = false;
    bool cuts_enabled = false;
    bool raw_view_enabled = false;
    bool filter2d_view_enabled = false;
};

struct Estimate
{
    Usage file;
    Usage input;
    Usage record;
    Usage output;
    Usage cuts;
    Usage total;
    bool overflow = false;
};

namespace detail
{
inline uint64_t add(uint64_t lhs, uint64_t rhs, bool& overflow)
{
    constexpr auto maximum = (std::numeric_limits<uint64_t>::max)();
    if (rhs > maximum - lhs)
    {
        overflow = true;
        return maximum;
    }
    return lhs + rhs;
}

inline uint64_t multiply(uint64_t lhs, uint64_t rhs, bool& overflow)
{
    constexpr auto maximum = (std::numeric_limits<uint64_t>::max)();
    if (rhs != 0 && lhs > maximum / rhs)
    {
        overflow = true;
        return maximum;
    }
    return lhs * rhs;
}

inline uint64_t multiply(uint64_t first, uint64_t second, uint64_t third, bool& overflow)
{
    if (first == 0 || second == 0 || third == 0)
        return 0;
    return multiply(multiply(first, second, overflow), third, overflow);
}
} // namespace detail

/*! \brief Estimate queue storage, including file-loading staging and input timestamps.
 *
 * FileFrameReadWorker initially allocates the complete file buffer on both devices, even when one copy is
 * subsequently released in CPU/GPU preload mode. Input capacity is rounded down to complete batches, as in
 * BatchInputQueue. Cuts include both XZ and YZ uint16 queues. Output includes any enabled raw/filter view queues.
 * Fixed processing workspaces, allocator overhead and CUDA stream resources are excluded; a budget reserve is
 * therefore still required. Arithmetic saturates instead of wrapping; callers should reject overflowed estimates.
 */
inline Estimate estimate(const Request& request)
{
    Estimate result;
    auto& overflow = result.overflow;

    uint64_t input_frame_bytes = 0;
    if (request.file_enabled || request.input_frames != 0 || (request.raw_view_enabled && request.output_frames != 0))
        input_frame_bytes = detail::multiply(request.width, request.height, request.input_bytes_per_pixel, overflow);

    if (request.file_enabled)
    {
        result.file.cpu = detail::multiply(input_frame_bytes, request.file_frames, overflow);
        // One extra packed frame is used by the file reader on the GPU.
        result.file.gpu = detail::add(result.file.cpu, input_frame_bytes, overflow);
    }

    // Treat a missing batch size as one to keep this helper defined for incomplete UI state.
    const auto batch_size = request.batch_size == 0 ? uint64_t{1} : request.batch_size;
    const auto input_frames = request.input_frames - request.input_frames % batch_size;
    result.input.gpu = detail::multiply(input_frame_bytes, input_frames, overflow);
    // Frame ID, synced timestamp, camera timestamp, and timestamp offset remain in CPU RAM.
    result.input.cpu = detail::multiply(input_frames, 4 * sizeof(uint64_t), overflow);

    const auto record_bytes =
        detail::multiply(request.record_bytes_per_frame, request.record_frames, request.record_buffer_count, overflow);
    if (request.record_on_gpu)
        result.record.gpu = record_bytes;
    else
        result.record.cpu = record_bytes;

    result.output.gpu = detail::multiply(request.output_bytes_per_frame, request.output_frames, overflow);
    if (request.raw_view_enabled)
        result.output.gpu = detail::add(result.output.gpu,
                                        detail::multiply(input_frame_bytes, request.output_frames, overflow),
                                        overflow);
    if (request.filter2d_view_enabled)
        result.output.gpu =
            detail::add(result.output.gpu,
                        detail::multiply(request.output_bytes_per_frame, request.output_frames, overflow),
                        overflow);

    if (request.cuts_enabled && request.cuts_frames != 0 && request.time_transformation_size != 0)
    {
        const auto cut_frame_bytes = detail::multiply(detail::add(request.width, request.height, overflow),
                                                      request.time_transformation_size,
                                                      sizeof(uint16_t),
                                                      overflow);
        result.cuts.gpu = detail::multiply(cut_frame_bytes, request.cuts_frames, overflow);
    }

    const Usage rows[] = {result.file, result.input, result.record, result.output, result.cuts};
    for (const auto usage : rows)
    {
        result.total.cpu = detail::add(result.total.cpu, usage.cpu, overflow);
        result.total.gpu = detail::add(result.total.gpu, usage.gpu, overflow);
    }
    return result;
}

inline uint64_t required_increase(uint64_t proposed, uint64_t current)
{
    return proposed > current ? proposed - current : 0;
}

inline Usage required_increase(const Usage& proposed, const Usage& current)
{
    return {required_increase(proposed.cpu, current.cpu), required_increase(proposed.gpu, current.gpu)};
}

/*! \brief Check additional allocations against currently available memory while retaining a reserve.
 *
 * Current must describe allocations that already exist, because they are already deducted from available.
 * Unchanged/smaller allocations remain permitted when free memory is below the reserve. CPU and GPU budgets
 * must be checked independently; releasing GPU memory cannot make room for a CPU allocation, or vice versa.
 */
inline bool budget_allows(uint64_t proposed, uint64_t current, uint64_t available, uint64_t reserve)
{
    const auto increase = required_increase(proposed, current);
    return increase == 0 || (available >= reserve && increase <= available - reserve);
}

inline bool budget_allows(const Usage& proposed, const Usage& current, const Usage& available, const Usage& reserve)
{
    return budget_allows(proposed.cpu, current.cpu, available.cpu, reserve.cpu) &&
           budget_allows(proposed.gpu, current.gpu, available.gpu, reserve.gpu);
}
} // namespace holovibes::buffer_memory
