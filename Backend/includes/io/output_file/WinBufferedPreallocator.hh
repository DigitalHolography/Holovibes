#pragma once
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <vector>
#endif

namespace holovibes::io_files
{

#ifdef _WIN32
struct WinBufferedPreallocator
{
    // configure before use
    size_t io_buf_size = 4u << 20;    // stdio buffer size (4 MiB default; try 1–16 MiB)
    uint64_t grow_chunk = 1ull << 30; // grow step (1 GiB; 256–1024 MiB are good for 1–20 GB files)
    bool durable_on_close = false;    // call _commit() at close/finalize if true

    // runtime state
    std::unique_ptr<char[]> io_buf; // stdio buffer storage
    uint64_t allocated_size = 0;    // how much NTFS space we've extended to
    uint64_t logical_size = 0;      // how many bytes we've actually written

    // Attach to an already-open FILE* (opened "wb" or "w+b")
    // Sets a large fully-buffered stdio buffer.
    void attach(FILE* f)
    {
        io_buf.reset(new char[io_buf_size]);
        if (setvbuf(f, io_buf.get(), _IOFBF, io_buf_size) != 0)
        {
            // Not fatal; continue with default buffering
        }
        // Initialize allocated_size from current EOF (in case header already written)
        HANDLE h = to_handle(f);
        LARGE_INTEGER size{};
        if (GetFileSizeEx(h, &size))
        {
            allocated_size = static_cast<uint64_t>(size.QuadPart);
            logical_size = allocated_size; // assume file pointer is at EOF initially
        }
    }

    // Ensure the file is large enough for a write ending at 'required_end'.
    // Grows the NTFS file in big steps to reduce metadata churn.
    void ensure_capacity(FILE* f, uint64_t required_end)
    {
        if (required_end <= allocated_size)
            return;

        // Round up to next multiple of grow_chunk
        uint64_t new_cap = ((required_end + grow_chunk - 1) / grow_chunk) * grow_chunk;

        // Save current stream position
        __int64 pos = _ftelli64(f);

        HANDLE h = to_handle(f);

        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(new_cap);

        // Move pointer and extend end-of-file
        if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN) || !SetEndOfFile(h))
        {
            // restore position and throw
            li.QuadPart = pos;
            SetFilePointerEx(h, li, nullptr, FILE_BEGIN);
            throw std::runtime_error("SetEndOfFile failed during preallocation");
        }

        // Restore position for stdio stream
        _fseeki64(f, pos, SEEK_SET);

        allocated_size = new_cap;
    }

    // Update logical byte count after a successful write of 'n' bytes.
    void on_bytes_written(size_t n) { logical_size += static_cast<uint64_t>(n); }

    // Flush stdio, trim any over-allocation down to logical_size,
    // and optionally force durability (_commit).
    void finalize(FILE* f)
    {
        fflush(f); // stdio -> OS cache

        // Trim to the exact logical size (so the file doesn't keep the last grow chunk)
        HANDLE h = to_handle(f);
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(logical_size);
        if (SetFilePointerEx(h, li, nullptr, FILE_BEGIN))
        {
            SetEndOfFile(h);
        }

        if (durable_on_close)
        {
            _commit(_fileno(f)); // OS cache -> device (FlushFileBuffers)
        }
    }

  private:
    static HANDLE to_handle(FILE* f)
    {
        int fd = _fileno(f);
        if (fd < 0)
            throw std::runtime_error("_fileno failed");
        intptr_t os = _get_osfhandle(fd);
        if (os == -1)
            throw std::runtime_error("_get_osfhandle failed");
        return reinterpret_cast<HANDLE>(os);
    }
};
#else

struct WinBufferedPreallocator
{
    size_t io_buf_size = 4u << 20;
    uint64_t grow_chunk = 1ull << 30;
    bool durable_on_close = false;
    void attach(FILE*) {}
    void ensure_capacity(FILE*, uint64_t) {}
    void on_bytes_written(size_t) {}
    void finalize(FILE*) {}
};
#endif

} // namespace holovibes::io_files