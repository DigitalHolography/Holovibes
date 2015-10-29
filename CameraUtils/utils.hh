#ifndef UTILS_HH
# define UTILS_HH

#ifdef UTILS_EXPORTS
# define UTILS_API __declspec(dllexport)
#else
# define UTILS_API __declspec(dllimport)
#endif

# include <fstream>
# include <string>

/* A set of various functions usable by all cameras if they wish to.
** This module factorizes common needs of cameras implementations.
*/
namespace camutils
{
  std::fstream logfile;
  static std::string filename;

  extern "C"
  {
    /* Logging functions
    */
    UTILS_API void create_logfile(std::string name);

    UTILS_API void log_msg(std::string msg);

    UTILS_API void close_logfile();

    /* "Pinned" memory allocation/deallocation on host RAM.
    ** Pinned (non-pageable) is required for data moving from host to GPU
    ** or the other way around, as GPUs do not handle other kinds of RAM memory.
    ** Setting such a memory to be non-pageable right from the allocation step
    ** allow faster memory accesses, and so faster memory copies.
    **
    ** \param size The number of bytes to allocate.
    */
    UTILS_API void allocate_memory(void** buf, const std::size_t size);

    UTILS_API void free_memory(void* buf);
  }
}

#endif /* !UTILS_HH */