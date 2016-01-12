/*! \file
 *
 * Functionnalities to log what happens inside the camera. */
#pragma once

#ifdef UTILS_EXPORTS
# define UTILS_API __declspec(dllexport)
#else
# define UTILS_API __declspec(dllimport)
#endif

# include <fstream>
# include <string>

namespace camutils
{
  std::fstream logfile;
  static std::string filename;

  /*! \defgroup CamUtils Camera Utilities
  * Camera utilities functions : various functions usable by cameras DLLs
  * to respond to their needs.
  *
  * Services provided are :
  * * Logging activity into a file
  * * Asking for pinned memory allocation on the host to speed up data transfer.
  * \{
  */
  extern "C"
  {
    /*! \brief Create a logfile for a given camera name.
     *
     * The file will be created in a log/ directory in the current
     * working directory of the executable. */
    UTILS_API void create_logfile(const std::string name);

    /*! \brief Log a single message. */
    UTILS_API void log_msg(const std::string msg);

    /*! \brief The user of the logging service has to manually close the stream.
     *
     * If the logfile is empty because no event was logged, it is deleted
     * to avoid cluttering. */
    UTILS_API void close_logfile();

    /*! \brief "Pinned" memory allocation/deallocation on host RAM.
     *
     * Pinned (non-pageable) memory is required for data moving from host RAM
     * to GPU RAM or the other way around, as GPUs do not handle other kinds
     * of RAM memory. In practice, using pageable memory on RAM forces
     * preparing a pinned intermediate memory space at each memory transaction.
     *
     * Setting allocated memory to be non-pageable right from the start allows
     * faster memory copies.
     *
     * \param buf The address of the pointer to the memory space needed.
     * If allocation fails, buf will point to a nullptr.
     * \param size The number of bytes to allocate.
     */
    UTILS_API void allocate_memory(void** buf, const std::size_t size);

    /*! \brief Memory allocated with allocate_memory() MUST be freed with this function. */
    UTILS_API void free_memory(void* buf);
  }
  /*! \} */ // End of CamUtils group
}