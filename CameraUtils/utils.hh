/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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

  /*void create_logfile(const std::string name);
  void log_msg(const std::string msg);
  void close_logfile();
  void allocate_memory(void** buf, const std::size_t size);
  void free_memory(void* buf);*/

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