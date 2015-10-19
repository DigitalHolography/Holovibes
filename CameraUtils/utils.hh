#ifndef UTILS_HH
# define UTILS_HH

#ifdef UTILS_EXPORTS
# define UTILS_API __declspec(dllexport)
#else
# define UTILS_API __declspec(dllimport)
#endif

# include <fstream>
# include <string>

namespace camutils
{
  static std::fstream logfile;
  static std::string filename;

  extern "C"
  {
    UTILS_API void create_logfile(std::string name);

    UTILS_API void log_msg(std::string msg);

    UTILS_API void close_logfile();
  }
}

#endif /* !UTILS_HH */