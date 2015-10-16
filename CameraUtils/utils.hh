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
  static std::ofstream logfile;

  extern "C"
  {
    void create_logfile(std::string name);

    void log_msg(std::string msg);
  }
}

#endif /* !UTILS_HH */