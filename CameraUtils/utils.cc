#include <time.h>

#include "utils.hh"

namespace camutils
{
  void create_logfile(std::string name)
  {
    time_t date = time(nullptr);
    std::string filename("");

    if (date != static_cast<time_t>(-1)) // If current date was successfully found
    {
      struct tm time_struct;
      localtime_s(&time_struct, &date);
      const size_t size = 100;
      char today[size] = { '\0' };

      strftime(today, size, "%y%m%d-%Hh%Mm%S_", &time_struct);
      // Appending current date to the file's name.
      filename.append(today);
    }
    filename.append(name);
    filename.append(".log");

    logfile.open(filename);
  }

  void log_msg(std::string msg)
  {
    logfile << msg << std::endl;
  }
}