#include <direct.h>
#include <cstdio>
#include <iostream>
#include <time.h>

#include "utils.hh"

namespace camutils
{
  void create_logfile(std::string name)
  {
    std::string folder = "log/";
    time_t date = time(nullptr);
    std::string file(folder);

    // If the "log" folder does not exist, create it.
    _mkdir(folder.c_str());

    if (date != static_cast<time_t>(-1)) // If current date was successfully found
    {
      struct tm time_struct;
      localtime_s(&time_struct, &date);
      const size_t size = 100;
      char today[size] = { '\0' };

      strftime(today, size, "%y%m%d-%Hh%Mm%S_", &time_struct);
      // Appending current date to the file's name.
      file.append(today);
    }
    file.append(name);
    file.append(".log");

    /* Opening with only ios::out flag first, so as to create
    ** automatically the file.
    */
    logfile.open(file, std::ios::out);
    logfile.close();

    // ios::in is needed later, for close_logfile.
    logfile.open(file, std::ios::in | std::ios::out);
    if (!logfile.is_open())
    {
      // Redirect the standard error stream.
      logfile.copyfmt(std::cerr);
      logfile.clear(std::cerr.rdstate());
      logfile.basic_ios<char>::rdbuf(std::cerr.rdbuf());
    }

    filename = file;
  }

  void log_msg(std::string msg)
  {
    logfile << msg << std::endl;
  }

  void close_logfile()
  {
    logfile.seekg(std::ios::beg);
    if (logfile.peek() == std::ios::traits_type::eof())
    {
      // File is empty, better erase it so as to avoid cluttering storage.
      logfile.close();
      remove(filename.c_str());
    }
  }
}