#include "recorder.hh"

#include <exception>
#include <cassert>
#include <thread>
#include <iostream>
#include <list>
#include <boost/filesystem.hpp>
#include <direct.h>

namespace holovibes
{
  // Helper functions
  namespace
  {
    void createFilePath(const std::string folderName)
    {
      std::list<std::string> folderLevels;
      char* c_str = (char*)folderName.c_str();

      // Point to end of the string
      char* strPtr = &c_str[strlen(c_str) - 1];

      // Create a list of the folders which do not currently exist
      do
      {
        if (boost::filesystem::exists(c_str))
          break;
        // Break off the last folder name, store in folderLevels list
        do
        {
          --strPtr;
        } while ((*strPtr != '\\') && (*strPtr != '/') && (strPtr >= c_str));
        folderLevels.push_front(std::string(strPtr + 1));
        strPtr[1] = 0;
      } while (strPtr >= c_str);

      if (folderLevels.empty())
        return;
      std::cout << folderLevels.back() << std::endl;
      folderLevels.pop_back();

      if (_chdir(c_str))
      {
        throw std::exception("[RECORDER] error cannot _chdir directory");
      }

      // Create the folders iteratively
      std::string startPath = boost::filesystem::current_path().string();
      for (std::list<std::string>::iterator it = folderLevels.begin(); it != folderLevels.end(); it++)
      {
        if (boost::filesystem::create_directory(it->c_str()) == 0)
          throw std::exception("[RECORDER] error cannot create directory");

        _chdir(it->c_str());
      }
      _chdir(startPath.c_str());
    }
  }

  Recorder::Recorder(
    Queue& queue,
    const std::string& filepath)
    : queue_(queue)
    , file_()
    , stop_requested_(false)
  {
    if (filepath.find('/') != std::string::npos)
      createFilePath(filepath);

#ifndef _DEBUG
    if (is_file_exist(filepath))
      throw std::exception("[RECORDER] error overwriting an existing file");
#endif /* Overwrite is useful while debugging. */

    file_.open(filepath, std::ios::binary | std::ios::trunc);
  }

  Recorder::~Recorder()
  {
  }

  void Recorder::record(const unsigned int n_images)
  {
    const size_t size = queue_.get_size();
    char* buffer = new char[size]();

    std::cout << "[RECORDER] started recording " <<
      n_images << " frames" << std::endl;

    for (unsigned int i = 0; !stop_requested_ && i < n_images; ++i)
    {
      while (queue_.get_current_elts() < 1)
        std::this_thread::yield();

      queue_.dequeue(buffer, cudaMemcpyDeviceToHost);
      file_.write(buffer, size);
    }

    std::cout << "[RECORDER] recording has been stopped" << std::endl;

    delete[] buffer;
  }

  void Recorder::stop()
  {
    stop_requested_ = true;
  }

  bool Recorder::is_file_exist(const std::string& filepath)
  {
    std::ifstream ifs(filepath);
    return ifs.good();
  }
}