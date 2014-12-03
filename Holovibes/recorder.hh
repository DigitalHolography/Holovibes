#ifndef RECORDER_HH
# define RECORDER_HH

#include <fstream>

#include "queue.hh"

namespace holovibes
{
  class Recorder
  {
  public:
    Recorder(
      Queue& queue,
      const std::string& filepath);
    ~Recorder();

    void record(unsigned int n_images);
    void stop();
  private:
    bool is_file_exist(const std::string& filepath);

  private:
    Queue& queue_;
    std::ofstream file_;
    bool stop_requested_;
  };
}

#endif /* !RECORDER_HH */