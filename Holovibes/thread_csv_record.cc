#include "thread_csv_record.hh"

namespace gui
{
    ThreadCSVRecord::ThreadCSVRecord(holovibes::Pipeline& pipeline,
      Deque& deque,
      std::string path,
      unsigned int nb_frames,
      QObject* parent)
      : QThread(parent),
      pipeline_(pipeline),
      deque_(deque),
      path_(path),
      nb_frames_(nb_frames)
    {
    }

    ThreadCSVRecord::~ThreadCSVRecord()
    {
    }

    void ThreadCSVRecord::stop()
    {
    }

    void ThreadCSVRecord::run()
    {
      deque_.clear();
      pipeline_.request_average_record(&deque_, nb_frames_);

      while (deque_.size() < nb_frames_)
        continue;

      std::cout << path_ << std::endl;
      std::ofstream of(path_);
      
      of << "signal,noise,average\n";

      for (auto it = deque_.begin(); it != deque_.end(); ++it)
      {
        std::tuple<float, float, float>& tuple = *it;
        of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0')
          << std::get<0>(tuple) << ","
          << std::get<1>(tuple) << ","
          << std::get<2>(tuple) << "\n";
      }
    }
}