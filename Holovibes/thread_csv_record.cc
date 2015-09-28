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
      nb_frames_(nb_frames),
      record_(true)
    {
    }

    ThreadCSVRecord::~ThreadCSVRecord()
    {
    }

    void ThreadCSVRecord::stop()
    {
      record_ = false;
    }

    void ThreadCSVRecord::run()
    {
      deque_.clear();
      pipeline_.request_average_record(&deque_, nb_frames_);

      while (deque_.size() < nb_frames_)
        continue;

      std::cout << path_ << "\n";
      std::ofstream of(path_);
      
      of << "signal,noise,average\n";

      unsigned int i = 0;
      unsigned int deque_size = static_cast<unsigned int>(deque_.size());
      while (i < deque_size && record_)
      {
        std::tuple<float, float, float>& tuple = deque_[i];
        of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0')
          << std::get<0>(tuple) << ","
          << std::get<1>(tuple) << ","
          << std::get<2>(tuple) << "\n";
        ++i;
      }

      pipeline_.request_refresh();
    }
}