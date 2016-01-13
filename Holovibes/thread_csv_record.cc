#include "thread_csv_record.hh"
#include "concurrent_deque.hh"
#include "holovibes.hh"

namespace gui
{
  ThreadCSVRecord::ThreadCSVRecord(holovibes::Holovibes& holo,
    Deque& deque,
    const std::string path,
    const unsigned int nb_frames,
    QObject* parent)
    : QThread(parent)
    , holo_(holo)
    , deque_(deque)
    , path_(path)
    , nb_frames_(nb_frames)
    , record_(true)
  {
  }

  ThreadCSVRecord::~ThreadCSVRecord()
  {
    this->stop();
  }

  void ThreadCSVRecord::stop()
  {
    record_ = false;
  }

  void ThreadCSVRecord::run()
  {
    deque_.clear();
    holo_.get_pipe()->request_average_record(&deque_, nb_frames_);

    while (deque_.size() < nb_frames_ && record_)
      continue;

    std::cout << path_ << "\n";
    std::ofstream of(path_);

    // Header displaying
    of << "[Phase number : " << holo_.get_compute_desc().nsamples
      << ", p : " << holo_.get_compute_desc().pindex
      << ", lambda : " << holo_.get_compute_desc().lambda
      << ", z : " << holo_.get_compute_desc().zdistance
      << "]" << std::endl;

    of << "[Column 1 : signal, Column 2 : noise, Column 3 : 10 * log10 (signal / noise)]" << std::endl;

    const unsigned int deque_size = static_cast<unsigned int>(deque_.size());
    unsigned int i = 0;
    while (i < deque_size && record_)
    {
      std::tuple<float, float, float, float>& tuple = deque_[i];
      of << std::fixed << std::setw(11) << std::setprecision(10) << std::setfill('0')
        << std::get<0>(tuple) << ","
        << std::get<1>(tuple) << ","
        << std::get<2>(tuple) << "\n";
      ++i;
    }

    holo_.get_pipe()->request_average_stop();
  }
}