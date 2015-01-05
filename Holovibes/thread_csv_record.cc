#include "thread_csv_record.hh"

namespace gui
{
    ThreadCSVRecord::ThreadCSVRecord(Deque& deque,
      unsigned int nb_frames,
      QObject* parent)
      : QThread(parent),
      deque_(deque)
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
    }
}