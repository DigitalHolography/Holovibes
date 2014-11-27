#include "thread_recorder.hh"
#include "recorder.hh"

namespace gui
{
  ThreadRecorder::ThreadRecorder(
    holovibes::Queue& queue,
    const std::string& filepath,
    unsigned int n_images,
    QObject* parent)
    : QThread(parent)
    , recorder_(queue, filepath)
    , n_images_(n_images)
  {}

  ThreadRecorder::~ThreadRecorder()
  {}

  void ThreadRecorder::stop()
  {
    recorder_.stop();
  }

  void ThreadRecorder::run()
  {
    recorder_.record(n_images_);
  }
}