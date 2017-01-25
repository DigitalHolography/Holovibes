#include "thread_recorder.hh"
#include "recorder.hh"
#include "queue.hh"

#include "info_manager.hh"

namespace gui
{
  ThreadRecorder::ThreadRecorder(
    holovibes::Queue& queue,
    const std::string& filepath,
    const unsigned int n_images,
    QObject* parent)
    : QThread(parent)
    , queue_(queue)
    , recorder_(queue, filepath)
    , n_images_(n_images)
  {
  }

  ThreadRecorder::~ThreadRecorder()
  {
  }

  void ThreadRecorder::stop()
  {
    recorder_.stop();
  }

  void ThreadRecorder::run()
  {
    QProgressBar*   progress_bar = InfoManager::get_manager()->get_progress_bar();

    queue_.flush();
    progress_bar->setMaximum(n_images_);
    connect(&recorder_, SIGNAL(value_change(int)), progress_bar, SLOT(setValue(int)));
    recorder_.record(n_images_);
  }
}