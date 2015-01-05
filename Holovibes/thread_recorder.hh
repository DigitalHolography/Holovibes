#ifndef THREAD_RECORDER_HH
# define THREAD_RECORDER_HH

# include <string>
# include <QThread>
# include <QObject>

# include "queue.hh"
# include "recorder.hh"

namespace gui
{
  class ThreadRecorder : public QThread
  {
    Q_OBJECT

  public:
    ThreadRecorder(
      holovibes::Queue& queue,
      const std::string& filepath,
      unsigned int n_images,
      QObject* parent = nullptr);

    virtual ~ThreadRecorder();

  public slots:
    void stop();
  private:
    void run() override;
  private:
    holovibes::Queue& queue_;
    holovibes::Recorder recorder_;
    unsigned int n_images_;
  };
}

#endif /* !THREAD_RECORDER_HH */