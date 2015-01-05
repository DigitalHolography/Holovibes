#ifndef THREAD_CSV_RECORD_HH
# define THREAD_CSV_RECORD_HH

# include <iostream>
# include <iomanip>
# include <fstream>
# include <QThread>
# include "concurrent_deque.hh"
# include "pipeline.hh"

namespace gui
{
  class ThreadCSVRecord : public QThread
  {
    Q_OBJECT

      typedef holovibes::ConcurrentDeque<std::tuple<float, float, float>> Deque;

  public:
    ThreadCSVRecord(holovibes::Pipeline& pipeline,
      Deque& deque,
      const std::string& path,
      unsigned int nb_frames,
      QObject* parent = nullptr);
    ~ThreadCSVRecord();

  public slots:
    void stop();

  private:
    void run() override;

  private:
    holovibes::Pipeline& pipeline_;
    Deque& deque_;
    const std::string& path_;
    unsigned int nb_frames_;
  };
}

#endif /* !THREAD_CSV_RECORD_HH */