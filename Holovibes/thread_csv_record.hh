#ifndef THREAD_CSV_RECORD_HH
# define THREAD_CSV_RECORD_HH

# include <QThread>
# include "concurrent_deque.hh"

namespace gui
{
  class ThreadCSVRecord : public QThread
  {
    Q_OBJECT

      typedef holovibes::ConcurrentDeque<std::tuple<float, float, float>> Deque;

  public:
    ThreadCSVRecord(Deque& deque,
      unsigned int nb_frames,
      QObject* parent = nullptr);
    ~ThreadCSVRecord();

  public slots:
    void stop();

  private:
    void run() override;

  private:
    Deque& deque_;
  };
}

#endif /* !THREAD_CSV_RECORD_HH */