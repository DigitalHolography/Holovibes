#ifndef THREAD_RECORDER_HH
# define THREAD_RECORDER_HH

# include <string>
# include <QThread>
# include <QObject>

# include "queue.hh"
# include "recorder.hh"

namespace gui
{
  /*! \class ThreadRecorder
  **
  ** Thread class used to record raw images.
  **
  ** It inherits QThread because it is the GUI that needs to launch the record and it has
  ** to know when it is finished (signal/slots system).
  */
  class ThreadRecorder : public QThread
  {
    Q_OBJECT

  public:
    /*! \brief ThreadRecorder constructor
    **
    ** \param queue Queue from where to fetch data
    ** \param filepath string containing output path of record
    ** \param n_images number of frames to record
    ** \param parent Qt parent
    */
    ThreadRecorder(
      holovibes::Queue& queue,
      const std::string& filepath,
      unsigned int n_images,
      QObject* parent = nullptr);

    virtual ~ThreadRecorder();

  public slots:
    /*! Stops the record by setting a flag */
    void stop();
  private:
    /*! \brief Overrided QThread run method, recording method
    **
    ** Ensure to flush the Queue before using it in order to record the frames
    ** from the moment the user started the record and not before.
    */
    void run() override;

  private:
    /*! Queue to record */
    holovibes::Queue& queue_;
    /*! Recorder object */
    holovibes::Recorder recorder_;
    /*! Number of frames to record */
    unsigned int n_images_;
  };
}

#endif /* !THREAD_RECORDER_HH */