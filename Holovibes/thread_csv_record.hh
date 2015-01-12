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
  /*! \class ThreadCSVRecord
  **
  ** Thread class used to record CSV files of ROI/average computations.
  **
  ** It inherits QThread because it is the GUI that needs to launch the record and it has
  ** to know when it is finished (signal/slots system).
  */
  class ThreadCSVRecord : public QThread
  {
    Q_OBJECT

      typedef holovibes::ConcurrentDeque<std::tuple<float, float, float>> Deque;

  public:
    /*! \brief ThreadCSVRecord constructor
    **
    ** \param pipeline pipeline of the program, see holovibes::Holovibes::get_pipeline()
    ** \param deque concurrent Deque containing the average values to record
    ** \param path string containing output path of record
    ** \param nb_frames number of frames i-e number of values to record
    ** \param parent Qt parent (default is null)
    */
    ThreadCSVRecord(holovibes::Pipeline& pipeline,
      Deque& deque,
      std::string path,
      unsigned int nb_frames,
      QObject* parent = nullptr);
    ~ThreadCSVRecord();

  public slots:
    /*! Stops the record by setting a flag */
    void stop();

  private:
    /*! \brief Overrided QThread run method, recording method
    **
    ** Ensure to flush the Deque before using it in order to record the frames
    ** from the moment the user started the record and not before.
    */
    void run() override;

  private:
    /*! Program's pipeline */
    holovibes::Pipeline& pipeline_;
    /*! Deque to record */
    Deque& deque_;
    /*! Output record path */
    std::string path_;
    /*! nb_frames number of frames i-e number of values to record */
    unsigned int nb_frames_;
    /*! Flag used to stop recording */
    bool record_;
  };
}

#endif /* !THREAD_CSV_RECORD_HH */