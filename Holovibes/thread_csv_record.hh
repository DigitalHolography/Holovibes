/*! \file
 *
 * Thread class used to record CSV files of ROI/average computations. */
#pragma once

# include <iostream>
# include <iomanip>
# include <fstream>
# include <QThread>

# include "pipe.hh"

/* Forward declarations. */
namespace holovibes
{
  class Holovibes;
  template <class T> class ConcurrentDeque;
}

namespace gui
{
  /*! \brief Thread class used to record CSV files of ROI/average computations.
  **
  ** It inherits QThread because it is the GUI that needs to launch the record and it has
  ** to know when it is finished (signals/slots system).
  */
  class ThreadCSVRecord : public QThread
  {
    Q_OBJECT

    typedef holovibes::ConcurrentDeque<std::tuple<float, float, float, float>> Deque;

signals :
	
	void value_change(int value);

  public:
    /*! \brief ThreadCSVRecord constructor
    **
    ** \param pipe pipe of the program, see holovibes::Holovibes::get_pipe()
    ** \param deque concurrent Deque containing the average values to record
    ** \param path string containing output path of record
    ** \param nb_frames number of frames i-e number of values to record
    ** \param parent Qt parent (default is null)
    */
    ThreadCSVRecord(holovibes::Holovibes& holo,
      Deque& deque,
      const std::string path,
      const unsigned int nb_frames,
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
    /*! Reference to the core class of the program. */
    holovibes::Holovibes& holo_;
    /*! Deque storing recorded data. */
    Deque& deque_;
    /*! Output record path */
    std::string path_;
    /*! Number of frames i-e number of values to record */
    unsigned int nb_frames_;
    /*! Flag used to stop recording */
    bool record_;
  };
}