/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * Specialised QThread used for video recording. */
#pragma once

# include <string>
# include <QThread>
# include <QObject>

# include "recorder.hh"

/* Forward declaration. */
namespace holovibes
{
  class Queue;
}

namespace gui
{
  /*! \brief Thread class used to record images in direct or hologram mode.
  **
  ** It inherits QThread because it is the GUI that needs to launch the record and it has
  ** to know when it is finished (signals/slots system).
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
      const unsigned int n_images,
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