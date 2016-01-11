/*! \file
 *
 * Store frames of a given queue in a file. */
#pragma once

#include <fstream>
#include <QObject>

/* Forward declaration. */
namespace holovibes
{
  class Queue;
}

namespace holovibes
{
  /*! \brief Store frames of given queue in file
   *
   * Image are stored in raw format at the given file path.
   * Recorder is thread safe and you can stop record anytime.
   *
   * Usage:
   * * Create Recorder
   * * Use record to record n_images any times you need it
   * * delete Recorder
   */
  class Recorder : public QObject
  {
    Q_OBJECT

signals :
    /*! \brief Inform that one frame has been recorded
    **
    ** value is number of frames recorded*/
    void value_change(int value);

  public:
    /*! \brief Open the given filepath.
     *
     * \param queue The source queue.
     * \param filepath The absolute path to the destination file.
     */
    Recorder(
      Queue& queue,
      const std::string& filepath);

    ~Recorder();

    /*! \brief record n_images to file
     *
     * Recorder is thread safe and you can stop this function anytime
     * by using the "stop" button.
     */
    void record(const unsigned int n_images);

    /*! \brief Stop current record */
    void stop();

  private:
    bool is_file_exist(const std::string& filepath);

  private:
    Queue& queue_;
    std::ofstream file_;
    bool stop_requested_;
  };
}