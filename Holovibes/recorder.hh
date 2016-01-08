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
   * Image are stored in raw format at the given file path
   * Recorder is thread safe and you can stop record at anytime
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
    /*! \brief inform, that one frame have been recorded
    **
    ** value is number of frames recorded*/
    void value_change(int value);
  public:
    /*! \brief Constructor
     *
     * Open given filepath
     */
    Recorder(
      Queue& queue,
      const std::string& filepath);

    ~Recorder();

    /*! \brief record n_images to file
     *
     * Recorder is thread safe and you can stop this function by using stop
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
