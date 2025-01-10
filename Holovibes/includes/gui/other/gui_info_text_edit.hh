/*! \file
 *
 * \brief Declares the widget InfoTextEdit, used for displaying information in the UI.
 */
#pragma once

#include <QObject>
#include <QString>
#include <QTimer>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QTWidgets/QTextEdit>

#include <string>

#include "chrono.hh"
#include "information_struct.hh"

namespace holovibes::gui
{
using MutexGuard = std::lock_guard<std::mutex>;

#define INPUT_Q_RED_COLORATION_RATIO 0.8f
#define INPUT_Q_ORANGE_COLORATION_RATIO 0.3f

/*! \class InfoTextEdit
 *
 * \brief Widget wrapping for a QTextEdit. It is used to display information to the user.
 * This class houses tools to compute and format the information in a readable form.
 */
class InfoTextEdit : public QTextEdit
{
    Q_OBJECT

  public:
    InfoTextEdit(QWidget* parent = nullptr)
        : QTextEdit(parent)
    {
    }

    /*!
     * \brief Does come calculations regarding the queues, throughputs and frames per second
     *
     * \param elapsed_time The time elapsed since the last function call
     */
    void display_information_slow(size_t elapsed_time);

    /*!
     * \brief Formats all available information and displays it in the UI
     *
     */
    void display_information();

  private:
    /*!
     * \brief Performs some simple multiplications with the respective fps to get the queue throughputs
     *
     * \param output_frame_res The RESOLUTION of the output frames (doesn't include time_transformation_size)
     * \param input_frame_size The size in memory of the input frames
     * \param record_frame_size The size in memory of the recorded frames
     */
    void compute_throughput(size_t output_frame_res, size_t input_frame_size, size_t record_frame_size);

    /*!
     * \brief Computes the average frames per second of the available streams (input, output, record)
     *
     * \param waited_time The time elapsed since the last function call
     */
    // void compute_fps(const long long waited_time);

    /*! \brief Input throughput */
    size_t input_throughput_ = 0;

    /*! \brief Output throughput */
    size_t output_throughput_ = 0;

    /*! \brief Saving throughput */
    size_t saving_throughput_ = 0;

    /*! \brief Structure that will be used to retrieve information from the API */
    Information information_;
};
} // namespace holovibes::gui