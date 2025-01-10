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
    /*! \brief Structure that will be used to retrieve information from the API */
    Information information_;
};
} // namespace holovibes::gui