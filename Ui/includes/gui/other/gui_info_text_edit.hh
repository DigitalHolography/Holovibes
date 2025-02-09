/*! \file gui_info_text_edit.hh
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

#define RED_COLORATION_RATIO 0.9f
#define ORANGE_COLORATION_RATIO 0.7f

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

    /*! \brief Formats all available information and displays it in the UI */
    void display_information();

  private:
    /*!
     * \brief Formats the GPU load into an html string
     *
     * \return std::string The formatted string
     */
    std::string gpu_load();

    /*!
     * \brief Formats GPU memory controller data into an html string
     *
     * \return std::string The formatted string
     */
    std::string gpu_memory_controller_load();

    /*!
     * \brief Formats the GPU memory percentage into an html string
     *
     * \return std::string The formatted string
     */
    std::string gpu_memory();

    /*! \brief Structure that will be used to retrieve information from the API */
    Information information_;
};
} // namespace holovibes::gui