/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Export panel
 */
#pragma once

#include "enum_record_mode.hh"
#include "panel.hh"
#include "PlotWindow.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class ExportPanel
 *
 * \brief Class representing the Export Panel in the GUI
 */
class ExportPanel : public Panel
{
    Q_OBJECT

  public:
    ExportPanel(QWidget* parent = nullptr);
    ~ExportPanel();

    void init() override;
    void on_notify() override;

  public slots:
    /*! \brief Opens file explorer on the fly to let the user chose the output file he wants with extension
     * replacement*/
    void browse_record_output_file();

    /*! \brief Enables or Disables number of frame restriction for recording
     *
     * \param value true: enable, false: disable
     */
    void set_nb_frames_mode(bool value);

    /*! \brief Browses output file */
    void browse_batch_input();

    /*! \brief Modifies the record mode
     *
     * \param value The new record mode
     */
    void set_record_mode(const QString& value);

    /*! \brief Stops the record */
    void stop_record();

    /*! \brief Resets ui on record finished
     *
     * \param record_mode The current record mode
     */
    void record_finished(RecordMode record_mode);

    /*! \brief Starts recording */
    void start_record();

    /*! \brief Creates Signal overlay */
    void activeSignalZone();

    /*! \brief Creates Noise overlay */
    void activeNoiseZone();

    /*! \brief Opens Chart window */
    void start_chart_display();

    /*! \brief Closes Chart window */
    void stop_chart_display();
};
} // namespace holovibes::gui
