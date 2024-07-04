/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Export panel
 */
#pragma once

#include "enum_record_mode.hh"
#include "panel.hh"
#include "PlotWindow.hh"
#include "lightui.hh"

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

    void set_record_frame_step(int step);
    int get_record_frame_step();

    void set_light_ui(std::shared_ptr<LightUI> light_ui);
    void set_output_file_name(std::string std_filepath);

  public slots:
    /*! \brief Opens file explorer on the fly to let the user chose the output file he wants with extension
     * replacement*/
    QString browse_record_output_file();

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

    /*! \brief Set the record device : GPU, with the pipeline and the realtime display, or CPU, with no pipe */
    void set_record_device(bool value);

    /*! \brief Creates Signal overlay */
    void activeSignalZone();

    /*! \brief Creates Noise overlay */
    void activeNoiseZone();

    /*! \brief Opens Chart window */
    void start_chart_display();

    /*! \brief Closes Chart window */
    void stop_chart_display();

    /**
     * @brief Handles the update of the batch mode enabled setting checkbox.
     */
    void update_batch_enabled();

    /**
     * @brief Handles the update of the record frame count enabled setting checkbox.
     */
    void update_record_frame_count_enabled();

    /**
     * @brief Handles the update of the record frame count setting spinbox.
     */
    void update_record_frame_count();

    /**
     * @brief Handles the update of the record file path setting line edit.
     */
    void update_record_file_path();

    /**
     * @brief Handles the update of the batch file path setting line edit.
     */
    void update_batch_file_path();

    /**
     * @brief Handles the update of the record mode setting combo box.
     */
    void update_record_mode();

    /**
     * @brief Handles the update of the record file extension setting combo box.
     */
    void update_record_file_extension(const QString& value);

  private:
    int record_frame_step_ = 512;
    std::shared_ptr<LightUI> light_ui_;
};
} // namespace holovibes::gui
