#ifndef LIGHTUI_HH
#define LIGHTUI_HH

#include <QMainWindow>

#include "notifier.hh"
#include "enum_record_mode.hh"

namespace Ui
{
class LightUI;
} // namespace Ui

namespace holovibes::gui
{
class MainWindow;
class ExportPanel;
class ImageRenderingPanel;

/**
 * @class LightUI
 * @brief The LightUI class is a QMainWindow derived class for the application's user interface.
 */
class LightUI : public QMainWindow
{
    Q_OBJECT

  public:
    /**
     * @brief Constructor for LightUI class.
     * @param parent The parent widget.
     * @param main_window Pointer to the MainWindow instance.
     * @param export_panel Pointer to the ExportPanel instance.
     */
    explicit LightUI(QWidget* parent,
                     MainWindow* main_window,
                     ExportPanel* export_panel);

    /**
     * @brief Destructor for LightUI class.
     */
    ~LightUI();

    /**
     * @brief Overridden showEvent handler.
     * @param event The QShowEvent instance.
     */
    void showEvent(QShowEvent* event) override;

    /**
     * @brief Updates the UI with the output file name for recording.
     * @param filename The name of the output file.
     */
    void actualise_record_output_file_ui(const QString& filename);

    /**
     * @brief Updates the UI with the Z distance.
     * @param z_distance The Z distance value.
     */
    void actualise_z_distance(const double z_distance);

    /**
     * @brief Resets the start button to its initial state.
     */
    void reset_start_button();
    void activate_start_button(bool activate);
    void set_progress_bar_value(int value);
    void set_progress_bar_maximum(int value);
    void progress_bar_recording();
    void progress_bar_stopped();
    void progress_bar_saving();

    /**
     * @brief Handles the start of a recording.
     * @param record The recording mode.
     */
    void on_record_start(RecordMode record);

    /**
     * @brief Handles the stop of a recording.
     * @param record The recording mode.
     */
    void on_record_stop(RecordMode record);

    /**
     * @brief Sets the state of the ui depending on the pipeline state.
     * @param active Boolean to set concerned widgets active (true) or inactive (false).
     */
    void pipeline_active(bool active);

  public slots:
    /**
     * @brief Opens the file explorer to let the user choose an output file with extension replacement.
     */
    void browse_record_output_file_ui();

    /**
     * @brief Opens the configuration UI.
     */
    void open_configuration_ui();

    /**
     * @brief Starts or stops the recording.
     * @param start Boolean to start (true) or stop (false) the recording.
     */
    void start_stop_recording(bool start);

    /**
     * @brief Slot for handling changes in Z value from a spin box.
     * @param z_distance The new Z distance value.
     */
    void z_value_changed_spinBox(int z_distance);

    /**
     * @brief Slot for handling changes in Z value from a slider.
     * @param z_distance The new Z distance value.
     */
    void z_value_changed_slider(int z_distance);

  protected:
    /**
     * @brief Overridden closeEvent handler.
     * @param event The QCloseEvent instance.
     */
    void closeEvent(QCloseEvent *event) override;
  
  private:
    Ui::LightUI* ui_; ///< Pointer to the UI instance.
    MainWindow* main_window_; ///< Pointer to the MainWindow instance.
    ExportPanel* export_panel_; ///< Pointer to the ExportPanel instance.
    bool visible_; ///< Boolean to track the visibility state of the UI.
    Subscriber<double> z_distance_subscriber_; ///< Subscriber for Z distance changes.
    Subscriber<RecordMode> record_start_subscriber_; ///< Subscriber for record start events.
    Subscriber<RecordMode> record_end_subscriber_; ///< Subscriber for record end events.
};

} // namespace holovibes::gui

#endif // LIGHTUI_HH
