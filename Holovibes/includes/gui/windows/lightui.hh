#ifndef LIGHTUI_HH
#define LIGHTUI_HH

#include <QMainWindow>
#include <tuple>

#include "notifier_struct.hh"
#include "notifier.hh"
#include "enum_record_mode.hh"
#include "enum_camera_kind.hh"

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
    explicit LightUI(QWidget* parent, MainWindow* main_window);

    /**
     * @brief Destructor for LightUI class.
     */
    ~LightUI();

    /**
     * @brief Overridden showEvent handler.
     * @param event The QShowEvent instance.
     */
    void showEvent(QShowEvent* event) override;

    /*!
     * \brief Update UI elements with the right value stored in the app
     */
    void notify();

    /**
     * @brief Updates the UI with the output file name for recording.
     * @param filename The name of the output file.
     */
    void actualise_record_output_file_ui(const std::filesystem::path file_path);

    /**
     * @brief Resets the start button to its initial state.
     */
    void reset_start_button();

    void activate_start_button(bool activate);
    void set_progress_bar_value(int value);
    void set_progress_bar_maximum(int value);

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
     * @brief Handles the updating of a recording ; used to update the progress bar.
     * @param record Contains info about the progress bar : its value and its max value.
     */
    void on_record_progress(const RecordProgressData& data);

    /**
     * @brief Handles the changes to the progress bar color.
     * @param data Contains the new color and text for the progress bar.
     */
    void on_record_progress_bar_color(const RecordBarColorData& data);

    void actualise_record_progress(const int value, const int max);
    void reset_record_progress_bar();

    /*! \brief Set the value of the record progress bar */
    void set_recordProgressBar_color(const QColor& color, const QString& text);

    /**
     * @brief Sets the state of the ui depending on the pipeline state.
     * @param active Boolean to set concerned widgets active (true) or inactive (false).
     */
    void pipeline_active(bool active);

    /**
     * @brief Sets the window size and position.
     */
    void set_window_size_position(int width, int height, int x, int y);

  public slots:
    /**
     * @brief Sets preset for given usage.
     */
    void set_preset();

    /**
     * @brief Opens the file explorer to let the user choose an output file with extension replacement.
     */
    void browse_record_output_file_ui();

    /**
     * @brief Sets the output filepath in the export manel with the name written
     * @param filename The name of the output file.
     */
    void set_record_file_name();

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
     * @brief Slot for handling changes in Z value from the ui.
     * @param z_distance The new Z distance value.
     */
    void z_value_changed(int z_distance);

    /*! \name Contrast settings
     * \{
     */
    /*!
     * \brief Enable or disable contrast
     *
     * \param enabled
     */
    void set_contrast_mode(bool enabled);

    /*!
     * \brief Set the min value for contrast
     *
     * \param value The new min value
     */
    void set_contrast_min(double value);

    /*!
     * \brief Set the contrast max value
     *
     * \param[in] value The new max value
     */
    void set_contrast_max(double value);

    /*!
     * \brief Enable or not the auto refresh
     *
     * \param[in] enabled
     */
    void set_contrast_auto_refresh(bool enabled);
    /*! \} */

    /*! \name Camera settings
     * \{
     */

    /*!
     * \brief Change the camera to the one in parameter
     *
     * \param[in] camera The camera to load
     */
    void change_camera(CameraKind camera);

    /*! \brief Button NONE in the UI, unload any loaded camera */
    void camera_none();

    /*! \brief Load the 710 Phantom camera */
    void camera_phantom();

    /*! \brief Load the 711 Phantom camera */
    void camera_ametek_s711_coaxlink_qspf_plus();

    /*! \brief Load the 991 Phantom camera */
    void camera_ametek_s991_coaxlink_qspf_plus();

    /*! \brief the Apply setting button in the UI. Load the settings file of the current camera */
    void configure_camera();

    /*! \brief Changes the recorded eye according to the current one */
    void update_recorded_eye();

    /*! \brief Sets the recorded eye to None */
    void reset_recorded_eye();

  protected:
    /**
     * @brief Overridden closeEvent handler.
     * @param event The QCloseEvent instance.
     */
    void closeEvent(QCloseEvent* event) override;

  private:
    Ui::LightUI* ui_;                                           ///< Pointer to the UI instance.
    MainWindow* main_window_;                                   ///< Pointer to the MainWindow instance.
    bool visible_;                                              ///< Boolean to track the visibility state of the UI.
    Subscriber<RecordMode> record_start_subscriber_;            ///< Subscriber for record start events.
    Subscriber<RecordMode> record_end_subscriber_;              ///< Subscriber for record end events.
    Subscriber<bool> record_finished_subscriber_;               ///< Subscriber for record finished events.
    Subscriber<RecordProgressData> record_progress_subscriber_; ///< Subscriber for record progress events.
    Subscriber<const std::filesystem::path>
        record_output_file_subscriber_; ///< Subscriber for record output file path events.
    Subscriber<RecordBarColorData>
        record_progress_bar_color_subscriber_; ///< Subscriber for record progress bar color events.
};

} // namespace holovibes::gui

#endif // LIGHTUI_HH
