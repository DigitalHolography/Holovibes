#ifndef MAIN_WINDOW_HH_
# define MAIN_WINDOW_HH_

# include <cmath>
# include <thread>
# include <iomanip>
# include <QMainWindow>
# include <QFileDialog>
# include <QShortcut>
# include <QMessageBox>
# include <QDesktopServices>
# include <boost/filesystem.hpp>
# include <boost/property_tree/ptree.hpp>
# include <boost/property_tree/ini_parser.hpp>
# include <cstring>
# include <vector>
# include <sys/stat.h>

# include "camera_exception.hh"
# include "../GPIB/gpib_controller.hh"
# include "ui_main_window.h"
# include "holovibes.hh"
# include "pipe.hh"
# include "compute_descriptor.hh"
# include "observer.hh"
# include "gui_gl_window.hh"
# include "gui_plot_window.hh"
# include "thread_recorder.hh"
# include "thread_csv_record.hh"
# include "concurrent_deque.hh"
# include <string>

namespace gui
{
  /*! \class MainWindow
  **
  ** Main class of the GUI. It regroup most of the slots used for user actions.
  ** These slots are divided into several sections:
  **
  ** * Menu: every action in the menu (e-g: configuration of .ini, camera selection ...).
  ** * Image rendering: phase number, p, z, lambda ...
  ** * View: log scale, shifted corner, contrast ...
  ** * Special: image ratio, average/ROI plot ...
  ** * Record: record of raw frames, average/ROI file ...
  **
  **
  */
  class MainWindow : public QMainWindow, public holovibes::Observer
  {
    Q_OBJECT

  public:
    /*! \brief MainWindow constructor
    **
    ** Set keyboard shortcuts, set visibility and load default values for the fields.
    **
    ** \param holovibes holovibes object
    ** \param parent Qt parent (should be null as it is the GUI hierarchy top class)
    */
    MainWindow(holovibes::Holovibes& holovibes, QWidget *parent = 0);

    /*! \brief MainWindow destructor
    **
    ** Destroy respectively compute and capture threads.
    */
    ~MainWindow();

    void notify() override;

    public slots:
    /*! \{ \name Menu */
    /*! \brief Open holovibes configuration file */
    void configure_holovibes();
    /*! \brief Call child OpenGL widget fullscreen method */
    void gl_full_screen();
    /*! \brief Set camera to NONE
    **
    ** Delete GL widget, destroy capture and/or compute thread then
    ** set visibility to false.
    */
    void camera_none();
    /*! \brief Change camera to Adimec */
    void camera_adimec();
    /*! \brief Change camera to EDGE */
    void camera_edge();
    /*! \brief Change camera to IDS */
    void camera_ids();
    /*! \brief Change camera to iXon */
    void camera_ixon();
    /*! \brief Change camera to Pike */
    void camera_pike();
    /*! \brief Change camera to Pixelfly */
    void camera_pixelfly();
    /*! \brief Change camera to XIQ */
    void camera_xiq();
    /*! \brief Open camera configuration file */
    void configure_camera();
    /*! \brief Display program's credits */
    void credits();
    /* \} */

    /*! \{ \name Image rendering
    **
    ** Lots of these methods stick to the following scheme:
    **
    ** * Get pipe
    ** * Set visibility to false
    ** * Check if value is correct/into slot specific bounds
    ** * Update a value in FrameDescriptor of the holovibes object
    ** * Request a pipe refresh
    ** * Set visibility to true
    */
    /*! \brief Set image mode either to direct or hologram mode
    **
    ** Check if Camera has been enabled, then create a new GuiGLWindow keeping
    ** its old position and size if it was previously opened, set visibility
    ** and call notify().
    **
    ** \param value true for direct mode, false for hologram mode.
    */
    void set_image_mode(bool value);
    /*! \brief Set phase number (also called 'n' in papers)
    ** \param value new phase number
    */
    void set_phase_number(int value);
    /*! \brief Set p-th frame to be displayed in OpenGl window
    ** \param value new p
    */
    void set_p(int value);
    /*! \brief Increment p (useful for keyboard shortcuts) */
    void increment_p();
    /*! \brief Decrement p (useful for keyboard shortcuts) */
    void decrement_p();
    /*! \brief Set wavelength/lambda
    ** \param value wavelength/lambda value in meters
    */
    void set_wavelength(double value);
    /*! \brief Set z/distance to object
    ** \param value z in meters
    */
    void set_z(double value);
    /*! \brief Increment z (useful for keyboard shortcuts) */
    void increment_z();
    /*! \brief Decrement z (useful for keyboard shortcuts) */
    void decrement_z();
    /*! \brief Set z step (useful for keyboard shortcuts)
    ** \param new z step
    */
    void set_z_step(double value);
    /*! \brief Set algorithm
    ** \param value algorithm "1FFT" or "2FFT"
    */
    void set_algorithm(QString value);
    /*! \} */

    /*! \{ \name View */
    /*! \brief Set view mode
    ** \param value view mode: "magnitude", "squarred magnitude" or "argument"
    */
    void set_view_mode(QString value);
    /*! \brief Set autofocus mode on
    **
    ** Set GLWidget selection mode to AUTOFOCUS.
    ** Check and set values mandatory for autofocus computation then connect end of
    ** selection signal of OpenGl widget to request_autofocus() slot. Then whenever the
    ** user has finished its selection, the request will be called.
    */
    void set_autofocus_mode();
    /*! \brief Request autofocus action in pipeline
    ** \param zone selection zone for computation
    */
    void request_autofocus(holovibes::Rectangle zone);
    /*! \brief Request stft roi action in zone
    ** \param zone selection is region of interest
    */
    void request_stft_roi_update(holovibes::Rectangle zone);
    /*! \brief Indicate stft roi selection is finish
    ** \param zone selection is region of interest
    */
    void request_stft_roi_end();
    /*! \brief Request to stop the autofocus currently
    **  occuring.
    */
    void request_autofocus_stop();
    /*! \brief Enable or disable contrast mode
    ** \param value true to enable coontrast, false otherwise.
    */
    void set_contrast_mode(bool value);
    /*! \brief Request autocontrast action in pipe
    **
    ** It will automatically fill contrast minimum and maximum values.
    */
    void set_auto_contrast();
    /*! \brief Set contrast minimum value
    ** \param value new contrast minimum value
    */
    void set_contrast_min(double value);
    /*! \brief Set contrast maximum value
    ** \param value new contrast maximum value
    */
    void set_contrast_max(double value);
    /*! \brief Enable or disable logarithmic scale */
    void set_log_scale(bool value);
    /*! \brief Enable or diable shift corners algorithm */
    void set_shifted_corners(bool value);
    /*! \} */

    /*! \{ \name Special */
    /*! \brief Enable or disable vibrometry/image ratio mode */
    void set_vibro_mode(bool value);
    /*! \brief Set p-th frame for vibrometry mode
    ** \param value new p vibrometry
    */
    void set_p_vibro(int value);
    /*! \brief Set q-th frame for vibrometry mode
    ** \param value new q vibrometry
    */
    void set_q_vibro(int value);
    /*! \brief Enable or disable average/ROI mode */
    void set_average_mode(bool value);
    /*! \brief Plot average/ROI computations */
    void set_average_graphic();
    /*! \brief Dispose average/ROI computations */
    void dispose_average_graphic();

    /*! \brief Browse average/ROI zone file for load/save */
    void browse_roi_file();
    /*! \brief Save ROI zone to file */
    void save_roi();
    /*! \brief Load ROI zone from file */
    void load_roi();
    /*! \} */

    /*! Phase unwrapping + colored output. */
    void set_unwrap(bool value);

    /*! \{ \name Record */
    /*! \brief Browse image record output file */
    void browse_file();
    /*! \brief Launch image record
    **
    ** A ThreadRecord is used for image recording in order not to block the
    ** GUI during recording time. When the record is done, it calls finished_image_record().
    */
    void set_record();
    /*! \brief Destroy ThreadRecord cleanly */
    void finished_image_record();
    /*! \brief Browse ROI/average record output file */
    void browse_roi_output_file();
    /*! \brief Launch ROI/average record
    **
    ** A ThreadCSVRecord is used for average/ROI values recording not to block
    ** the GUI.
    ** Close the plot if opened previously before starting the record. Call
    ** finished_average_record() when done.
    */
    void average_record();
    /*! \brief Destroy ThreadCSVRecord cleanly */
    void finished_average_record();
    /*! \brief Browse batch instruction file */
    void browse_batch_input();
    /*! \brief Configure image batch record */
    void image_batch_record();
    /*! \brief Configure average/ROI batch record */
    void csv_batch_record();
    /*! \brief Launch batch record
    **
    ** Checks for errors, select which Queue to record, format path, execute
    ** GPIB first block then connect to batch_next_record.
    **
    ** \param path output mantissa
    */
    void batch_record(const std::string& path);
    /*! \brief Execute next batch record
    **
    ** Execute GPIB instruction and record thread alternatively until there is
    ** no more GPIB instructions.
    */
    void batch_next_record();
    /*! \brief Destroy batch record threads cleanly */
    void batch_finished_record();
    /*! \brief Stop image record */
    void stop_image_record();
    /*! \brief Stop average/ROI record */
    void stop_csv_record();

    /*! \brief Set import file src */
    void import_browse_file();
    /*! \brief Run thread_reader */
    void import_file();
    /*! \brief Stop thread_reader, and launch thread_capture */
    void import_file_stop();
    /*! \brief Update end to start if start > end */
    void import_start_spinbox_update();
    /*! \brief Update start to end if start < end */
    void import_end_spinbox_update();

    /*! \} */

  protected:
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    /*! \{ \name Visibility
    **
    ** All these methods are used to enable or disable parts of the GUI
    ** in order for the user not to do actions during the executions of
    ** computations or without having no camera.
    */
    void global_visibility(bool value);
    void camera_visible(bool value);
    void contrast_visible(bool value);
    void record_visible(bool value);
    void record_but_cancel_visible(bool value);
    void image_ratio_visible(bool value);
    void average_visible(bool value);
    void average_record_but_cancel_visible(bool value);
    /*! \} */

    /*! \brief Change camera
    **
    ** Delete real time OpenGL display window then destroy compute and
    ** capture thread cleanly then change camera type to the given one.
    **
    ** \param camera_type new camera type
    */
    void change_camera(holovibes::Holovibes::camera_type camera_type);
    /*! \brief Display error message
    ** \param msg error message
    */
    void display_error(std::string msg);
    /*! \brief Display information message
    ** \param msg information message
    */
    void display_info(std::string msg);
    /*! \brief Open a file
    ** \param path file path
    */
    void open_file(const std::string& path);
    /*! \brief Load holovibe configuration file */
    void load_ini(const std::string& path);
    /*! \brief Save holovibes configuration file */
    void save_ini(const std::string& path);
    /*! \brief Split a string each delim
    ** \param str string to split
    ** \param delim delimitor
    ** \param elts tokens vector
    */
    void split_string(const std::string& str, char delim, std::vector<std::string>& elts);
    /*! \brief Format batch output file name
    **
    ** Example:
    **
    ** * Path: test.txt
    ** * Index: 1
    ** * Result: test_000001.txt
    **
    ** \param path batch output file path
    ** \param index index of the file
    ** \return path with _index up to 10^6
    */
    std::string format_batch_output(const std::string& path, unsigned int index);

  private:
    Ui::MainWindow ui;
    /*! Reference to Holovibes object */
    holovibes::Holovibes& holovibes_;
    /*! OpenGL window */
    std::unique_ptr<GuiGLWindow> gl_window_;
    /*! true if in direct mode, false otherwise */
    bool is_direct_mode_;
    /*! true if a camera is loaded, false otherwise */
    bool is_enabled_camera_;
    /*! true if average mode is enabled, false otherwise */
    bool is_enabled_average_;
    /*! true if batch record is an image batch record, false if it is an average/ROI record */
    bool is_batch_img_;
    /*! true if batch has been interrupted by user, false otherwise */
    bool is_batch_interrupted_;
    /*! z step used for keyboard shortcuts */
    double z_step_;
    /*! current camera type */
    holovibes::Holovibes::camera_type camera_type_;

    /*! Plot/graphic window of average/ROI computations */
    std::unique_ptr<PlotWindow> plot_window_;

    /*! Image record thread */
    std::unique_ptr<ThreadRecorder> record_thread_;
    /*! ROI/average record thread */
    std::unique_ptr<ThreadCSVRecord> CSV_record_thread_;
    /*! Number of frames to record */
    unsigned int nb_frames_;

    /*! File index used in batch recording */
    unsigned int file_index_;

    std::unique_ptr<gpib::VisaInterface> gpib_interface_;

    /*! \brief Queue max size */
    unsigned int q_max_size_;

    /*! \{ \name Shortcuts */
    QShortcut* z_up_shortcut_;
    QShortcut* z_down_shortcut_;
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;
    QShortcut* gl_full_screen_;
    QShortcut* gl_normal_screen_;
    QShortcut* autofocus_ctrl_c_shortcut_;
    /*! \} */
  };
}

#endif /* !MAIN_WINDOW_HH_ */