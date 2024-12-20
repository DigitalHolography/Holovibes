/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Import panel
 */
#pragma once

#include "panel.hh"
#include "frame_desc.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class ImportPanel
 *
 * \brief Class representing the Import Panel in the GUI
 */
class ImportPanel : public Panel
{
    Q_OBJECT

  public:
    ImportPanel(QWidget* parent = nullptr);
    ~ImportPanel();

    void on_notify() override;

    void load_gui(const json& j_us) override;
    void save_gui(json& j_us) override;

    std::string& get_file_input_directory();

  public slots:
    /*! \brief Sets the start stop buttons object accessibility
     *
     * \param value accessibility
     */
    void set_start_stop_buttons(bool value);

    /*! \brief Opens file explorer to let the user chose the file he wants to import */
    void import_browse_file();

    /*! \brief Creates an input file to gather data from it.
     *
     * \param filename The chosen file
     */
    void import_file(const QString& filename);

    /*! \brief Setups attributes for launching and launchs the imported file */
    void import_start();
    /*! \brief Reset ui and stop holovibes' compute worker and file read worker */
    void import_stop();

    /*! \brief Handles the ui input fps changes
     */
    void update_fps();

    /*!
     * \brief Handles the update of the import file path in the UI.
     */
    void update_import_file_path();

    /*!
     * \brief Handles the update of the load file in GPU in the UI.
     *
     * \param enabled[in] Whether or not to enable loading the file in GPU
     */
    void update_load_file_in_gpu(bool enabled);

    /*!
     * \brief Handles the update of the load file in RAM in the UI.
     *
     * \param enabled[in] Whether or not to enable loading the file in RAM
     */
    void update_load_file_in_ram(bool enabled);

    /*!
     * \brief Handles the update of the input file start index in the UI.
     */
    void update_input_file_start_index();

    /*!
     * \brief Handles the update of the input file end index in the UI.
     */
    void update_input_file_end_index();
};
} // namespace holovibes::gui
