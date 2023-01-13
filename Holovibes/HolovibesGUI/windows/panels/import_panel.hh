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

    /*! \brief Handles the ui start index */
    void import_start_spinbox_update();

    /*! \brief Handles the ui end index*/
    void import_end_spinbox_update();

    /*! \brief Handles the ui input fps changes
     *
     * \param value the new value to set
     */
    void on_input_fps_change(int value);
};
} // namespace holovibes::gui
