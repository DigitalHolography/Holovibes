/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Import panel
 */
#pragma once

#include "panel.hh"

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

    enum ImportType
    {
        None,
        Camera,
        File,
    };

    void on_notify() override;

    void load_ini(const boost::property_tree::ptree& ptree) override;
    void save_ini(boost::property_tree::ptree& ptree) override;

    ImportType get_import_type();
    void set_import_type(ImportType type);
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

    /*! \brief Sets ui values and constraints + launch FileReadWroker */
    void init_holovibes_import_mode();

    /*! \brief Setups attributes for launching and launchs the imported file */
    void import_start();
    /*! \brief Reset ui and stop holovibes' compute worker and file read worker */
    void import_stop();

    /*! \brief Handles the ui input fps */
    void import_start_spinbox_update();

    /*! \brief Handles the ui output fps */
    void import_end_spinbox_update();

  private:
    ImportType import_type_ = ImportType::None;

    std::string file_input_directory_;
};
} // namespace holovibes::gui
