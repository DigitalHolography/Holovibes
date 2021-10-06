/*! \file */
#pragma once

#include "panel.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class ImportPanel */
class ImportPanel : public Panel
{
    Q_OBJECT

  public:
    ImportPanel(QWidget* parent = nullptr);
    ~ImportPanel();

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
    enum ImportType
    {
        None,
        Camera,
        File,
    };

    ImportType import_type_ = ImportType::None;

    MainWindow* parent_;
};
} // namespace holovibes::gui
