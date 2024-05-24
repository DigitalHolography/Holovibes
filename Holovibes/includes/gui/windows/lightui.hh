#ifndef LIGHTUI_HH
#define LIGHTUI_HH

#include <QMainWindow>

#include "export_panel.hh"

namespace Ui
{
class LightUI;
} // namespace Ui


namespace holovibes::gui
{
class MainWindow;

class LightUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit LightUI(QWidget *parent = nullptr, MainWindow* main_window = nullptr, ExportPanel* export_panel = nullptr);
    ~LightUI();

public slots:
    /*! \brief Opens file explorer on the fly to let the user chose the output file he wants with extension
     * replacement*/
    void browse_record_output_file_ui();

    void open_configuration_ui();

    /*! \brief Start/Stops the record */
    void start_stop_recording(bool start);

private:
    Ui::LightUI *ui_;
    MainWindow* main_window_;
    ExportPanel* export_panel_;
};
}

#endif // LIGHTUI_HH
