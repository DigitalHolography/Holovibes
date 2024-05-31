#ifndef LIGHTUI_HH
#define LIGHTUI_HH

#include <QMainWindow>

namespace Ui
{
class LightUI;
} // namespace Ui

namespace holovibes::gui
{
class MainWindow;
class ExportPanel;
class ImageRenderingPanel;

class LightUI : public QMainWindow
{
    Q_OBJECT

  public:
    explicit LightUI(QWidget* parent,
                     MainWindow* main_window,
                     ExportPanel* export_panel,
                     ImageRenderingPanel* image_rendering_panel);
    ~LightUI();
    void showEvent(QShowEvent* event) override;
    void actualise_record_output_file_ui(const QString& filename);
    void actualise_z_distance(const double z_distance);

  public slots:
    /*! \brief Opens file explorer on the fly to let the user chose the output file he wants with extension
     * replacement*/
    void browse_record_output_file_ui();

    void open_configuration_ui();

    /*! \brief Start/Stops the record */
    void start_stop_recording(bool start);
    void z_value_changed(int z_distance);

  private:
    Ui::LightUI* ui_;
    MainWindow* main_window_;
    ExportPanel* export_panel_;
    ImageRenderingPanel* image_rendering_panel_;
    bool visible_;
};
} // namespace holovibes::gui

#endif // LIGHTUI_HH
