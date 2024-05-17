#ifndef LIGHTUI_HH
#define LIGHTUI_HH

#include <QDialog>

namespace Ui
{
class LightUI;
} // namespace Ui


namespace holovibes::gui
{
class MainWindow;

class LightUI : public QDialog
{
    Q_OBJECT

public:
    explicit LightUI(QWidget *parent = nullptr, MainWindow* main_window = nullptr);
    ~LightUI();

private:
    Ui::LightUI *ui_;
    MainWindow* main_window_;
};
}

#endif // LIGHTUI_HH
