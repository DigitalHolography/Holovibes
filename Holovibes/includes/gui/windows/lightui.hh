#ifndef LIGHTUI_HH
#define LIGHTUI_HH

#include <QDialog>

namespace Ui
{
class LightUI;
} // namespace Ui


namespace holovibes::gui
{

class LightUI : public QDialog
{
    Q_OBJECT

public:
    explicit LightUI(QWidget *parent = nullptr);
    ~LightUI();

private:
    Ui::LightUI *ui;
};
}

#endif // LIGHTUI_HH
