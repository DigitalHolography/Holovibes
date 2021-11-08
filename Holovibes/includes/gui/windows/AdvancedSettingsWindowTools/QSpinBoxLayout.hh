#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>

namespace holovibes::gui
{

class QSpinBoxLayout : public QHBoxLayout
{
  public:
    QSpinBoxLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr, const std::string& name = "");

    ~QSpinBoxLayout();

    void setLabel(const std::string& name);

  private:
    QLabel* label_;
};
} // namespace holovibes::gui