#pragma once

#include "QSpinBoxLayout.hh"
#include <QDoubleSpinBox>

namespace holovibes::gui
{
class QDoubleSpinBoxLayout : public QSpinBoxLayout
{
  public:
    QDoubleSpinBoxLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr, const std::string& name = "");
    ~QDoubleSpinBoxLayout();

    QDoubleSpinBoxLayout* setValue(double default_value);

  private:
    QDoubleSpinBox* spin_box_;
};
} // namespace holovibes::gui