#pragma once

#include "QSpinBoxLayout.hh"
#include <QSpinBox>

namespace holovibes::gui
{
class QIntSpinBoxLayout : public QSpinBoxLayout
{
  public:
    QIntSpinBoxLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr, const std::string& name = "");
    ~QIntSpinBoxLayout();

    QIntSpinBoxLayout* setValue(int default_value);

  private:
    QSpinBox* spin_box_;
};
} // namespace holovibes::gui