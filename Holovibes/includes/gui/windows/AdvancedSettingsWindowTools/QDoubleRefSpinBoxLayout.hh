#pragma once

#include "QDoubleSpinBoxLayout.hh"

namespace holovibes::gui
{
class QDoubleRefSpinBoxLayout : public QDoubleSpinBoxLayout
{
    Q_OBJECT

  public:
    QDoubleRefSpinBoxLayout(QWidget* parent_widget = nullptr,
                            const std::string& name = "",
                            double* value_ptr = nullptr);

    ~QDoubleRefSpinBoxLayout();

    QDoubleSpinBoxLayout* set_value(double value);

  private slots:
    void refresh_value();

  protected:
    double* value_ptr_;
};
} // namespace holovibes::gui