/*! \file
 *
 * \brief Qt widget embeded into layout.
 */
#pragma once

#include "QSpinBoxLayout.hh"
#include <QDoubleSpinBox>

namespace holovibes::gui
{
/*! \class QDoubleSpinBoxLayout
 *
 * \brief Specialization of QSpinBoxLayout with double value.
 */
class QDoubleSpinBoxLayout : public QSpinBoxLayout
{
  public:
    QDoubleSpinBoxLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr, const std::string& name = "");

    ~QDoubleSpinBoxLayout();

    /*! \brief Sets the Value object
     *
     * \param value the new value
     * \return QDoubleSpinBoxLayout* this, for linked initilizer purposes
     */
    QDoubleSpinBoxLayout* setValue(double value);

  private:
    QDoubleSpinBox* spin_box_;
};
} // namespace holovibes::gui