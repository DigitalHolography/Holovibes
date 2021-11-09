/*! \file
 *
 * \brief Qt widget embeded into layout.
 */
#pragma once

#include "QSpinBoxLayout.hh"
#include <QSpinBox>

namespace holovibes::gui
{
/*! \class QIntSpinBoxLayout
 *
 * \brief Specialization of QSpinBoxLayout with int value.
 */
class QIntSpinBoxLayout : public QSpinBoxLayout
{
    Q_OBJECT

  public:
    QIntSpinBoxLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr, const std::string& name = "");
    ~QIntSpinBoxLayout();

    /*! \brief Sets the Value object
     *
     * \param value the new value
     * \return QIntSpinBoxLayout* this, for linked initilizer purposes
     */
    QIntSpinBoxLayout* setValue(int default_value);

  signals:
    /*! \brief Calls when spin box is spinned*/
    void value_changed();

  private:
    QSpinBox* spin_box_;
};
} // namespace holovibes::gui