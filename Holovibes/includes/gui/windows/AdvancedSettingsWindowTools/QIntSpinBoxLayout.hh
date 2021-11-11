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
    QIntSpinBoxLayout(QMainWindow* parent = nullptr, const std::string& name = "");
    ~QIntSpinBoxLayout();

    /*! \brief Sets the Value object
     *
     * \param value the new value
     * \return QIntSpinBoxLayout* this, for linked initilizer purposes
     */
    QIntSpinBoxLayout* set_value(int default_value);

    /*! \brief Sets lower bound value
     *
     * \param minimum_value the new lower bound
     * \return QIntSpinBoxLayout* this, for linked initilizer purposes
     */
    QIntSpinBoxLayout* set_minimum_value(int minimum_value);

    /*! \brief Sets upper bound value
     *
     * \param maximum_value the new upper bound
     * \return QIntSpinBoxLayout* this, for linked initilizer purposes
     */
    QIntSpinBoxLayout* set_maximum_value(int maximum_value);

    /*! \brief Gets the value of spin box
     *
     * \return int: the new value
     */
    int get_value();

  signals:
    /*! \brief Calls when spin box is spinned*/
    void value_changed();

  private:
    QSpinBox* spin_box_;
};
} // namespace holovibes::gui