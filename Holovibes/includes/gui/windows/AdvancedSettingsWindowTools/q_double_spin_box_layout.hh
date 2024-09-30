/*! \file
 *
 * \brief Qt widget embedded into layout.
 */
#pragma once

#include "q_spin_box_layout.hh"
#include <QDoubleSpinBox>

namespace holovibes::gui
{
/*! \class QDoubleSpinBoxLayout
 *
 * \brief Specialization of QSpinBoxLayout with double value.
 */
class QDoubleSpinBoxLayout : public QSpinBoxLayout
{
    Q_OBJECT

  public:
    QDoubleSpinBoxLayout(QMainWindow* parent = nullptr, const std::string& name = "");
    ~QDoubleSpinBoxLayout();

    /*! \brief Sets the Value object
     *
     * \param value the new value
     * \return QDoubleSpinBoxLayout* this, for linked initilizer purposes
     */
    virtual QDoubleSpinBoxLayout* set_value(double value);

    /*! \brief Sets lower bound value
     *
     * \param minimum_value the new lower bound
     * \return QDoubleSpinBoxLayout* this, for linked initializer purposes
     */
    QDoubleSpinBoxLayout* set_minimum_value(double minimum_value);

    /*! \brief Sets upper bound value
     *
     * \param maximum_value the new upper bound
     * \return QDoubleSpinBoxLayout* this, for linked initializer purposes
     */
    QDoubleSpinBoxLayout* set_maximum_value(double maximum_value);

    /*! \brief Sets single step value
     *
     * \param step the new step
     * \return QDoubleSpinBoxLayout* this, for linked initializer purposes
     */
    QDoubleSpinBoxLayout* set_single_step(double step);

    /*! \brief Sets the number of decimals
     *
     * \param precision the new number of decimals
     * \return QDoubleSpinBoxLayout* this, for linked initializer purposes
     */
    QDoubleSpinBoxLayout* set_decimals(int precision);

    /*! \brief Gets the value of spin box
     *
     * \return double: the new value
     */
    double get_value();

  signals:
    /*! \brief Calls when spin box is spinned */
    void value_changed();

  protected:
    QDoubleSpinBox* spin_box_;
};
} // namespace holovibes::gui
