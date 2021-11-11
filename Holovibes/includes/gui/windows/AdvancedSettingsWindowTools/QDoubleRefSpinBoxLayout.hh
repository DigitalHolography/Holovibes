/*! \file
 *
 * \brief Qt widget embeded into layout linked to a referenced value.
 */
#pragma once

#include "QDoubleSpinBoxLayout.hh"

namespace holovibes::gui
{
/*! \class QDoubleRefSpinBoxLayout
 *
 * \brief Specialization of QDoubleSpinBoxLayout with double referenced value.
 */
class QDoubleRefSpinBoxLayout : public QDoubleSpinBoxLayout
{
    Q_OBJECT

  public:
    QDoubleRefSpinBoxLayout(QMainWindow* parent = nullptr, const std::string& name = "", double* value_ptr = nullptr);

    ~QDoubleRefSpinBoxLayout();

    /*! \brief Sets the Value object with performing state checks
     *
     * \param value the new value
     * \return QDoubleSpinBoxLayout* this, for linked initilizer purposes
     */
    QDoubleSpinBoxLayout* set_value(double value) override;

  private slots:
    /*! \brief Sets the attribute value to the spin box value */
    void refresh_value();

  protected:
    double* value_ptr_;
};
} // namespace holovibes::gui