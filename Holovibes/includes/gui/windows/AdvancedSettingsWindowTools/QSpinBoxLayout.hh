/*! \file
 *
 * \brief Qt widget embeded into layout.
 */
#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>

namespace holovibes::gui
{
/*! \class QSpinBoxLayout
 *
 * \brief Enhancement of Qt QSpinBox display.
 */
class QSpinBoxLayout : public QHBoxLayout
{
  protected:
    /*! \brief QSpinBoxLayout object constructor
     *
     * \param parent_widget the object that will embed the object
     * \param name the name to display for the created QGroupBox
     */
    QSpinBoxLayout(QWidget* parent_widget = nullptr, const std::string& name = "");

    /*! \brief QSpinBoxLayout object desctructor */
    ~QSpinBoxLayout();

  public:
    /*! \brief Sets the Label object
     *
     * \param name the new name
     */
    void setLabel(const std::string& name);

  private:
    QLabel* label_;
};
} // namespace holovibes::gui