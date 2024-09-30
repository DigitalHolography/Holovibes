/*! \file
 *
 * \brief Qt widget embedded into layout.
 */
#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include "logger.hh"

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
     * \param parent the window that will embed the object
     * \param name the name to display beside the spin box
     */
    QSpinBoxLayout(QMainWindow* parent = nullptr, const std::string& name = "");

    /*! \brief QSpinBoxLayout object desctructor */
    ~QSpinBoxLayout();

  public:
    /*! \brief Sets the Label object
     *
     * \param name the new name
     */
    void set_label(const std::string& name);

    void set_label_min_size(int width, int height);

  private:
    QLabel* label_;
};
} // namespace holovibes::gui
