/*! \file
 *
 * \brief Qt widget embeded into layout.
 */
#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QToolbutton>
#include "QLineEditLayout.hh"

namespace holovibes::gui
{
/*! \class QPathSelectorLayout
 *
 * \brief Enhancement of path selector system display.
 */
class QPathSelectorLayout : public QLineEditLayout
{
    Q_OBJECT

  public:
    QPathSelectorLayout(QMainWindow* parent = nullptr, const std::string& name = "");
    ~QPathSelectorLayout();

  private slots:
    /*! \brief Calls on new path selection*/
    void change_folder();

  protected:
    QToolButton* browse_button_;
};
} // namespace holovibes::gui