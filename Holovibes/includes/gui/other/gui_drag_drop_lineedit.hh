/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <QDragEnterEvent>
#include <QLineEdit>
#include <QObject>
#include <QWidget>

namespace holovibes::gui
{
/*! \class Drag_drop_lineedit
 *
 * \brief #TODO Add a description for this class
 */
class Drag_drop_lineedit : public QLineEdit
{
    Q_OBJECT
  public:
    Drag_drop_lineedit(QWidget* parent = nullptr);

  public slots:
    void dropEvent(QDropEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* e) override { e->acceptProposedAction(); }
};
} // namespace holovibes::gui
