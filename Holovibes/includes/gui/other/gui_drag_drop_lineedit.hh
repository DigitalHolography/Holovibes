/*! \file
 *
 * \brief declaration of the Drag_drop_lineedit class
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
 * \brief class that inherits from QLineEdit and allows drag and drop of files
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
