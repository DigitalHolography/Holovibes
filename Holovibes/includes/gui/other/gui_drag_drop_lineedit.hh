/*! \file
 *
 * \brief Defines the Drag_drop_lineedit class for handling drag and drop in a QLineEdit.
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
 * \brief Extends QLineEdit to support drag and drop functionality.
 *
 * This class allows a QLineEdit widget to accept dragged items and handle drop events,
 * enabling users to drag and drop content into the line edit.
 */
class Drag_drop_lineedit : public QLineEdit
{
    Q_OBJECT
  public:
    /*! \brief Constructor
     *
     * \param parent Pointer to the parent widget. Defaults to nullptr.
     */
    Drag_drop_lineedit(QWidget* parent = nullptr);

  public slots:
    /*! \brief Handles the drop event
     *
     * This method is called when an item is dropped onto the line edit.
     *
     * \param event Pointer to the QDropEvent.
     */
    void dropEvent(QDropEvent* event) override;

    /*! \brief Handles the drag enter event
     *
     * This method is called when a drag action enters the line edit's area.
     *
     * \param e Pointer to the QDragEnterEvent.
     */
    void dragEnterEvent(QDragEnterEvent* e) override { e->acceptProposedAction(); }
};
} // namespace holovibes::gui