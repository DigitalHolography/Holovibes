/*! \file
 *
 * \brief Qt widget embeded into layout.
 */
#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QToolbutton>

namespace holovibes::gui
{
/*! \class QPathSelectorLayout
 *
 * \brief Enhancement of path selector system display.
 */
class QPathSelectorLayout : public QHBoxLayout
{
    Q_OBJECT
  public:
    /*! \brief QPathSelectorLayout object constructor
     *
     * \param parent the object that will embed the layouts
     * \param parent_widget the object that will embed the object
     */
    QPathSelectorLayout(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);

    /*! \brief QPathSelectorLayout object destructor */
    ~QPathSelectorLayout();

    /*! \brief Sets the selected path into the line edit box
     *
     * \param text the new text
     * \return QPathSelectorLayout* this, for linked initilizer purposes
     */
    QPathSelectorLayout* setText(const std::string& text);

    /*! \brief Sets the name/label of the layout
     *
     * \param name the new name
     * \return QPathSelectorLayout* this, for linked initilizer purposes
     */
    QPathSelectorLayout* setName(const std::string& name);

  private slots:
    /*! \brief Calls on new path selection*/
    void change_folder();

  private:
    QLabel* label_;
    QLineEdit* line_edit_;
    QToolButton* browse_button_;
};
} // namespace holovibes::gui