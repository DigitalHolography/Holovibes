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
     * \param parent the window that will embed the object
     * \param name the name to display beside the line edit
     */
    QPathSelectorLayout(QMainWindow* parent = nullptr, const std::string& name = "");

    /*! \brief QPathSelectorLayout object destructor */
    ~QPathSelectorLayout();

    /*! \brief Sets the selected path into the line edit box
     *
     * \param text the new text
     * \return QPathSelectorLayout* this, for linked initilizer purposes
     */
    QPathSelectorLayout* set_text(const std::string& text);

    /*! \brief Sets the name/label of the layout
     *
     * \param name the new name
     * \return QPathSelectorLayout* this, for linked initilizer purposes
     */
    QPathSelectorLayout* set_name(const std::string& name);

    /*! \brief Get the line edit's text
     *
     * \return const std::string the new text
     */
    const std::string get_text();

  private slots:
    /*! \brief Calls on new path selection*/
    void change_folder();

  signals:
    /*! \brief Triggered when the folder has actually changed */
    void folder_changed();

  private:
    QLabel* label_;
    QLineEdit* line_edit_;
    QToolButton* browse_button_;
};
} // namespace holovibes::gui