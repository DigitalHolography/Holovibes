/*! \file
 *
 * \brief Qt window displaying the hologram in XY view.
 */
#pragma once

#include "display_queue.hh"

#include "RawWindow.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class HoloWindow
 *
 * \brief class that represents a hologram window in the GUI.
 */
class HoloWindow : public RawWindow
{
  public:
    HoloWindow(QPoint p,
               QSize s,
               DisplayQueue* q,
               std::unique_ptr<SliceWindow>& xz,
               std::unique_ptr<SliceWindow>& yz,
               float ratio);
    virtual ~HoloWindow();

    void update_slice_transforms();

    void resetTransform() override;
    void setTransform() override;

  protected:
    void initShaders() override;

    void focusInEvent(QFocusEvent* e) override;

  private:
    std::unique_ptr<SliceWindow>& xz_slice_;
    std::unique_ptr<SliceWindow>& yz_slice_;
};
} // namespace holovibes::gui
