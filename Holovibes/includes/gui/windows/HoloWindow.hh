/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Qt window displaying the hologram in XY view. */
#pragma once

#include "icompute.hh"
#include "RawWindow.hh"

namespace holovibes
{
namespace gui
{
class MainWindow;
using SharedPipe = std::shared_ptr<ICompute>;

class HoloWindow : public RawWindow
{
  public:
    HoloWindow(QPoint p,
               QSize s,
               Queue* q,
               SharedPipe ic,
               std::unique_ptr<SliceWindow>& xy,
               std::unique_ptr<SliceWindow>& yy,
               MainWindow* main_window = nullptr);
    virtual ~HoloWindow();

    void update_slice_transforms();

    SharedPipe getPipe();

    void resetTransform() override;
    void setTransform() override;

  protected:
    SharedPipe Ic;

    virtual void initShaders() override;

    void focusInEvent(QFocusEvent* e) override;

  private:
    MainWindow* main_window_;

    std::unique_ptr<SliceWindow>& xz_slice_;
    std::unique_ptr<SliceWindow>& yz_slice_;
};
} // namespace gui
} // namespace holovibes
