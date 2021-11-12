/*! \file
 *
 * \brief Qt window displaying the hologram in XY view.
 */
#pragma once

#include "icompute.hh"
#include "RawWindow.hh"

namespace holovibes
{
namespace gui
{
class MainWindow;
using SharedPipe = std::shared_ptr<ICompute>;

/*! \class HoloWindow
 *
 * \brief #TODO Add a description for this class
 */
class HoloWindow : public RawWindow
{
  public:
    HoloWindow(QPoint p,
               QSize s,
               ComputeDescriptor* cd,
               DisplayQueue* q,
               SharedPipe ic,
               std::unique_ptr<SliceWindow>& xz,
               std::unique_ptr<SliceWindow>& yz);
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
    std::unique_ptr<SliceWindow>& xz_slice_;
    std::unique_ptr<SliceWindow>& yz_slice_;
};
} // namespace gui
} // namespace holovibes
