#ifndef CAMERA_PCO_EDGE_HH
# define CAMERA_PCO_EDGE_HH

# include "../CameraPCO/camera_pco.hh"

# include <Windows.h>
# include <SC2_SDKStructures.h>
# include <SC2_CamExport.h>

namespace camera
{
  class CameraPCOEdge : public CameraPCO
  {
  public:
    CameraPCOEdge();
    virtual ~CameraPCOEdge();

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    /* Custom camera parameters. */
    
    /*! * 0x0000: auto trigger.
     *  * 0x0001: software trigger.
     *  * 0x0002: extern exposure & software trigger.
     *  * 0x0003: extern exposure control.
     */
    WORD triggermode_;

    /* Number of images recorded per millisecond.
    ** Please note that in the .ini file, fps are expressed
    ** in frames per second (less awkward).
    */ 
    DWORD framerate_;
    // Computation mode (adjusting between exposure time and fps)
    WORD framerate_mode_;

    /* Binning mode. Binning value can be 1 pixel (no binning),
    ** or 2 or 4 pixels, for each dimension; e.g. binning of 2x4
    ** means aggregating 2 pixels horizontally and 4 vertically
    ** to obtain one pixel in the output image.
    */
    WORD hz_binning_;
    WORD vt_binning_;

    /* Region Of Interest (ROI) selection.
    ** The ROI shall be no greater than the camera's frame dimensions.
    ** Using a ROI speeds up image readout by limiting the number of pixels
    ** which need to be output, but reduces accuracy.
    **
    ** Points p0 and p1 define the rectangle zone of the ROI, like this :
    **
    **   (p0) *---------------------*
    **        |                     |
    **        |                     |
    **        |                     |
    **        *---------------------* (p1)
    **
    ** Please note that ROI settings depend on binning and
    ** sensor format.
    ** e.g. binning of 2x2 on a 1600x1200 image makes maximum
    ** horizontal value to 800 and vertical to 600.
    */
    WORD p0_x_;
    WORD p0_y_;

    WORD p1_x_;
    WORD p1_y_;

    /* Also called pixel clock frequency.
    ** Frequency in Hertz determining image readout frequency.
    ** For available values, please  see the camera's manual.
    ** In the .ini file, this is expressed in MHz for commodity.
    */
    DWORD pixel_rate_;

    /* Conversion factor : the amount of electrons per "count"
    ** when interpreting electron quantity as light intensity.
    ** In our context (interferometry), a large dynamic range is
    ** already provided (the camera is very exposed to light),
    ** so this value is very low.
    */
    WORD conversion_factor_;

    /* Timeout configurations
    ** The SDK handles this with an array of size 3.
    ** (the third parameter only concerns Firewire interfaces,
    ** so we don't care about it).
    ** [0] Command timeout
    ** [1] Image request timeout
    **
    ** After x milliseconds without a response from the camera,
    ** abort command execution / image acquisition request.
    ** Please note that in the .ini file durations are expressed
    ** in seconds for commodity.
    */
    unsigned int timeouts_[3];

    /* Hardware input/output configuration
    ** The Edge 4.2 LT model comprises 4 I/O hardware links.
    ** Each set of options for one I/O interface corresponds to
    ** a PCO_Signal (see sc2_SDKStructures.h) struct.
    */
    PCO_Signal io_0_conf;
    PCO_Signal io_1_conf;
    PCO_Signal io_2_conf;
    PCO_Signal io_3_conf;
  };
}

#endif /* !CAMERA_PCO_EDGE_HH */