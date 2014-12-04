#ifndef CAMERA_PCO_EDGE_HH
# define CAMERA_PCO_EDGE_HH

# include "camera_pco.hh"

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

    /*! Format of sensor. The standard format uses only effective pixels,
     * while the extended format shows all pixels inclusive effective.
     */
    bool extended_sensor_format_;

    /* Frequency for shifting the pixels out of the sensor shift registers.
     * The pixel clock sets the clock frequency and therefore the image sensor
     * readout speed. At 12 MHz the image quality will be higher due to very
     * low readout noise. At 25 MHz the image sensor is read out with nearly
     * double speed, achieving higher frame rates. The pixel_rate_ field unit
     * is in MHz.
     */
    unsigned int pixel_rate_;

    /*! Binning combines neighboring pixels to form super pixels.
     * It increases the light signal of the resulting pixels and decreases the
     * spatial resolution of the total image.
     * The binning_ field enables a x2 square binning.
     */
    bool binning_;
  };
}

#endif /* !CAMERA_PCO_EDGE_HH */