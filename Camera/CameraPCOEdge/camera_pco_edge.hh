#ifndef CAMERA_PCO_EDGE_HH
# define CAMERA_PCO_EDGE_HH

# include "../CameraPCO/camera_pco.hh"

# include <Windows.h>
# include <SC2_SDKStructures.h>
# include <SC2_CamExport.h>

namespace camera
{
  class CAMERA_API CameraPCOEdge : public CameraPCO
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
  };
}

#endif /* !CAMERA_PCO_EDGE_HH */