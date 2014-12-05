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

  };
}

#endif /* !CAMERA_PCO_EDGE_HH */