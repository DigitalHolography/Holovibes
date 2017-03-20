/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * This class factorizes a bit code for PCO cameras. */
#pragma once

# include <iostream>
# include <array>
# include <Windows.h>
# include <PCO_err.h>
# include <PCO_errt.h>
# include <SC2_SDKStructures.h>
# include <SC2_CamExport.h>

# include <camera.hh>

namespace camera
{
  //!< Number of buffers to be handled by the camera.
  static const SHORT NBUFFERS = 2;

  /*! This class factorizes a bit code for PCO cameras.
   * Note that it is useless by itself; further derived classes are used. */
  class CameraPCO : public Camera
  {
  public:
    /*! The CameraPCO constructor factorizes buffer preparation. */
    CameraPCO(
      const std::string& ini_filepath,
      WORD camera_type);

    virtual ~CameraPCO();

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  protected:
    /* \brief Returns the actual x resolution. */
    WORD get_actual_res_x()
    {
      return actual_res_x_;
    }

    /* \brief Returns the actual y resolution. */
    WORD get_actual_res_y()
    {
      return actual_res_y_;
    }

  protected:
    HANDLE device_; //!< Pointer to the camera object provided by the SDK.

  private:
    //!< Get current frame dimensions at runtime.
    /*! \return The error status obtained. */
    int get_sensor_sizes();

    //!< Ask the camera to allocate appropriate buffers for acquisition.
    /*! \return The error status obtained. */
    int allocate_buffers();

  private:
    WORD camera_type_; //!< The camera's type id, as defined in the PCO SDK.

    std::array<HANDLE, NBUFFERS> buffers_events_;
    std::array<unsigned short*, NBUFFERS> buffers_;
    std::array<DWORD, NBUFFERS> buffers_driver_status_;

    DWORD buffer_size_; //!< Size of a buffer in bytes.

    //!\{
    /*! Current dimensions of the frame may be smaller than the maximum
     * due to downsampling and/or ROI. */
    WORD actual_res_x_;
    WORD actual_res_y_;
    //!\}
  };
}