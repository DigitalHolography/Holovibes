#ifndef PIXEFLY_HH
# define PIXEFLY_HH

#include "windows.h"
#include <iostream>
#include "camera.hh"
#include "sc2_SDKStructures.h"
#include "SC2_CamExport.h"
#include "pco_errt.h"

namespace camera
{
  class Pixefly : public Camera
  {
  public:
    enum endianness
    {
      BIG_ENDIAN,
      LITTLE_ENDIAN
    };

    Pixefly();

    virtual ~Pixefly()
    {
    }

    // Getters
    const std::string& get_name() const
    {
      return name_;
    }

    enum endianness get_endianness() const
    {
      return endianness_;
    }

    int get_frame_width() const
    {
      return frame_width_;
    }

    int get_frame_height() const
    {
      return frame_height_;
    }

    // Virtual methods
    virtual void init_camera();
    virtual void start_acquisition();
    virtual void stop_acquisition();
    virtual void shutdown_camera();
    void set_sensor();
    void check_error(DWORD error, std::string);
    int get_frame_size(); // should always be called after init_camera

    virtual void* get_frame();

    //internal methods
    void buff_alloc();

  protected:
    std::string name_;

  private:
    enum endianness endianness_;

    WORD frame_height_;
    WORD frame_width_;
    WORD max_frame_height_;
    WORD max_frame_width_;
    WORD bining_x_;
    WORD bining_y_;
    unsigned char frame_bit_depth_;
    HANDLE my_cam_;
    HANDLE refreshEvent_;
    WORD *frame_buffer_;
    bool inited_;
    bool internal_buff_alloc_;
    bool irSensitivityEnabled_;
    bool extendedSensorFormatEnabled_;
    bool acquiring_;
    DWORD buff_size;
    int fps_;
    float pixel_rate_;
    DWORD error_;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;
  };
}

#endif