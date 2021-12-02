#pragma once

#include <BFType.h>
#include <BiDef.h>
#include "camera.hh"

#include "camera_exception.hh"

namespace camera
{
class CameraPhantomBitflow : public Camera
{
  public:
    CameraPhantomBitflow();

    virtual ~CameraPhantomBitflow() {}

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    void open_boards();
    void create_buffers();
    BFU32 get_circ_options(size_t i);

    /*! \brief Number of boards to use  (1, 2, 4) */
    int nb_boards = 0;
    /*! \brief Board numbers to open */
    int board_nums[4];
    /*! \brief Handle to the opened BitFlow board. */
    Bd boards[4];
    /*! \brief BufArray containing Bitflow related data */
    BIBA buf_arrays[4];

    /*! \brief Size of 1 frame in bytes */
    BFU32 bitmap_size;
    /*! \brief Number of allocated buffers (frames) */
    BFU32 nb_buffers = 256;
    /*! \brief nb_buffers * bitmap_size + PAGE_SIZE */
    BFU32 total_mem_size;
    /*! \brief Array of pointers to the beginning of frames */
    PBFU32* frames;
    /*! \brief Frame data */
    PBFU32 data;
    /*! \brief Frame interrupt */
    CiSIGNAL eod_signal;
    /*! \brief Bitflow API calls return value */
    BFRC RV;

    /*! \brief Total number of captured images */
    BFU32 captured = 0;
    /*! \brief Previous total number of captured images */
    BFU32 old_captured = 0;
};
} // namespace camera
