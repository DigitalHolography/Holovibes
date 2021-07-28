#pragma once

#include <BFType.h>
#include <BiDef.h>
#include <camera.hh>

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

    size_t nb_boards = 1; //!< Number of boards to use  (1, 2, 4)
    Bd boards[4];         //!< Handle to the opened BitFlow board.
    BIBA buf_arrays[4];   //!< BufArray containing Bitflow related data

    BFU32 bitmap_size;      //!< Size of 1 frame in bytes
    BFU32 nb_buffers = 256; //!< Number of allocated buffers (frames)
    BFU32 total_mem_size;   //!< nb_buffers * bitmap_size + PAGE_SIZE
    PBFU32* frames;         //!< Array of pointers to the beginning of frames
    PBFU32 data;            //!< Frame data
    CiSIGNAL eod_signal;    //!< Frame interrupt
    BFRC RV;                //!< Bitflow API calls return value

    BFU32 captured = 0;     //!< Total number of captured images
    BFU32 old_captured = 0; //!< previous total number of captured images
};
} // namespace camera
