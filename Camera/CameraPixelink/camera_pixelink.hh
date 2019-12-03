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
 * Camera Pixelink's. */
#pragma once

# include <Windows.h>

# include "pixeLINKApi.h"

# include <camera.hh>
# include <string>
# include <memory>

namespace camera
{
    class CameraPixelink : public Camera
    {
    public:
        CameraPixelink();

        virtual ~CameraPixelink();

        virtual void init_camera() override;
        virtual void start_acquisition() override;
        virtual void stop_acquisition() override;
        virtual void shutdown_camera() override;
        virtual void* get_frame() override;

    private:
        virtual void load_default_params() override;
        virtual void load_ini_params() override;
        virtual void bind_params() override;
        void allocate_data_buffer();
        void release_data_buffer();

        void query_camera_name();

    private:
        HANDLE device_;
        std::string name_;

        std::unique_ptr<unsigned short[]> data_buffer_;
        U32 buffer_size_;

        FRAME_DESC PxL_fd_;

        //At this point, just go read the detailed descriptions on Pixelink's website
        //https://support.pixelink.com/support/solutions/articles/3000044618-features
        float f_roi[4];
        float f_brightness;
        float f_frame_rate;
        float f_pixel_addressing[2];
        float f_pixel_format;
        float f_shutter;
        float f_trigger;
    };
}
