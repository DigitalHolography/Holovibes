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

#pragma once

#include <string>

namespace holovibes
{
	/*! \brief    Used to get information from .cine files
	*
	*   \details  Reads the header of a file and if it is a .cine file
    */
    class CineFile
    {
    public:
        /*! \brief   Used to store information about the images */
        // We had here a bug, if we swap the fields pixel size and pixel bits,
        // the value of pixel size change without any reason later in the execution.
        // Maybe a pointer does an illegal memory access and changes the value.
        struct ImageInfo
        {
			/*! Width of 1 image in pixels */
            int32_t img_width;
			/*! Height of 1 image in pixels */
            int32_t img_height;
            /*! Size of a pixel in micron */
            float pixel_size;
			/*! Number of bits in 1 pixel */
            uint16_t pixel_bits;
        };

        /*! \brief    Creates the singleton instance
        *
        *   \param    file_path   Path to the .cine file
        *
        *   \return   A pointer to the newly created instance,
        *             or nullptr if an error occured
        */
        static CineFile* new_instance(const std::string& file_path);

        /*! \brief    Retrieves the singleton instance
        *
        *   \return   A pointer to the singleton instance
        */
        static CineFile* get_instance();

        /*! \brief    Deletes the singleton instance */
        static void delete_instance();

		/*! \brief    Returns the current file's image information */
        const ImageInfo& get_image_info() const;

    private:
		/*! \brief    Creates a CineFile object from an existing file path and reads all of the required data
		*
		*   \param file_path Path of the .cine file to process
        */
        CineFile(const std::string& file_path);

        /*! \brief    Default destructor */
        ~CineFile() = default;

        /*! \brief    Default copy constructor */
        CineFile(const CineFile&) = default;

        /*! \brief    Default copy operator */
        CineFile& operator=(const CineFile&) = default;

		/*! Path of the .cine file */
        const std::string cine_file_path_;

		/*! Information related to the images of the cine file */
        ImageInfo image_info_;

		/*! True if there was no error while creating the instance */
		/*! If false, new_instance method deletes the instance and returns nullptr */
        bool is_valid_instance_ = false;

		/*! Singleton instance */
        static CineFile* instance;
    };
}
