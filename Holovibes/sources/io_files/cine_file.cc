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

#include "cine_file.hh"
#include "logger.hh"

namespace holovibes
{
    CineFile* CineFile::instance = nullptr;

    CineFile* CineFile::new_instance(const std::string& file_path)
    {
        if (instance != nullptr)
            delete instance;

        instance = new CineFile(file_path);

        if (!instance->is_valid_instance_)
            delete_instance();

        return instance;
    }

    CineFile* CineFile::get_instance()
    {
        if (instance == nullptr)
            LOG_WARN("CineFile instance is null (get_instance)");

        return instance;
    }

    void CineFile::delete_instance()
    {
        delete instance;
        instance = nullptr;
    }

    const CineFile::ImageInfo& CineFile::get_image_info() const
    {
        return image_info_;
    }

    CineFile::CineFile(const std::string& file_path)
        : cine_file_path_(file_path)
    {
        // TODO: perform more checks on the fields of the header

        std::ifstream file(file_path, std::ios::in | std::ios::binary);

        if (!file)
        {
            LOG_ERROR("Could not open file: " + file_path);
            return;
        }

        // check type field in the header
        // should always be equal to "CI"
        char type[2];
        file.read(reinterpret_cast<char*>(&type), sizeof(uint16_t));
        if (file.gcount() != sizeof(uint16_t) || strncmp("CI", type, 2) != 0)
        {
            LOG_ERROR("Invalid CINE type");
            return;
        }

        int32_t img_width = 0;
        int32_t img_height = 0;
        uint16_t pixel_bits = 0;
        int32_t x_pixels_per_meter = 0;

        // skip CINEFILEHEADER i.e. go to BITMAPINFOHEADER
        file.seekg(42, std::ios_base::cur);
        // skip biSize
        file.seekg(sizeof(uint32_t), std::ios_base::cur);
        // read biWidth
        file.read(reinterpret_cast<char*>(&img_width), sizeof(int32_t));
        // read biHeight
        file.read(reinterpret_cast<char*>(&img_height), sizeof(int32_t));
        // skip biPlanes
        file.seekg(sizeof(uint16_t), std::ios_base::cur);
        // read biBitCount
        file.read(reinterpret_cast<char*>(&pixel_bits), sizeof(uint16_t));
        // skip biCompression and biSizeImage
        file.seekg(2 * sizeof(uint32_t), std::ios_base::cur);
        // read biXPelsPerMeter
        file.read(reinterpret_cast<char*>(&x_pixels_per_meter), sizeof(int32_t));

        if (file.bad() || file.fail() || file.eof())
        {
            LOG_ERROR("The file is invalid or an error was encountered while reading the file");
            return;
        }

        // compute pixel size
        float pixel_size = (1 / static_cast<double>(x_pixels_per_meter)) * 1e6;

        image_info_ = {img_width, img_height, pixel_size, pixel_bits};
        is_valid_instance_ = true;
        LOG_INFO("Loaded cine file: " + file_path);
    }
}
