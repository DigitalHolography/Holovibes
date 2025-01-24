/*! \file holo_file_converter.hh
 *
 * \brief Regroup functions related to the conversion of holovibes files header and footer from different versions
 */
#pragma once

#include <string>
#include <functional>

#include "compute_settings.hh"
#include "enum_api_code.hh"
#include "input_holo_file.hh"

namespace holovibes::version
{

/*! \class HoloSettingsConverter
 *
 * \brief Struct that contains all information to perform a convertion
 */
struct HoloSettingsConverter
{
    HoloSettingsConverter(int version,
                          std::string patch_file,
                          std::function<void(io_files::InputHoloFile&, json&, const json&)> converter)
        : version(version)
        , patch_file(patch_file)
        , converter(converter)
    {
    }

    /*! \brief Source version */
    int version;

    /*! \brief patch file name */
    std::string patch_file;

    /*! \brief convertion function */
    std::function<void(io_files::InputHoloFile&, json&, const json&)> converter;
};

/*! \class HoloFileConverter
 *
 * \brief Class that holds all conversion functions for holo files
 */
class HoloFileConverter
{
  public:
    inline static const int latest_version = 6;

  public:
    /*! \brief Convert a holo_file to the latest version. Version is deduced from the header of the file.
     *
     * \param[in] input The current holo file
     *
     * \return ApiCode the status of the operation
     */
    static ApiCode convert_holo_file(io_files::InputHoloFile& input);

  private:
    /*! \brief List of all available converters */
    inline static std::vector<HoloSettingsConverter> converters_;

    /*! \brief path to json patch directories  */
    inline static const auto patches_folder = GET_EXE_DIR / "assets/json_patches_holofile";

  private:
    /*! \brief Register all available converters */
    static void init();

    /*! \brief Default conversion function that applies a patch to the compute settings of the holo file
     *
     * \param[in] input      The current holo file
     * \param[in] data       The current compute settings of the holo file
     * \param[in] json_patch The patch to apply to the compute settings
     */
    static void convert_default(io_files::InputHoloFile& input, json& data, const json& json_patch)
    {
        data = data.patch(json_patch);
    }

    /*! \brief Convert holo file footer from version 3 to 4: Passing from int to enum for ComputeMode,
     * TimeTransformation and SpaceTransformation
     *
     * \param[in] input      The current holo file
     * \param[in] data       The current compute settings of the holo file
     * \param[in] json_patch The patch to apply to the compute settings
     */
    static void convert_v3_to_v4(io_files::InputHoloFile& input, json& data, const json& json_patch);

    /*! \brief Convert holo file footer from version 4 to 5: Renaming of image types and renaming info part
     *
     * \param[in] input      The current holo file
     * \param[in] data       The current compute settings of the holo file
     * \param[in] json_patch The patch to apply to the compute settings
     */
    static void convert_v4_to_v5(io_files::InputHoloFile& input, json& data, const json& json_patch);
};
}; // namespace holovibes::version