/*! \file compute_settings_converter.hh
 *
 * \brief Regroup functions related to the conversion of compute settings from different versions
 */
#pragma once

#include <string>
#include <functional>

#include "compute_settings.hh"
#include "enum_compute_settings_version.hh"
#include "enum_api_code.hh"

namespace holovibes::version
{

/*! \class ComputeSettingsConverterEntry
 *
 * \brief Struct that contains all information to perform a convertion
 */
struct ComputeSettingsConverterEntry
{
    ComputeSettingsConverterEntry(ComputeSettingsVersion version,
                                  std::string patch_file,
                                  std::function<void(json&, const json&)> converter)
        : version(version)
        , patch_file(patch_file)
        , converter(converter)
    {
    }

    /*! \brief Source version */
    ComputeSettingsVersion version;

    /*! \brief patch file name */
    std::string patch_file;

    /*! \brief convertion function */
    std::function<void(json&, const json&)> converter;
};

/*! \class ComputeSettingsConverter
 *
 * \brief Class that holds all conversion functions for compute settings
 */
class ComputeSettingsConverter
{
  public:
    inline static const ComputeSettingsVersion latest_version = ComputeSettingsVersion::V1;

  public:
    /*! \brief Convert a compute settings. If the version is not the latest, it will apply all patches to reach the
     * latest version. If the compute settings has no version, all patches will be applied.
     *
     * \param[in] input The current compute settings
     *
     * \return ApiCode the status of the operation
     */
    static ApiCode convert_compute_settings(json& input);

  private:
    /*! \brief List of all available converters */
    inline static std::vector<ComputeSettingsConverterEntry> converters_;

    /*! \brief path to json patch directories  */
    inline static const auto patches_folder = GET_EXE_DIR / "assets/json_patches_settings";

  private:
    /*! \brief Register all available converters */
    static void init();

    /*! \brief Default conversion function that applies a patch to the compute settings of the holo file
     *
     * \param[in] data       The current compute settings of the holo file
     * \param[in] json_patch The patch to apply to the compute settings
     */
    static void convert_default(json& data, const json& json_patch);
};
}; // namespace holovibes::version