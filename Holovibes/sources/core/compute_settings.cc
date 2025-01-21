/*! \file
 *
 * \brief Contains functions related to the compute settings
 *
 */

#include "enum_theme.hh"
#include "internals_struct.hh"
#include "compute_settings_struct.hh"
#include "compute_settings.hh"
#include <iomanip>
#include <spdlog/spdlog.h>
#include "API.hh"

#include "logger.hh"

namespace holovibes::api
{

/*! \brief Merge two json objects
 *
 * \param[in] base_json The json object to merge into
 * \param[in] update_json The json object to merge from
 */
void merge_json(json& base_json, const json& update_json)
{
    for (auto& [key, value] : update_json.items())
    {
        if (!base_json.contains(key))
            throw std::runtime_error("Error: Key '" + key + "' found in update_json but not in base_json");

        if (base_json[key].is_object() && value.is_object())
            merge_json(base_json[key], value);
        else
            base_json[key] = value;
    }
}

void ComputeSettingsApi::load_compute_settings(const std::string& json_path) const
{
    LOG_FUNC(json_path);
    if (json_path.empty())
    {
        LOG_WARN("Configuration file not found.");
        return;
    }

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    auto compute_settings = ComputeSettings();
    compute_settings.Update();
    json old_one;
    to_json(old_one, compute_settings);

    try
    {
        // this allows a compute_settings to not have all the fields and to still work
        merge_json(old_one, j_cs);
        from_json(old_one, compute_settings);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("{} is an invalid compute settings", json_path);
        throw std::exception(e);
    }

    compute_settings.Assert();
    compute_settings.Load();
    compute_settings.Dump("cli_load_compute_settings");

    LOG_INFO("Compute settings loaded from : {}", json_path);
}

void ComputeSettingsApi::import_buffer(const std::string& json_path) const
{

    LOG_FUNC(json_path);
    if (json_path.empty())
    {
        LOG_WARN("Configuration file not found.");
        return;
    }

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    auto advanced_settings = AdvancedSettings();

    auto buffer_settings = AdvancedSettings::BufferSizes();

    try
    {
        from_json(j_cs, buffer_settings);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("{} is an invalid buffer settings", json_path);
        throw std::exception(e);
    }

    buffer_settings.Assert();
    buffer_settings.Load();

    LOG_INFO("Buffer settings loaded from : {}", json_path);
}

// clang-format off

json ComputeSettingsApi::compute_settings_to_json() const
{
   auto compute_settings = ComputeSettings();
   compute_settings.Update();
   json new_footer;
   to_json(new_footer, compute_settings);
   return new_footer;
}

// clang-format on

void ComputeSettingsApi::save_compute_settings(const std::string& json_path) const
{
    LOG_FUNC(json_path);

    if (json_path.empty())
        return;

    std::ofstream file(json_path);
    file << std::setw(1) << compute_settings_to_json();

    LOG_DEBUG("Compute settings overwritten at : {}", json_path);
}
} // namespace holovibes::api

namespace holovibes
{
inline static const std::filesystem::path dir(GET_EXE_DIR);

/*! \class JsonSettings
 *
 * \brief Struct that help with Json convertion
 *
 */
struct JsonSettings
{

    /*! \brief latest version of holo file version */
    inline static const auto latest_version = ComputeSettingsVersion::V5;

    /*! \brief path to json patch directories  */
    inline static const auto patches_folder = dir / "assets/json_patches_holofile";

    /*! \brief default convertion function */
    static void convert_default(json& data, const json& json_patch) { data = data.patch(json_patch); }

    /*! \brief convert holo file footer from version 3 to 4 */
    static void convert_v3_to_v4(json& data, const json& json_patch)
    {
        convert_default(data, json_patch);

        data["compute settings"]["image rendering"]["space transformation"] = static_cast<SpaceTransformation>(
            static_cast<int>(data["compute settings"]["image rendering"]["space transformation"]));
        data["compute settings"]["image rendering"]["image mode"] =
            static_cast<Computation>(static_cast<int>(data["compute settings"]["image rendering"]["image mode"]) - 1);
        data["compute settings"]["image rendering"]["time transformation"] = static_cast<TimeTransformation>(
            static_cast<int>(data["compute settings"]["image rendering"]["time transformation"]));
    }

    /*! \brief convert holo file footer from version 4 to 5 */
    static void convert_v4_to_v5(json& data, const json& json_patch)
    {
        if (data.contains("file info"))
        {
            data["info"] = data["file info"];
            data["info"]["input fps"] = 1;
            data["info"]["contiguous"] = 1;
        }

        convert_default(data, json_patch);

        if (data["compute_setting"]["view"]["image_type"] == "PHASEINCREASE")
        {
            data["compute_setting"]["view"]["image_type"] = "PHASE_INCREASE";
        }
        else if (data["compute_setting"]["view"]["image_type"] == "SQUAREDMODULUS")
        {
            data["compute_setting"]["view"]["image_type"] = "SQUARED_MODULUS";
        }
    }

    /*! \class ComputeSettingsConverter
     *
     * \brief Struct that contains all information to perform a convertion
     *
     */
    struct ComputeSettingsConverter
    {
        ComputeSettingsConverter(ComputeSettingsVersion from,
                                 ComputeSettingsVersion to,
                                 std::string patch_file,
                                 std::function<void(json&, const json&)> converter = convert_default)
            : from(from)
            , to(to)
            , patch_file(patch_file)
            , converter(converter)
        {
        }

        /*! \brief source version */
        ComputeSettingsVersion from;

        /*! \brief destination version */
        ComputeSettingsVersion to;

        /*! \brief patch file name */
        std::string patch_file;

        /*! \brief convertion function */
        std::function<void(json&, const json&)> converter;
    };

    /*! \brief vector that contains all available converters */
    inline static const std::vector<ComputeSettingsConverter> converters = {
        {ComputeSettingsVersion::V2, ComputeSettingsVersion::V3, "patch_v2_to_v3.json", convert_default},
        {ComputeSettingsVersion::V3, ComputeSettingsVersion::V4, "patch_v3_to_v4.json", convert_v3_to_v4},
        {ComputeSettingsVersion::V4, ComputeSettingsVersion::V5, "patch_v4_to_v5.json", convert_v4_to_v5},
    };
};

/*! \brief convert a json based on the source version
 *
 *
 * \param data: json footer
 * \param from: source version
 */
void ComputeSettings::convert_json(json& data, ComputeSettingsVersion from)
{
    auto it = std::find_if(JsonSettings::converters.begin(),
                           JsonSettings::converters.end(),
                           [=](auto converter) -> bool { return converter.from == from; });

    if (it == JsonSettings::converters.end())
        throw std::out_of_range("No converter found");

    std::for_each(it,
                  JsonSettings::converters.end(),
                  [&data](const JsonSettings::ComputeSettingsConverter& converter)
                  {
                      LOG_TRACE("Applying patch version v{}", static_cast<int>(converter.to) + 2);
                      std::ifstream patch_file{JsonSettings::patches_folder / converter.patch_file};
                      try
                      {
                          converter.converter(data, json::parse(patch_file));
                      }
                      catch (const std::exception&)
                      {
                      }
                  });
}
} // namespace holovibes
