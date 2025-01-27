#include "compute_settings_converter.hh"

#include "logger.hh"

namespace holovibes::version
{

void ComputeSettingsConverter::init()
{
    converters_ = {
        {ComputeSettingsVersion::NONE, "patch_v1.json", convert_default},
    };
}

void ComputeSettingsConverter::convert_default(json& data, const json& json_patch)
{
    if (json_patch.empty())
        return;
    data = data.patch(json_patch);
}

ApiCode ComputeSettingsConverter::convert_compute_settings(json& input)
{
    if (converters_.empty())
        init();

    ComputeSettingsVersion version = ComputeSettingsVersion::NONE;
    if (!input.contains("version"))
        LOG_WARN("No version found in the compute settings file interpreting it as version v0");
    else
    {
        try
        {
            version = input["version"];
        }
        catch (const std::exception&)
        {
            std::string version_str = input["version"];
            LOG_ERROR("Unknown compute settings version found : v{}. Latest supported is v{}",
                      version_str,
                      static_cast<int>(latest_version));
            return ApiCode::FAILURE;
        }
    }

    if (version == latest_version)
        return ApiCode::OK;

    LOG_INFO("Converting compute settings from version v{} to v{}",
             static_cast<int>(version),
             static_cast<int>(latest_version));

    while (version != latest_version)
    {
        int version_int = static_cast<int>(version);

        auto it = std::find_if(converters_.begin(),
                               converters_.end(),
                               [=](auto converter) -> bool { return converter.version == version; });

        if (it == converters_.end())
        {
            LOG_ERROR("No compute settings converter found for version v{}", version_int);
            return ApiCode::FAILURE;
        }

        LOG_TRACE("Applying compute settings patch version v{} to v{}", version_int, version_int + 1);

        json patch = {};
        if (!it->patch_file.empty())
        {
            try
            {
                std::ifstream patch_file{patches_folder / it->patch_file};
                patch = json::parse(patch_file);
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("File does not exist or is not a valid json file: {}. Error: {}", it->patch_file, e.what());
                return ApiCode::FAILURE;
            }
        }

        try
        {
            it->converter(input, patch);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Failed to apply compute settings patch for version {}: {}", version_int, e.what());
            return ApiCode::FAILURE;
        }

        version = static_cast<ComputeSettingsVersion>(version_int + 1);
    }

    return ApiCode::OK;
}

}; // namespace holovibes::version