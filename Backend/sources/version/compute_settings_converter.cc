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

ApiCode ComputeSettingsConverter::convert_compute_settings(json& input)
{
    if (converters_.empty())
        init();

    ComputeSettingsVersion version = ComputeSettingsVersion::NONE;
    if (!input.contains("version"))
        LOG_WARN("No version found in the compute settings file interpreting it as version 0");
    else
    {
        try
        {
            version = input["version"];
        }
        catch (const std::exception&)
        {
            std::string version_str = input["version"];
            LOG_ERROR("Unknown compute settings version found : {}. Latest supported is v{}",
                      version_str,
                      static_cast<int>(latest_version));
            return ApiCode::FAILURE;
        }
    }

    if (version == latest_version)
        return ApiCode::OK;

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

        LOG_WARN("Applying holo file patch version v{} to v{}", version_int, version_int + 1);

        if (!it->patch_file.empty())
        {
            std::filesystem::path path = patches_folder / it->patch_file;
            std::ifstream patch_file{patches_folder / it->patch_file};
            try
            {
                it->converter(input, json::parse(patch_file));
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("Failed to apply compute settings patch v{}: {}", version_int, e.what());
                return ApiCode::FAILURE;
            }
        }

        version = static_cast<ComputeSettingsVersion>(version_int + 1);
    }

    return ApiCode::OK;
}

}; // namespace holovibes::version