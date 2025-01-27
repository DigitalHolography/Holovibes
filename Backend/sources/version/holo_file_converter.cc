#include "holo_file_converter.hh"

namespace holovibes::version
{

void HoloFileConverter::init()
{
    converters_ = {
        {2, "patch_v2_to_v3.json", convert_default},
        {3, "patch_v3_to_v4.json", convert_v3_to_v4},
        {4, "patch_v4_to_v5.json", convert_v4_to_v5},
        {5, "", convert_default},
        {6, "", convert_default},
    };

    // Version 6 was skipped because of a versioning error, it is considered the same as 5
    // Version 7 just added a new entry in the header, so the footer is the same as version 6
}

void HoloFileConverter::convert_default(io_files::InputHoloFile&, json& data, const json& json_patch)
{
    if (json_patch.empty())
        return;

    data = data.patch(json_patch);
}

void HoloFileConverter::convert_v3_to_v4(io_files::InputHoloFile& input, json& data, const json& json_patch)
{
    convert_default(input, data, json_patch);

    data["compute settings"]["image rendering"]["space transformation"] = static_cast<SpaceTransformation>(
        static_cast<int>(data["compute settings"]["image rendering"]["space transformation"]));
    data["compute settings"]["image rendering"]["image mode"] =
        static_cast<Computation>(static_cast<int>(data["compute settings"]["image rendering"]["image mode"]) - 1);
    data["compute settings"]["image rendering"]["time transformation"] = static_cast<TimeTransformation>(
        static_cast<int>(data["compute settings"]["image rendering"]["time transformation"]));
}

/*! \brief convert holo file footer from version 4 to 5 */
void HoloFileConverter::convert_v4_to_v5(io_files::InputHoloFile& input, json& data, const json& json_patch)
{
    if (data.contains("file info"))
    {
        data["info"] = data["file info"];
        data["info"]["input fps"] = 1;
        data["info"]["contiguous"] = 1;
    }

    convert_default(input, data, json_patch);

    if (data["compute_setting"]["view"]["image_type"] == "PHASEINCREASE")
        data["compute_setting"]["view"]["image_type"] = "PHASE_INCREASE";
    else if (data["compute_setting"]["view"]["image_type"] == "SQUAREDMODULUS")
        data["compute_setting"]["view"]["image_type"] = "SQUARED_MODULUS";
}

ApiCode HoloFileConverter::convert_holo_file(io_files::InputHoloFile& input)
{
    if (converters_.empty())
        init();

    int version = input.holo_file_header_.version;

    if (version == latest_version)
        return ApiCode::OK;

    if (version > latest_version)
    {
        LOG_ERROR("Holo file version {} not supported, latest supported is {}", version, latest_version);
        return ApiCode::FAILURE;
    }

    // Start at version 2 since before that there was no versioning
    if (version < 2)
        version = 2;

    LOG_INFO("Converting holo file from version {} to {}", version, latest_version);

    for (; version < latest_version; ++version)
    {
        auto it = std::find_if(converters_.begin(),
                               converters_.end(),
                               [=](auto converter) -> bool { return converter.version == version; });

        if (it == converters_.end())
        {
            LOG_ERROR("No holo file converter found for version {}", version);
            return ApiCode::FAILURE;
        }

        LOG_TRACE("Applying holo file patch version {} to {}", version, version + 1);

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
            it->converter(input, input.meta_data_, patch);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Failed to apply holo file patch for version {}: {}", version, e.what());
            return ApiCode::FAILURE;
        }
    }

    return ApiCode::OK;
}

}; // namespace holovibes::version