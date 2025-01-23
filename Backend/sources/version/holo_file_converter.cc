#include "holo_file_converter.hh"

namespace holovibes::version
{

void HoloFileConverter::init()
{
    converters_ = {
        {2, "patch_v2_to_v3.json", convert_default},
        {3, "patch_v3_to_v4.json", convert_v3_to_v4},
        {4, "patch_v4_to_v5.json", convert_v4_to_v5},
        {5, "patch_v5_to_v6.json", convert_default},
        {6,
         "empty.json",
         convert_default}, // Version 6 was skipped because of a versioning error, it is considered the same as 5
    };
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
    int version = input.holo_file_header_.version;

    if (version == latest_version)
        return ApiCode::OK;

    if (version > latest_version)
    {
        LOG_ERROR("Holo file version v{} not supported, latest supported is v{}", version, latest_version);
        return ApiCode::FAILURE;
    }

    // Start at version 2 since before that there was no versioning
    if (version < 2)
        version = 2;

    while (version != latest_version)
    {
        auto it = std::find_if(converters_.begin(),
                               converters_.end(),
                               [=](auto converter) -> bool { return converter.version == version; });

        if (it == converters_.end())
        {
            LOG_ERROR("No holo file converter found for version {}", version);
            return ApiCode::FAILURE;
        }

        LOG_WARN("Applying holo file patch version v{} to v{}", version, version + 1);

        std::ifstream patch_file{patches_folder / it->patch_file};
        try
        {
            it->converter(input, input.meta_data_, json::parse(patch_file));
        }
        catch (const std::exception&)
        {
            LOG_ERROR("Failed to apply holo file patch v{}", version);
            return ApiCode::FAILURE;
        }

        version++;
    }

    // TODO(etienne): Needs to apply patch to the compute settings here

    return ApiCode::OK;
}

}; // namespace holovibes::version