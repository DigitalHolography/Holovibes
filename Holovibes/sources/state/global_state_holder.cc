#include "global_state_holder.hh"

#include "holovibes.hh"
#include "API.hh"

namespace holovibes
{
static inline const std::filesystem::path dir(get_exe_dir());
GSH::GSH()
    : cache_dispatcher_(advanced_cache_,
                        compute_cache_,
                        export_cache_,
                        composite_cache_,
                        filter2d_cache_,
                        view_cache_,
                        zone_cache_,
                        import_cache_,
                        file_read_cache_,
                        request_cache_)
{
    set_caches_as_refs();
}

void GSH::set_caches_as_refs()
{
    AdvancedCache::RefSingleton::set_main_ref(advanced_cache_);
    ComputeCache::RefSingleton::set_main_ref(compute_cache_);
    ExportCache::RefSingleton::set_main_ref(export_cache_);
    CompositeCache::RefSingleton::set_main_ref(composite_cache_);
    Filter2DCache::RefSingleton::set_main_ref(filter2d_cache_);
    ViewCache::RefSingleton::set_main_ref(view_cache_);
    ZoneCache::RefSingleton::set_main_ref(zone_cache_);
    ImportCache::RefSingleton::set_main_ref(import_cache_);
    FileReadCache::RefSingleton::set_main_ref(file_read_cache_);
    RequestCache::RefSingleton::set_main_ref(request_cache_);
}

GSH::~GSH() { remove_caches_as_refs(); }

void GSH::remove_caches_as_refs()
{
    AdvancedCache::RefSingleton::remove_main_ref(advanced_cache_);
    ComputeCache::RefSingleton::remove_main_ref(compute_cache_);
    ExportCache::RefSingleton::remove_main_ref(export_cache_);
    CompositeCache::RefSingleton::remove_main_ref(composite_cache_);
    Filter2DCache::RefSingleton::remove_main_ref(filter2d_cache_);
    ViewCache::RefSingleton::remove_main_ref(view_cache_);
    ZoneCache::RefSingleton::remove_main_ref(zone_cache_);
    ImportCache::RefSingleton::remove_main_ref(import_cache_);
    FileReadCache::RefSingleton::remove_main_ref(file_read_cache_);
    RequestCache::RefSingleton::remove_main_ref(request_cache_);
}

GSH& GSH::instance()
{
    static GSH instance_;
    return instance_;
}
/*! \class JsonSettings
 *
 * \brief Struct that help with Json convertion
 *
 */
struct JsonSettings
{

    /*! \brief latest version of holo file version */
    inline static const auto latest_version = GSH::ComputeSettingsVersion::V5;

    /*! \brief path to json patch directories  */
    inline static const auto patches_folder = dir / "json_patches_holofile";

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
    }

    /*! \class ComputeSettingsConverter
     *
     * \brief Struct that contains all information to perform a convertion
     *
     */
    struct ComputeSettingsConverter
    {
        ComputeSettingsConverter(GSH::ComputeSettingsVersion from,
                                 GSH::ComputeSettingsVersion to,
                                 std::string patch_file,
                                 std::function<void(json&, const json&)> converter = convert_default)
            : from(from)
            , to(to)
            , patch_file(patch_file)
            , converter(converter)
        {
        }

        /*! \brief source version */
        GSH::ComputeSettingsVersion from;

        /*! \brief destination version */
        GSH::ComputeSettingsVersion to;

        /*! \brief patch file name */
        std::string patch_file;

        /*! \brief convertion function */
        std::function<void(json&, const json&)> converter;
    };

    /*! \brief vector that contains all available converters */
    inline static const std::vector<ComputeSettingsConverter> converters = {
        {GSH::ComputeSettingsVersion::V2, GSH::ComputeSettingsVersion::V3, "patch_v2_to_v3.json", convert_default},
        {GSH::ComputeSettingsVersion::V3, GSH::ComputeSettingsVersion::V4, "patch_v3_to_v4.json", convert_v3_to_v4},
        {GSH::ComputeSettingsVersion::V4, GSH::ComputeSettingsVersion::V5, "patch_v4_to_v5.json", convert_v4_to_v5},
    };
};

/*! \brief convert a json based on the source version
 *
 *
 * \param data: json footer
 * \param from: source version
 */
void GSH::convert_json(json& data, GSH::ComputeSettingsVersion from)
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
                      LOG_TRACE(main, "Applying patch version v{}", static_cast<int>(converter.to) + 2);
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
