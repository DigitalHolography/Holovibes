#include "camera_param_map.hh"

namespace camera
{

CameraParamMap::CameraParamMap(std::string prefix)
    : prefix_(prefix)
{
}

CameraParamMap::~CameraParamMap()
{
    for (const auto& [key, value] : map_)
        delete value;
}

bool CameraParamMap::has(const std::string key) const { return map_.contains(key); }

void CameraParamMap::set_from_ini(const boost::property_tree::ptree& pt)
{
    for (const auto& [key, value] : map_)
        value->set_from_ini(pt);
}

} // namespace camera