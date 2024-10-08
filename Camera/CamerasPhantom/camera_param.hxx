#pragma once

#include "camera_param.hh"

#include <string>
#include "camera.hh"

namespace camera
{

template <typename T>
CameraParam<T>::CameraParam(T value, std::string name, std::string prefix, bool is_inside_ini)
    : value_(value)
    , name_(name)
    , prefix_(prefix)
    , is_inside_ini_(is_inside_ini)
{
}

template <typename T>
T CameraParam<T>::get_value()
{
    return value_;
}

template <typename T>
void CameraParam<T>::set_value(T value)
{
    value_ = value;
}

template <>
inline void CameraParam<std::vector<unsigned int>>::set_from_ini(const boost::property_tree::ptree& pt)
{
    for (size_t i = 0; i < value_.size() && is_inside_ini_; ++i)
        value_[i] = pt.get<unsigned int>(prefix_ + "." + name_ + std::to_string(i), value_[i]);
}

template <typename T>
void CameraParam<T>::set_from_ini(const boost::property_tree::ptree& pt)
{

    if (is_inside_ini_)
        value_ = pt.get<T>(prefix_ + "." + name_, value_);
}

template <typename T>
std::optional<T> CameraParamMap::get(const std::string key) const
{
    if (!map_.contains(key))
        return {};

    auto* tmp = (*map_.find(key)).second;
    if (CameraParam<T>* p = dynamic_cast<CameraParam<T>*>(tmp); p)
        return {p->get_value()};
    return {};
}

template <typename T>
T& CameraParamMap::at(const std::string key) const
{
    auto opt = get<T>(key);
    assert(opt);
    return opt.value();
}
template <typename T>
void CameraParamMap::set(const std::string key, T value, bool is_inside_ini)
{
    if (!map_.contains(key))
        map_[key] = new CameraParam<T>(value, key, prefix_, is_inside_ini);
    else
    {
        CameraParam<T>* p = dynamic_cast<CameraParam<T>*>(map_[key]);
        assert(p);
        p->set_value(value);
    }
}

} // namespace camera