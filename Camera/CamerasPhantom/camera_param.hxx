#pragma once

#include "camera_param.hh"

#include "spdlog/spdlog.h"

#include <string>

namespace camera
{

#define NB_MAX_GRABBER 4

template <typename T>
CameraParam<T>::CameraParam(T value, std::string name, std::string prefix)
    : value_(value)
    , name_(name)
    , prefix_(prefix)
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

template <typename T>
void CameraParam<T>::set_ini(const boost::property_tree::ptree& pt)
{
    value_ = pt.get<T>(prefix_ + "." + name_, value_);
}

void CameraParam<unsigned int[NB_MAX_GRABBER]>::set_ini(const boost::property_tree::ptree& pt)
{
    for (size_t i = 0; i < NB_MAX_GRABBER; ++i)
        value_[i] = pt.get<unsigned int>(prefix_ + "." + name_ + std::to_string(i), value_[i]);
}

} // namespace camera