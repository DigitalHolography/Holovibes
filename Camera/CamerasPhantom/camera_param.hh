#pragma once

#include <string>
#include <map>
#include "spdlog/spdlog.h"
#include "camera.hh"

namespace camera
{
class CameraParamInt
{
  public:
    virtual void set_from_ini(const boost::property_tree::ptree& pt) = 0;
};

template <typename T>
class CameraParam : public CameraParamInt
{
  public:
    CameraParam(T value, std::string name, std::string prefix, bool is_inside_ini = true);

    T get_value();
    void set_value(T value);

    void set_from_ini(const boost::property_tree::ptree& pt) override;

  private:
    T value_;
    std::string name_;
    std::string prefix_;
    bool is_inside_ini_;
};

// TODO comment
class CameraParamMap
{
  public:
    CameraParamMap(std::string prefix);

    ~CameraParamMap();

    bool has(const std::string key) const;

    template <typename T>
    std::optional<T> get(const std::string key) const;

    template <typename T>
    T at(const std::string key) const;

    template <typename T>
    void set(const std::string key, T value, bool is_inside_ini = true);

    void set_from_ini(const boost::property_tree::ptree& pt);

  private:
    std::string prefix_;
    std::map<const std::string, CameraParamInt*> map_;
};

} // namespace camera

#include "camera_param.hxx"
