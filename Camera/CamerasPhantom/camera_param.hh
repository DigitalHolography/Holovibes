#pragma once

#include <string>

#include "camera_param.hxx"

namespace camera
{
class CameraParamInt
{
  public:
    virtual void set_ini(const boost::property_tree::ptree& pt) = 0;
};

template <typename T>
class CameraParam : public CameraParamInt
{
  public:
    CameraParam(T value, std::string name, std::string prefix);

    T get_value();
    void set_value(T value);

    void set_ini(const boost::property_tree::ptree& pt) override;

  private:
    T value_;
    std::string name_;
    std::string prefix_;
};

} // namespace camera