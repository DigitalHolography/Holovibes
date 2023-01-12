#pragma once

#include "logger.hh"
#include "compute_cache.hh"

namespace holovibes
{
class ImportGSHOnChange
{
  public:
    template <typename T>
    void operator()(typename T::ValueType&)
    {
    }

    template <typename T>
    bool change_accepted(typename T::ConstRefType)
    {
        return true;
    }

  public:
    template <>
    void operator()<ImportFrameDescriptor>(FrameDescriptor& new_value);
    template <>
    void operator()<ImportType>(ImportTypeEnum& new_value);
    template <>
    void operator()<ImportFilePath>(std::string& filename);
    template <>
    void operator()<CurrentCameraKind>(CameraKind& camera);
    template <>
    void operator()<StartFrame>(uint& new_value);
    template <>
    void operator()<EndFrame>(uint& new_value);
    template <>
    void operator()<FileNumberOfFrame>(uint& new_value);

    template <>
    bool change_accepted<StartFrame>(uint new_value);
    template <>
    bool change_accepted<EndFrame>(uint new_value);
    template <>
    bool change_accepted<FileNumberOfFrame>(uint new_value);
};
} // namespace holovibes
