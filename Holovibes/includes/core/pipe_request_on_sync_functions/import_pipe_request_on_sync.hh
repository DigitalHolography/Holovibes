#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class ImportPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, Pipe&)
    {
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value, [[maybe_unused]] typename T::ConstRefType, Pipe& pipe)
    {
        operator()<T>(new_value, pipe);
    }

  public:
    template <>
    void on_sync<ImportType>(ImportTypeEnum new_value, ImportTypeEnum old_value, Pipe& pipe);
    template <>
    void operator()<ImportType>(ImportTypeEnum new_value, Pipe& pipe);

    template <>
    void on_sync<CurrentCameraKind>(CameraKind new_value, CameraKind old_value, Pipe& pipe);
    template <>
    void operator()<CurrentCameraKind>(CameraKind new_value, Pipe& pipe);
};
} // namespace holovibes
