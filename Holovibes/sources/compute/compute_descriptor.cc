#include "compute_descriptor.hh"
#include "user_interface_descriptor.hh"

#include "holovibes.hh"
#include "tools.hh"
#include "API.hh"

namespace holovibes
{
using LockGuard = std::lock_guard<std::mutex>;

ComputeDescriptor::ComputeDescriptor()
    : Observable()
{
}

ComputeDescriptor::~ComputeDescriptor() {}

} // namespace holovibes
