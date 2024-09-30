#include "worker.hh"

namespace holovibes::worker
{
void Worker::stop() { stop_requested_ = true; }
} // namespace holovibes::worker
