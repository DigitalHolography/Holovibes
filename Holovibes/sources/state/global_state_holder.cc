#include "global_state_holder.hh"

#include "holovibes.hh"

namespace holovibes
{

GSH& GSH::instance()
{
    static GSH instance_;
    return instance_;
}

void GSH::batch_command(entities::BatchCommand cmd)
{
    if (cmd.batch_size > Holovibes::instance().get_cd().input_buffer_size)
        cmd.batch_size = Holovibes::instance().get_cd().input_buffer_size.load();

    if (Holovibes::instance().get_cd().time_transformation_stride < cmd.batch_size)
        Holovibes::instance().get_cd().set_time_transformation_stride(cmd.batch_size);
    // Go to lower multiple
    if (Holovibes::instance().get_cd().time_transformation_stride % cmd.batch_size != 0)
        Holovibes::instance().get_cd().set_time_transformation_stride(
            Holovibes::instance().get_cd().time_transformation_stride -
            Holovibes::instance().get_cd().time_transformation_stride % cmd.batch_size);

    batch_cache.set_batch_size(cmd.batch_size);
}

void GSH::load_ptree(const boost::property_tree::ptree& ptree)
{
    batch_cache.set_batch_size(ptree.get<uint>("image_rendering.batch_size", 1));
}

void GSH::dump_ptree(boost::property_tree::ptree& ptree) const
{
    ptree.put<uint>("image_rendering.batch_size", batch_cache.get_batch_size());
}

} // namespace holovibes
