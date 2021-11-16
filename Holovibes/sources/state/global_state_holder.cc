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

    if (Holovibes::instance().get_cd().time_transformation_stride < cmd.value)
        Holovibes::instance().get_cd().set_time_transformation_stride(cmd.value);
    // Go to lower multiple
    if (Holovibes::instance().get_cd().time_transformation_stride % cmd.value != 0)
        Holovibes::instance().get_cd().set_time_transformation_stride(
            Holovibes::instance().get_cd().time_transformation_stride -
            Holovibes::instance().get_cd().time_transformation_stride % cmd.value);

    compute_cache.set_batch_size(cmd.value);
}

void GSH::time_transformation_size_command(entities::TimeTranformationSizeCommand cmd)
{
	compute_cache.set_time_transformation_size(cmd.value);
}

void GSH::load_ptree(const boost::property_tree::ptree& ptree)
{
    compute_cache.set_batch_size(ptree.get<uint>("image_rendering.batch_size", 1));
	compute_cache.set_time_transformation_size(
        std::max(ptree.get<ushort>("image_rendering.time_transformation_size", 1), 1));
}

void GSH::dump_ptree(boost::property_tree::ptree& ptree) const
{
    ptree.put<uint>("image_rendering.batch_size", compute_cache.get_batch_size());
    ptree.put<uint>("image_rendering.batch_size", compute_cache.get_time_transformation_size());
}

} // namespace holovibes
