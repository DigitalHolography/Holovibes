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
    if (cmd.value > Holovibes::instance().get_cd().input_buffer_size)
        cmd.value = Holovibes::instance().get_cd().input_buffer_size.load();

    if (compute_cache_.get_time_transformation_stride() < cmd.value)
        compute_cache_.set_time_transformation_stride(cmd.value);
    // Go to lower multiple
    if (compute_cache_.get_time_transformation_stride() % cmd.value != 0)
        compute_cache_.set_time_transformation_stride(compute_cache_.get_time_transformation_stride() -
                                                      compute_cache_.get_time_transformation_stride() % cmd.value);

    compute_cache_.set_batch_size(cmd.value);
}

void GSH::time_transformation_size_command(entities::TimeTransformationSizeCommand cmd)
{
    // FIXME: temporary fix due to ttsize change in pipe.make_request
    std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_time_transformation_size(cmd.value);
}

void GSH::time_transformation_stride_command(entities::TimeTransformationStrideCommand cmd)
{
    // cringe toi meme FIXME: temporary fix due to ttstride change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_time_transformation_stride(cmd.value);

    if (compute_cache_.get_batch_size() > cmd.value)
        compute_cache_.set_time_transformation_stride(compute_cache_.get_batch_size());
    // Go to lower multiple
    if (cmd.value % compute_cache_.get_batch_size() != 0)
        compute_cache_.set_time_transformation_stride(cmd.value - cmd.value % compute_cache_.get_batch_size());
}

entities::BatchQuery GSH::batch_query() const { return {compute_cache_.get_batch_size()}; }
entities::TimeTransformationSizeQuery GSH::time_transformation_size_query() const
{
    return {compute_cache_.get_time_transformation_size()};
}

entities::TimeTransformationStrideQuery GSH::time_transformation_stride_query() const
{
    return {compute_cache_.get_time_transformation_stride()};
}

void GSH::load_ptree(const boost::property_tree::ptree& ptree)
{
    compute_cache_.set_batch_size(ptree.get<uint>("image_rendering.batch_size", 1));
    compute_cache_.set_time_transformation_size(
        std::max<ushort>(ptree.get<ushort>("image_rendering.time_transformation_size", 1), 1));
    compute_cache_.set_time_transformation_stride(ptree.get<ushort>("image_rendering.time_transformation_stride", 1));

    filter2d_cache_.set_filter2d_n1(ptree.get<int>("image_rendering.filter2d_n1", 0));
    filter2d_cache_.set_filter2d_n2(ptree.get<int>("image_rendering.filter2d_n2", 1));
}

void GSH::dump_ptree(boost::property_tree::ptree& ptree) const
{
    ptree.put<uint>("image_rendering.batch_size", compute_cache_.get_batch_size());
    ptree.put<uint>("image_rendering.time_transformation_size", compute_cache_.get_time_transformation_size());
    ptree.put<ushort>("image_rendering.time_transformation_stride", compute_cache_.get_time_transformation_stride());
}

} // namespace holovibes
