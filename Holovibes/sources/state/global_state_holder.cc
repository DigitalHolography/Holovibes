#include "global_state_holder.hh"

#include "holovibes.hh"

namespace holovibes
{

GSH& GSH::instance()
{
    static GSH instance_;
    return instance_;
}

#pragma region SETTERS

void GSH::set_batch_size(uint value)
{
    if (value > Holovibes::instance().get_cd().input_buffer_size)
        value = Holovibes::instance().get_cd().input_buffer_size.load();

    if (compute_cache_.get_time_transformation_stride() < value)
        compute_cache_.set_time_transformation_stride(value);
    // Go to lower multiple
    if (compute_cache_.get_time_transformation_stride() % value != 0)
        compute_cache_.set_time_transformation_stride(compute_cache_.get_time_transformation_stride() -
                                                      compute_cache_.get_time_transformation_stride() % value);

    compute_cache_.set_batch_size(value);
}

void GSH::set_time_transformation_size(uint value)
{
    // FIXME: temporary fix due to ttsize change in pipe.make_request
    std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_time_transformation_size(value);
}

void GSH::set_time_transformation_stride(uint value)
{
    // FIXME: temporary fix due to ttstride change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_time_transformation_stride(value);

    if (compute_cache_.get_batch_size() > value)
        compute_cache_.set_time_transformation_stride(compute_cache_.get_batch_size());
    // Go to lower multiple
    if (value % compute_cache_.get_batch_size() != 0)
        compute_cache_.set_time_transformation_stride(value - value % compute_cache_.get_batch_size());
}

void GSH::set_space_transformation(const SpaceTransformation& value) { compute_cache_.set_space_transformation(value); }
void GSH::set_space_transformation_from_string(const std::string& value)
{
    try
    {
        set_space_transformation(string_to_space_transform.at(value));
    }
    catch (const std::out_of_range&)
    {
        set_space_transformation(SpaceTransformation::None);
        LOG_ERROR << "Unknown space transform: " << value << ", falling back to None";
    }
}
void GSH::set_time_transformation(const TimeTransformation& value) { compute_cache_.set_time_transformation(value); }
void GSH::set_time_transformation_from_string(const std::string& value)
{
    try
    {
        set_time_transformation(string_to_time_transform.at(value));
    }
    catch (const std::out_of_range&)
    {
        set_time_transformation(TimeTransformation::NONE);
        LOG_ERROR << "Unknown time transform: " << value << ", falling back to None";
    }
}

void GSH::set_filter2d_n1(int value) { filter2d_cache_.set_filter2d_n1(value); }
void GSH::set_filter2d_n2(int value) { filter2d_cache_.set_filter2d_n2(value); }

#pragma endregion

#pragma region GETTERS

uint GSH::get_batch_size() const { return compute_cache_.get_batch_size(); }

uint GSH::get_time_transformation_size() const { return compute_cache_.get_time_transformation_size(); }

uint GSH::get_time_transformation_stride() const { return compute_cache_.get_time_transformation_stride(); }

SpaceTransformation GSH::get_space_transformation() const { return compute_cache_.get_space_transformation(); }
TimeTransformation GSH::get_time_transformation() const { return compute_cache_.get_time_transformation(); };

int GSH::get_filter2d_n1() const { return filter2d_cache_.get_filter2d_n1(); }

int GSH::get_filter2d_n2() const { return filter2d_cache_.get_filter2d_n2(); }

#pragma endregion

void GSH::load_ptree(const boost::property_tree::ptree& ptree)
{
    compute_cache_.set_batch_size(ptree.get<uint>("image_rendering.batch_size", 1));
    compute_cache_.set_time_transformation_size(
        std::max<ushort>(ptree.get<ushort>("image_rendering.time_transformation_size", 1), 1));
    compute_cache_.set_time_transformation_stride(ptree.get<ushort>("image_rendering.time_transformation_stride", 1));
    compute_cache_.set_space_transformation(static_cast<SpaceTransformation>(
        ptree.get<int>("image_rendering.space_transformation", static_cast<int>(SpaceTransformation::None))));
    compute_cache_.set_time_transformation(static_cast<TimeTransformation>(
        ptree.get<int>("image_rendering.time_transformation", static_cast<int>(TimeTransformation::STFT))));

    filter2d_cache_.set_filter2d_n1(ptree.get<int>("image_rendering.filter2d_n1", 0));
    filter2d_cache_.set_filter2d_n2(ptree.get<int>("image_rendering.filter2d_n2", 1));
}

// void GSH::load_advanced(const boost::property_tree::ptree& ptree) {

// }

void GSH::dump_ptree(boost::property_tree::ptree& ptree) const
{
    ptree.put<uint>("image_rendering.batch_size", compute_cache_.get_batch_size());
    ptree.put<uint>("image_rendering.time_transformation_size", compute_cache_.get_time_transformation_size());
    ptree.put<ushort>("image_rendering.time_transformation_stride", compute_cache_.get_time_transformation_stride());
    ptree.put<int>("image_rendering.space_transformation", static_cast<int>(compute_cache_.get_space_transformation()));
    ptree.put<int>("image_rendering.time_transformation", static_cast<int>(compute_cache_.get_time_transformation()));

    ptree.put<int>("image_rendering.filter2d_n1", filter2d_cache_.get_filter2d_n1());
    ptree.put<int>("image_rendering.filter2d_n2", filter2d_cache_.get_filter2d_n2());
}

} // namespace holovibes
