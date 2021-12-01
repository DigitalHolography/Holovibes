#include "global_state_holder.hh"

#include "holovibes.hh"
#include "API.hh"
namespace holovibes
{

GSH& GSH::instance()
{
    static GSH* instance_ = nullptr;
    if (instance_ == nullptr)
        instance_ = new GSH();

    return *instance_;
}

#pragma region GETTERS

const View_Window& GSH::get_current_window() const
{
    switch (view_cache_.get_current_window())
    {
    case (WindowKind::XYview):
        return view_cache_.get_xy_const_ref();
    case (WindowKind::XZview):
        return view_cache_.get_xz_const_ref();
    case (WindowKind::YZview):
        return view_cache_.get_yz_const_ref();
    default: // case (WindowKind::Filter2D):
        return view_cache_.get_filter2d_const_ref();
    }
}

/* private */
View_Window& GSH::get_current_window()
{
    switch (view_cache_.get_current_window())
    {
    case (WindowKind::XYview):
        return view_cache_.get_xy_ref();
    case (WindowKind::XZview):
        return view_cache_.get_xz_ref();
    case (WindowKind::YZview):
        return view_cache_.get_yz_ref();
    default: // case (WindowKind::Filter2D):
        return view_cache_.get_filter2d_ref();
    }
}

bool GSH::is_current_window_xyz_type() const
{
    static const std::set<WindowKind> types = {WindowKind::XYview, WindowKind::XZview, WindowKind::YZview};
    return types.contains(view_cache_.get_current_window());
}

float GSH::get_contrast_min() const
{
    return get_current_window().log_scale_slice_enabled ? get_current_window().contrast_min
                                                        : log10(get_current_window().contrast_min);
}

float GSH::get_contrast_max() const
{
    return get_current_window().log_scale_slice_enabled ? get_current_window().contrast_max
                                                        : log10(get_current_window().contrast_max);
}

double GSH::get_rotation() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const View_XYZ&>(get_current_window());
    return w.rot;
}

bool GSH::get_flip_enabled() const
{

    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const View_XYZ&>(get_current_window());
    return w.flip_enabled;
}

bool GSH::get_img_log_scale_slice_enabled() const { return get_current_window().log_scale_slice_enabled; }

unsigned GSH::get_img_accu_level() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const View_XYZ&>(get_current_window());
    return w.img_accu_level;
}
#pragma endregion

#pragma region SETTERS

void GSH::set_batch_size(uint value)
{
    if (value > advanced_cache_.get_input_buffer_size())
        value = advanced_cache_.get_input_buffer_size();

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
    // std::lock_guard<std::mutex> lock(mutex_);
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

void GSH::set_convolution_enabled(bool value)
{
    // std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_convolution_enabled(value);
}

void GSH::set_contrast_enabled(bool contrast_enabled) { get_current_window().contrast_enabled = contrast_enabled; }

void GSH::set_contrast_auto_refresh(bool contrast_auto_refresh)
{
    get_current_window().contrast_auto_refresh = contrast_auto_refresh;
}

void GSH::set_contrast_invert(bool contrast_invert) { get_current_window().contrast_invert = contrast_invert; }

void GSH::set_contrast_min(float value)
{
    get_current_window().contrast_min = get_current_window().log_scale_slice_enabled ? value : pow(10, value);
}

void GSH::set_contrast_max(float value)
{
    get_current_window().contrast_max = get_current_window().log_scale_slice_enabled ? value : pow(10, value);
}

void GSH::set_log_scale_slice_enabled(bool value) { get_current_window().log_scale_slice_enabled = value; }

void GSH::set_accumulation_level(int value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    reinterpret_cast<View_XYZ&>(get_current_window()).img_accu_level = value;
}

void GSH::set_rotation(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<View_XYZ&>(get_current_window());
    w.rot = value;
}

void GSH::set_flip_enabled(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<View_XYZ&>(get_current_window());
    w.flip_enabled = value;
}

void GSH::set_fft_shift_enabled(bool value)
{
    view_cache_.set_fft_shift_enabled(value);
    api::pipe_refresh();
}

#pragma endregion

/*! \brief Change the window according to the given index */
void GSH::change_window(uint index) { view_cache_.set_current_window(static_cast<WindowKind>(index)); }

static void load_image_rendering(const boost::property_tree::ptree& ptree,
                                 ComputeCache::Ref& compute_cache_,
                                 Filter2DCache::Ref& filter2d_cache_)
{
    // Should we sey default value of load as chache default values
    compute_cache_.set_batch_size(ptree.get<uint>("image_rendering.batch_size", 1));
    compute_cache_.set_time_transformation_size(
        std::max<ushort>(ptree.get<ushort>("image_rendering.time_transformation_size", 1), 1));
    compute_cache_.set_time_transformation_stride(ptree.get<ushort>("image_rendering.time_transformation_stride", 1));
    compute_cache_.set_space_transformation(static_cast<SpaceTransformation>(
        ptree.get<int>("image_rendering.space_transformation", static_cast<int>(SpaceTransformation::NONE))));
    compute_cache_.set_time_transformation(static_cast<TimeTransformation>(
        ptree.get<int>("image_rendering.time_transformation", static_cast<int>(TimeTransformation::STFT))));
    compute_cache_.set_lambda(ptree.get<float>("image_rendering.lambda", 852e-9f));
    compute_cache_.set_z_distance(ptree.get<float>("image_rendering.z_distance", 1.50f));
    compute_cache_.set_convolution_enabled(ptree.get<bool>("image_rendering.convolution_enabled", false));
    compute_cache_.set_divide_convolution_enabled(ptree.get<bool>("image_rendering.divide_convolution_enabled", false));
    compute_cache_.set_compute_mode(
        static_cast<Computation>(ptree.get<int>("image_rendering.image_mode", static_cast<int>(Computation::Raw))));

    filter2d_cache_.set_filter2d_n1(ptree.get<int>("image_rendering.filter2d_n1", 0));
    filter2d_cache_.set_filter2d_n2(ptree.get<int>("image_rendering.filter2d_n2", 1));
}

void xyzf_load(const boost::property_tree::ptree& ptree, const std::string name, View_Window& view)
{
    view.log_scale_slice_enabled = ptree.get<bool>("view." + name + "_log_scale_enabled", view.log_scale_slice_enabled);

    view.contrast_enabled = ptree.get<bool>("view." + name + "_contrast_enabled", view.contrast_enabled);
    view.contrast_auto_refresh = ptree.get<bool>("view." + name + "_auto_contrast_enabled", view.contrast_auto_refresh);
    view.contrast_invert = ptree.get<bool>("view." + name + "_invert_enabled", view.contrast_invert);
    view.contrast_min = ptree.get<float>("view." + name + "_contrast_min", view.contrast_min);
    view.contrast_max = ptree.get<float>("view." + name + "_contrast_max", view.contrast_max);
}

void xyz_load(const boost::property_tree::ptree& ptree, const std::string name, View_XYZ& view)
{
    view.flip_enabled = ptree.get<bool>("view." + name + "_flip_enabled", view.flip_enabled);
    view.rot = ptree.get<float>("view." + name + "_rot", view.rot);

    view.img_accu_level = ptree.get<ushort>("view." + name + "_img_accu_level", view.img_accu_level);

    xyzf_load(ptree, name, view);
}

static void load_view(const boost::property_tree::ptree& ptree, ViewCache::Ref& view_cache_)
{
    view_cache_.set_img_type(
        static_cast<ImgType>(ptree.get<int>("view.view_type", static_cast<int>(ImgType::Modulus))));
    view_cache_.set_x(View_XY{View_Accu{ptree.get<short>("view.x_accu", 0)}, ptree.get<ushort>("view.x_cuts", 0)});
    view_cache_.set_y(View_XY{View_Accu{ptree.get<short>("view.y_accu", 0)}, ptree.get<ushort>("view.y_cuts", 0)});
    view_cache_.set_p(View_PQ{View_Accu{ptree.get<short>("view.p_accu", 0)}, ptree.get<uint>("view.p_index", 0)});
    view_cache_.set_q(View_PQ{View_Accu{ptree.get<short>("view.q_accu", 0)}, ptree.get<uint>("view.q_index", 0)});

    xyz_load(ptree, "xy", view_cache_.get_xy_ref());
    xyz_load(ptree, "xz", view_cache_.get_xz_ref());
    xyz_load(ptree, "yz", view_cache_.get_yz_ref());
    xyzf_load(ptree, "filter2d", view_cache_.get_filter2d_ref());

    view_cache_.set_filter2d_enabled(ptree.get<bool>("image_rendering.filter2d_enabled", false));
    // FIXME: Currently not working.
    // The app crash when one of the visibility is already set at when the app begins.
    // Possible problem: Concurrency between maindisplay and the other displays
    // view_cache_.set_lens_view_enabled(ptree.get<bool>("view.lens_view_enabled", false));
    // view_cache_.set_filter2d_view_enabled(ptree.get<bool>("image_rendering.filter2d_view_enabled", false));
    view_cache_.set_fft_shift_enabled(ptree.get<bool>("view.fft_shift_enabled", false));
    // view_cache_.set_raw_view_enabled(ptree.get<bool>("view.raw_view_enabled", false));
    // GSH::instance().set_cuts_view_enabled(ptree.get<bool>("view.3d_cuts_enabled",
    // false));

    api::toggle_renormalize(ptree.get<bool>("view.renorm_enabled", true));
}

static void load_composite(const boost::property_tree::ptree& ptree, CompositeCache::Ref& composite_cache_)
{
    composite_cache_.set_composite_kind(
        static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(CompositeKind::RGB))));
    composite_cache_.set_composite_auto_weights(ptree.get<bool>("composite.auto_weights_enabled", false));
}

static void load_advanced(const boost::property_tree::ptree& ptree,
                          AdvancedCache::Ref& advanced_cache_,
                          FileReadCache::Ref& file_read_cache_)
{
    advanced_cache_.set_display_rate(ptree.get<float>("advanced.display_rate", 30));
    file_read_cache_.set_file_buffer_size(ptree.get<ushort>("advanced.file_buffer_size", 32));
    advanced_cache_.set_input_buffer_size(ptree.get<ushort>("advanced.input_buffer_size", 256));
    advanced_cache_.set_record_buffer_size(ptree.get<ushort>("advanced.record_buffer_size", 64));
    advanced_cache_.set_output_buffer_size(ptree.get<ushort>("advanced.output_buffer_size", 64));
}

void GSH::load_ptree(const boost::property_tree::ptree& ptree)
{
    load_image_rendering(ptree, compute_cache_, filter2d_cache_);
    load_view(ptree, view_cache_);
    load_composite(ptree, composite_cache_);
    load_advanced(ptree, advanced_cache_, file_read_cache_);
}

// void GSH::load_advanced(const boost::property_tree::ptree& ptree) {

// }

static void save_image_rendering(boost::property_tree::ptree& ptree,
                                 const ComputeCache::Ref& compute_cache_,
                                 const Filter2DCache::Ref& filter2d_cache_)
{
    ptree.put<uint>("image_rendering.batch_size", compute_cache_.get_batch_size());
    ptree.put<uint>("image_rendering.time_transformation_size", compute_cache_.get_time_transformation_size());
    ptree.put<ushort>("image_rendering.time_transformation_stride", compute_cache_.get_time_transformation_stride());
    ptree.put<int>("image_rendering.space_transformation", static_cast<int>(compute_cache_.get_space_transformation()));
    ptree.put<int>("image_rendering.time_transformation", static_cast<int>(compute_cache_.get_time_transformation()));
    ptree.put<float>("image_rendering.lambda", compute_cache_.get_lambda());
    ptree.put<float>("image_rendering.z_distance", compute_cache_.get_z_distance());
    ptree.put<bool>("image_rendering.convolution_enabled", compute_cache_.get_convolution_enabled());
    ptree.put<bool>("image_rendering.divide_convolution_enabled", compute_cache_.get_divide_convolution_enabled());
    ptree.put<int>("image_rendering.image_mode", static_cast<int>(compute_cache_.get_compute_mode()));

    ptree.put<int>("image_rendering.filter2d_n1", filter2d_cache_.get_filter2d_n1());
    ptree.put<int>("image_rendering.filter2d_n2", filter2d_cache_.get_filter2d_n2());
}

static void xyzf_save(boost::property_tree::ptree& ptree, const std::string& name, View_Window view)
{
    ptree.put<bool>("view." + name + "_log_scale_enabled", view.log_scale_slice_enabled);
    ptree.put<bool>("view." + name + "_contrast_enabled", view.contrast_enabled);
    ptree.put<bool>("view." + name + "_contrast_void_enabled", view.contrast_auto_refresh);
    ptree.put<bool>("view." + name + "_contrast_invert_enabled", view.contrast_invert);
    ptree.put<float>("view." + name + "_contrast_min", view.contrast_min);
    ptree.put<float>("view." + name + "_contrast_max", view.contrast_max);
}

static void xyz_save(boost::property_tree::ptree& ptree, const std::string& name, const View_XYZ& view)
{
    ptree.put<bool>("view." + name + "_flip_enabled", view.flip_enabled);
    ptree.put<int>("view." + name + "_rot", view.rot);
    ptree.put<ushort>("view." + name + "_img_accu_level", view.img_accu_level);

    xyzf_save(ptree, name, view);
}

static void save_view(boost::property_tree::ptree& ptree, const ViewCache::Ref& view_cache_)
{
    View_XY x = view_cache_.get_x();
    View_XY y = view_cache_.get_y();
    View_PQ p = view_cache_.get_p();
    View_PQ q = view_cache_.get_q();

    ptree.put<int>("view.view_type", static_cast<int>(view_cache_.get_img_type()));

    ptree.put<short>("view.x_accu_level", x.accu_level);
    ptree.put<short>("view.y_accu_level", y.accu_level);
    ptree.put<ushort>("view.x_cuts", x.cuts);
    ptree.put<ushort>("view.y_cuts", y.cuts);

    ptree.put<short>("view.p_accu_level", p.accu_level);
    ptree.put<short>("view.q_accu_level", q.accu_level);
    ptree.put<uint>("view.p_index", p.index);
    ptree.put<uint>("view.q_index", q.index);

    xyz_save(ptree, "xy", view_cache_.get_xy());
    xyz_save(ptree, "xz", view_cache_.get_xz());
    xyz_save(ptree, "yz", view_cache_.get_yz());
    xyzf_save(ptree, "filter2d", view_cache_.get_filter2d());

    ptree.put<bool>("view.lens_view_enabled", view_cache_.get_lens_view_enabled());
    ptree.put<bool>("image_rendering.filter2d_enabled", static_cast<int>(view_cache_.get_filter2d_enabled()));
    ptree.put<bool>("image_rendering.filter2d_view_enabled", static_cast<int>(view_cache_.get_filter2d_view_enabled()));
    ptree.put<bool>("view.fft_shift_enabled", view_cache_.get_fft_shift_enabled());
    ptree.put<bool>("view.raw_view_enabled", view_cache_.get_raw_view_enabled());
    ptree.put<bool>("view.3d_cuts_enabled", view_cache_.get_cuts_view_enabled());

    ptree.put<bool>("view.renorm_enabled", view_cache_.get_renorm_enabled());
}

static void save_composite(boost::property_tree::ptree& ptree, const CompositeCache::Ref& composite_cache_)
{
    ptree.put<int>("composite.mode", static_cast<int>(composite_cache_.get_composite_kind()));
    ptree.put<bool>("composite.auto_weights_enabled", composite_cache_.get_composite_auto_weights());
}

static void save_advanced(boost::property_tree::ptree& ptree,
                          const AdvancedCache::Ref& advanced_cache_,
                          const FileReadCache::Ref& file_read_cache_)
{
    ptree.put<ushort>("advanced.display_rate", static_cast<ushort>(advanced_cache_.get_display_rate()));
    ptree.put<uint>("advanced.file_buffer_size", static_cast<ushort>(file_read_cache_.get_file_buffer_size()));
    ptree.put<uint>("advanced.input_buffer_size", advanced_cache_.get_input_buffer_size());
    ptree.put<uint>("advanced.record_buffer_size", advanced_cache_.get_record_buffer_size());
    ptree.put<uint>("advanced.output_buffer_size", advanced_cache_.get_output_buffer_size());
}

void GSH::dump_ptree(boost::property_tree::ptree& ptree) const
{
    save_image_rendering(ptree, compute_cache_, filter2d_cache_);
    save_view(ptree, view_cache_);
    save_composite(ptree, composite_cache_);
    save_advanced(ptree, advanced_cache_, file_read_cache_);
}

} // namespace holovibes
