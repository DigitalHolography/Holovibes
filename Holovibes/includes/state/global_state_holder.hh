#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "caches.hh"
#include "entities.hh"

namespace holovibes
{

/*! \class GSH
 *
 * \brief The GSH (global state holder), is where all the global state of the program is stored.
 *
 * Its goal is to register changes commanded by the API (which comes from user events), dispatch the changes
 * to each worker, and provide informations queried by the API. It relies on several structures and mecanisms :
 *
 * Queries and Commands : every data needed by the API shall be obtained in the form of structured Queries, provided by
 * the GSH. This guarantees that only needed data is accessed to, and grants better code readability.
 * The same principle is applied to changes comming from the API, which come in the form of structured Commands.
 *
 * MicroCache : local state holder belonging to each worker. Previously, each worker had to fetch data at the same
 * place ; therefore, all variables had to be atomic, with the aim to be thread safe. In order to avoid that, each
 * worker now possess MicroCaches, containing all the state they need. Those MicroCaches are accessed only by their
 * worker, and are synchronized with the GSH when each worker chooses to (using a trigger system).
 * More informations and explanations concerning their synchronization with the GSH are provided in files micro-cache.hh
 * and micro-cache.hxx.
 *
 * FastUpdateHolder : the fastUpdateHolder is a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
class GSH
{
  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    static GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

#pragma region(collapsed) GETTERS

    uint get_time_transformation_stride() const noexcept { return compute_cache_.get_time_transformation_stride(); }

    SpaceTransformation get_space_transformation() const noexcept { return compute_cache_.get_space_transformation(); }

    TimeTransformation get_time_transformation() const noexcept { return compute_cache_.get_time_transformation(); };

    uint get_batch_size() const noexcept { return compute_cache_.get_batch_size(); }
    uint get_time_transformation_size() const noexcept { return compute_cache_.get_time_transformation_size(); }
    float get_lambda() const noexcept { return compute_cache_.get_lambda(); }
    float get_z_distance() const noexcept { return compute_cache_.get_z_distance(); };
    bool get_convolution_enabled() const noexcept { return compute_cache_.get_convolution_enabled(); }

    int get_filter2d_n1() const noexcept { return filter2d_cache_.get_filter2d_n1(); }
    int get_filter2d_n2() const noexcept { return filter2d_cache_.get_filter2d_n2(); }
    ImgType get_img_type() const noexcept { return view_cache_.get_img_type(); }

    View_XY get_x() const noexcept { return view_cache_.get_x(); }
    int get_x_accu_level() const noexcept { return view_cache_.get_x().accu_level; }
    int get_x_cuts() const noexcept { return view_cache_.get_x().cuts; }

    View_XY get_y() const noexcept { return view_cache_.get_y(); }
    int get_y_accu_level() const noexcept { return view_cache_.get_y().accu_level; }
    int get_y_cuts() const noexcept { return view_cache_.get_y().cuts; }

    View_PQ get_p() const noexcept { return view_cache_.get_p(); }
    int get_p_accu_level() const noexcept { return view_cache_.get_p().accu_level; }
    uint get_p_index() const noexcept { return view_cache_.get_p().index; }

    View_PQ get_q() const noexcept { return view_cache_.get_q(); }
    int get_q_accu_level() const noexcept { return view_cache_.get_q().accu_level; }
    uint get_q_index() const noexcept { return view_cache_.get_q().index; }

    View_XYZ get_xy() const noexcept { return view_cache_.get_xy(); }
    bool get_xy_flip_enabled() const noexcept { return view_cache_.get_xy().flip_enabled; }
    float get_xy_rot() const noexcept { return view_cache_.get_xy().flip_enabled; }
    uint get_xy_img_accu_level() const noexcept { return view_cache_.get_xy().img_accu_level; }
    bool get_xy_log_scale_slice_enabled() const noexcept { return view_cache_.get_xy().log_scale_slice_enabled; }
    bool get_xy_contrast_enabled() const noexcept { return view_cache_.get_xy().contrast_enabled; }
    bool get_xy_contrast_auto_refresh() const noexcept { return view_cache_.get_xy().contrast_auto_refresh; }
    bool get_xy_contrast_invert() const noexcept { return view_cache_.get_xy().contrast_invert; }
    float get_xy_contrast_min() const noexcept { return view_cache_.get_xy().contrast_min; }
    float get_xy_contrast_max() const noexcept { return view_cache_.get_xy().contrast_max; }
    bool get_xy_img_accu_enabled() const noexcept { return view_cache_.get_xy().img_accu_level > 1; }

    View_XYZ get_xz() const noexcept { return view_cache_.get_xz(); }
    bool get_xz_flip_enabled() const noexcept { return view_cache_.get_xz().flip_enabled; }
    float get_xz_rot() const noexcept { return view_cache_.get_xz().flip_enabled; }
    uint get_xz_img_accu_level() const noexcept { return view_cache_.get_xz().img_accu_level; }
    bool get_xz_log_scale_slice_enabled() const noexcept { return view_cache_.get_xz().log_scale_slice_enabled; }
    bool get_xz_contrast_enabled() const noexcept { return view_cache_.get_xz().contrast_enabled; }
    bool get_xz_contrast_auto_refresh() const noexcept { return view_cache_.get_xz().contrast_auto_refresh; }
    bool get_xz_contrast_invert() const noexcept { return view_cache_.get_xz().contrast_invert; }
    float get_xz_contrast_min() const noexcept { return view_cache_.get_xz().contrast_min; }
    float get_xz_contrast_max() const noexcept { return view_cache_.get_xz().contrast_max; }
    bool get_xz_img_accu_enabled() const noexcept { return view_cache_.get_xz().img_accu_level > 1; }

    View_XYZ get_yz() const noexcept { return view_cache_.get_yz(); }
    bool get_yz_flip_enabled() const noexcept { return view_cache_.get_yz().flip_enabled; }
    float get_yz_rot() const noexcept { return view_cache_.get_yz().flip_enabled; }
    uint get_yz_img_accu_level() const noexcept { return view_cache_.get_yz().img_accu_level; }
    bool get_yz_log_scale_slice_enabled() const noexcept { return view_cache_.get_yz().log_scale_slice_enabled; }
    bool get_yz_contrast_enabled() const noexcept { return view_cache_.get_yz().contrast_enabled; }
    bool get_yz_contrast_auto_refresh() const noexcept { return view_cache_.get_yz().contrast_auto_refresh; }
    bool get_yz_contrast_invert() const noexcept { return view_cache_.get_yz().contrast_invert; }
    float get_yz_contrast_min() const noexcept { return view_cache_.get_yz().contrast_min; }
    float get_yz_contrast_max() const noexcept { return view_cache_.get_yz().contrast_max; }
    bool get_yz_img_accu_enabled() const noexcept { return view_cache_.get_yz().img_accu_level > 1; }

    View_Window get_filter2d() const noexcept { return view_cache_.get_filter2d(); }
    bool get_filter2d_contrast_enabled() const noexcept { return view_cache_.get_filter2d().contrast_enabled; }
    bool get_filter2d_contrast_invert() const noexcept { return view_cache_.get_filter2d().contrast_invert; }
    float get_filter2d_contrast_min() const noexcept { return view_cache_.get_filter2d().contrast_min; }
    float get_filter2d_contrast_max() const noexcept { return view_cache_.get_filter2d().contrast_max; }
    bool get_filter2d_log_scale_slice_enabled() const noexcept
    {
        return view_cache_.get_filter2d().log_scale_slice_enabled;
    }
    bool get_filter2d_contrast_auto_refresh() const noexcept
    {
        return view_cache_.get_filter2d().contrast_auto_refresh;
    }

    WindowKind get_current_window_type() const noexcept { return view_cache_.get_current_window(); }

    bool get_contrast_auto_refresh() const noexcept { return get_current_window().contrast_auto_refresh; }
    bool get_contrast_invert() const noexcept { return get_current_window().contrast_invert; }
    bool get_contrast_enabled() const noexcept { return get_current_window().contrast_enabled; }

    bool is_current_window_xyz_type() const;

    const View_Window& get_current_window() const;

    float get_contrast_min() const;
    float get_contrast_max() const;

    double get_rotation() const;
    bool get_flip_enabled() const;

    bool get_img_log_scale_slice_enabled() const;
    unsigned get_img_accu_level() const;

#pragma endregion

#pragma region(collapsed) SETTERS
    void set_batch_size(uint value);
    void set_time_transformation_size(uint value);
    void set_time_transformation_stride(uint value);

    void set_space_transformation(const SpaceTransformation value) noexcept
    {
        compute_cache_.set_space_transformation(value);
    }

    void set_time_transformation(const TimeTransformation value) noexcept
    {
        compute_cache_.set_time_transformation(value);
    }

    void set_lambda(float value) noexcept { compute_cache_.set_lambda(value); }

    void set_z_distance(float value) noexcept { compute_cache_.set_z_distance(value); }
    void set_convolution_enabled(bool value);

    void set_filter2d_n1(int value) noexcept { filter2d_cache_.set_filter2d_n1(value); }
    void set_filter2d_n2(int value) noexcept { filter2d_cache_.set_filter2d_n2(value); }

    void set_img_type(ImgType value) noexcept { view_cache_.set_img_type(value); }

    void set_x(View_XY value) noexcept { view_cache_.set_x(value); }
    void set_x_accu_level(int value) noexcept { view_cache_.get_x_ref().accu_level = value; }
    void set_x_cuts(int value) noexcept { view_cache_.get_x_ref().cuts = value; }

    void set_y(View_XY value) noexcept { view_cache_.set_y(value); }
    void set_y_accu_level(int value) noexcept { view_cache_.get_y_ref().accu_level = value; }
    void set_y_cuts(int value) noexcept { view_cache_.get_y_ref().cuts = value; }

    void set_p(View_PQ value) noexcept { view_cache_.set_p(value); }
    void set_p_accu_level(int value) noexcept { view_cache_.get_p_ref().accu_level = value; }
    void set_p_index(uint value) noexcept { view_cache_.get_p_ref().index = value; }

    void set_q(View_PQ value) noexcept { view_cache_.set_q(value); }
    void set_q_accu_level(int value) noexcept { view_cache_.get_q_ref().accu_level = value; }
    void set_q_index(uint value) noexcept { view_cache_.get_q_ref().index = value; }

    void set_xy(View_XYZ value) noexcept { view_cache_.set_xy(value); }
    void set_xy_flip_enabled(bool value) noexcept { view_cache_.get_xy_ref().flip_enabled = value; }
    void set_xy_rot(float value) noexcept { view_cache_.get_xy_ref().rot = value; }
    void set_xy_img_accu_level(uint value) noexcept { view_cache_.get_xy_ref().img_accu_level = value; }
    void set_xy_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_xy_ref().log_scale_slice_enabled = value;
    }
    void set_xy_contrast_enabled(bool value) noexcept { view_cache_.get_xy_ref().contrast_enabled = value; }
    void set_xy_contrast_auto_refresh(bool value) noexcept { view_cache_.get_xy_ref().contrast_auto_refresh = value; }
    void set_xy_contrast_invert(bool value) noexcept { view_cache_.get_xy_ref().contrast_invert = value; }
    void set_xy_contrast_min(float value) noexcept
    {
        view_cache_.get_xy_ref().contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_xy_contrast_max(float value) noexcept
    {
        view_cache_.get_xy_ref().contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_xz(View_XYZ value) noexcept { view_cache_.set_xz(value); }
    void set_xz_flip_enabled(bool value) noexcept { view_cache_.get_xz_ref().flip_enabled = value; }
    void set_xz_rot(float value) noexcept { view_cache_.get_xz_ref().rot = value; }
    void set_xz_img_accu_level(uint value) noexcept { view_cache_.get_xz_ref().img_accu_level = value; }
    void set_xz_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_xz_ref().log_scale_slice_enabled = value;
    }
    void set_xz_contrast_enabled(bool value) noexcept { view_cache_.get_xz_ref().contrast_enabled = value; }
    void set_xz_contrast_auto_refresh(bool value) noexcept { view_cache_.get_xz_ref().contrast_auto_refresh = value; }
    void set_xz_contrast_invert(bool value) noexcept { view_cache_.get_xz_ref().contrast_invert = value; }
    void set_xz_contrast_min(float value) noexcept
    {
        view_cache_.get_xz_ref().contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_xz_contrast_max(float value) noexcept
    {
        view_cache_.get_xz_ref().contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_yz(View_XYZ value) noexcept { view_cache_.set_yz(value); }
    void set_yz_flip_enabled(bool value) noexcept { view_cache_.get_yz_ref().flip_enabled = value; }
    void set_yz_rot(float value) noexcept { view_cache_.get_yz_ref().rot = value; }
    void set_yz_img_accu_level(uint value) noexcept { view_cache_.get_yz_ref().img_accu_level = value; }
    void set_yz_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_yz_ref().log_scale_slice_enabled = value;
    }
    void set_yz_contrast_enabled(bool value) noexcept { view_cache_.get_yz_ref().contrast_enabled = value; }
    void set_yz_contrast_auto_refresh(bool value) noexcept { view_cache_.get_yz_ref().contrast_auto_refresh = value; }
    void set_yz_contrast_invert(bool value) noexcept { view_cache_.get_yz_ref().contrast_invert = value; }
    void set_yz_contrast_min(float value) noexcept
    {
        view_cache_.get_yz_ref().contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_yz_contrast_max(float value) noexcept
    {
        view_cache_.get_yz_ref().contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_filter2d(View_Window value) noexcept { view_cache_.set_filter2d(value); }
    void set_filter2d_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_filter2d_ref().log_scale_slice_enabled = value;
    }
    void set_filter2d_contrast_enabled(bool value) noexcept { view_cache_.get_filter2d_ref().contrast_enabled = value; }
    void set_filter2d_contrast_auto_refresh(bool value) noexcept
    {
        view_cache_.get_filter2d_ref().contrast_auto_refresh = value;
    }
    void set_filter2d_contrast_invert(bool value) noexcept { view_cache_.get_filter2d_ref().contrast_invert = value; }
    void set_filter2d_contrast_min(float value) noexcept
    {
        view_cache_.get_filter2d_ref().contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_filter2d_contrast_max(float value) noexcept
    {
        view_cache_.get_filter2d_ref().contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_log_scale_filter2d_enabled(bool log_scale_filter2d_enabled) noexcept
    {
        view_cache_.get_filter2d_ref().log_scale_slice_enabled = log_scale_filter2d_enabled;
    }

    void set_contrast_enabled(bool contrast_enabled);
    void set_contrast_auto_refresh(bool contrast_auto_refresh);
    void set_contrast_invert(bool contrast_invert);
    void set_contrast_min(float value);
    void set_contrast_max(float value);
    void set_log_scale_slice_enabled(bool value);
    void set_accumulation_level(int value);
    void set_rotation(double value);
    void set_flip_enabled(double value);

#pragma endregion

    void change_window(uint index);

    void load_ptree(const boost::property_tree::ptree& ptree);

    void dump_ptree(boost::property_tree::ptree& ptree) const;

  private:
    GSH() noexcept {}

    View_Window& get_current_window();

    ComputeCache::Ref compute_cache_;
    Filter2DCache::Ref filter2d_cache_;
    ViewCache::Ref view_cache_;

    mutable std::mutex mutex_;
};

} // namespace holovibes
