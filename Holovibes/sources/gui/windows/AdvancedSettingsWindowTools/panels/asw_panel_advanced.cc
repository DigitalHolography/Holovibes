#include "asw_panel_advanced.hh"
#include "API.hh"
namespace holovibes::gui
{

#define DEFAULT_DISPLAY_RATE_VALUE api::get_display_rate()
#define DEFAULT_FILTER2D_SMOOTH_LOW_VALUE api::get_filter2d_smooth_low()
#define DEFAULT_FILTER2D_SMOOTH_HIGH_VALUE api::get_filter2d_smooth_high()
#define DEFAULT_CONTRAST_LOWER_THRESHOLD_VALUE api::get_contrast_lower_threshold()
#define DEFAULT_CONTRAST_UPPER_THRESHOLD_VALUE api::get_contrast_upper_threshold()
#define DEFAULT_RENORM_CONSTANT_VALUE api::get_renorm_constant()
#define DEFAULT_CUTS_CONTRAST_P_OFFSET_VALUE api::get_cuts_contrast_p_offset()

ASWPanelAdvanced::ASWPanelAdvanced()
    : AdvancedSettingsWindowPanel("Advanced")
{
    advanced_layout_ = new QVBoxLayout();

    // Widgets creation
    create_display_rate_widget();
    create_filter2d_smooth_low_widget();
    create_filter2d_smooth_high_widget();
    create_contrast_lower_threshold_widget();
    create_contrast_upper_threshold_widget();
    create_renorm_constant_widget();
    create_cuts_contrast_p_offset_widget();

    setLayout(advanced_layout_);
}

ASWPanelAdvanced::~ASWPanelAdvanced() {}

#pragma region WIDGETS

void ASWPanelAdvanced::create_display_rate_widget()
{
    // Display rate spin box
    display_rate_ = new QDoubleSpinBoxLayout(nullptr, "DisplayRate");
    display_rate_->set_value(DEFAULT_DISPLAY_RATE_VALUE);
    advanced_layout_->addItem(display_rate_);
    connect(display_rate_, SIGNAL(value_changed()), this, SLOT(on_change_display_rate_value()));
}

void ASWPanelAdvanced::create_filter2d_smooth_low_widget()
{
    // Filter2d smooth low spin box
    filter2d_smooth_low_ = new QDoubleSpinBoxLayout(nullptr, "Filter2D_smooth_low");
    filter2d_smooth_low_->set_value(DEFAULT_FILTER2D_SMOOTH_LOW_VALUE);
    advanced_layout_->addItem(filter2d_smooth_low_);
    connect(filter2d_smooth_low_, SIGNAL(value_changed()), this, SLOT(on_change_filter2d_smooth_low_value()));
}

void ASWPanelAdvanced::create_filter2d_smooth_high_widget()
{
    // Filter2d smooth high spin box
    filter2d_smooth_high_ = new QDoubleSpinBoxLayout(nullptr, "Filter2D_smooth_high");
    filter2d_smooth_high_->set_value(DEFAULT_FILTER2D_SMOOTH_HIGH_VALUE);
    advanced_layout_->addItem(filter2d_smooth_high_);
    connect(filter2d_smooth_high_, SIGNAL(value_changed()), this, SLOT(on_change_filter2d_smooth_high_value()));
}

void ASWPanelAdvanced::create_contrast_lower_threshold_widget()
{
    // Contrast lower threshold spin box
    contrast_lower_threshold_ = new QDoubleSpinBoxLayout(nullptr, "Contrast_lower_threshold");
    contrast_lower_threshold_->set_value(DEFAULT_CONTRAST_LOWER_THRESHOLD_VALUE);
    advanced_layout_->addItem(contrast_lower_threshold_);
    connect(contrast_lower_threshold_, SIGNAL(value_changed()), this, SLOT(on_change_contrast_lower_threshold_value()));
}

void ASWPanelAdvanced::create_contrast_upper_threshold_widget()
{
    // Contrast upper threshold spin box
    contrast_upper_threshold_ = new QDoubleSpinBoxLayout(nullptr, "Contrast_upper_threshold");
    contrast_upper_threshold_->set_value(DEFAULT_CONTRAST_UPPER_THRESHOLD_VALUE);
    advanced_layout_->addItem(contrast_upper_threshold_);
    connect(contrast_upper_threshold_, SIGNAL(value_changed()), this, SLOT(on_change_contrast_upper_threshold_value()));
}
void ASWPanelAdvanced::create_renorm_constant_widget()
{
    // Renorm constant spin box
    renorm_constant_ = new QIntSpinBoxLayout(nullptr, "Renorm_constant");
    renorm_constant_->set_value(DEFAULT_RENORM_CONSTANT_VALUE);
    advanced_layout_->addItem(renorm_constant_);
    connect(renorm_constant_, SIGNAL(value_changed()), this, SLOT(on_change_renorm_constant_value()));
}

void ASWPanelAdvanced::create_cuts_contrast_p_offset_widget()
{
    // Cuts contrast p offset cuts spin box
    cuts_contrast_p_offset_ = new QIntSpinBoxLayout(nullptr, "Cuts_contrast_p_offset");
    cuts_contrast_p_offset_->set_value(DEFAULT_CUTS_CONTRAST_P_OFFSET_VALUE);
    advanced_layout_->addItem(cuts_contrast_p_offset_);
    connect(cuts_contrast_p_offset_, SIGNAL(value_changed()), this, SLOT(on_change_cuts_contrast_p_offset_value()));
}

#pragma endregion

#pragma region SLOTS

void ASWPanelAdvanced::on_change_display_rate_value()
{
    LOG_INFO << display_rate_->get_value();
    api::set_display_rate(display_rate_->get_value());
}

void ASWPanelAdvanced::on_change_filter2d_smooth_low_value()
{
    LOG_INFO << filter2d_smooth_low_->get_value();
    api::set_filter2d_smooth_low(filter2d_smooth_low_->get_value());
}

void ASWPanelAdvanced::on_change_filter2d_smooth_high_value()
{
    LOG_INFO << filter2d_smooth_high_->get_value();
    api::set_filter2d_smooth_high(filter2d_smooth_high_->get_value());
}

void ASWPanelAdvanced::on_change_contrast_lower_threshold_value()
{
    LOG_INFO << contrast_lower_threshold_->get_value();
    api::set_contrast_lower_threshold(contrast_lower_threshold_->get_value());
}

void ASWPanelAdvanced::on_change_contrast_upper_threshold_value()
{
    LOG_INFO << contrast_upper_threshold_->get_value();
    api::set_contrast_upper_threshold(contrast_upper_threshold_->get_value());
}

void ASWPanelAdvanced::on_change_renorm_constant_value()
{
    LOG_INFO << renorm_constant_->get_value();
    api::set_renorm_constant(renorm_constant_->get_value());
}

void ASWPanelAdvanced::on_change_cuts_contrast_p_offset_value()
{
    LOG_INFO << cuts_contrast_p_offset_->get_value();
    api::set_cuts_contrast_p_offset(cuts_contrast_p_offset_->get_value());
}
#pragma endregion

} // namespace holovibes::gui