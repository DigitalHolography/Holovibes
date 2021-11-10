#include "asw_panel_advanced.hh"

namespace holovibes::gui
{

// TODO: change by API getter call
#define DEFAULT_DISPLAY_RATE_VALUE 0
#define DEFAULT_FILTER2D_SMOOTH_LOW_VALUE 0
#define DEFAULT_FILTER2D_SMOOTH_HIGH_VALUE 0.5f
#define DEFAULT_CONTRAST_UPPER_THRESHOLD_VALUE DBL_MAX
//#define DEFAULT_CONTRAST_UPPER_THRESHOLD_VALUE 99.5f
#define DEFAULT_RENORM_CONSTANT_VALUE INT_MAX
// #define DEFAULT_RENORM_CONSTANT_VALUE 5
#define DEFAULT_CUTS_CONTRAST_P_OFFSET_VALUE 0

ASWPanelAdvanced::ASWPanelAdvanced(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Advanced")
{
    advanced_layout_ = new QVBoxLayout();

    // Widgets creation
    display_rate_widget();
    filter2d_smooth_low_widget();
    filter2d_smooth_high_widget();
    contrast_upper_threshold_widget();
    renorm_constant_widget();
    cuts_contrast_p_offset_widget();

    setLayout(advanced_layout_);
}

ASWPanelAdvanced::~ASWPanelAdvanced() {}

#pragma region WIDGETS

void ASWPanelAdvanced::display_rate_widget()
{
    // Display rate spin box
    display_rate_ = new QDoubleSpinBoxLayout(parent_, parent_widget_, "DisplayRate");
    display_rate_->setValue(DEFAULT_DISPLAY_RATE_VALUE);
    advanced_layout_->addItem(display_rate_);
    connect(display_rate_, SIGNAL(value_changed()), this, SLOT(on_change_display_rate_value()));
}

void ASWPanelAdvanced::filter2d_smooth_low_widget()
{
    // Filter2d smooth low spin box
    filter2d_smooth_low_ = new QDoubleSpinBoxLayout(parent_, parent_widget_, "Filter2D_smooth_low");
    filter2d_smooth_low_->setValue(DEFAULT_FILTER2D_SMOOTH_LOW_VALUE);
    advanced_layout_->addItem(filter2d_smooth_low_);
    connect(filter2d_smooth_low_, SIGNAL(value_changed()), this, SLOT(on_change_filter2d_smooth_low_value()));
}

void ASWPanelAdvanced::filter2d_smooth_high_widget()
{
    // Filter2d smooth high spin box
    filter2d_smooth_high_ = new QDoubleSpinBoxLayout(parent_, parent_widget_, "Filter2D_smooth_high");
    filter2d_smooth_high_->setValue(DEFAULT_FILTER2D_SMOOTH_HIGH_VALUE);
    advanced_layout_->addItem(filter2d_smooth_high_);
    connect(filter2d_smooth_high_, SIGNAL(value_changed()), this, SLOT(on_change_filter2d_smooth_high_value()));
}

void ASWPanelAdvanced::contrast_upper_threshold_widget()
{
    // Contrast upper threshold spin box
    contrast_upper_threshold_ = new QDoubleSpinBoxLayout(parent_, parent_widget_, "Contrast_upper_threshold");
    contrast_upper_threshold_->setValue(DEFAULT_CONTRAST_UPPER_THRESHOLD_VALUE);
    advanced_layout_->addItem(contrast_upper_threshold_);
    connect(contrast_upper_threshold_, SIGNAL(value_changed()), this, SLOT(on_change_contrast_upper_threshold_value()));
}
void ASWPanelAdvanced::renorm_constant_widget()
{
    // Renorm constant spin box
    renorm_constant_ = new QIntSpinBoxLayout(parent_, parent_widget_, "Renorm_constant");
    renorm_constant_->setValue(DEFAULT_RENORM_CONSTANT_VALUE);
    advanced_layout_->addItem(renorm_constant_);
    connect(renorm_constant_, SIGNAL(value_changed()), this, SLOT(on_change_renorm_constant_value()));
}

void ASWPanelAdvanced::cuts_contrast_p_offset_widget()
{
    // Cuts contrast p offset cuts spin box
    cuts_contrast_p_offset_ = new QIntSpinBoxLayout(parent_, parent_widget_, "Cuts_contrast_p_offset");
    cuts_contrast_p_offset_->setValue(DEFAULT_CUTS_CONTRAST_P_OFFSET_VALUE);
    advanced_layout_->addItem(cuts_contrast_p_offset_);
    connect(cuts_contrast_p_offset_, SIGNAL(value_changed()), this, SLOT(on_change_cuts_contrast_p_offset_value()));
}

#pragma endregion

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelAdvanced::on_change_display_rate_value() { LOG_INFO << display_rate_->get_value(); };

void ASWPanelAdvanced::on_change_filter2d_smooth_low_value() { LOG_INFO << filter2d_smooth_low_->get_value(); };

void ASWPanelAdvanced::on_change_filter2d_smooth_high_value() { LOG_INFO << filter2d_smooth_high_->get_value(); };

void ASWPanelAdvanced::on_change_contrast_upper_threshold_value()
{
    LOG_INFO << contrast_upper_threshold_->get_value();
};

void ASWPanelAdvanced::on_change_renorm_constant_value() { LOG_INFO << renorm_constant_->get_value(); };

void ASWPanelAdvanced::on_change_cuts_contrast_p_offset_value() { LOG_INFO << cuts_contrast_p_offset_->get_value(); };
#pragma endregion

} // namespace holovibes::gui