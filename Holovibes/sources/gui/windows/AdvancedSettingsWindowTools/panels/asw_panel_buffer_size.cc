#include "asw_panel_buffer_size.hh"
#include "API.hh"

namespace holovibes::gui
{

#define DEFAULT_FILE_VALUE api::get_file_buffer_size()
#define DEFAULT_INPUT_VALUE api::get_input_buffer_size()
#define DEFAULT_RECORD_VALUE api::get_record_buffer_size()
#define DEFAULT_OUTPUT_VALUE api::get_output_buffer_size()
#define DEFAULT_CUTS_VALUE api::get_time_transformation_cuts_output_buffer_size()

ASWPanelBufferSize::ASWPanelBufferSize()
    : AdvancedSettingsWindowPanel("Buffer size")
{
    buffer_size_layout_ = new QVBoxLayout();

    // Widgets creation
    create_file_widget();
    create_input_widget();
    create_record_widget();
    create_output_widget();
    create_cuts_widget();

    setLayout(buffer_size_layout_);
}

ASWPanelBufferSize::~ASWPanelBufferSize() {}

#pragma region WIDGETS

void ASWPanelBufferSize::create_file_widget()
{
    // File spin box
    file_ = new QIntSpinBoxLayout(nullptr, "file");
    file_->set_value(DEFAULT_FILE_VALUE);
    buffer_size_layout_->addItem(file_);
    connect(file_, SIGNAL(value_changed()), this, SLOT(on_change_file_value()));
}

void ASWPanelBufferSize::create_input_widget()
{
    // Input spin box
    input_ = new QIntSpinBoxLayout(nullptr, "input");
    input_->set_value(DEFAULT_INPUT_VALUE);
    buffer_size_layout_->addItem(input_);
    connect(input_, SIGNAL(value_changed()), this, SLOT(on_change_input_value()));
}

void ASWPanelBufferSize::create_record_widget()
{
    // Record spin box
    record_ = new QIntSpinBoxLayout(nullptr, "record");
    record_->set_value(DEFAULT_RECORD_VALUE);
    buffer_size_layout_->addItem(record_);
    connect(record_, SIGNAL(value_changed()), this, SLOT(on_change_record_value()));
}

void ASWPanelBufferSize::create_output_widget()
{
    // Output spin box
    output_ = new QIntSpinBoxLayout(nullptr, "output");
    output_->set_value(DEFAULT_OUTPUT_VALUE);
    buffer_size_layout_->addItem(output_);
    connect(output_, SIGNAL(value_changed()), this, SLOT(on_change_output_value()));
}

void ASWPanelBufferSize::create_cuts_widget()
{
    // 3D cuts spin box
    cuts_ = new QIntSpinBoxLayout(nullptr, "3D cuts");
    cuts_->set_value(DEFAULT_CUTS_VALUE);
    buffer_size_layout_->addItem(cuts_);
    connect(cuts_, SIGNAL(value_changed()), this, SLOT(on_change_cuts_value()));
}

#pragma endregion

#pragma region SLOTS

void ASWPanelBufferSize::on_change_file_value()
{
    LOG_INFO << file_->get_value();
    api::set_file_buffer_size(file_->get_value());
}

void ASWPanelBufferSize::on_change_input_value()
{
    LOG_INFO << input_->get_value();
    api::set_input_buffer_size(input_->get_value());
}

void ASWPanelBufferSize::on_change_record_value()
{
    LOG_INFO << record_->get_value();
    api::set_record_buffer_size(record_->get_value());
}

void ASWPanelBufferSize::on_change_output_value()
{
    LOG_INFO << output_->get_value();
    api::set_output_buffer_size(output_->get_value());
}

void ASWPanelBufferSize::on_change_cuts_value()
{
    LOG_INFO << cuts_->get_value();
    api::set_time_transformation_cuts_output_buffer_size(cuts_->get_value());
}

#pragma endregion

} // namespace holovibes::gui