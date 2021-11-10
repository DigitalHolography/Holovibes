#include "asw_panel_buffer_size.hh"

namespace holovibes::gui
{

// TODO: change by API getter call
#define DEFAULT_FILE_VALUE 32
#define DEFAULT_INPUT_VALUE 256
#define DEFAULT_RECORD_VALUE 64
#define DEFAULT_OUTPUT_VALUE 64
#define DEFAULT_CUTS_VALUE 64

ASWPanelBufferSize::ASWPanelBufferSize(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Buffer size")
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
    file_ = new QIntSpinBoxLayout(parent_widget_, "file");
    file_->setValue(DEFAULT_FILE_VALUE);
    buffer_size_layout_->addItem(file_);
    connect(file_, SIGNAL(value_changed()), this, SLOT(on_change_file_value()));
}

void ASWPanelBufferSize::create_input_widget()
{
    // Input spin box
    input_ = new QIntSpinBoxLayout(parent_widget_, "input");
    input_->setValue(DEFAULT_INPUT_VALUE);
    buffer_size_layout_->addItem(input_);
    connect(input_, SIGNAL(value_changed()), this, SLOT(on_change_input_value()));
}

void ASWPanelBufferSize::create_record_widget()
{
    // Record spin box
    record_ = new QIntSpinBoxLayout(parent_widget_, "record");
    record_->setValue(DEFAULT_RECORD_VALUE);
    buffer_size_layout_->addItem(record_);
    connect(record_, SIGNAL(value_changed()), this, SLOT(on_change_record_value()));
}

void ASWPanelBufferSize::create_output_widget()
{
    // Output spin box
    output_ = new QIntSpinBoxLayout(parent_widget_, "output");
    output_->setValue(DEFAULT_OUTPUT_VALUE);
    buffer_size_layout_->addItem(output_);
    connect(output_, SIGNAL(value_changed()), this, SLOT(on_change_output_value()));
}

void ASWPanelBufferSize::create_cuts_widget()
{
    // 3D cuts spin box
    cuts_ = new QIntSpinBoxLayout(parent_widget_, "3D cuts");
    cuts_->setValue(DEFAULT_CUTS_VALUE);
    buffer_size_layout_->addItem(cuts_);
    connect(cuts_, SIGNAL(value_changed()), this, SLOT(on_change_cuts_value()));
}

#pragma endregion

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelBufferSize::on_change_file_value() { LOG_INFO << file_->get_value(); }

void ASWPanelBufferSize::on_change_input_value() { LOG_INFO << input_->get_value(); }

void ASWPanelBufferSize::on_change_record_value() { LOG_INFO << record_->get_value(); }

void ASWPanelBufferSize::on_change_output_value() { LOG_INFO << output_->get_value(); }

void ASWPanelBufferSize::on_change_cuts_value() { LOG_INFO << cuts_->get_value(); }

#pragma endregion

} // namespace holovibes::gui