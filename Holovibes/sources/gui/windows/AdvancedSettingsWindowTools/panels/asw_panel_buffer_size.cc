#include "asw_panel_buffer_size.hh"

namespace holovibes::gui
{

ASWPanelBufferSize::ASWPanelBufferSize(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Buffer size")
{
    buffer_size_layout_ = new QVBoxLayout();

    // File spin box
    file_ = new QIntSpinBoxLayout(parent, parent_widget, "file");
    file_->setValue(32);
    buffer_size_layout_->addItem(file_);
    connect(file_, SIGNAL(value_changed()), this, SLOT(on_change_file_value()));

    // Input spin box
    input_ = new QIntSpinBoxLayout(parent, parent_widget, "input");
    input_->setValue(256);
    buffer_size_layout_->addItem(input_);
    connect(input_, SIGNAL(value_changed()), this, SLOT(on_change_input_value()));

    // Record spin box
    record_ = new QIntSpinBoxLayout(parent, parent_widget, "record");
    record_->setValue(64);
    buffer_size_layout_->addItem(record_);
    connect(record_, SIGNAL(value_changed()), this, SLOT(on_change_record_value()));

    // Output spin box
    output_ = new QIntSpinBoxLayout(parent, parent_widget, "output");
    output_->setValue(64);
    buffer_size_layout_->addItem(output_);
    connect(output_, SIGNAL(value_changed()), this, SLOT(on_change_output_value()));

    // 3D cuts spin box
    cuts_ = new QIntSpinBoxLayout(parent, parent_widget, "3D cuts");
    cuts_->setValue(64);
    buffer_size_layout_->addItem(cuts_);
    connect(cuts_, SIGNAL(value_changed()), this, SLOT(on_change_cuts_value()));

    setLayout(buffer_size_layout_);
}

ASWPanelBufferSize::~ASWPanelBufferSize() {}

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelBufferSize::on_change_file_value() { LOG_INFO; }

void ASWPanelBufferSize::on_change_input_value() { LOG_INFO; };

void ASWPanelBufferSize::on_change_record_value() { LOG_INFO; };

void ASWPanelBufferSize::on_change_output_value() { LOG_INFO; };

void ASWPanelBufferSize::on_change_cuts_value() { LOG_INFO; };

#pragma endregion

} // namespace holovibes::gui