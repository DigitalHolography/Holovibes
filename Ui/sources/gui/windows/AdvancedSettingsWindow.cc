#include "ui_advancedsettingswindow.h"
#include "AdvancedSettingsWindow.hh"
#include "API.hh"
#include "record_trigger_tcp_server.hh"
#include "user_interface_descriptor.hh"
#include "buffer_memory.hh"
#include <QScrollArea>
#include <QScrollBar>
#include <QTimer>
#include <spdlog/spdlog.h>

namespace holovibes::gui
{
namespace
{
namespace memory = holovibes::buffer_memory;
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t GiB = 1024 * MiB;

QString memory_size(uint64_t bytes)
{
    if (bytes >= GiB)
        return QString::number(static_cast<double>(bytes) / GiB, 'f', 2) + " GiB";
    return QString::number(static_cast<double>(bytes) / MiB, 'f', 1) + " MiB";
}

memory::Request buffer_memory_request()
{
    auto& api = API;
    const auto fd = api.input.get_input_fd();
    memory::Request request;
    request.width = fd.width;
    request.height = fd.height;
    request.input_bytes_per_pixel = fd.depth;
    const uint64_t pixels = request.width * request.height;
    request.output_bytes_per_frame = pixels * (api.compute.get_compute_mode() == Computation::Raw ? fd.depth
                                               : api.compute.get_img_type() == ImgType::Composite ? 6
                                                                                                  : 2);
    switch (api.record.get_record_mode())
    {
    case RecordMode::RAW:
        request.record_bytes_per_frame = pixels * fd.depth;
        break;
    case RecordMode::HOLOGRAM:
        request.record_bytes_per_frame = pixels * (api.compute.get_img_type() == ImgType::Composite ? 6 : 2);
        break;
    case RecordMode::CUTS_XZ:
        request.record_bytes_per_frame = request.width * api.transform.get_time_transformation_size() * 2;
        break;
    case RecordMode::CUTS_YZ:
        request.record_bytes_per_frame = request.height * api.transform.get_time_transformation_size() * 2;
        break;
    case RecordMode::MOMENTS:
    case RecordMode::OCT_CUBE_FLOAT:
        request.record_bytes_per_frame = pixels * 4;
        break;
    case RecordMode::OCT_CUBE:
        request.record_bytes_per_frame = pixels * 8;
        break;
    default:
        break;
    }
    request.file_frames = api.input.get_file_buffer_size();
    if (api.input.get_file_load_kind() != FileLoadKind::REGULAR)
    {
        const auto start = api.input.get_input_file_start_index();
        const auto end = api.input.get_input_file_end_index();
        request.file_frames = end > start ? end - start : 0;
    }
    request.file_enabled = api.input.get_import_type() == ImportType::File;
    request.input_frames = api.input.get_input_buffer_size();
    request.record_frames = api.record.get_record_buffer_size();
    request.output_frames = api.compute.get_output_buffer_size();
    request.cuts_frames = api.transform.get_time_transformation_cuts_output_buffer_size();
    request.time_transformation_size = api.transform.get_time_transformation_size();
    request.batch_size = api.transform.get_batch_size();
    request.record_buffer_count = api.record.get_record_queue_multibuffering_enabled() ? 3 : 1;
    request.record_on_gpu = api.record.get_record_queue_location() == Device::GPU;
    // Hologram mode can allocate the configured cuts even while hidden. Raw mode does not use these queues.
    request.cuts_enabled = api.compute.get_compute_mode() == Computation::Hologram;
    request.raw_view_enabled = api.view.get_raw_view_enabled();
    request.filter2d_view_enabled = api.view.get_filter2d_view_enabled();
    return request;
}

// Credit only queues that actually exist. Pending settings are not evidence of allocated memory.
memory::Estimate allocated_buffer_memory()
{
    memory::Estimate result;
    result.file = worker::FileFrameReadWorker::get_allocated_memory();
    if (const auto input = Holovibes::instance().get_input_queue())
    {
        const auto fd = input->get_fd();
        result.input.gpu = uint64_t{fd.width} * fd.height * fd.depth * input->get_total_nb_frames();
        result.input.cpu = uint64_t{input->get_total_nb_frames()} * 4 * sizeof(uint64_t);
    }
    // Stopped processing can retain its queues. They still occupy memory and must also receive credit.
    std::shared_ptr<Pipe> pipe;
    try
    {
        pipe = Holovibes::instance().get_compute_pipe();
    }
    catch (const std::runtime_error&)
    {
        // No pipe has been initialized for this source yet.
    }
    if (pipe)
        if (const auto output = pipe->get_output_queue())
        {
            const auto fd = output->get_fd();
            result.output.gpu = uint64_t{fd.width} * fd.height * fd.depth * output->get_max_size();
        }
    // Optional view/cut queues can be replaced by the compute thread. Avoid reading their unique_ptrs here;
    // reserving their full future size also covers replacement allocations that coexist with the old queue.
    // The active queue is an alias of one of the three registered, physically distinct recording queues.
    if (const auto record = Holovibes::instance().get_record_queue().load())
    {
        const auto fd = record->get_fd();
        const uint64_t frame_bytes = uint64_t{fd.width} * fd.height * fd.depth;
        auto& queues = FastUpdatesMap::map<QueueType>;
        for (const auto type : {QueueType::RECORD_QUEUE, QueueType::RECORD_QUEUE_2, QueueType::RECORD_QUEUE_3})
        {
            const auto queue = queues.find(type);
            if (queue == queues.end())
                continue;
            const auto bytes = frame_bytes * std::get<1>(*queue->second).load();
            if (std::get<2>(*queue->second).load() == Device::GPU)
                result.record.gpu += bytes;
            else
                result.record.cpu += bytes;
        }
    }
    return result;
}
} // namespace

AdvancedSettingsWindow::AdvancedSettingsWindow(QMainWindow* parent)
    : QMainWindow(parent)
{

    ui.setupUi(this);
    auto* contents = takeCentralWidget();
    auto* scroll_area = new QScrollArea(this);
    scroll_area->setFrameShape(QFrame::NoFrame);
    scroll_area->setWidget(contents);
    scroll_area->setWidgetResizable(true);
    setCentralWidget(scroll_area);
    setWindowIcon(QIcon(":/assets/icons/holovibes_logo.png"));
    this->setAttribute(Qt::WA_DeleteOnClose);
    set_current_values();

    const std::array<QSpinBox*, 5> buffers = {ui.FileBSSpinBox,
                                              ui.InputBSSpinBox,
                                              ui.RecordBSSpinBox,
                                              ui.OutputBSSpinBox,
                                              ui.Cuts3DBSSpinBox};
    for (size_t index = 0; index < buffers.size(); ++index)
    {
        buffers[index]->setMinimum(1);
        buffers[index]->setKeyboardTracking(false);
        buffer_memory_labels_[index] = new QLabel(ui.BufferSizeGroupBox);
        buffer_memory_labels_[index]->setObjectName(QString("BufferMemory%1Label").arg(index));
        ui.BufferSizeLayout->addWidget(buffer_memory_labels_[index], static_cast<int>(index), 2);
        connect(buffers[index],
                qOverload<int>(&QSpinBox::valueChanged),
                this,
                [this] { update_buffer_memory_estimate(); });
    }
    buffer_memory_summary_ = new QLabel(ui.BufferSizeGroupBox);
    buffer_memory_summary_->setObjectName("BufferMemorySummaryLabel");
    buffer_memory_summary_->setWordWrap(true);
    buffer_memory_summary_->setFixedWidth(276);
    buffer_memory_summary_->setMinimumHeight(150);
    ui.BufferSizeLayout->addWidget(buffer_memory_summary_, 5, 0, 1, 3);
    connect(ui.RecordQueueLocationCheckBox, &QCheckBox::toggled, this, [this] { update_buffer_memory_estimate(); });
    connect(ui.RecordQueueMultibufferingCheckBox,
            &QCheckBox::toggled,
            this,
            [this] { update_buffer_memory_estimate(); });
    auto* memory_timer = new QTimer(this);
    connect(memory_timer, &QTimer::timeout, this, [this] { update_buffer_memory_estimate(); });
    memory_timer->start(2000);
    update_buffer_memory_estimate();
    resize_to_contents();
    this->show();
}

AdvancedSettingsWindow::~AdvancedSettingsWindow()
{
    UserInterfaceDescriptor::instance().advanced_settings_window_.release();
}

#pragma region SLOTS

void AdvancedSettingsWindow::closeEvent(QCloseEvent* event) { emit closed(); }

void AdvancedSettingsWindow::set_ui_values()
{
    // Refresh the budgets at Save as other applications may have consumed memory since the last edit.
    if (!update_buffer_memory_estimate())
        return;

    auto& api = API;

    api.record.set_record_buffer_configuration(ui.RecordBSSpinBox->value(),
                                               ui.RecordQueueLocationCheckBox->isChecked() ? Device::GPU : Device::CPU,
                                               ui.RecordQueueMultibufferingCheckBox->isChecked());

    api.input.set_file_buffer_size(static_cast<int>(ui.FileBSSpinBox->value()));
    api.input.set_input_buffer_size(static_cast<int>(ui.InputBSSpinBox->value()));
    api.compute.set_output_buffer_size(static_cast<int>(ui.OutputBSSpinBox->value()));
    api.transform.set_time_transformation_cuts_output_buffer_size(static_cast<int>(ui.Cuts3DBSSpinBox->value()));

    UserInterfaceDescriptor::instance().display_rate_ = ui.DisplayRateSpinBox->value();

    api.filter2d.set_filter2d_smooth_low(ui.Filter2DLowSpinBox->value());
    api.filter2d.set_filter2d_smooth_high(ui.Filter2DHighSpinBox->value());
    api.contrast.set_contrast_lower_threshold(ui.ContrastLowerSpinBox->value());
    api.contrast.set_contrast_upper_threshold(ui.ContrastUpperSpinBox->value());
    api.global_pp.set_renorm_constant(ui.RenormConstantSpinBox->value());
    api.contrast.set_cuts_contrast_p_offset(ui.CutsContrastSpinBox->value());

    UserInterfaceDescriptor::instance().output_filename_ = ui.OutputNameLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().record_output_directory_ = ui.InputFolderPathLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().file_input_directory_ = ui.OutputFolderPathLineEdit->text().toStdString();
    UserInterfaceDescriptor::instance().matlab_record_signal_enabled_ = ui.MatlabRecordSignalCheckBox->isChecked();
    UserInterfaceDescriptor::instance().matlab_record_signal_port_ =
        static_cast<unsigned short>(ui.MatlabRecordSignalPortSpinBox->value());
    RecordTriggerTcpServer::instance().configure(UserInterfaceDescriptor::instance().matlab_record_signal_enabled_,
                                                 UserInterfaceDescriptor::instance().matlab_record_signal_port_);

    UserInterfaceDescriptor::instance().auto_scale_point_threshold_ = ui.autoScalePointThresholdSpinBox->value();

    api.window_pp.set_raw_bitshift(ui.rawBitShiftSpinBox->value());

    if (callback_)
        callback_();

    this->close();
}

void AdvancedSettingsWindow::change_input_folder_path() { change_folder(ui.InputFolderPathLineEdit); }
void AdvancedSettingsWindow::change_output_folder_path() { change_folder(ui.OutputFolderPathLineEdit); }

#pragma endregion

void AdvancedSettingsWindow::change_folder(Drag_drop_lineedit* lineEdit)
{
    QString foldername =
        QFileDialog::getExistingDirectory(this,
                                          tr("Open Directory"),
                                          lineEdit->text(),
                                          QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (foldername.isEmpty())
        return;

    lineEdit->setText(foldername);
}

void AdvancedSettingsWindow::set_current_values()
{
    auto& api = API;

    ui.FileBSSpinBox->setValue(api.input.get_file_buffer_size());
    ui.InputBSSpinBox->setValue(api.input.get_input_buffer_size());
    ui.RecordBSSpinBox->setValue(api.record.get_record_buffer_size());
    ui.OutputBSSpinBox->setValue(static_cast<int>(api.compute.get_output_buffer_size()));
    ui.Cuts3DBSSpinBox->setValue(api.transform.get_time_transformation_cuts_output_buffer_size());

    ui.DisplayRateSpinBox->setValue(UserInterfaceDescriptor::instance().display_rate_);
    ui.Filter2DLowSpinBox->setValue(api.filter2d.get_filter2d_smooth_low());
    ui.Filter2DHighSpinBox->setValue(api.filter2d.get_filter2d_smooth_high());
    ui.ContrastLowerSpinBox->setValue(api.contrast.get_contrast_lower_threshold());
    ui.ContrastUpperSpinBox->setValue(api.contrast.get_contrast_upper_threshold());
    ui.RenormConstantSpinBox->setValue(api.global_pp.get_renorm_constant());
    ui.CutsContrastSpinBox->setValue(api.contrast.get_cuts_contrast_p_offset());

    ui.RecordQueueLocationCheckBox->setChecked(api.record.get_record_queue_location() == Device::GPU);
    ui.RecordQueueMultibufferingCheckBox->setChecked(api.record.get_record_queue_multibuffering_enabled());

    ui.OutputNameLineEdit->setText(UserInterfaceDescriptor::instance().output_filename_.c_str());
    ui.InputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().record_output_directory_.c_str());
    ui.OutputFolderPathLineEdit->setText(UserInterfaceDescriptor::instance().file_input_directory_.c_str());
    ui.MatlabRecordSignalCheckBox->setChecked(UserInterfaceDescriptor::instance().matlab_record_signal_enabled_);
    ui.MatlabRecordSignalPortSpinBox->setValue(UserInterfaceDescriptor::instance().matlab_record_signal_port_);

    ui.autoScalePointThresholdSpinBox->setValue(
        static_cast<int>(UserInterfaceDescriptor::instance().auto_scale_point_threshold_));

    ui.rawBitShiftSpinBox->setValue(api.window_pp.get_raw_bitshift());
}

bool AdvancedSettingsWindow::update_buffer_memory_estimate()
{
    const auto current = buffer_memory_request();
    auto proposed = current;
    if (API.input.get_file_load_kind() == FileLoadKind::REGULAR)
        proposed.file_frames = ui.FileBSSpinBox->value();
    proposed.input_frames = ui.InputBSSpinBox->value();
    proposed.record_frames = ui.RecordBSSpinBox->value();
    proposed.output_frames = ui.OutputBSSpinBox->value();
    proposed.cuts_frames = ui.Cuts3DBSSpinBox->value();
    proposed.record_on_gpu = ui.RecordQueueLocationCheckBox->isChecked();
    proposed.record_buffer_count = ui.RecordQueueMultibufferingCheckBox->isChecked() ? 3 : 1;

    QString problem;
    if (proposed.input_frames < proposed.batch_size || proposed.record_frames < proposed.batch_size)
        problem = tr("Input and Record must each hold at least one batch (%1 frames).")
                      .arg(static_cast<qulonglong>(proposed.batch_size));
    else if (proposed.raw_view_enabled && proposed.output_frames < proposed.batch_size)
        problem = tr("Output must hold at least one batch while Raw view is enabled.");

    if (current.width == 0 || current.height == 0 || current.input_bytes_per_pixel == 0)
    {
        const bool increasing =
            proposed.file_frames > current.file_frames || proposed.input_frames > current.input_frames ||
            proposed.record_frames * proposed.record_buffer_count >
                current.record_frames * current.record_buffer_count ||
            proposed.record_on_gpu != current.record_on_gpu || proposed.output_frames > current.output_frames ||
            proposed.cuts_frames > current.cuts_frames;
        for (auto* label : buffer_memory_labels_)
            label->setText(tr("Unknown"));
        buffer_memory_summary_->setText(
            problem.isEmpty() ? tr("Open a file or camera to estimate RAM and VRAM usage. Buffer increases "
                                   "are unavailable until the image dimensions are known.")
                              : problem);
        const bool allowed = !increasing && problem.isEmpty();
        ui.ApplyPushButton->setEnabled(allowed);
        ui.ApplyPushButton->setToolTip(buffer_memory_summary_->text());
        resize_to_contents();
        return allowed;
    }

    const auto estimate = memory::estimate(proposed);
    auto allocated = allocated_buffer_memory();
    // Always include deferred settings in the final configuration. Crediting unchanged but unallocated sizes
    // would allow successive edits to reserve the same free memory several times before the next restart.

    const std::array<memory::Usage, 5> usages = {estimate.file,
                                                 estimate.input,
                                                 estimate.record,
                                                 estimate.output,
                                                 estimate.cuts};
    const std::array<memory::Usage, 5> existing = {allocated.file,
                                                   allocated.input,
                                                   allocated.record,
                                                   allocated.output,
                                                   allocated.cuts};
    memory::Usage required;
    for (size_t index = 0; index < usages.size(); ++index)
    {
        const auto usage = usages[index];
        auto* label = buffer_memory_labels_[index];
        label->setText(usage.cpu == 0 && usage.gpu == 0 ? tr("Inactive")
                       : usage.cpu && usage.gpu         ? memory_size(usage.cpu + usage.gpu) + tr(" total")
                                                : memory_size(usage.cpu + usage.gpu) + (usage.cpu ? " RAM" : " GPU"));
        label->setToolTip(tr("RAM: %1\nVRAM: %2").arg(memory_size(usage.cpu), memory_size(usage.gpu)));
        const auto increase = memory::required_increase(usage, existing[index]);
        required.cpu += increase.cpu;
        required.gpu += increase.gpu;
    }
    buffer_memory_labels_[2]->setToolTip(
        buffer_memory_labels_[2]->toolTip() +
        tr("\nIncludes %1 distinct recording buffer(s).").arg(static_cast<qulonglong>(proposed.record_buffer_count)));
    buffer_memory_labels_[4]->setToolTip(
        buffer_memory_labels_[4]->toolTip() +
        tr("\nBoth XZ and YZ queues in Hologram mode, including while hidden. Their configured "
           "memory is included in the budget for future use."));
    ui.FileBSSpinBox->setToolTip(
        API.input.get_file_load_kind() == FileLoadKind::REGULAR
            ? tr("Frames per file read. File loading uses both RAM and VRAM.")
            : tr("Whole-file loading is enabled; its selected frame range determines memory usage, "
                 "and this buffer size is ignored."));

    MEMORYSTATUSEX system_memory{};
    system_memory.dwLength = sizeof(system_memory);
    const bool cpu_known = GlobalMemoryStatusEx(&system_memory) != FALSE;
    size_t gpu_free = 0, gpu_total = 0;
    const bool gpu_known = cudaMemGetInfo(&gpu_free, &gpu_total) == cudaSuccess;
    const memory::Usage available{cpu_known ? (std::min)(system_memory.ullAvailPhys, system_memory.ullAvailPageFile)
                                            : 0,
                                  gpu_known ? gpu_free : 0};
    const memory::Usage reserve{(std::max)(GiB, system_memory.ullTotalPhys / uint64_t{10}),
                                (std::max)(256 * MiB, uint64_t{gpu_total} / 10)};
    if (estimate.overflow)
        problem = tr("These buffer sizes are too large to calculate safely. Reduce them before saving.");
    else if (problem.isEmpty() && ((!cpu_known && required.cpu) || (!gpu_known && required.gpu)))
        problem = tr("Memory availability could not be checked. Buffer increases cannot be saved right now.");
    else if (problem.isEmpty() && !memory::budget_allows(required, {}, available, reserve))
        problem = tr("Not enough memory headroom. Reduce buffer sizes, disable queued recordings, or change the "
                     "record queue location.");

    const QString totals = tr("Buffers: RAM %1 / VRAM %2\nAvailable: RAM %3 / VRAM %4\nReserved: RAM %5 / VRAM %6")
                               .arg(memory_size(estimate.total.cpu),
                                    memory_size(estimate.total.gpu),
                                    cpu_known ? memory_size(available.cpu) : tr("unknown"),
                                    gpu_known ? memory_size(available.gpu) : tr("unknown"),
                                    memory_size(reserve.cpu),
                                    memory_size(reserve.gpu));
    buffer_memory_summary_->setText(
        totals + "\n" + tr("Additional: RAM %1 / VRAM %2").arg(memory_size(required.cpu), memory_size(required.gpu)) +
        "\n" + (problem.isEmpty() ? tr("Within memory budget for the current input and modes.") : problem));
    buffer_memory_summary_->setToolTip(
        tr("Estimates use %1 x %2 input pixels and the current display/record modes.\n"
           "Keeps 10% of total memory free (at least 1 GiB RAM and 256 MiB VRAM) for Windows and processing.\n"
           "Only existing allocations are credited; reductions in other buffers are not used to fund increases.\n"
           "Estimates exclude processing workspaces and cannot guarantee future memory availability.")
            .arg(static_cast<qulonglong>(current.width))
            .arg(static_cast<qulonglong>(current.height)));
    buffer_memory_summary_->setStyleSheet(problem.isEmpty() ? "" : "color: #ff8c69;");
    // The main window can restart processing even when only unrelated preferences were edited.
    const bool allowed = problem.isEmpty();
    ui.ApplyPushButton->setEnabled(allowed);
    ui.ApplyPushButton->setToolTip(problem);
    resize_to_contents();
    return allowed;
}

void AdvancedSettingsWindow::resize_to_contents()
{
    ui.gridLayout->activate();
    const auto content_size = ui.gridLayoutWidget->sizeHint() + QSize(20, 20);
    ui.CentralWidget->setMinimumSize(content_size);
    const int available_height = (std::max)(300, screen()->availableGeometry().height() - 80);
    const int height = (std::min)(content_size.height(), available_height);
    const auto* scroll_area = qobject_cast<QScrollArea*>(centralWidget());
    const int scroll_width = height < content_size.height() ? scroll_area->verticalScrollBar()->sizeHint().width() : 0;
    setFixedSize(content_size.width() + scroll_width, height);
}

} // namespace holovibes::gui
