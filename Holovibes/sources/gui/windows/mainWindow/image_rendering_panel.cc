#include <filesystem>

#include "image_rendering_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

namespace holovibes::gui
{
ImageRenderingPanel::ImageRenderingPanel(QWidget* parent)
    : Panel(parent)
{
}

ImageRenderingPanel::~ImageRenderingPanel() {}

void ImageRenderingPanel::init() { ui_->ZDoubleSpinBox->setSingleStep(z_step_); }

void ImageRenderingPanel::on_notify()
{
    const bool is_raw = parent_->is_raw_mode();

    ui_->TimeTransformationStrideSpinBox->setEnabled(!is_raw);

    const uint input_queue_capacity = global::global_config.input_queue_max_size;

    ui_->TimeTransformationStrideSpinBox->setValue(parent_->cd_.time_transformation_stride);
    ui_->TimeTransformationStrideSpinBox->setSingleStep(parent_->cd_.batch_size);
    ui_->TimeTransformationStrideSpinBox->setMinimum(parent_->cd_.batch_size);

    ui_->BatchSizeSpinBox->setEnabled(!is_raw && !ui_->ExportPanel->is_recording);

    parent_->cd_.check_batch_size_limit(input_queue_capacity);
    ui_->BatchSizeSpinBox->setValue(parent_->cd_.batch_size);
    ui_->BatchSizeSpinBox->setMaximum(input_queue_capacity);

    ui_->SpaceTransformationComboBox->setEnabled(!is_raw && !parent_->cd_.time_transformation_cuts_enabled);
    ui_->SpaceTransformationComboBox->setCurrentIndex(static_cast<int>(parent_->cd_.space_transformation.load()));
    ui_->TimeTransformationComboBox->setEnabled(!is_raw);
    ui_->TimeTransformationComboBox->setCurrentIndex(static_cast<int>(parent_->cd_.time_transformation.load()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui_->timeTransformationSizeSpinBox->setEnabled(!is_raw && !parent_->cd_.time_transformation_cuts_enabled);
    ui_->timeTransformationSizeSpinBox->setValue(parent_->cd_.time_transformation_size);

    ui_->WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui_->WaveLengthDoubleSpinBox->setValue(parent_->cd_.lambda * 1.0e9f);
    ui_->ZDoubleSpinBox->setEnabled(!is_raw);
    ui_->ZDoubleSpinBox->setValue(parent_->cd_.zdistance);
    ui_->BoundaryLineEdit->setText(QString::number(parent_->holovibes_.get_boundary()));

    // Filter2D
    ui_->Filter2D->setEnabled(!is_raw);
    ui_->Filter2D->setChecked(!is_raw && parent_->cd_.filter2d_enabled);
    ui_->Filter2DView->setEnabled(!is_raw && parent_->cd_.filter2d_enabled);
    ui_->Filter2DView->setChecked(!is_raw && parent_->cd_.filter2d_view_enabled);
    ui_->Filter2DN1SpinBox->setEnabled(!is_raw && parent_->cd_.filter2d_enabled);
    ui_->Filter2DN1SpinBox->setValue(parent_->cd_.filter2d_n1);
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);
    ui_->Filter2DN2SpinBox->setEnabled(!is_raw && parent_->cd_.filter2d_enabled);
    ui_->Filter2DN2SpinBox->setValue(parent_->cd_.filter2d_n2);

    // Convolution
    ui_->ConvoCheckBox->setEnabled(parent_->cd_.compute_mode == Computation::Hologram);
    ui_->ConvoCheckBox->setChecked(parent_->cd_.convolution_enabled);
    ui_->DivideConvoCheckBox->setChecked(parent_->cd_.convolution_enabled && parent_->cd_.divide_convolution_enabled);
}

void ImageRenderingPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    const float z_step_ = ptree.get<float>("image_rendering.z_step", z_step_);
    if (z_step_ > 0.0f)
        ui_->ZDoubleSpinBox->setSingleStep(z_step_);
}

void ImageRenderingPanel::save_ini(boost::property_tree::ptree& ptree)
{
    ptree.put<double>("image_rendering.z_step", z_step_);
}

void ImageRenderingPanel::set_image_mode(QString mode)
{
    if (mode != nullptr)
    {
        // Call comes from ui
        if (ui_->ImageModeComboBox->currentIndex() == 0)
            set_raw_mode();
        else
            set_holographic_mode();
    }
    else if (parent_->cd_.compute_mode == Computation::Raw)
        set_raw_mode();
    else if (parent_->cd_.compute_mode == Computation::Hologram)
        set_holographic_mode();
}

void ImageRenderingPanel::set_raw_mode()
{
    parent_->close_windows();
    parent_->close_critical_compute();

    if (parent_->is_enabled_camera_)
    {
        QPoint pos(0, 0);
        const camera::FrameDescriptor& fd = parent_->holovibes_.get_gpu_input_queue()->get_fd();
        unsigned short width = fd.width;
        unsigned short height = fd.height;

        get_good_size(width, height, parent_->window_max_size);
        QSize size(width, height);
        parent_->init_image_mode(pos, size);
        parent_->cd_.set_compute_mode(Computation::Raw);
        parent_->createPipe();

        parent_->mainDisplay.reset(new RawWindow(pos, size, parent_->holovibes_.get_gpu_input_queue().get()));
        parent_->mainDisplay->setTitle(QString("XY view"));
        parent_->mainDisplay->setCd(&(parent_->cd_));
        parent_->mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));

        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::INPUT_FORMAT,
                                                                  fd_info);
        set_convolution_mode(false);
        set_divide_convolution_mode(false);
        parent_->notify();
        parent_->layout_toggled();
    }
}

void ImageRenderingPanel::set_holographic_mode()
{
    // That function is used to reallocate the buffers since the Square
    // input mode could have changed
    /* Close windows & destory thread compute */
    parent_->close_windows();
    parent_->close_critical_compute();

    /* ---------- */
    try
    {
        parent_->cd_.set_compute_mode(Computation::Hologram);
        /* Pipe & Window */
        parent_->createPipe();
        parent_->createHoloWindow();
        /* Info Manager */
        const camera::FrameDescriptor& fd = parent_->holovibes_.get_gpu_output_queue()->get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::OUTPUT_FORMAT,
                                                                  fd_info);
        /* Contrast */
        parent_->cd_.set_contrast_enabled(true);

        /* Filter2D */
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

        /* Record Frame Calculation */
        ui_->NumberOfFramesSpinBox->setValue(
            ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                 (float)ui_->TimeTransformationStrideSpinBox->value()));

        /* Notify */
        parent_->notify();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot set holographic mode: " << e.what();
    }
}

void ImageRenderingPanel::set_computation_mode()
{
    if (ui_->ImageModeComboBox->currentIndex() == 0)
    {
        parent_->cd_.set_compute_mode(Computation::Raw);
    }
    else if (ui_->ImageModeComboBox->currentIndex() == 1)
    {
        parent_->cd_.set_compute_mode(Computation::Hologram);
    }
}

void ImageRenderingPanel::update_batch_size()
{
    if (parent_->is_raw_mode())
        return;

    int value = ui_->BatchSizeSpinBox->value();

    if (value == parent_->cd_.batch_size)
        return;

    auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            parent_->cd_.set_batch_size(value);
            parent_->cd_.adapt_time_transformation_stride();
            parent_->holovibes_.get_compute_pipe()->request_update_batch_size();
            parent_->notify();
        });
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

void ImageRenderingPanel::update_time_transformation_stride()
{
    if (parent_->is_raw_mode())
        return;

    int value = ui_->TimeTransformationStrideSpinBox->value();

    if (value == parent_->cd_.time_transformation_stride)
        return;

    auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            parent_->cd_.set_time_transformation_stride(value);
            parent_->cd_.adapt_time_transformation_stride();
            parent_->holovibes_.get_compute_pipe()->request_update_time_transformation_stride();
            ui_->NumberOfFramesSpinBox->setValue(
                ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                     (float)ui_->TimeTransformationStrideSpinBox->value()));
            parent_->notify();
        });
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

void ImageRenderingPanel::set_filter2d(bool checked)
{
    if (parent_->is_raw_mode())
        return;

    if (!checked)
    {
        parent_->cd_.set_filter2d_enabled(checked);
        cancel_filter2d();
    }
    else
    {
        const camera::FrameDescriptor& fd = parent_->holovibes_.get_gpu_input_queue()->get_fd();

        // Set the input box related to the filter2d
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
        set_filter2d_n2(ui_->Filter2DN2SpinBox->value());
        set_filter2d_n1(ui_->Filter2DN1SpinBox->value());

        if (auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(WindowKind::XYview);
        parent_->cd_.set_filter2d_enabled(checked);
    }
    parent_->pipe_refresh();
    parent_->notify();
}

void ImageRenderingPanel::cancel_filter2d()
{
    if (parent_->is_raw_mode())
        return;

    if (parent_->cd_.filter2d_view_enabled)
        update_filter2d_view(false);
    parent_->pipe_refresh();
    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_pipe()
{
    if (auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (parent_->cd_.time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (parent_->cd_.filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
    }

    parent_->pipe_refresh();
    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_n1(int n)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_filter2d_n1(n);
    set_filter2d_pipe();
}

void ImageRenderingPanel::set_filter2d_n2(int n)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_filter2d_n2(n);
    set_filter2d_pipe();
}

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    if (parent_->is_raw_mode())
        return;

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = parent_->mainDisplay->framePosition() + QPoint(parent_->mainDisplay->width() + 310, 0);
            auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get());
            if (pipe)
            {
                pipe->request_filter2d_view();

                const camera::FrameDescriptor& fd = parent_->holovibes_.get_gpu_output_queue()->get_fd();
                ushort filter2d_window_width = fd.width;
                ushort filter2d_window_height = fd.height;
                get_good_size(filter2d_window_width, filter2d_window_height, parent_->auxiliary_window_max_size);

                // Wait for the filter2d view to be enabled for notify
                while (pipe->get_filter2d_view_requested())
                    continue;

                filter2d_window.reset(new Filter2DWindow(pos,
                                                         QSize(filter2d_window_width, filter2d_window_height),
                                                         pipe->get_filter2d_view_queue().get(),
                                                         parent_));

                filter2d_window->setTitle("Filter2D view");
                filter2d_window->setCd(&(parent_->cd_));

                connect(filter2d_window.get(), SIGNAL(destroyed()), this, SLOT(disable_filter2d_view()));
                parent_->cd_.set_log_scale_slice_enabled(WindowKind::Filter2D, true);
                pipe->autocontrast_end_pipe(WindowKind::Filter2D);
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
        }
    }

    else
    {
        disable_filter2d_view();
        filter2d_window.reset(nullptr);
    }

    parent_->pipe_refresh();
    parent_->notify();
}

void ImageRenderingPanel::disable_filter2d_view()
{

    auto pipe = parent_->holovibes_.get_compute_pipe();
    pipe->request_disable_filter2d_view();

    // Wait for the filter2d view to be disabled for notify
    while (pipe->get_disable_filter2d_view_requested())
        continue;

    if (filter2d_window)
    {
        // Remove the on triggered event

        disconnect(filter2d_window.get(), SIGNAL(destroyed()), this, SLOT(disable_filter2d_view()));
    }

    // Change the focused window
    parent_->change_window();

    parent_->notify();
}

void ImageRenderingPanel::set_space_transformation(const QString& value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_space_transformation_from_string(value.toStdString());
    set_holographic_mode();
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_time_transformation_from_string(value.toStdString());
    set_holographic_mode();
}

void ImageRenderingPanel::set_time_transformation_size()
{
    if (parent_->is_raw_mode())
        return;

    int time_transformation_size = ui_->timeTransformationSizeSpinBox->value();
    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == parent_->cd_.time_transformation_size)
        return;
    parent_->notify();
    auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            parent_->cd_.set_time_transformation_size(time_transformation_size);
            parent_->holovibes_.get_compute_pipe()->request_update_time_transformation_size();
            ui_->ViewPanel->set_p_accu();
            // This will not do anything until
            // SliceWindow::changeTexture() isn't coded.
        });
    }
}

void ImageRenderingPanel::set_wavelength(const double value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_lambda(static_cast<float>(value) * 1.0e-9f);
    parent_->pipe_refresh();
}

void ImageRenderingPanel::set_z(const double value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_zdistance(static_cast<float>(value));
    parent_->pipe_refresh();
}

void ImageRenderingPanel::set_z_step(const double value)
{
    z_step_ = value;
    ui_->ZDoubleSpinBox->setSingleStep(value);
}

void ImageRenderingPanel::increment_z()
{
    if (parent_->is_raw_mode())
        return;

    set_z(parent_->cd_.zdistance + z_step_);
    ui_->ZDoubleSpinBox->setValue(parent_->cd_.zdistance);
}

void ImageRenderingPanel::decrement_z()
{
    if (parent_->is_raw_mode())
        return;

    set_z(parent_->cd_.zdistance - z_step_);
    ui_->ZDoubleSpinBox->setValue(parent_->cd_.zdistance);
}

void ImageRenderingPanel::set_convolution_mode(const bool value)
{
    parent_->cd_.set_convolution(value, ui_->KernelQuickSelectComboBox->currentText().toStdString());

    try
    {
        auto pipe = parent_->holovibes_.get_compute_pipe();

        if (value)
        {
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        else
        {
            pipe->request_disable_convolution();
            // Wait for the convolution to be disabled for notify
            while (pipe->get_disable_convolution_requested())
                continue;
        }
    }
    catch (const std::exception& e)
    {
        parent_->cd_.set_convolution_enabled(false);
        LOG_ERROR << e.what();
    }

    parent_->notify();
}

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    if (parent_->cd_.convolution_enabled)
    {
        parent_->cd_.set_convolution(true, ui_->KernelQuickSelectComboBox->currentText().toStdString());

        try
        {
            auto pipe = parent_->holovibes_.get_compute_pipe();
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        catch (const std::exception& e)
        {
            parent_->cd_.set_convolution_enabled(false);
            LOG_ERROR << e.what();
        }

        parent_->notify();
    }
}

void ImageRenderingPanel::set_divide_convolution_mode(const bool value)
{
    parent_->cd_.set_divide_convolution_mode(value);

    parent_->pipe_refresh();
    parent_->notify();
}
} // namespace holovibes::gui
