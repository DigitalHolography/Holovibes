/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "info_manager.hh"
#include "HoloWindow.hh"
#include "MainWindow.hh"
#include "SliceWindow.hh"

namespace holovibes
{
	namespace gui
	{
		HoloWindow::HoloWindow(QPoint p, QSize s, std::unique_ptr<Queue>& q, SharedPipe ic, std::unique_ptr<SliceWindow>& xz, std::unique_ptr<SliceWindow>& yz, MainWindow *main_window)
			: DirectWindow(p, s, q, KindOfView::Hologram)
			, Ic(ic)
			, main_window_(main_window)
			, xz_slice_(xz)
			, yz_slice_(yz)
		{}

		HoloWindow::~HoloWindow()
		{}

		std::shared_ptr<ICompute> HoloWindow::getPipe()
		{
			return Ic;
		}

		void	HoloWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.holo.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.tex.glsl");
			Program->link();
			overlay_manager_.create_default();
		}
		
		void	HoloWindow::focusInEvent(QFocusEvent *e)
		{
			QOpenGLWindow::focusInEvent(e);
			Cd->current_window = WindowKind::XYview;
			Cd->notify_observers();
		}

		void	HoloWindow::update_slice_transforms()
		{
			if (xz_slice_)
			{
				xz_slice_->setTranslate(translate_[0], 0);
				xz_slice_->setScale(getScale());
			}
			if (yz_slice_)
			{
				yz_slice_->setTranslate(0, translate_[1]);
				yz_slice_->setScale(getScale());
			}
		}

		void	HoloWindow::update_stft_zoom_buffer(units::RectFd zone)
		{
			auto pipe = dynamic_cast<Pipe *>(Ic.get());
			if (pipe)
			{
				pipe->run_end_pipe([=]() {
					Cd->setZoomedZone(zone);
					if (Cd->croped_stft)
					{
						std::stringstream ss;
						ss << "(X1,Y1,X2,Y2) = (" << zone.x() << "," << zone.y() << "," << zone.right() << "," << zone.bottom() << ")";
						InfoManager::get_manager()->update_info("STFT Zone", ss.str());
						Ic->request_update_n(Cd->nsamples);
					}
				});
			}
		}

		void	HoloWindow::resetTransform()
		{
			if (Cd && Cd->locked_zoom)
				return;
			if (Fd.frame_res() != Cd->getZoomedZone().area())
			{
				units::ConversionData convert(this);
				update_stft_zoom_buffer(units::RectFd(convert, 0, 0, Fd.width, Fd.height));
			}
			BasicOpenGLWindow::resetTransform();
		}

		void HoloWindow::setTransform()
		{
			BasicOpenGLWindow::setTransform();
			update_slice_transforms();
		}
	}
}
