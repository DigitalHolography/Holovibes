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

#include "Overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "DirectWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
	namespace gui
	{
		Overlay::Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
			: QOpenGLFunctions()
			, zone_(0, 0)
			, kOverlay_(overlay)
			, verticesIndex_(0)
			, colorIndex_(0)
			, elemIndex_(0)
			//, Vao_()
			, Program_(nullptr)
			, active_(true)
			, display_(false)
			, parent_(parent)
		{
		}

		Overlay::~Overlay()
		{
		}

		const Rectangle& Overlay::getZone() const
		{
			return zone_;
		}

		const KindOfOverlay Overlay::getKind() const
		{
			return kOverlay_;
		}

		const Color Overlay::getColor() const
		{
			return color_;
		}

		const bool Overlay::isDisplayed() const
		{
			return display_;
		}

		const bool Overlay::isActive() const
		{
			return active_;
		}

		void Overlay::disable()
		{
			glDeleteBuffers(1, &elemIndex_);
			glDeleteBuffers(1, &verticesIndex_);
			glDeleteBuffers(1, &colorIndex_);
			delete Program_;
			active_ = false;
		}

		void Overlay::press(QPoint pos)
		{
			zone_.setTopLeft(pos);
			zone_.setBottomRight(zone_.topLeft());
		}

		void Overlay::initProgram()
		{
			initializeOpenGLFunctions();
			Program_ = new QOpenGLShaderProgram();
			addShaders();
			Program_->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.color.glsl");
			if (!Program_->bind())
				std::cerr << "[Error] " << Program_->log().toStdString() << std::endl;
			init();
			Program_->release();
		}

		void Overlay::print()
		{
			std::cout << "Kind: " << kOverlay_ << ", zone: " << zone_
				<< ", active: " << active_ << ", display: " << display_
				<< std::endl;
		}
	}
} 