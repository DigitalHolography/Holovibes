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

#pragma once

#ifndef _HAS_AUTO_PTR_ETC
#define _HAS_AUTO_PTR_ETC 1
#endif // !_HAS_AUTO_PTR_ETC

#include <atomic>
#include <QOpenGLWindow.h>
#include <QOpenGLFunctions.h>
#include <QOpenGLVertexArrayObject.h>
#include <QOpenGLShaderProgram.h>
#include <QEvent.h>
#include <cuda_gl_interop.h>

#include <glm\gtc\matrix_transform.hpp>

#include "overlay_manager.hh"
#include "tools_conversion.cuh"
#include "queue.hh"

namespace holovibes
{
	namespace gui
	{
		using KindOfView =
		enum
		{
			Direct = 1,
			Hologram,
			SliceXZ,
			SliceYZ,
			Vision3D
		};

		class BasicOpenGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
		{
		public:
			// Constructor & Destructor
			BasicOpenGLWindow(QPoint p, QSize s, Queue& q, KindOfView k);
			virtual ~BasicOpenGLWindow();

			const KindOfView	getKindOfView() const;
			const KindOfOverlay getKindOfOverlay() const;
			QOpenGLVertexArrayObject& getVao();
			void				resetSelection();

			void	setCd(ComputeDescriptor* cd);
			ComputeDescriptor* getCd();
			const FrameDescriptor& getFd() const;
			OverlayManager& getOverlayManager();

			// Transform functions ------
			void resetTransform();
			void setScale(float);
			float getScale() const;
			void setAngle(float a);
			float getAngle() const;
			void setFlip(bool f);
			bool getFlip() const;
			void setTranslate(float x, int y);
			glm::vec2 getTranslate() const;

			const glm::mat3x3& getTransformMatrix() const;
			const glm::mat3x3& getTransformInverseMatrix() const;

		protected:
			// Fields -------------------

			Qt::WindowState			winState;
			QPoint					winPos;
			Queue&					Qu;
			ComputeDescriptor		*Cd;
			const FrameDescriptor&	Fd;
			const KindOfView		kView;
			bool					fullScreen_;

			OverlayManager	overlay_manager_;

			void setTransform();

			// CUDA Objects -------------
			cudaGraphicsResource_t	cuResource;
			cudaStream_t			cuStream;

			void*	cuPtrToPbo;
			size_t	sizeBuffer;

			// OpenGL Objects -----------
			QOpenGLShaderProgram	*Program;
			QOpenGLVertexArrayObject	Vao;
			GLuint	Vbo, Ebo, Pbo;
			GLuint	Tex;

			// Virtual Pure Functions ---
			virtual void initShaders() = 0;
			virtual void initializeGL() = 0;
			virtual void resizeGL(int width, int height);
			virtual void paintGL() = 0;

			// Event functions ----------
			void	timerEvent(QTimerEvent *e);
			void	keyPressEvent(QKeyEvent *e);
			void	wheelEvent(QWheelEvent *e);

		private:
			glm::vec4 translate_;
			float scale_;
			/// Angle in degree
			float angle_;
			bool flip_;

			glm::mat3x3 transform_matrix_;
			glm::mat3x3 transform_inverse_matrix_;

		};
	}
}
