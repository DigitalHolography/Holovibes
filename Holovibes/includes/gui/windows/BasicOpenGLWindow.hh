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

/*! \file
*
* Interface implemented by each Qt window. */
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
	/*! \brief Contains all function to display the graphical user interface */
	namespace gui
	{
		/*! \brief Describes the kind of window */
		enum KindOfView
		{
			Direct = 1, /**< Simply displaying the input frames */
			Hologram, /**< Applying the demodulation and computations on the input frames */
			SliceXZ, /**< Displaying the XZ view of the hologram */
			SliceYZ, /**< Displaying the YZ view of the hologram */
			Vision3D /**< Displaying the Hologram in a special 3D mode */
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
			const camera::FrameDescriptor& getFd() const;
			OverlayManager& getOverlayManager();

			// Transform functions ------
			void	setTransform();
			void	resetTransform();
			void	setAngle(float a);
			void	setFlip(int f);

		protected:
			// Fields -------------------

			Qt::WindowState			winState;
			QPoint					winPos;
			Queue&					Qu;
			ComputeDescriptor		*Cd;
			const camera::FrameDescriptor&	Fd;
			const KindOfView		kView;
			bool					fullScreen_;

			OverlayManager	overlay_manager_;

			glm::vec4	Translate;
			float		Scale;
			float		Angle;
			int			Flip;

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
		};
	}
}
