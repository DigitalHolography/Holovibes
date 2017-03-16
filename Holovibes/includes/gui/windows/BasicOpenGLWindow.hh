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

#include <array>
#include <atomic>
#include <QOpenGLWindow.h>
#include <QOpenGLFunctions.h>
#include <QOpenGLVertexArrayObject.h>
#include <QOpenGLShaderProgram.h>
#include <QEvent.h>
#include <cuda_gl_interop.h>

#include "tools_conversion.cuh"
#include "compute_descriptor.hh"
#include "queue.hh"

#ifndef vertCoord
# define vertCoord 1.0f
#endif

#ifndef DisplayRate
# define DisplayRate 1000.f/30.f
#endif

namespace gui
{
	using KindOfView =
	enum
	{
		Direct = 1,
		Hologram,
		Slice
	};
	
	class BasicOpenGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
	{
		Q_OBJECT
		public:
			// Constructor & Destructor
			BasicOpenGLWindow(QPoint p, QSize s, holovibes::Queue& q,
				holovibes::ComputeDescriptor &cd, KindOfView k);
			virtual ~BasicOpenGLWindow();

			const KindOfView getKindOfView() const;

		protected:
			// Fields -----------
			QSize	winSize;
			holovibes::ComputeDescriptor&	Cd;
			holovibes::Queue&				Queue;
			const camera::FrameDescriptor&  Fd;
			const KindOfView	kView;

			static std::atomic<bool>	slicesAreLocked;
			std::array<float, 2>	Translate;
			float	Scale;

			// CUDA Objects -----
			cudaGraphicsResource_t	cuResource;
			cudaStream_t			cuStream;
			cudaArray_t				cuArray;
			cudaResourceDesc		cuArrRD;
			cudaSurfaceObject_t		cuSurface;

			// OpenGL Objects ---
			QOpenGLShaderProgram	*Program;
			QOpenGLVertexArrayObject	Vao;
			GLuint	Vbo, Ebo;
			GLuint	Tex;
			
			// Virtual Pure Functions
			virtual void initializeGL() = 0;
			virtual void resizeGL(int w, int h) = 0;
			virtual void paintGL() = 0;

			void	timerEvent(QTimerEvent *e);
			void	keyPressEvent(QKeyEvent* e);

			// Transform functions
			void	setTranslate();
			void	setScale();
			void	resetTransform();
	};
}