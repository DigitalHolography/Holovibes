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

#include "Selection.hh"
#include "tools_conversion.cuh"
#include "queue.hh"

#ifndef DisplayRate
# define DisplayRate 1000.f/30.f
#endif

namespace holovibes
{
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
		public:
			// Constructor & Destructor
			BasicOpenGLWindow(QPoint p, QSize s, Queue& q, KindOfView k);
			virtual ~BasicOpenGLWindow();

			const KindOfView		getKindOfView() const;
			void					setKindOfSelection(KindOfSelection k);
			const KindOfSelection	getKindOfSelection() const;
			void					resetTransform();
			void					resetSelection();
			void					setAngle(float a);
			void					setFlip(int f);

		protected:
			// Fields -----------
			Queue&					Qu;
			const FrameDescriptor&	Fd;
			const KindOfView	kView;

			std::array<float, 2>	Translate;
			float	Scale;
			float	Angle;
			int		Flip;

			// CUDA Objects -----
			cudaGraphicsResource_t	cuResource;
			cudaStream_t			cuStream;

			cudaArray_t				cuArray;
			cudaResourceDesc		cuArrRD;
			cudaSurfaceObject_t		cuSurface;

			void*	cuPtrToPbo;
			size_t	sizeBuffer;

			// OpenGL Objects ---
			QOpenGLShaderProgram	*Program;
			QOpenGLVertexArrayObject	Vao;
			GLuint	Vbo, Ebo, Pbo;
			GLuint	Tex;

			Selection	zoneSelected;
			static std::atomic<bool>	slicesAreLocked;

			// Virtual Pure Functions
			virtual void initShaders() = 0;
			virtual void initializeGL() = 0;
			virtual void resizeGL(int w, int h) = 0;
			virtual void paintGL() = 0;

			void	timerEvent(QTimerEvent *e);
			void	keyPressEvent(QKeyEvent* e);

			// Transform functions
			void	setTranslate();
			void	setScale();
		};
	}
}
