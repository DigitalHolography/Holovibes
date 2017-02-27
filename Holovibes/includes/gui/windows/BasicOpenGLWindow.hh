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

#include <QOpenGLWindow.h>
#include <QOpenGLFunctions.h>
#include <QOpenGLVertexArrayObject.h>
#include <QOpenGLShaderProgram.h>
#include <cuda_gl_interop.h>

#include "tools_conversion.cuh"
#include "queue.hh"

#ifndef vertCoord
# define vertCoord 1.0f
#endif
#ifndef texCoord
# define texCoord 1.0f
#endif

namespace gui
{
	typedef
	enum	KindOfView
	{
		Direct = 1,
		Hologram,
		Slice
	}		t_KindOfView;

	class BasicOpenGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
	{
		Q_OBJECT
		public:
			// Constructor & Destructor
			BasicOpenGLWindow(QPoint p, QSize s, holovibes::Queue& q, t_KindOfView k);
			virtual ~BasicOpenGLWindow();

		protected:
			// Fields -----------
			QPoint	winPos;
			QSize	winSize;
			holovibes::Queue&	Queue;
			t_KindOfView	Kind;

			// CUDA Objects -----
			struct cudaGraphicsResource*	cuResource;
			cudaStream_t					cuStream;

			// OpenGL Objects ---
			QOpenGLShaderProgram	*Program;
			QOpenGLVertexArrayObject	Vao;
			GLuint	Vbo, Ebo;
			GLuint	Tex;
			
			// Virtual Pure Functions
			virtual void initializeGL() = 0;
			virtual void resizeGL(int w, int h) = 0;
			virtual void paintGL() = 0;
	};
}