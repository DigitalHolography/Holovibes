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

# include <QOpenGLWidget>
# include <QOpenGLFunctions.h>
# include <qopenglbuffer.h>
# include <qopenglvertexarrayobject.h>
# include <qopenglshaderprogram.h>
# include <qopengltexture.h>
# include <cuda_gl_interop.h>
# include <qtimer.h>

# include "holovibes.hh"
# include "tools_conversion.cuh"

#ifndef vertCoord
# define vertCoord 0.85f
#endif
#ifndef texCoord
# define texCoord 1.0f
#endif

namespace gui {
	
	class BasicWidget : public QOpenGLWidget, protected QOpenGLFunctions
	{
		Q_OBJECT
		public:
			BasicWidget(const uint w, const uint h, QWidget* parent = 0);
			virtual ~BasicWidget();

		protected:
			const uint	Width;
			const uint	Height;

			// CUDA
			struct cudaGraphicsResource*	cuResource;
			cudaStream_t					cuStream;

			// OpenGL Objects
			QOpenGLVertexArrayObject	Vao;
			GLuint	Vbo, Ebo;
			GLuint	Tex; // , Pbo;

			// OpenGL Shaders Objects
			QOpenGLShaderProgram	*Program;
			QOpenGLShader			*Vertex;
			QOpenGLShader			*Fragment;
			
			virtual void initShaders() = 0;
			virtual void initTexture() = 0;

			virtual void initializeGL() = 0;
			virtual void resizeGL(int w, int h) = 0;
			virtual void paintGL() = 0;
		private:
			QTimer	timer;
	};

}