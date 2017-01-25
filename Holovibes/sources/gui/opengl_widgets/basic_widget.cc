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

#include "basic_widget.hh"

namespace gui {

	BasicWidget::BasicWidget(const uint w, const uint h, QWidget* parent) :
							QOpenGLWidget(parent),
							QOpenGLFunctions(),
							Width(w), Height(h),
							cuBuffer(nullptr),
							Ebo(QOpenGLBuffer::IndexBuffer),
							Vbo(QOpenGLBuffer::VertexBuffer), Vao(0),
							Tex(nullptr),
							Program(nullptr), Vertex(nullptr), Fragment(nullptr)
	{
		if (cudaStreamCreate(&cuStream) != cudaSuccess)
			cuStream = 0;
		resize(QSize(w, h));
	}

	BasicWidget::~BasicWidget()
	{
		makeCurrent();

		cudaGraphicsUnregisterResource(cuBuffer);
		cudaStreamDestroy(cuStream);
		
		Ebo.destroy();
		Vao.destroy();
		Vbo.destroy();
		Tex->destroy();

		delete Tex;
		delete Fragment;
		delete Vertex;
		delete Program;
		
		doneCurrent();
	}

}
