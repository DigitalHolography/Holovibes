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
							cuResource(nullptr),
							Vao(0),
							Vbo(0), Ebo(0),
							Tex(0),// Pbo(0),
							Program(nullptr), Vertex(nullptr), Fragment(nullptr),
							timer(this)
	{
		connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
		timer.start(1000 / 60);
		if (cudaStreamCreate(&cuStream) != cudaSuccess)
			cuStream = 0;
		resize(QSize(w, h));
	}

	BasicWidget::~BasicWidget()
	{
		makeCurrent();

		cudaGraphicsUnregisterResource(cuResource);
		cudaStreamDestroy(cuStream);
		
		if (Ebo) glDeleteBuffers(1, &Ebo);
		if (Vbo) glDeleteBuffers(1, &Vbo);
		if (Tex) glDeleteBuffers(1, &Tex);
		//if (Pbo) glDeleteBuffers(1, &Pbo);
		Vao.destroy();

		delete Fragment;
		delete Vertex;
		delete Program;
		
		doneCurrent();
	}

}
