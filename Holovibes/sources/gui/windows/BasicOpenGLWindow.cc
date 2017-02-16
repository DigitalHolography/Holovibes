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

#include <qtimer.h>
#include "BasicOpenGLWindow.hh"

namespace gui
{
	BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, holovibes::Queue& q) :
		QOpenGLWindow(), QOpenGLFunctions(),
		winPos(p), winSize(s),
		Queue(q),
		cuResource(nullptr),
		Program(nullptr),
		Vao(0),
		Vbo(0), Ebo(0),
		Tex(0)
	{
		/*static QTimer timer(this);
		connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
		timer.start(1000 / 60);*/
		//Ui::GLSliceWindow ui;
		//ui.setupUi(this);

		/*QSurfaceFormat format;
		format.setDepthBufferSize(24);
		format.setStencilBufferSize(8);
		setFormat(format);*/
		if (cudaStreamCreate(&cuStream) != cudaSuccess)
			cuStream = 0;
		resize(winSize);
		setFramePosition(winPos);
		setIcon(QIcon("icon1.ico"));

		show();
	}

	BasicOpenGLWindow::~BasicOpenGLWindow()
	{
		makeCurrent();

		cudaGraphicsUnregisterResource(cuResource);
		cudaStreamDestroy(cuStream);

		if (Tex) glDeleteBuffers(1, &Tex);
		if (Ebo) glDeleteBuffers(1, &Ebo);
		if (Vbo) glDeleteBuffers(1, &Vbo);
		Vao.destroy();
		delete Program;

		doneCurrent();
	}

}