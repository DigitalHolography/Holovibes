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

#include "slice_widget.hh"

namespace gui {

	SliceWidget::SliceWidget(holovibes::Queue& q,
							const uint w, const uint h, QWidget* parent) :
							BasicWidget(w, h, parent),
							HQueue(q), Fd(HQueue.get_frame_desc())
	{}

	SliceWidget::~SliceWidget()
	{}

	void SliceWidget::initShaders()
	{
		Vertex = new QOpenGLShader(QOpenGLShader::Vertex);
		Vertex->compileSourceCode(
			"#version 400\n"
			"layout(location = 0) in vec2 xy;\n"
			"layout(location = 1) in vec2 uv;\n"
			"out vec2 frag_uv;\n"
			"void main() {\n"
			"	frag_uv = uv;\n"
			"   gl_Position = vec4(xy, 0.0f, 1.0f);\n"
			"}"
		);
		Fragment = new QOpenGLShader(QOpenGLShader::Fragment);
		Fragment->compileSourceCode(
			"#version 400\n"
			"in vec2 frag_uv;\n"
			"out vec4 out_color;\n"
			"uniform sampler2D tex;\n"
			"void main() {\n"
			"	out_color = texture(tex, frag_uv).rgba;\n"
			"}\n"
		);
	}

	void SliceWidget::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		initShaders();
		Program = new QOpenGLShaderProgram();
		Program->addShader(Vertex);
		Program->addShader(Fragment);
		if (Program->link())
		{
			/*glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glBlendEquation(GL_FUNC_ADD);*/
			glClear(GL_COLOR_BUFFER_BIT);

			glGenTextures(1, &Tex);
			glBindTexture(GL_TEXTURE_2D, Tex);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glBindTexture(GL_TEXTURE_2D, 0);

			cudaGraphicsGLRegisterBuffer(
				&cuda_buffer_,
				Tex,
				cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);

		}
		else
			std::cerr << Program->log().toStdString() << '\n';
	}

	void SliceWidget::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void SliceWidget::paintGL()
	{

	}

}