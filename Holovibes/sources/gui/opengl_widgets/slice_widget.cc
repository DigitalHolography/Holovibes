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

	SliceWidget::~SliceWidget() {}

	void SliceWidget::initShaders()
	{
		Vertex = new QOpenGLShader(QOpenGLShader::Vertex);
		Vertex->compileSourceCode(
			"#version 450\n"
			"layout(location = 0) in vec2 xy;\n"
			//"layout(location = 1) in vec2 uv;\n"
			"out vec2 frag_uv;\n"
			"void main() {\n"
			"	frag_uv = xy;\n" //uv;\n"
			"   gl_Position = vec4(xy, 0.0f, 1.0f);\n"
			"}\n"
		);
		if (!Vertex->isCompiled())
			std::cerr << "[Error] Vertex Shader is not compiled\n";
		Fragment = new QOpenGLShader(QOpenGLShader::Fragment);
		Fragment->compileSourceCode(
			"#version 450\n"
			"in vec2 frag_uv;\n"
			"out vec4 out_color;\n"
			"uniform sampler2D tex;\n"
			"void main() {\n"
			"	out_color = texture(tex, frag_uv).rgba;\n"
			"}\n"
		);
		if (!Fragment->isCompiled())
			std::cerr << "[Error] Fragment Shader is not compiled\n";
	}

	void SliceWidget::initTexture()
	{
		//if (!Tex.create()) std::cerr << "[Error] Tex create() fail\n";
		//Tex.bind();
		Tex = new QOpenGLTexture(QOpenGLTexture::Target2D);
		Tex->setMagnificationFilter(QOpenGLTexture::Nearest);
		Tex->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
		Tex->setFormat(QOpenGLTexture::RGB16U);
		//Tex->setWrapMode(QOpenGLTexture::DirectionS, QOpenGLTexture::ClampToEdge);
		//Tex->setWrapMode(QOpenGLTexture::DirectionT, QOpenGLTexture::ClampToEdge);

		const auto	pixelFormat = (Fd.depth == 8) ? QOpenGLTexture::RG : QOpenGLTexture::Red;
		const auto	pixelType = (Fd.depth == 1) ? QOpenGLTexture::UInt8 : QOpenGLTexture::UInt16;
		Tex->setSize(Fd.width, Fd.height, Fd.depth);
		Tex->allocateStorage(pixelFormat, pixelType);
	}

	void SliceWidget::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		std::cout << glGetString(GL_VERSION) << '\n';
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		
		initShaders();
		Program = new QOpenGLShaderProgram();
		Program->addShader(Vertex);
		Program->addShader(Fragment);
		if (!Program->link())
			std::cerr << "[Error] " << Program->log().toStdString() << '\n';
		/*Program->enableAttributeArray("xy");
		Program->setAttributeBuffer("xy", GL_FLOAT, 0, 0);*/
		
		if (!Vao.create())
			std::cerr << "[Error] Vao create() fail\n";
		Vao.bind();

		if (!Vbo.create())
			std::cerr << "[Error] Vbo create() fail\n";
		Vbo.bind();
		Vbo.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		const float	data[8] = {
			// Top-left
			0.0f, 0.0f,
			// Top-right
			1.0f, 0.0f,
			// Bottom-right
			1.0f, 1.0f,
			// Bottom-left
			0.0f, 1.0f
		};
		Vbo.allocate(data, 8 * sizeof(float));

		initTexture();
		cudaGraphicsGLRegisterBuffer(
			&cuBuffer,
			Tex->textureId(),
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);

		if (!Ebo.create())
			std::cerr << "[Error] Ebo create() fail\n";
		Ebo.bind();
		Ebo.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		const GLuint elements[6] = {
			0, 1, 2,
			2, 3, 0
		};
		Ebo.allocate(elements, 6 * sizeof(GLuint));
		
		/*glGenBuffers(1, &ebo);
		GLuint elements[6] = {
			0, 1, 2,
			2, 3, 0
		};

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);*/

		Tex->release();
		Ebo.release();
		Vbo.release();
		Vao.release();
		Program->release();

		doneCurrent();
	}

	void SliceWidget::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void SliceWidget::paintGL()
	{
		const void* frame = HQueue.get_last_images(1);
		/* ----------- */
		#pragma region Cuda
		/* Map the buffer for access by CUDA. */
		cudaGraphicsMapResources(1, &cuBuffer, cuStream);
		size_t	size;
		void*	glBuffer;
		cudaGraphicsResourceGetMappedPointer(&glBuffer, &size, cuBuffer);
		/* CUDA memcpy of the frame to opengl buffer. */
		const uint resolution = Fd.frame_res();
		if (Fd.depth == 4)
		{
			float_to_ushort(
				static_cast<const float *>(frame),
				static_cast<unsigned short *> (glBuffer),
				resolution);
		}
		else if (Fd.depth == 8)
		{
			complex_to_ushort(
				static_cast<const cufftComplex *>(frame),
				static_cast<unsigned int *> (glBuffer),
				resolution);
		}
		else
			cudaMemcpy(glBuffer, frame, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

		/* Unmap the buffer for access by CUDA. */
		cudaGraphicsUnmapResources(1, &cuBuffer, cuStream);
		#pragma endregion
		/* ----------- */
		makeCurrent();
		glClear(GL_COLOR_BUFFER_BIT);
		if (!Program->bind())
			std::cerr << "[Error] Program bind() fail\n";
		Vao.bind();
		if (!Vbo.bind())
			std::cerr << "[Error] Vbo bind() fail\n";
		if (!Ebo.bind())
			std::cerr << "[Error] Ebo bind() fail\n";
		
		const auto	pixelFormat = (Fd.depth == 8) ? QOpenGLTexture::RG : QOpenGLTexture::Red;
		const auto	pixelType = (Fd.depth == 1) ? QOpenGLTexture::UInt8 : QOpenGLTexture::UInt16;
		Tex->bind();
		//Tex->setData(QOpenGLTexture::RGB_Integer, QOpenGLTexture::UInt16, frame);
		//Tex->setData(pixelFormat, pixelType, frame);
				
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		Ebo.release();
		Tex->release();
		Vbo.release();
		Vao.release();
		Program->release();
		doneCurrent();
	}

}