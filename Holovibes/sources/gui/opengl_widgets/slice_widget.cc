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
			"layout(location = 1) in vec2 uv;\n"
			"out vec2 texCoord;\n"
			"void main() {\n"
			"	texCoord = uv;\n"
			"   gl_Position = vec4(xy, 0.0f, 1.0f);\n"
			"}\n"
		);
		if (!Vertex->isCompiled())
			std::cerr << "[Error] Vertex Shader is not compiled\n";
		Fragment = new QOpenGLShader(QOpenGLShader::Fragment);
		Fragment->compileSourceCode(
			"#version 450\n"
			"in vec2	texCoord;\n"
			"out vec4	out_color;\n"
			//"uniform sampler2D	tex;\n"
			"void main() {\n"
			//"	out_color  = texture(tex, texCoord);\n"
			"	out_color  = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
			"}\n"
		);
		if (!Fragment->isCompiled())
			std::cerr << "[Error] Fragment Shader is not compiled\n";
	}

	void SliceWidget::initTexture()
	{
		/*//if (!Tex.create()) std::cerr << "[Error] Tex create() fail\n";
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
		Tex->allocateStorage(pixelFormat, pixelType);*/
		
		glGenBuffers(1, &Tex);
		glBindBuffer(GL_TEXTURE_BUFFER, Tex);

		unsigned int size = Fd.frame_size();
		if (Fd.depth == 4 || Fd.depth == 8)
			size /= 2;
		glBufferData(GL_TEXTURE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_TEXTURE_BUFFER, 0);

		cudaGraphicsGLRegisterBuffer(
			&cuBuffer,
			Tex,
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
		
	}

	void SliceWidget::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		///* ---------- */
		#pragma region Shaders
		initShaders();
		Program = new QOpenGLShaderProgram();
		Program->addShader(Vertex);
		Program->addShader(Fragment);
		if (!Program->bind()) std::cerr << "[Error] " << Program->log().toStdString() << '\n';
		#pragma endregion
		/* ---------- */
		if (!Vao.create())
				std::cerr << "[Error] Vao create() fail\n";
		Vao.bind();
		/* ---------- */
		#pragma region Texture
		/*glGenTextures(1, &Tex);
		glBindTexture(GL_TEXTURE_2D, Tex);

		if (Fd.endianness == camera::BIG_ENDIAN)
			glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
		else
			glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);

		//auto pixelType = (Fd.depth == 1) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;

		auto pixelFormat = GL_RED; // crash
		if (Fd.depth == 8)
			pixelFormat = GL_RG;

		glTexImage2D(GL_TEXTURE_2D, 0,
			GL_RGBA, Fd.width, Fd.height, 0, GL_RG, GL_UNSIGNED_BYTE,
			HQueue.get_last_images(1));
			
		cudaGraphicsGLRegisterImage(&cuBuffer, Tex, GL_TEXTURE_2D, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
		glBindTexture(GL_TEXTURE_2D, 0);*/
		#pragma endregion
		/* ---------- */
		#pragma region Vertex Buffer Object
		const float	data[16] = {
			// Top-left
			-vertCoord, vertCoord,	// vertex coord (-1.0f <-> 1.0f)
			0.0f, 0.0f,				// texture coord (0.0f <-> 1.0f)
			// Top-right
			vertCoord, vertCoord,
			1.0f, 0.0f,
			// Bottom-right
			vertCoord, -vertCoord,
			1.0f, 1.0f,
			// Bottom-left
			-vertCoord, -vertCoord,
			0.0f, 1.0f
		};
		glGenBuffers(1, &Vbo);
		glBindBuffer(GL_ARRAY_BUFFER, Vbo);
		glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat), data, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
			reinterpret_cast<void*>(2 * sizeof(float)));

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		#pragma endregion
		/* ---------- */
		#pragma region Element Buffer Object
		const GLuint elements[6] = {
			0, 1, 2,
			2, 3, 0
		};
		glGenBuffers(1, &Ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_STATIC_DRAW);
		#pragma endregion
		/* ---------- */
		Vao.release();
		Program->release();
		glViewport(0, 0, Width, Height);
		doneCurrent();
	}

	void SliceWidget::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void SliceWidget::paintGL()
	{
		const void* frame = HQueue.get_last_images(1);
		makeCurrent();
		glClear(GL_COLOR_BUFFER_BIT);
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
		#pragma region Texture update
		/*glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Tex);
		GLuint texId;
		glGenTextures(1, &texId);
		glBindTexture(GL_TEXTURE_2D, texId);
		glTexSubImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Fd.width, Fd.height, 0,
			GL_RGBA, GL_UNSIGNED_BYTE, 0);

		glBindTexture(GL_TEXTURE_2D, Tex);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Fd.width, Fd.height, 0,
			GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glGenerateMipmap(GL_TEXTURE_2D);
		glUniform1i(glGetUniformLocation(Program->programId(), "tex"), 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		if (Fd.depth == 8)
		{
			//We replace the green color by the blue one for complex display
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
		}
		else
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
		}
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);*/
		#pragma endregion
		/* ----------- */
		Program->bind();
		Vao.bind();

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);

		Vao.release();
		Program->release();

		doneCurrent();
	}

}