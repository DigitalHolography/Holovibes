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

#include "texture_update.cuh"
#include "slice_widget.hh"

namespace gui {

	SliceWidget::SliceWidget(holovibes::Queue& q,
							const uint w, const uint h, QWidget* parent) :
							BasicWidget(w, h, parent),
							HQueue(q), Fd(HQueue.get_frame_desc()) {}

	SliceWidget::~SliceWidget() {}

	void SliceWidget::initShaders()
	{
		Vertex = new QOpenGLShader(QOpenGLShader::Vertex);
		Vertex->compileSourceCode(
			"#version 450\n"
			"layout(location = 0) in vec2 xy;\n"
			"layout(location = 1) in vec2 uv;\n"
			"out vec2	texCoord;\n"
			"void main()"
			"{\n"
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
			"uniform sampler2D	tex;\n"
			"void main()"
			"{\n"
			//"	vec4 clrTex = texture(tex, texCoord);\n"
			//"	vec4 clr = vec4(0.8f, 0.0f, 0.0f, 1.0f);\n"
			//"	out_color = mix(clrTex, clr, 0.5);\n"
			"	out_color = texture(tex, texCoord);\n"
			"}\n"
		);
		if (!Fragment->isCompiled())
			std::cerr << "[Error] Fragment Shader is not compiled\n";
	}

	void SliceWidget::initTexture()
	{		
		/*glGenBuffers(1, &Tex);
		glBindBuffer(GL_TEXTURE_BUFFER, Tex);

		unsigned int size = Fd.frame_size();
		if (Fd.depth == 4 || Fd.depth == 8)
			size /= 2;
		glBufferData(GL_TEXTURE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_TEXTURE_BUFFER, 0);

		cudaGraphicsGLRegisterBuffer(
			&cuResource, Tex,
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);*/
	}

	void SliceWidget::initializeGL()
	{
		makeCurrent();
		initializeOpenGLFunctions();
		glClearColor(0.128f, 0.128f, 0.128f, 1.0f);
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
		glGenTextures(1, &Tex);
		glBindTexture(GL_TEXTURE_2D, Tex);

		uint	size = Fd.frame_size();
		uint	res = Fd.frame_res();
		ushort	*mTexture = new ushort[size];
		
		std::memset(mTexture, 0x00, size * 2);

		/*for (uint i = 0; i < size; i += 2)
		{
			mTexture[i] = 0xc000;
			mTexture[i + 1] = 0x4000;
		}*/

		/*for (uint i = 0; i < size; i += 4)
		{
			mTexture[i] = 0xff;
			mTexture[i + 1] = 0x00;
			mTexture[i + 2] = 0x00;
			mTexture[i + 3] = 0xff;
			i += 4;
			if (i < size)
			{
				mTexture[i] = 0x00;
				mTexture[i + 1] = 0xff;
				mTexture[i + 2] = 0x00;
				mTexture[i + 3] = 0xff;
				i += 4;
			}
			if (i < size)
			{
				mTexture[i] = 0x00;
				mTexture[i + 1] = 0x00;
				mTexture[i + 2] = 0xff;
				mTexture[i + 3] = 0xff;
			}
		}*/

		glTexImage2D(GL_TEXTURE_2D, 0,
			GL_RGBA,
			Fd.width, Fd.height, 0,
			GL_RG, GL_UNSIGNED_SHORT, mTexture);
		
		glUniform1i(glGetUniformLocation(Program->programId(), "tex"), 0);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
		delete[] mTexture;
		cudaGraphicsGLRegisterImage(&cuResource, Tex, GL_TEXTURE_2D,
			cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
		// cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone
		// cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore
		#pragma endregion
		/* ---------- */
		#pragma region Vertex Buffer Object
		const float	data[16] = {
			// Top-left
			-vertCoord, vertCoord,	// vertex coord (-1.0f <-> 1.0f)
			0.0f, 0.0f,				// texture coord (0.0f <-> 1.0f)
			// Top-right
			vertCoord, vertCoord,
			texCoord, 0.0f,
			// Bottom-right
			vertCoord, -vertCoord,
			texCoord, texCoord,
			// Bottom-left
			-vertCoord, -vertCoord,
			0.0f, texCoord
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
		/*
		// unregister
		cudaGraphicsUnregisterResource(viewCudaResource);
        // resize
		glBindTexture(GL_TEXTURE_2D, viewGLTexture);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
				view.getWidth(), view.getHeight(), 0,
				GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// register back
		cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D,
			cudaGraphicsRegisterFlagsWriteDiscard);
		*/
	}

	void SliceWidget::paintGL()
	{
		makeCurrent();
		glClear(GL_COLOR_BUFFER_BIT);
		/* ----------- */
		#pragma region Cuda
		/* Map the buffer for access by CUDA. */
		cudaGraphicsMapResources(1, &cuResource, cuStream);
		cudaArray_t cuArr = nullptr;

		cudaGraphicsSubResourceGetMappedArray(&cuArr, cuResource, 0, 0);
		cudaResourceDesc cuArrRD;
		{
			cuArrRD.resType = cudaResourceTypeArray;
			cuArrRD.res.array.array = cuArr;
		}
		cudaSurfaceObject_t cuSurface;
		cudaCreateSurfaceObject(&cuSurface, &cuArrRD);
		{
			textureUpdate(cuSurface, HQueue.get_last_images(1), Fd.width, Fd.height);
			/*dim3 threads(32, 32);
			dim3 blocks(Fd.width >> 5, Fd.height >> 5);

			kernelTextureUpdate <<< blocks, threads >>>(
				reinterpret_cast<unsigned short*>(HQueue.get_last_images(1)),
				cuSurface, dim3(Fd.widt, Fd.height));*/
		}
		cudaDestroySurfaceObject(cuSurface);
		
		// Unmap the buffer for access by CUDA.
		cudaGraphicsUnmapResources(1, &cuResource, cuStream);
		cudaStreamSynchronize(cuStream);

		#pragma endregion
		/* ----------- */
		glBindTexture(GL_TEXTURE_2D, Tex);
		glGenerateMipmap(GL_TEXTURE_2D);
		Program->bind();
		Vao.bind();

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);

		Vao.release();
		Program->release();
		glBindTexture(GL_TEXTURE_2D, 0);
		GLenum error = glGetError();
		auto err_string = glGetString(error);
		if (error != GL_NO_ERROR && err_string)
			std::cerr << "[GL] " << err_string << '\n';

		doneCurrent();
	}

}