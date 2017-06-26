/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8x   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "3DVision.hh"
#include <stdlib.h>

namespace holovibes
{
	namespace gui
	{
		Vision3DWindow::Vision3DWindow(QPoint p, QSize s, Queue& q, ComputeDescriptor& cd, const FrameDescriptor& fd, Queue& stft_queue) :
			BasicOpenGLWindow(p, s, q, KindOfView::Vision3D),
			vertex_shader_path_("shaders/vertex.3d.glsl"),
			fragment_shader_path_("shaders/fragment.3d.glsl"),
			queue_(stft_queue),
			compute_desc_(cd),
			colorBufferObject_(0),
			vertexBufferObject_(0),
			voxel_(fd.frame_res() * cd.nsamples.load()),
			frame_desc_(fd),
			color_buffer_(nullptr),
			translate(glm::vec3(0, 0, 10)),
			rotate(glm::vec3(M_PI, 0, M_PI)),
			scale(10 / static_cast<float>(fd.width))
		{
			cuResource = nullptr;
		}

		Vision3DWindow::~Vision3DWindow()
		{
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			if (cuResource)
				cudaGraphicsUnmapResources(1, &cuResource, cuStream);
		}

		GLint *Vision3DWindow::get_vertex_buffer()
		{
			GLint *vertex_buffer = new GLint[voxel_ * 3];
			ushort middle_samples = compute_desc_.nsamples.load() / 2;
			ushort middle_width = frame_desc_.width / 2;
			ushort middle_height = frame_desc_.height / 2;

			for (ushort z = 0; z < compute_desc_.nsamples.load(); ++z)
				for (ushort y = 0; y < frame_desc_.height; ++y)
					for (ushort x = 0; x < frame_desc_.width; ++x)
					{
						uint index = (x + y * frame_desc_.width + z * frame_desc_.frame_res()) * 3;
						vertex_buffer[index] = (x - middle_width);
						vertex_buffer[index + 1] = (y - middle_height);
						vertex_buffer[index + 2] = (z - middle_samples);
					}
			return (vertex_buffer);
		}

		GLuint	Vision3DWindow::push_gl_matrix(glm::mat4 matrix, char *name)
		{
			GLuint matrix_id = glGetUniformLocation(Program->programId(), name);
			glUniformMatrix4fv(matrix_id, 1, GL_FALSE, glm::value_ptr(matrix));
			return (matrix_id);
		}

		GLuint Vision3DWindow::load_matrix(glm::vec3 Translate, glm::vec3 const & Rotate, float scale)
		{
			glm::mat4 Projection = glm::perspective(glm::radians(45.0f), frame_desc_.width / static_cast<float>(frame_desc_.height), 0.1f, 100.f);
			glm::mat4 View = glm::translate(glm::mat4(1.0f), -Translate);
			View = glm::rotate(View, Rotate.z, glm::vec3(0.0f, 0.0f, -1.0f));
			View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
			View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
			glm::mat4 MVP = Projection * View * Model;
			GLuint matrix_id = glGetUniformLocation(Program->programId(), "MVP");
			glUniformMatrix4fv(matrix_id, 1, GL_FALSE, glm::value_ptr(MVP));
			return (matrix_id);
		}

		void Vision3DWindow::initializeGL()
		{
			GLint	*vertex_buffer = get_vertex_buffer();

			initializeOpenGLFunctions();

			glClearColor(0.f, 0.f, 0.f, 0.f);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);


			Vao.create();
			Vao.bind();
			initShaders();

			glGenBuffers(1, &vertexBufferObject_);
			glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject_);
			glBufferData(GL_ARRAY_BUFFER, voxel_ * 3 * sizeof(GLint), vertex_buffer, GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_INT, GL_FALSE, 0, NULL);
			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glGenBuffers(1, &colorBufferObject_);
			glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject_);
			glBufferData(GL_ARRAY_BUFFER, voxel_ * sizeof(GLint), color_buffer_, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL);
			glDisableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			cudaGraphicsGLRegisterBuffer(&cuResource, colorBufferObject_,
				cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
			cudaGraphicsMapResources(1, &cuResource, cuStream);
			cudaGraphicsResourceGetMappedPointer(&cuPtrToPbo, &sizeBuffer, cuResource);

			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, colorBufferObject_);

			startTimer(1000 / static_cast<float>(compute_desc_.display_rate.load()));

			delete[] vertex_buffer;
		}

		void Vision3DWindow::paintGL()
		{
			uint offset = compute_desc_.pindex.load() * frame_desc_.frame_res();
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			load_matrix(translate, rotate, scale);

			cudaMemcpy((char *)cuPtrToPbo + offset, (char *)queue_.get_buffer() + offset, sizeBuffer - offset, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
			cudaStreamSynchronize(cuStream);

			glBufferSubData(GL_ARRAY_BUFFER, offset, sizeBuffer, color_buffer_);
			glDrawArrays(GL_POINTS, offset, voxel_);
		}

		void Vision3DWindow::keyPressEvent(QKeyEvent *e)
		{
			float rotation_step = 0.1f * static_cast<float>(M_PI);
			float translate_step = 10.f * scale;
			switch (e->key())
			{
			case Qt::Key::Key_8:
				translate.y += translate_step;
				break;
			case Qt::Key::Key_2:
				translate.y -= translate_step;
				break;
			case Qt::Key::Key_6:
				translate.x += translate_step;
				break;
			case Qt::Key::Key_4:
				translate.x -= translate_step;
				break;
			case Qt::Key::Key_7:
				translate.z -= translate_step;
				break;
			case Qt::Key::Key_9:
				translate.z += translate_step;
				break;
			case Qt::Key::Key_A:
				rotate.x += rotation_step;
				break;
			case Qt::Key::Key_D:
				rotate.x -= rotation_step;
				break;
			case Qt::Key::Key_W:
				rotate.y -= rotation_step;
				break;
			case Qt::Key::Key_S:
				rotate.y += rotation_step;
				break;
			case Qt::Key::Key_Q:
				rotate.z -= rotation_step;
				break;
			case Qt::Key::Key_E:
				rotate.z += rotation_step;
				break;
			case Qt::Key::Key_Space:
				rotate = glm::vec3(M_PI, 0, M_PI);
				translate = glm::vec3(0, 0, 10);
				scale = 1.f;
				break;
			}
			load_matrix(translate, rotate, scale);
		}

		void Vision3DWindow::keyReleaseEvent(QKeyEvent *e)
		{

		}

		void Vision3DWindow::wheelEvent(QWheelEvent *e)
		{
			float scale_step = 0.1f * scale;

			if (e->angleDelta().y() > 0)
				scale += scale_step;
			else if (e->angleDelta().y() < 0)
				scale -= scale_step;
		}

		void Vision3DWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, vertex_shader_path_.c_str());
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, fragment_shader_path_.c_str());
			Program->bind();
		}
	}
}