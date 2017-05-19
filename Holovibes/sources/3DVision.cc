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

#include "3DVision.hh"
#include <stdlib.h>

namespace holovibes
{
	namespace gui
	{
		Vision3DWindow::Vision3DWindow(QPoint p, QSize s, Queue& q, ComputeDescriptor& cd, const FrameDescriptor& fd, Queue& stft_queue) :
			BasicOpenGLWindow(p, s, q, KindOfView::Vision3D),
			vertex_shader_(""),
			fragment_shader_(""),
			queue_(stft_queue),
			compute_desc_(cd),
			colorbufferobject_(0),
			voxel_(fd.frame_res() * compute_desc_.nsamples.load()),
			frame_desc_(fd),
			translate(glm::vec3(0, 0, 10)),
			rotate(glm::vec3(M_PI, 0, M_PI)),
			scale(1),
			gpu_color_buffer_(nullptr),
			color_buffer_(nullptr)
		{
			color_buffer_ = new GLfloat[voxel_];
		}

		Vision3DWindow::~Vision3DWindow()
		{
			delete[] color_buffer_;
		}

		GLfloat *Vision3DWindow::get_vertex_buffer()
		{
			GLfloat *vertex_buffer = new GLfloat[voxel_ * 3];

			for (int k = 0; k < compute_desc_.nsamples.load(); ++k)
				for (int j = 0; j < frame_desc_.height; ++j)
					for (int i = 0; i < frame_desc_.width; ++i)
					{
						int index = i * 3 + j * frame_desc_.width * 3 + k * frame_desc_.frame_res() * 3;
						vertex_buffer[index] = (i - frame_desc_.width / 2) * SPACE_BETWEEN_POINTS;
						vertex_buffer[index + 1] = (j - frame_desc_.height / 2) * SPACE_BETWEEN_POINTS;
						vertex_buffer[index + 2] = (k - compute_desc_.nsamples.load() / 2) * SPACE_BETWEEN_POINTS;
					}
			return (vertex_buffer);
		}

		GLuint	Vision3DWindow::push_gl_matrix(glm::mat4 matrix, char *name)
		{
			GLuint	matrix_id;

			matrix_id = glGetUniformLocation(Program->programId(), name);
			glUniformMatrix4fv(matrix_id, 1, GL_FALSE, glm::value_ptr(matrix));
			return (matrix_id);
		}

		void Vision3DWindow::load_matrix(glm::vec3 Translate, glm::vec3 const & Rotate, float scale)
		{
			glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 2048.f / 2048.f, 0.1f, 100.f);
			glm::mat4 View = glm::translate(glm::mat4(1.0f), -Translate);
			View = glm::rotate(View, Rotate.z, glm::vec3(0.0f, 0.0f, -1.0f));
			View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
			View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
			glm::mat4 MVP = Projection * View * Model;
			push_gl_matrix(MVP, "MVP");
		}

		GLuint Vision3DWindow::create_gl_buffer(GLuint *gl_buffer, const void *data, size_t nb_vertex, size_t elts_per_vec)
		{
			static GLuint index = 0;

			glGenBuffers(1, gl_buffer);
			glBindBuffer(GL_ARRAY_BUFFER, *gl_buffer);
			glBufferData(GL_ARRAY_BUFFER, nb_vertex * sizeof(GLfloat), data, GL_STREAM_DRAW);
			glEnableVertexAttribArray(index);
			glVertexAttribPointer(index, elts_per_vec, GL_FLOAT, GL_FALSE, 0, NULL);
			glDisableVertexAttribArray(index);
			index++;
			return (index - 1);
		}

		void Vision3DWindow::initializeGL()
		{
			makeCurrent();
			initializeOpenGLFunctions();
			glClearColor(0.f, 0.f, 0.f, 0.f);

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);

			GLfloat	*vertex_buffer = get_vertex_buffer();

			cudaMemcpy(color_buffer_, queue_.get_buffer(), voxel_ * sizeof(GLfloat), cudaMemcpyDeviceToHost);

			Vao.create();
			Vao.bind();
			initShaders();
			create_gl_buffer(&Vbo, vertex_buffer, voxel_ * 3, 3);
			create_gl_buffer(&colorbufferobject_, color_buffer_, voxel_, 1);

			delete[] vertex_buffer;

			startTimer(DISPLAY_RATE);
		}

		void Vision3DWindow::paintGL()
		{
			makeCurrent();
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


			//Program->bind();
			load_matrix(translate, rotate, scale);

			cudaMemcpyAsync(color_buffer_, queue_.get_buffer(), voxel_ * sizeof(GLfloat), cudaMemcpyDeviceToHost);

			glBindBuffer(GL_ARRAY_BUFFER, colorbufferobject_);
			glBufferSubData(GL_ARRAY_BUFFER, 0, voxel_ * sizeof(GLfloat), color_buffer_);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glDrawArrays(GL_POINTS, compute_desc_.pindex.load() * frame_desc_.frame_res(), voxel_);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(0);

			//Program->release();

		}

		void Vision3DWindow::keyPressEvent(QKeyEvent *e)
		{
			switch (e->key())
			{
			case Qt::Key::Key_8:
				translate.y -= 0.1f;
				break;
			case Qt::Key::Key_2:
				translate.y += 0.1f;
				break;
			case Qt::Key::Key_6:
				translate.x -= 0.1f;
				break;
			case Qt::Key::Key_4:
				translate.x += 0.1f;
				break;
			case Qt::Key::Key_7:
				translate.z -= 0.1f;
				break;
			case Qt::Key::Key_9:
				translate.z += 0.1f;
				break;
			case Qt::Key::Key_A:
				rotate.x += 0.1f * M_PI;
				break;
			case Qt::Key::Key_D:
				rotate.x -= 0.1f * M_PI;
				break;
			case Qt::Key::Key_W:
				rotate.y += 0.1f * M_PI;
				break;
			case Qt::Key::Key_S:
				rotate.y -= 0.1f * M_PI;
				break;
			case Qt::Key::Key_Q:
				rotate.z += 0.1f * M_PI;
				break;
			case Qt::Key::Key_E:
				rotate.z -= 0.1f * M_PI;
				break;
			case Qt::Key::Key_Space:
				rotate = glm::vec3(M_PI, 0, M_PI);
				translate = glm::vec3(0, 0, 10);
				scale = 1.f;
				break;
			}
		}

		void Vision3DWindow::keyReleaseEvent(QKeyEvent *e)
		{

		}

		void Vision3DWindow::wheelEvent(QWheelEvent *e)
		{
			if (e->angleDelta().y() > 0)
			{
				scale += 0.1f * scale;
			}
			else if (e->angleDelta().y() < 0)
			{
				scale -= 0.1f * scale;
			}
		}

		void Vision3DWindow::initShaders()
		{
			Program = new QOpenGLShaderProgram();
			Program->addShaderFromSourceFile(QOpenGLShader::Vertex, "shaders/vertex.3d.glsl");
			Program->addShaderFromSourceFile(QOpenGLShader::Fragment, "shaders/fragment.3d.glsl");
			Program->bind();
		}
	}
}