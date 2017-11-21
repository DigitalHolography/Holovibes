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

/*! \file
 *
 * Qt window displaying the 3D mode. */
#pragma once

# include "BasicOpenGLWindow.hh"

namespace holovibes
{
	namespace gui
	{
		class Vision3DWindow : public BasicOpenGLWindow
		{
		public:
			Vision3DWindow(QPoint p, QSize s, Queue& q, ComputeDescriptor& cd, const camera::FrameDescriptor& fd, Queue& stft_queue);
			~Vision3DWindow();

			virtual void initializeGL() override;
			virtual void initShaders() override;
			virtual void paintGL() override;

			void keyPressEvent(QKeyEvent *e) override;
			void keyReleaseEvent(QKeyEvent *e) override;
			void wheelEvent(QWheelEvent *e) override;


		private:
			std::string				vertex_shader_path_;
			std::string				fragment_shader_path_;
			Queue&					queue_;
			ComputeDescriptor&		compute_desc_;
			const camera::FrameDescriptor&	frame_desc_;
			int						voxel_;

			GLuint					colorBufferObject_;
			GLuint					vertexBufferObject_;
			float					*gpu_color_buffer_;
			GLfloat					*color_buffer_;

			glm::vec3				translate;
			glm::vec3				rotate;
			float					scale;

			GLint	*get_vertex_buffer();
			GLuint	load_matrix(glm::vec3 Translate, glm::vec3 const & Rotate, float scale);
			GLuint	push_gl_matrix(glm::mat4 mat, char *name);
		};
	}
}