#include "TextureWindowHelper.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"
#include <string>
namespace holovibes::gui
{

TextureWindowHelper::TextureWindowHelper(QPoint p, QSize s, DisplayQueue* q, KindOfView k)
    : BasicOpenGLWindow(p, s, q, KindOfView::Filter2D)
{
    LOG_ERROR("Creating TextureWindowHelper");
}

TextureWindowHelper::~TextureWindowHelper() { LOG_ERROR("Destroying TextureWindowHelper"); }

// Initialization
void TextureWindowHelper::initializeGL()
{

    makeCurrent();
    initializeOpenGLFunctions();
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);

    initShaders();
    Vao.create();
    Vao.bind();
    Program->bind();

#pragma region Texture
    cudaTexture = new CudaTexture(fd_.width, fd_.height, fd_.depth, cuStream);
    if (!cudaTexture->init())
    {
        LOG_ERROR("Failed to initialize CUDA Texture");
    }
#pragma endregion

#pragma region Vertex Buffer Object
    const float data[] = {// Top-left
                          -1.f,
                          1.f, // vertex coord (-1.0f <-> 1.0f)
                          0.0f,
                          0.0f, // texture coord (0.0f <-> 1.0f)
                                // Top-right
                          1.f,
                          1.f,
                          1.f,
                          0.0f,
                          // Bottom-right
                          1.f,
                          -1.f,
                          1.f,
                          1.f,
                          // Bottom-left
                          -1.f,
                          -1.f,
                          0.0f,
                          1.f};
    glGenBuffers(1, &Vbo);
    glBindBuffer(GL_ARRAY_BUFFER, Vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#pragma endregion

#pragma region Element Buffer Object
    const GLuint elements[] = {0, 1, 2, 2, 3, 0};
    glGenBuffers(1, &Ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#pragma endregion

    setTransform();

    Program->release();
    Vao.release();

    glViewport(0, 0, width(), height());
    startTimer(1000 / UserInterfaceDescriptor::instance().display_rate_);
}
void TextureWindowHelper::initShaders()
{
    std::string vertex_shader_path = "vertex.holo.glsl";
    std::string fragment_shader_path = "fragment.tex.glsl";
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / vertex_shader_path).string()));
    Program->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / fragment_shader_path).string()));
    Program->link();
}
// Rendering
void TextureWindowHelper::paintGL()
{
    void* frame = output_->get_last_image();
    if (!frame)
        return;

    makeCurrent();
    glClear(GL_COLOR_BUFFER_BIT);
    Vao.bind();
    Program->bind();

    cudaTexture->update(frame, output_->get_fd());

    glBindTexture(GL_TEXTURE_2D, cudaTexture->getTextureID());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    Program->release();
    Vao.release();

    overlay_manager_.draw();
}

} // namespace holovibes::gui