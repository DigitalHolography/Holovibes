#include "API.hh"
#include "rainbow_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "API.hh"

namespace holovibes::gui
{
RainbowOverlay::RainbowOverlay(BasicOpenGLWindow* parent)
    : Overlay(Rainbow, parent)
{
    LOG_FUNC();

    alpha_ = 0.2f;
    display_ = true;
}

void RainbowOverlay::init()
{
    // Program_ already bound by caller (initProgram)

    Vao_.bind();

    /* Set vertices position
     * There is 3 vertices on each horizontal line of the rectangular area
     * in order to have two gradient in the same rectangle. */
    const float vertices[] = {
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
        0.f,
    };
    glGenBuffers(1, &verticesIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(verticesShader_);
    glVertexAttribPointer(verticesShader_, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glDisableVertexAttribArray(verticesShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set color
    const float colorData[] = {
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
    };
    glGenBuffers(1, &colorIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(colorShader_);
    glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /* Set vertices order :
     *
     *   0---1---2
     *   |       |
     *   5---4---3
     */
    const GLuint elements[] = {0, 1, 4, 4, 5, 0, 1, 2, 3, 3, 4, 1};
    glGenBuffers(1, &elemIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Vao_.release();

    // Program_ released by caller (initProgram)
}

void RainbowOverlay::draw()
{
    parent_->makeCurrent();
    setBuffer();
    Vao_.bind();
    Program_->bind();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);
    setUniform();

    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, nullptr);

    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Program_->release();
    Vao_.release();
}

void RainbowOverlay::setBuffer()
{
    parent_->makeCurrent();
    Program_->bind();

    units::ConversionData convert(parent_);
    auto fd = parent_->getFd();

    int red;
    int blue;
    if (api::get_composite_kind() == CompositeKind::RGB)
    {
        red = api::get_composite_p_red();
        blue = api::get_composite_p_blue();
    }
    else
    {
        red = api::get_composite_p_min_h();
        blue = api::get_composite_p_max_h();
    }
    int green = (red + blue) / 2;
    units::PointFd red1;
    units::PointFd red2;
    units::PointFd green1;
    units::PointFd green2;
    units::PointFd blue1;
    units::PointFd blue2;

    if (parent_->getKindOfView() == KindOfView::SliceXZ)
    {
        red1 = units::PointFd(convert, 0, red);
        red2 = units::PointFd(convert, fd.width, red);
        green1 = units::PointFd(convert, 0, green);
        green2 = units::PointFd(convert, fd.width, green);
        blue1 = units::PointFd(convert, 0, blue);
        blue2 = units::PointFd(convert, fd.width, blue);
    }
    else
    {
        red1 = units::PointFd(convert, red, 0);
        red2 = units::PointFd(convert, red, fd.height);
        green1 = units::PointFd(convert, green, 0);
        green2 = units::PointFd(convert, green, fd.height);
        blue1 = units::PointFd(convert, blue, 0);
        blue2 = units::PointFd(convert, blue, fd.height);
    }

    units::PointOpengl red1_gl = red1;
    units::PointOpengl red2_gl = red2;
    units::PointOpengl green1_gl = green1;
    units::PointOpengl green2_gl = green2;
    units::PointOpengl blue1_gl = blue1;
    units::PointOpengl blue2_gl = blue2;

    const float subVertices[] = {red1_gl.x(),
                                 red1_gl.y(),
                                 green1_gl.x(),
                                 green1_gl.y(),
                                 blue1_gl.x(),
                                 blue1_gl.y(),
                                 blue2_gl.x(),
                                 blue2_gl.y(),
                                 green2_gl.x(),
                                 green2_gl.y(),
                                 red2_gl.x(),
                                 red2_gl.y()};

    // Updating the buffer at verticesIndex_ with new coordinates
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Program_->release();
}

void RainbowOverlay::move(QMouseEvent* e)
{
    if (e->buttons() != Qt::LeftButton)
        return;

    zone_.setDst(getMousePos(e->pos()));
    if (parent_->getKindOfView() == KindOfView::SliceYZ)
    {
        if (api::get_composite_kind() == CompositeKind::RGB)
            api::set_rgb_p(check_interval(zone_.src().x()), check_interval(zone_.dst().x()));
        else
            api::set_composite_p_h(check_interval(zone_.src().x()), check_interval(zone_.dst().x()));
    }
    else
    {
        if (api::get_composite_kind() == CompositeKind::RGB)
            api::set_rgb_p(check_interval(zone_.src().y()), check_interval(zone_.dst().y()));
        else
            api::set_composite_p_h(check_interval(zone_.src().y()), check_interval(zone_.dst().y()));
    }
}

unsigned int RainbowOverlay::check_interval(int x)
{
    const int max = api::get_time_transformation_size() - 1;

    return static_cast<unsigned int>(std::min(max, std::max(x, 0)));
}
} // namespace holovibes::gui
