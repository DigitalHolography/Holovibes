#include "rainbow_overlay.hh"

#include "API.hh"
#include "BasicOpenGLWindow.hh"
#include "point_gl.hh"

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
    initDraw();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, nullptr);

    endDraw();
}

void RainbowOverlay::setBuffer()
{
    parent_->makeCurrent();
    Program_->bind();

    auto fd = parent_->getFd();

    int red;
    int blue;
    if (API.composite.get_composite_kind() == CompositeKind::RGB)
    {
        red = API.composite.get_composite_p_red();
        blue = API.composite.get_composite_p_blue();
    }
    else
    {
        red = API.composite.get_composite_p_min_h();
        blue = API.composite.get_composite_p_max_h();
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
        red1 = units::PointFd(0, red);
        red2 = units::PointFd(fd.width, red);
        green1 = units::PointFd(0, green);
        green2 = units::PointFd(fd.width, green);
        blue1 = units::PointFd(0, blue);
        blue2 = units::PointFd(fd.width, blue);
    }
    else
    {
        red1 = units::PointFd(red, 0);
        red2 = units::PointFd(red, fd.height);
        green1 = units::PointFd(green, 0);
        green2 = units::PointFd(green, fd.height);
        blue1 = units::PointFd(blue, 0);
        blue2 = units::PointFd(blue, fd.height);
    }

    PointGL red1_gl(*parent_, red1);
    PointGL red2_gl(*parent_, red2);
    PointGL green1_gl(*parent_, green1);
    PointGL green2_gl(*parent_, green2);
    PointGL blue1_gl(*parent_, blue1);
    PointGL blue2_gl(*parent_, blue2);

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
        if (API.composite.get_composite_kind() == CompositeKind::RGB)
            API.composite.set_rgb_p(check_interval(zone_.src().x()), check_interval(zone_.dst().x()));
        else
            API.composite.set_composite_p_h(check_interval(zone_.src().x()), check_interval(zone_.dst().x()));
    }
    else
    {
        if (API.composite.get_composite_kind() == CompositeKind::RGB)
            API.composite.set_rgb_p(check_interval(zone_.src().y()), check_interval(zone_.dst().y()));
        else
            API.composite.set_composite_p_h(check_interval(zone_.src().y()), check_interval(zone_.dst().y()));
    }
}

unsigned int RainbowOverlay::check_interval(int x)
{
    const int max = API.transform.get_time_transformation_size() - 1;

    return static_cast<unsigned int>(std::min(max, std::max(x, 0)));
}
} // namespace holovibes::gui
