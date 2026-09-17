// Validate the features Holovibes needs without requiring CUDA or camera hardware.
#include <QApplication>
#include <QChart>
#include <QLineSeries>
#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        QApplication app(argc, argv);
        QChart chart;
        auto* series = new QLineSeries;
        series->append(1, 2);
        chart.addSeries(series);
        boost::program_options::options_description options("smoke");
        options.add_options()("frames", boost::program_options::value<int>()->default_value(2));
        const auto settings = nlohmann::json::parse(R"({"frames":2})");
        const glm::vec2 dimensions(32, 32);

        const char* hdfPath = "dependency-smoke.h5";
        {
            H5::H5File file(hdfPath, H5F_ACC_TRUNC);
            const hsize_t count[] = {1};
            H5::DataSpace space(1, count);
            auto data = file.createDataSet("frames", H5::PredType::NATIVE_INT, space);
            int value = settings.at("frames").get<int>();
            data.write(&value, H5::PredType::NATIVE_INT);
            int read = 0;
            data.read(&read, H5::PredType::NATIVE_INT);
            if (read != value) throw std::runtime_error("HDF5 round trip failed");
        }
        std::filesystem::remove(hdfPath);
        for (const auto& format : {std::pair{"avi", cv::VideoWriter::fourcc('M','J','P','G')},
                                  std::pair{"mp4", cv::VideoWriter::fourcc('m','p','4','v')}})
        {
            const auto path = std::string("dependency-smoke.") + format.first;
            cv::VideoWriter writer(path, cv::CAP_FFMPEG, format.second, 10,
                                   cv::Size(static_cast<int>(dimensions.x), static_cast<int>(dimensions.y)));
            if (!writer.isOpened()) throw std::runtime_error("Cannot encode " + path);
            cv::Mat frame(32, 32, CV_8UC3, cv::Scalar(30, 100, 200));
            writer.write(frame);
            writer.write(frame);
            writer.release();
            cv::VideoCapture reader(path, cv::CAP_FFMPEG);
            if (!reader.read(frame) || frame.rows != 32 || frame.cols != 32)
                throw std::runtime_error("Cannot decode " + path);
            reader.release();
            std::filesystem::remove(path);
        }
        spdlog::info("Qt Charts, HDF5 C++, AVI/MP4, Boost, JSON and GLM smoke test passed");
        return 0;
    }
    catch (const H5::Exception& error) { std::cerr << error.getDetailMsg() << '\n'; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
    return 1;
}
