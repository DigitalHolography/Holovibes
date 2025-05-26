/*! \file
 *
 */

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <sstream>

#include "input_mraw_file.hh"

#include "API.hh"
#include "all_struct.hh"
#include "compute_settings_struct.hh"
#include "file_exception.hh"
#include "holovibes_config.hh"
#include "logger.hh"

bool extract_xml_from_cihx(const std::string& path, std::string& xml)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
        return false;

    std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    // On cherche le tout premier '<?xml' (souvent le vrai début)
    size_t xml_start = content.find("<?xml");
    if (xml_start == std::string::npos)
        xml_start = content.find("<cih"); // Parfois ils zappent la déclaration XML

    size_t xml_end = content.rfind("</cih>");
    if (xml_start == std::string::npos || xml_end == std::string::npos)
        return false;
    xml_end += std::string("</cih>").length();

    xml = content.substr(xml_start, xml_end - xml_start);

    // Enlève tout ce qui suit le XML (zéro, retour-chariot, binaire)
    while (!xml.empty() && (xml.back() == 0 || xml.back() == '\r' || xml.back() == '\n'))
        xml.pop_back();

    return true;
}

namespace holovibes::io_files
{
InputMrawFile::InputMrawFile(const std::string& file_path, const std::string& cih)
    : InputFrameFile(file_path)
    , MrawFile()
{

    LOG_FUNC(file_path);

    InputMrawFile::load_cih(cih);

    frame_size_ = fd_.get_frame_size();
}

void InputMrawFile::set_pos_to_frame(size_t frame_id)
{
    std::fpos_t frame_offset = frame_size_ * frame_id;

    if (std::fsetpos(file_, &frame_offset) != 0)
        throw FileException("Unable to seek the frame requested");
}

void InputMrawFile::load_cih(const std::string& cih)
{
    LOG_FUNC();

    // Initialisation des valeurs
    img_nb = 0;
    fd_.byteEndian = camera::Endianness::LittleEndian;
    fd_.width = 0;
    fd_.height = 0;
    fd_.depth = camera::PixelDepth::Bits0;

    // Détecter l'extension
    auto ext_pos = cih.find_last_of('.');
    std::string ext = (ext_pos == std::string::npos) ? "" : cih.substr(ext_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "cihx") // Format XML
    {
        std::string xml;
        if (extract_xml_from_cihx(cih, xml))
        {
            try
            {
                std::stringstream ss(xml);
                boost::property_tree::ptree pt;
                LOG_INFO("First 200 chars: {}", xml.substr(0, 200));
                LOG_INFO("Last 200 chars: {}", xml.substr(xml.length() - 200));
                boost::property_tree::read_xml(ss, pt);

                boost::optional<int> totalFrame = pt.get_optional<int>("cih.frameInfo.totalFrame");
                boost::optional<int> recordedFrame = pt.get_optional<int>("cih.frameInfo.recordedFrame");
                if (totalFrame)
                    img_nb = totalFrame.get();
                else if (recordedFrame)
                    img_nb = recordedFrame.get();

                // width / height
                boost::optional<int> width = pt.get_optional<int>("cih.imageDataInfo.resolution.width");
                boost::optional<int> height = pt.get_optional<int>("cih.imageDataInfo.resolution.height");
                if (!width) // fallback sur imageFileInfo
                    width = pt.get_optional<int>("cih.imageFileInfo.resolution.width");
                if (!height)
                    height = pt.get_optional<int>("cih.imageFileInfo.resolution.height");

                if (width)
                    fd_.width = width.get();
                if (height)
                    fd_.height = height.get();

                // bit depth
                boost::optional<int> bit = pt.get_optional<int>("cih.imageDataInfo.colorInfo.bit");
                if (bit)
                    fd_.depth = camera::PixelDepth(bit.get() / 8);
            }
            catch (std::exception& e)
            {
                LOG_ERROR("Error parsing CIHX file: {}", e.what());
            }
        }
        else
        {
            LOG_ERROR("Cannot extract XML from CIHX: {}", cih);
        }
    }
    else // Format texte .cih
    {
        std::ifstream fin(cih);
        if (!fin)
        {
            LOG_ERROR("Cannot open CIH file: {}", cih);
            return;
        }

        std::string line;
        while (std::getline(fin, line))
        {
            // Enlève les commentaires éventuels ou espaces
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

            // Split clé: valeur
            std::size_t sep = line.find(':');
            if (sep == std::string::npos)
                continue;

            std::string key = line.substr(0, sep);
            std::string value = line.substr(sep + 1);

            // Trim key et value
            key.erase(key.find_last_not_of(" \t") + 1);
            key.erase(0, key.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));

            // Cas insensible à la casse sur la clé
            std::string lkey = key;
            std::transform(lkey.begin(), lkey.end(), lkey.begin(), ::tolower);

            if (lkey == "total frame" || lkey == "original total frame")
                img_nb = std::stoi(value);
            else if (lkey == "image width")
                fd_.width = std::stoi(value);
            else if (lkey == "image height")
                fd_.height = std::stoi(value);
            else if (lkey == "color bit")
                fd_.depth = camera::PixelDepth(std::stoi(value) / 8);

            if (img_nb && fd_.width && fd_.height && fd_.depth)
                break;
        }
    }

    LOG_ERROR("img_nb={}, width={}, height={}, depth={}", img_nb, fd_.width, fd_.height, static_cast<int>(fd_.depth));
    LOG_ERROR("Exiting");
}

json InputMrawFile::import_compute_settings(void) { return json{}; }

void InputMrawFile::import_info(void) const { LOG_ERROR("unused"); }

} // namespace holovibes::io_files