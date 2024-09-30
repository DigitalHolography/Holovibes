#include "stdio.h"
#include <iostream>
#include <vector>

const int BMP_IDENTIFICATOR_SIZE = 2;

// Windows BMP-specific format data
struct bmp_identificator
{
    unsigned char identificator[BMP_IDENTIFICATOR_SIZE];
};

struct bmp_header
{
    unsigned int file_size;
    unsigned short creator1;
    unsigned short creator2;
    unsigned int bmp_offset;
};

struct bmp_device_independant_info
{
    unsigned int header_size;
    int width;
    int height;
    unsigned short num_planes;
    unsigned short bits_per_pixel;
    unsigned int compression;
    unsigned int bmp_byte_size;
    int hres;
    int vres;
    unsigned int num_colors;
    unsigned int num_important_colors;
};

#pragma pack(2)
struct HoloFileHeader
{
    char magic_number[4] = {'H', 'O', 'L', 'O'};
    uint16_t version = 5;
    uint16_t bits_per_pixel = 8;
    uint32_t img_width;
    uint32_t img_height;
    uint32_t img_nb = 1;
    uint64_t total_data_size;
    uint8_t endianness = 1;
    char padding[35];
};

void read_bmp(const char* path, const char* out_path)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL)
    {
        std::cerr << "InputFilter::read_bmp: IO error could not find file";
        exit(0);
    }

    HoloFileHeader* HoloHeader = new HoloFileHeader();

    bmp_identificator identificator;
    int e = fread(identificator.identificator, sizeof(identificator), 1, f);
    if (e < 0)
    {
        std::cerr << "InputFilter::read_bmp: IO error file too short (identificator)";
        exit(0);
    }

    // Check to make sure that the first two bytes of the file are the "BM"
    // identifier that identifies a bitmap image.
    if (identificator.identificator[0] != 'B' || identificator.identificator[1] != 'M')
    {
        std::cerr << "provided file is not in proper BMP format.\n";
        exit(0);
    }

    bmp_header header;
    e = fread((char*)(&header), sizeof(header), 1, f);
    if (e < 0)
    {
        std::cerr << "InputFilter::read_bmp: IO error file too short (header)";
        exit(0);
    }

    bmp_device_independant_info di_info;
    e = fread((char*)(&di_info), sizeof(di_info), 1, f);
    if (e < 0)
    {
        std::cerr << "InputFilter::read_bmp: IO error file too short (di_info)";
        exit(0);
    }

    // Check for this here and so that we know later whether we need to insert
    // each row at the bottom or top of the image.
    if (di_info.height < 0)
    {
        di_info.height = -di_info.height;
    }

    memset(HoloHeader->padding, 0, sizeof(HoloHeader->padding));

    // Extract image height and width from header
    HoloHeader->img_width = di_info.width;
    HoloHeader->img_height = di_info.height;

    HoloHeader->total_data_size =
        HoloHeader->img_height * HoloHeader->img_width * HoloHeader->img_nb * (HoloHeader->bits_per_pixel / 8);

    // Only support for 24-bit images
    if (di_info.bits_per_pixel != 24)
    {
        exit(0);
    }

    // No support for compressed images
    if (di_info.compression != 0)
    {
        exit(0);
    }

    // Write Holofile header
    FILE* out = fopen(out_path, "w+");
    fwrite(HoloHeader, sizeof(HoloFileHeader), 1, out);

    // Skip to bytecode
    e = fseek(f, header.bmp_offset, 0);

    // Read the rest of the data pixel by pixel
    unsigned char* pixel = new unsigned char[3];
    char color;

    // Store the image in temp buffer and flatten the colors to B&W
    auto temp = std::vector<std::vector<unsigned char>>(HoloHeader->img_height);

    printf("%d,%d\n", HoloHeader->img_width, HoloHeader->img_height);

    for (size_t i = 0; i < HoloHeader->img_height; i++)
    {
        auto vec = std::vector<unsigned char>(HoloHeader->img_width);
        for (size_t j = 0; j < HoloHeader->img_width; j++)
        {
            e = fread(pixel, sizeof(unsigned char), 3, f);
            color = (pixel[0] + pixel[1] + pixel[2]) / 3;
            vec[j] = color;
        }
        temp[HoloHeader->img_height - i - 1] = vec;
    }

    for (size_t i = 0; i < HoloHeader->img_height; i++)
        for (size_t j = 0; j < HoloHeader->img_width; j++)
            fwrite(&(temp[i][j]), sizeof(char), 1, out);

    fclose(f);
    fclose(out);
}

int main(int argc, char** argv)
{
    if (argc == 2)
    {
        read_bmp(argv[1], "out.holo");
        return 0;
    }
    if (argc == 3)
    {
        read_bmp(argv[1], argv[2]);
        return 0;
    }

    return 1;
}