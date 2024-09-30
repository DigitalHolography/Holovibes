#include "stdio.h"
#include <iostream>
#include <vector>

#pragma pack(2)
struct HoloFileHeader
{
    char magic_number[4] = {'H', 'O', 'L', 'O'};
    uint16_t version = 5;
    uint16_t bits_per_pixel = 8;
    uint32_t img_width;
    uint32_t img_height;
    uint32_t img_nb;
    uint64_t total_data_size;
    uint8_t endianness = 1;
    char padding[35];
};

void square_holo(const char* path, const char* square_path)
{
    // READING
    FILE* f = fopen(path, "rb");
    if (f == NULL)
    {
        std::cerr << "Square_holo: IO error could not find file (1)";
        exit(0);
    }

    // Header
    HoloFileHeader* HoloHeader = new HoloFileHeader();
    fread(HoloHeader, sizeof(HoloFileHeader), 1, f);

    // Images
    uint32_t w = HoloHeader->img_width;
    uint32_t h = HoloHeader->img_height;

    uint32_t big_dim = w > h ? w : h;
    HoloHeader->img_height = big_dim;
    HoloHeader->img_width = big_dim;

    uint32_t nb = HoloHeader->img_nb;
    HoloHeader->total_data_size = (size_t)big_dim * big_dim * HoloHeader->img_nb;

    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * w * h * nb);
    fread(data, sizeof(unsigned char) * w * h, nb, f);

    fclose(f);

    // WRITING
    FILE* square_f = fopen(square_path, "w+");
    if (f == NULL)
    {
        std::cerr << "Square_holo: IO error could not find file (2)";
        free(data);
        exit(0);
    }

    fwrite(HoloHeader, sizeof(HoloFileHeader), 1, square_f);
    unsigned char* padding;

    if (h < big_dim)
    {
        std::cout << "A" << std::endl;

        padding = (unsigned char*)calloc((big_dim - h) * w, sizeof(unsigned char));

        for (size_t i = 0; i < nb; i++)
        {
            std::cout << "(" << i << ") ";
            fwrite(data + i * h * w, sizeof(unsigned char), h * w, square_f);
            fwrite(padding, sizeof(unsigned char), (big_dim - h) * w, square_f);
        }
    }
    else
    {
        std::cout << "B" << std::endl;

        padding = (unsigned char*)calloc(big_dim - w, sizeof(unsigned char));

        for (size_t i = 0; i < nb; i++)
        {
            for (size_t j = 0; j < big_dim; j++)
            {
                std::cout << "(" << i << ", " << j << ") ";
                fwrite(data + i * h * w + j * w, sizeof(unsigned char), w, square_f);
                if (w < big_dim)
                    fwrite(padding, sizeof(unsigned char), big_dim - w, square_f);
            }
        }
    }

    free(data);
    free(padding);
    fclose(square_f);
}

int main(int argc, char** argv)
{
    if (argc == 2)
    {
        square_holo(argv[1], "out.holo");
        return 0;
    }

    if (argc == 3)
    {
        square_holo(argv[1], argv[2]);
        return 0;
    }

    return 1;
}