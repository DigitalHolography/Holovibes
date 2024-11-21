#include "common.cuh"
#include "tools_debug.hh"
#include "cuda_memory.cuh"

namespace
{

void write_1D_float_array_to_file(const float* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
                outFile << " "; // Separate values in a row by a space
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}
} // namespace

float* load_CSV_to_float_array(const std::filesystem::path& path)
{
    std::string filename = path.string();
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return nullptr;
    }

    std::vector<float> values;
    std::string line;

    // Lire le fichier ligne par ligne
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        // Lire chaque valeur séparée par des virgules (ou espaces, selon le fichier)
        while (std::getline(ss, value, ','))
        {
            try
            {
                values.push_back(std::stof(value)); // Convertir la valeur en float et l'ajouter au vecteur
            }
            catch (const std::invalid_argument&)
            {
                std::cerr << "Erreur de conversion de valeur : " << value << std::endl;
            }
        }
    }

    file.close();

    // Copier les valeurs dans un tableau float*
    float* dataArray = new float[values.size()];
    for (int i = 0; i < values.size(); ++i)
    {
        dataArray[i] = values[i];
    }

    return dataArray;
}

void load_bin_video_file(const std::filesystem::path& path, float* output, cudaStream_t stream)
{
    const int width = 512;
    const int height = 512;
    const int frames = 506;
    const int total_size = width * height * frames;

    // Allouer un tableau pour stocker les données
    float* video_data = new float[total_size];

    // Ouvrir le fichier binaire en mode lecture
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier video.bin" << std::endl;
        return;
    }

    // Lire les données dans le tableau
    file.read(reinterpret_cast<char*>(video_data), total_size * sizeof(float));
    if (!file)
    {
        std::cerr << "Erreur : Lecture du fichier incomplète" << std::endl;
        delete[] video_data;
        return;
    }

    file.close();

    cudaXMemcpyAsync(output, video_data, sizeof(float) * total_size, cudaMemcpyHostToDevice, stream);
    delete[] video_data;
}

void print_in_file_gpu(float* input, uint rows, uint col, std::string filename, cudaStream_t stream)
{
    if (input == nullptr)
        return;
    float* result = new float[rows * col];
    cudaXMemcpyAsync(result, input, rows * col * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);
    write_1D_float_array_to_file(result, rows, col, "test_" + filename + ".txt");
}

void print_in_file_cpu(float* input, uint rows, uint col, std::string filename)
{
    if (input == nullptr)
        return;
    write_1D_float_array_to_file(input, rows, col, "test_" + filename + ".txt");
}
