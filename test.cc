#include <iostream>
#include <vector>

void print_vector(std::vector<float> v, int n)
{
    std::cout << "Vector elements: ";
    for (int i = 0; i < n; ++i)
    {
        std::cout << v[i] << " "; // Will print: 0 0 0 0 0
    }
    std::cout << std::endl;
}

void fft_freqs(int time_transformation_size, int input_fps)
{
    std::vector<float> f0(time_transformation_size);
    std::vector<float> f1(time_transformation_size);
    std::vector<float> f2(time_transformation_size);

    // initialize f0 (f0 = [1, ..., 1])
    for (auto i = 0; i < time_transformation_size; i++)
        f0[i] = 1;

    // initialize f1
    // f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] * fs / n   if n is even
    if (time_transformation_size % 2 == 0)
    {
        for (auto i = 0; i < time_transformation_size / 2; i++)
            f1[i] = i * (float)(input_fps) / time_transformation_size;

        for (auto i = time_transformation_size / 2; i < time_transformation_size; i++)
            f1[i] = -(time_transformation_size - i) * (float)(input_fps) / time_transformation_size;
    }
    // f = [0, 1, ..., (n - 1) / 2, -(n - 1) / 2, ..., -1] * fs / n if n is odd
    else
    {
        for (auto i = 0; i < (time_transformation_size + 1) / 2; i++)
            f1[i] = i * (float)(input_fps) / time_transformation_size;

        for (auto i = time_transformation_size - 1; i > (time_transformation_size) / 2; i--)
            f1[i] = (i - time_transformation_size) * (float)(input_fps) / time_transformation_size;
    }

    // initialize f2 (f2 = f1^2)
    for (auto i = 0; i < time_transformation_size; i++)
        f2[i] = f1[i] * f1[i];

    print_vector(f0, time_transformation_size);
    print_vector(f1, time_transformation_size);
    print_vector(f2, time_transformation_size);
}

int main() { fft_freqs(11, 20000); }