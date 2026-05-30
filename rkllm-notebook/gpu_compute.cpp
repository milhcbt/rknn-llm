#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <CL/opencl.hpp>

// Source code Kernel OpenCL yang akan dieksekusi oleh core GPU Mali
const char* kernel_source = 
"__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, int N) {"
"    int id = get_global_id(0);"
"    if (id < N) {"
"        C[id] = A[id] + B[id];"
"    }"
"}";

int main() {
    const int N = 10000000; // 10 Juta Elemen Data
    size_t bytes = N * sizeof(float);

    // 1. Alokasi memori di sisi CPU (Host)
    std::vector<float> h_A(N, 1.0f); // Isinya angka 1.0 semua
    std::vector<float> h_B(N, 2.0f); // Isinya angka 2.0 semua
    std::vector<float> h_C(N, 0.0f); // Tempat menampung hasil (Harus jadi 3.0)

    try {
        // 2. Inisialisasi Hardware GPU
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices.front();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        std::cout << "🚀 Menjalankan Komputasi pada: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "📊 Mengolah " << N << " elemen data (Total: " << (bytes / (1024 * 1024)) * 3 << " MB alokasi VRAM)..." << std::endl;

        // Start timer pencatatan performa total GPU
        auto start = std::chrono::high_resolution_clock::now();

        // 3. Alokasi Buffer di Memori GPU (Device)
        cl::Buffer d_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data());
        cl::Buffer d_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data());
        cl::Buffer d_C(context, CL_MEM_WRITE_ONLY, bytes);

        // 4. Kompilasi runtime kode Kernel OpenCL untuk arsitektur Valhall Mali-G610
        cl::Program::Sources sources;
        sources.push_back({kernel_source, strlen(kernel_source)});
        cl::Program program(context, sources);
        program.build({device});

        cl::Kernel kernel(program, "vector_add");

        // 5. Atur Argumen Fungsi Kernel
        kernel.setArg(0, d_A);
        kernel.setArg(1, d_B);
        kernel.setArg(2, d_C);
        kernel.setArg(3, N);

        // 6. Eksekusi Komputasi Paralel di GPU
        cl::NDRange global_size(N);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
        queue.finish(); // Tunggu sampai seluruh core GPU selesai menghitung

        // 7. Salin kembali hasil dari VRAM GPU ke RAM CPU
        queue.enqueueReadBuffer(d_C, CL_TRUE, 0, bytes, h_C.data());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        // 8. Validasi Hasil Hitungan GPU oleh CPU
        bool benar = true;
        for (int i = 0; i < N; i++) {
            if (h_C[i] != 3.0f) {
                benar = false;
                break;
            }
        }

        if (benar) {
            std::cout << "✅ BERHASIL! GPU memberikan hasil perhitungan matematika yang 100% akurat." << std::endl;
            std::cout << "⏱️ Waktu Eksekusi GPU (termasuk transfer memori): " << duration.count() << " ms" << std::endl;
        } else {
            std::cout << "❌ GAGAL! Terjadi korupsi data atau kesalahan hitung pada inti pemrosesan GPU." << std::endl;
        }

    } catch (cl::Error& err) {
        std::cerr << "❌ OpenCL Error: " << err.what() << " (Kode: " << err.err() << ")" << std::endl;
        return 1;
    }

    return 0;
}
