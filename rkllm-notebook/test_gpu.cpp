#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS // Mengaktifkan cl::Error untuk C++
#include <iostream>
#include <vector>
#include <CL/opencl.hpp> // Menggunakan nama header baru yang direkomendasikan

int main() {
    try {
        // 1. Ambil platform OpenCL yang tersedia
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            std::cerr << "❌ Error: Tidak ada platform OpenCL yang ditemukan!" << std::endl;
            return 1;
        }

        std::cout << "=== INFORMASI GPU RK3588 (OpenCL C++) ===" << std::endl;

        for (auto& platform : platforms) {
            std::cout << "\n[Platform]" << std::endl;
            std::cout << "Nama Vendor  : " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
            std::cout << "Nama Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::cout << "Versi OpenCL : " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

            // 2. Ambil semua device (GPU) dari platform ini
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

            if (devices.empty()) {
                std::cout << "⚠️ Peringatan: Tidak ada GPU ditemukan di platform ini." << std::endl;
                continue;
            }

            for (auto& device : devices) {
                std::cout << "\n  [Device / Hardware GPU]" << std::endl;
                std::cout << "  Nama Perangkat: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << "  Tipe Perangkat: GPU" << std::endl;
                std::cout << "  Driver Versi  : " << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
                
                cl_ulong global_mem_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                std::cout << "  Memori Global : " << (global_mem_size / (1024 * 1024)) << " MB" << std::endl;
                
                // Tes inisialisasi Konteks Hardware
                try {
                    cl::Context context(device);
                    std::cout << "  Status Akses  : ✅ Sukses! GPU aktif dan merespon kode C++." << std::endl;
                } catch (cl::Error& err) {
                    std::cerr << "  Status Akses  : ❌ Gagal membuat konteks! Kode Error: " << err.err() << " (" << err.what() << ")" << std::endl;
                }
            }
        }
        std::cout << "\n=========================================" << std::endl;
    } catch (cl::Error& err) {
        std::cerr << "❌ Terjadi error OpenCL global: " << err.what() << " (Kode: " << err.err() << ")" << std::endl;
        return 1;
    }
    return 0;
}
