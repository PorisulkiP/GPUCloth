#pragma once

#ifdef MAIN_EXPORTS
#define MAIN_API __declspec(dllexport)
#else
#define MAIN_API __declspec(dllimport)
#endif

// Вывод в консоль сообщения о том, что всё работает
__declspec(dllexport) extern "C" MAIN_API void print_info();

// Запуск вычислений
__declspec(dllexport) extern "C" MAIN_API int calculate();