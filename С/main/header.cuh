#pragma once

#ifdef MATHLIBRARY_EXPORTS
#define MATHLIBRARY_API __declspec(dllexport)
#else
#define MATHLIBRARY_API __declspec(dllimport)
#endif

// Рекуррентное соотношение Фибоначчи описывает последовательность F
// где F(n) равно { n = 0, a
// { n = 1, b
// { n > 1, F(n-2) + F(n-1)
// для некоторых начальных интегральных значений a и b.
// Если последовательность инициализирована F(0) = 1, F(1) = 1,
// то это соотношение порождает хорошо известное Фибоначчи
// последовательность: 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

// Инициализация последовательности отношений Фибоначчи
// такие, что F(0) = a, F(1) = b.
// Эта функция должна быть вызвана перед любой другой функцией.
extern "C" MATHLIBRARY_API void fibonacci_init(
    const unsigned long long a, const unsigned long long b);


// Произведите следующее значение в последовательности.
// Возвращает true при успешном выполнении и обновляет текущее значение и индекс;
// false при переполнении оставляет текущее значение и индекс неизменными.
extern "C" MATHLIBRARY_API bool fibonacci_next();

// Get the current value in the sequence.
extern "C" MATHLIBRARY_API unsigned long long fibonacci_current();

// Get the position of the current value in the sequence.
extern "C" MATHLIBRARY_API unsigned fibonacci_index();