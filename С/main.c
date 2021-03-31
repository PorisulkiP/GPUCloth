#include <stdio.h>
#include <windows.h> 

// a sample exported function
void __declspec(dllexport) SomeFunction(const LPCSTR sometext)
{
    MessageBoxA(0, sometext, "DLL Message", MB_OK | MB_ICONINFORMATION);
}

extern "C" DLL_EXPORT BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
    switch (fdwReason)
    {
        case DLL_PROCESS_ATTACH:
            printf("Load DLL in Python\n");

            printf("HINSTANCE = %p\n",hinstDLL); // Вывод описателя экземпляра DLL

            if (lpvReserved)                     // Определение способа загрузки
              printf("DLL loaded with implicit layout\n"); 
            else
              printf("DLL loaded with explicit layout\n");          
            return 1;                            // Успешная инициализация

        case DLL_PROCESS_DETACH:
            printf("DETACH DLL\n");
            break;

        case DLL_THREAD_ATTACH:
            break;

        case DLL_THREAD_DETACH:
            break;
    }
    return TRUE; // succesful
}