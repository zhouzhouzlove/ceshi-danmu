#ifndef MODEL_INIT_H
	#define MODEL_INIT_H
	struct inferx_handler;

	#ifdef __cplusplus
	extern "C"
	{
	#endif // __cplusplus

#ifdef _MSC_VER
        HMODULE inferx_model_init(char* pcDllName, struct inferx_handler* hd);
#else
        void*   inferx_model_init(char* pcDllName, struct inferx_handler* hd);
#endif

#ifdef _MSC_VER
        void inferx_model_free(HMODULE hDll);
#else
        void inferx_model_free(void*   hDll);
#endif

	#ifdef __cplusplus
	}
	#endif // __cplusplus
#endif // MODEL_INIT_H
