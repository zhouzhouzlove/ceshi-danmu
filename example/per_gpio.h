#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
int gpio_out(int value);
int gpio_read(int pin); 
int gpio_int();

#ifdef __cplusplus
}
#endif
