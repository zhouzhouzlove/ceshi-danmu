#include "stdlib.h"  
#include "stdio.h"  
#include "string.h"
#include "unistd.h"
#include "fcntl.h" 
#include "poll.h"
#include "per_gpio.h"

#define DIRECTION_MAX 64
#define MSG printf

//int gpio_out(int value);
//int gpio_read(int pin); 
//int gpio_int();

int gpio_int()

{
	FILE *p=NULL;
        
        p = fopen("/sys/class/gpio/gpio42/direction","w");
        if(p==NULL)
	{
	p = fopen("/sys/class/gpio/export","w");
	fprintf(p,"%d",42);
	fclose(p);
	
        
        p = fopen("/sys/class/gpio/export","w");
        fprintf(p,"%d",41);
        fclose(p);
       
	}

	p = fopen("/sys/class/gpio/gpio42/direction","w");
	fprintf(p,"in");  //配置成IN
        fclose(p);
	p = fopen("/sys/class/gpio/gpio41/direction","w");
        fprintf(p,"out");  //配置成OUT
        fclose(p);
	p = fopen("/sys/class/gpio/gpio41/value","w");
	fprintf(p,"%d",0);  //LOW
	fclose(p);


	return 0;

}
/*
int main(int argc, char *argv[])  
{
     
   int i = 1;  
   int count=0;
   int state=0;
   
   gpio_int(); //set gpio 41 "out" and 42 "in"

    while(1)
	{
	i= gpio_read(42);
	if(i==0 && state==0)
	{
		count++;
                state=1; //Some on comming
		printf("Someone %d Comming!!\n",count);
		gpio_out(0);
		state=1; //Some on comming
	}
        if(i==1 && state==1)
        
	{
         state=0;//someone go!
         printf("Someone %d GO!!!!!!!!!\n",count);
	}
	
	}  
    return 0;  
}  

*/

int gpio_read(int pin)  
{  
    char path[DIRECTION_MAX];  
    char value_str[3];  
    int fd;  
  
    snprintf(path, DIRECTION_MAX, "/sys/class/gpio/gpio%d/value", pin);  
    fd = open(path, O_RDONLY);  
    if (fd < 0) 
{

	MSG("Failed to open gpio value for reading!\n");  
        fprintf(stderr, "failed to open gpio value for reading!\n");  
        return -1;  
    }  
  
    if (
	read(fd, value_str, 3) < 0) {  
        fprintf(stderr, "failed to read value!\n");  
        return -1;  
    }  
  
    close(fd);  
    return (atoi(value_str));  
}

int gpio_out(int value)

{
 FILE *p=NULL;

p = fopen("/sys/class/gpio/gpio41/value","w");
fprintf(p,"%d",1);
fclose(p);
sleep(1);
p = fopen("/sys/class/gpio/gpio41/value","w");
fprintf(p,"%d",0);
fclose(p);

return 0;


}  
