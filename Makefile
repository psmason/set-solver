SRCS  = main.m.cpp
SRCS += cards.cpp
SRCS += attributes.cpp
SRCS += color.cpp
SRCS += symbol.cpp
SRCS += shading.cpp
SRCS += solver.cpp
SRCS += paintmatches.cpp
SRCS += utils.cpp
SRCS += client.cpp

FLAGS  = -I.
FLAGS += `pkg-config --cflags --libs opencv`

CC = g++ -std=c++11 -Wall -Wextra

task:
	$(CC) $(SRCS) $(FLAGS)
