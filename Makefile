SRCS  = main.m.cpp
SRCS += cards.cpp
SRCS += attributes.cpp
SRCS += color.cpp

FLAGS  = -I.
FLAGS += `pkg-config --cflags --libs opencv`

task:
	g++ -std=c++11 $(SRCS) $(FLAGS)
