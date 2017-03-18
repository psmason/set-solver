SRCS  = main.m.cpp
SRCS += cards.cpp
FLAGS  = -I.
FLAGS += `pkg-config --cflags --libs opencv`

task:
	g++ -std=c++11 $(SRCS) $(FLAGS)
