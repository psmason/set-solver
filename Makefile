SRCDIR = ./src
SRCS   = $(wildcard $(SRCDIR)/*.cpp)

FLAGS  = -I./src
FLAGS += `pkg-config --cflags --libs opencv`

CC = g++ -std=c++11 -Wall -Wextra

task:
	$(CC) $(SRCS) $(FLAGS)
