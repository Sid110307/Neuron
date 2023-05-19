CC=clang
EXECUTABLE_DIR=bin

CFLAGS=-std=c11 -Wall -Wextra
LIBS=-lm

OBJECTS=$(patsubst src/%.c,bin/%.o,$(wildcard src/*.c))

.PHONY: all
all: neuron

bin/%.o: src/%.c
	mkdir -p $(EXECUTABLE_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

neuron: $(OBJECTS)
	mkdir -p $(EXECUTABLE_DIR)
	$(CC) $(CFLAGS) -o $(EXECUTABLE_DIR)/$@ $^ $(LIBS)
	sudo ./$(EXECUTABLE_DIR)/$@

clean:
	rm -rf $(EXECUTABLE_DIR)
	rm -rf $(OBJECTS)
