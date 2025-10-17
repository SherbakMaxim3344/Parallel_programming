CXX = g++
CXXFLAGS = -std=c++17 -pthread -O2 -Wall -Wextra
TARGET = coarse_grained_benchmark
SOURCES = main.cpp
HEADERS = ThreadSafeList.hpp CoarseGrainedList.hpp

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

release: CXXFLAGS += -O3 -DNDEBUG
release: $(TARGET)

clean:
	rm -f $(TARGET) *.csv benchmark_results.txt

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run debug release