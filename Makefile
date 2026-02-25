CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -Werror -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wstack-usage=4096 -Wno-error=stack-usage= \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wredundant-decls -Wwrite-strings \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE -fcf-protection \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/amdgpu
LDFLAGS = -pie
# Linux/ELF only: -Wl,-z,relro,-z,now -Wl,-z,noexecstack

SOURCES = src/main.c \
          src/fe/preproc.c src/fe/lexer.c src/fe/parser.c src/fe/sema.c \
          src/ir/bir.c src/ir/bir_print.c src/ir/bir_lower.c src/ir/bir_mem2reg.c \
          src/amdgpu/isel.c src/amdgpu/emit.c src/amdgpu/encode.c src/amdgpu/enc_tab.c
OBJECTS = $(SOURCES:.c=.o)
TARGET  = barracuda

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(TARGET).exe

.PHONY: all clean
