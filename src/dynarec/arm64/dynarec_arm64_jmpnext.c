#include <stdint.h>
#include <stdio.h> // Para printf (opcional, apenas para log de erro)

#include "arm64_emitter.h"

#define EMIT(A)       \
    do {              \
        *block = (A); \
        ++block;      \
    } while (0)

void CreateJmpNext(void* addr, void* next)
{
    // Verifica se os endereços são válidos
    if (!addr || !next) {
        printf("Erro: endereço inválido em CreateJmpNext: addr=%p, next=%p\n", addr, next);
        return; // Retorna sem fazer nada se os endereços forem inválidos
    }

    uint32_t* block = (uint32_t*)addr;

    // Calcula o offset entre next e addr e carrega em x2
    LDRx_literal(x2, (intptr_t)next - (intptr_t)addr);
    
    // Realiza o salto para o endereço calculado
    BR(x2);
}
