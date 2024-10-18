#ifndef __DYNAREC_ARM_PRIVATE_H_
#define __DYNAREC_ARM_PRIVATE_H_

#include "../dynarec_private.h"

// Estruturas e tipos de dados para o Dynarec ARM64

typedef struct x64emu_s x64emu_t;
typedef struct dynablock_s dynablock_t;
typedef struct instsize_s instsize_t;

#define BARRIER_MAYBE   8

// Flags nativas
typedef enum {
    NF_EQ   = (1 << 0),
    NF_SF   = (1 << 1),
    NF_VF   = (1 << 2)
} native_flags_t;

// Operações de flags nativas
typedef enum {
    NAT_FLAG_OP_NONE        = 0,
    NAT_FLAG_OP_TOUCH       = 1,
    NAT_FLAG_OP_UNUSABLE    = 2,
    NAT_FLAG_OP_CANCELED    = 3
} nat_flag_op_t;

// Caches de Neon
typedef enum {
    NEON_CACHE_NONE     = 0,
    NEON_CACHE_ST_D     = 1,
    NEON_CACHE_ST_F     = 2,
    NEON_CACHE_ST_I64   = 3,
    NEON_CACHE_MM       = 4,
    NEON_CACHE_XMMW     = 5,
    NEON_CACHE_XMMR     = 6,
    NEON_CACHE_YMMW     = 7,
    NEON_CACHE_YMMR     = 8,
    NEON_CACHE_SCR      = 9
} neon_cache_type_t;

typedef union neon_cache_s {
    int8_t v;
    struct {
        uint8_t t:4;   // tipo de registro
        uint8_t n:4;   // número do registro
    };
} neon_cache_t;

// Cache de SSE
typedef union sse_cache_s {
    int8_t v;
    struct {
        uint8_t reg:7;
        uint8_t write:1;
    };
} sse_cache_t;

typedef struct neoncache_s {
    // Cache do Neon
    neon_cache_t neoncache[32];
    int8_t stack;          // Contador da pilha
    int8_t stack_next;     // Próximo valor na pilha
    int8_t stack_pop;      // Valor a ser removido da pilha
    int8_t stack_push;     // Valor a ser adicionado à pilha
    // Outros campos omitidos para brevidade
} neoncache_t;

typedef struct flagcache_s {
    int pending;           // Existem flags pendentes?
    uint8_t dfnone;       // Flags adiadas estão definidas como df_none?
    uint8_t dfnone_here;  // Flags adiadas limpas nesta opcode?
} flagcache_t;

// Estruturas de instrução ARM64
typedef struct instruction_arm64_s {
    instruction_x64_t x64;
    uintptr_t address;    // Endereço da instrução emitida
    uintptr_t epilog;     // Epílogo da instrução atual
    // Outros campos omitidos para brevidade
} instruction_arm64_t;

// Estruturas do Dynarec ARM
typedef struct dynarec_arm_s {
    instruction_arm64_t *insts;
    int32_t size;
    int32_t cap;
    // Outros campos omitidos para brevidade
} dynarec_arm_t;

// Funções para manipulação de jumps e tabelas
void add_next(dynarec_arm_t *dyn, uintptr_t addr);
uintptr_t get_closest_next(dynarec_arm_t *dyn, uintptr_t addr);
void add_jump(dynarec_arm_t *dyn, int ninst);
int get_first_jump(dynarec_arm_t *dyn, int next);
int get_first_jump_addr(dynarec_arm_t *dyn, uintptr_t next);
int is_nops(dynarec_arm_t *dyn, uintptr_t addr, int n);
int is_instructions(dynarec_arm_t *dyn, uintptr_t addr, int n);
int Table64(dynarec_arm_t *dyn, uint64_t val, int pass); // Adiciona um valor à tabela64

void CreateJmpNext(void* addr, void* next);

// Macro para rastreamento
#define GO_TRACE(A, B, s0)  \
    GETIP(addr);            \
    MOVx_REG(x1, xRIP);     \
    MRS_nzvc(s0);           \
    STORE_XEMU_CALL(xRIP);  \
    MOV32w(x2, B);          \
    CALL_(A, -1, s0);       \
    MSR_nzvc(s0);           \
    LOAD_XEMU_CALL(xRIP)

#endif //__DYNAREC_ARM_PRIVATE_H_
