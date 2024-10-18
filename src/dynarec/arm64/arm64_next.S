// arm update linker table for dynarec
// called with pointer to emu as 1st parameter
// and address of table to as 2nd parameter
// ip is at r12

.text
.align 4

.extern LinkNext

.global arm64_next

    .8byte  0   // NULL pointer before arm64_next, for getDB

arm64_next:
    // emu is r0
    // IP address is r1

    // Reserva espaço na pilha
    sub     sp,  sp,  (8 * 12)

    // Armazena os registradores no espaço reservado
    stp     x0, x1,  [sp, (8 *  0)] // emu e IP
    stp     x10, x11, [sp, (8 *  2)]
    stp     x12, x13, [sp, (8 *  4)]
    stp     x14, x15, [sp, (8 *  6)]
    stp     x16, x17, [sp, (8 *  8)]
    stp     x18, x27, [sp, (8 * 10)] // também salva x27 (RIP) para permitir mudança em LinkNext

    mov     x2, x30      // "from" está em lr, então coloca em x2
    add     x3, sp, 8*11 // x3 é o endereço para mudar RIP

    // Chama a função LinkNext
    bl      LinkNext

    // Verificação de erro: se o endereço retornado for nulo
    cbz     x0, handle_error // Se x0 (resultado) for zero, chama tratamento de erro

    // Preserva o valor de retorno
    mov     x3, x0

    // Restaura os registradores
    ldp     x0, x1,   [sp, (8 *  0)]
    ldp     x10, x11, [sp, (8 *  2)]
    ldp     x12, x13, [sp, (8 *  4)]
    ldp     x14, x15, [sp, (8 *  6)]
    ldp     x16, x17, [sp, (8 *  8)]
    ldp     x18, x27, [sp, (8 * 10)]

    // Libera o espaço da pilha
    add     sp,  sp, (8 * 12)

    // Retorna para o endereço de salto
    br      x3

handle_error:
    // Aqui você pode adicionar código para lidar com o erro, como registrar ou retornar um valor padrão
    // Exemplo:
    mov     x3, #0 // Define x3 como 0 ou outro valor padrão
    // Você pode também usar um mecanismo de log ou mesmo um retorno padrão
    br      x3 // Retorna para o endereço de erro
