.text
.align 4

.global arm64_lock_read_b
.global arm64_lock_write_b
.global arm64_lock_read_h
.global arm64_lock_write_h
.global arm64_lock_read_d
.global arm64_lock_write_d
.global arm64_lock_read_dd
.global arm64_lock_write_dd
.global arm64_lock_read_dq
.global arm64_lock_write_dq
.global arm64_lock_xchg_dd
.global arm64_lock_xchg_d
.global arm64_lock_xchg_h
.global arm64_lock_xchg_b
.global arm64_lock_storeifnull
.global arm64_lock_storeifnull_d
.global arm64_lock_storeifref
.global arm64_lock_storeifref_d
.global arm64_lock_storeifref2_d
.global arm64_lock_decifnot0b
.global arm64_lock_storeb
.global arm64_lock_incif0
.global arm64_lock_decifnot0
.global arm64_lock_store
.global arm64_lock_store_dd
.global arm64_lock_get_b
.global arm64_lock_get_d
.global arm64_lock_get_dd

error_handler:
    // Rotina de erro para ponteiros nulos
    mov     x0, #-1  // Retorna -1 para indicar erro
    ret

arm64_lock_read_b:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldaxrb  w0, [x0]
    ret

arm64_lock_write_b:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    mov     x2, x0
    stlxrb  w0, w1, [x2]
    dmb     ish
    ret

arm64_lock_read_h:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldaxrh  w0, [x0]
    ret

arm64_lock_write_h:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    mov     x2, x0
    stlxrh  w0, w1, [x2]
    dmb     ish
    ret

arm64_lock_read_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldaxr   w0, [x0]
    ret

arm64_lock_write_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    mov     x2, x0
    stlxr   w0, w1, [x2]
    dmb     ish
    ret

arm64_lock_read_dd:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldaxr   x0, [x0]
    ret

arm64_lock_write_dd:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    mov     x2, x0
    stlxr   w0, x1, [x2]
    dmb     ish
    ret

arm64_lock_read_dq:
    cbz     x2, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldaxp   x4, x3, [x2]
    str     x4, [x0]
    str     x3, [x1]
    ret

arm64_lock_write_dq:
    cbz     x2, error_handler   // Verificação de ponteiro nulo
    stlxp   w3, x0, x1, [x2]
    mov     w0, w3
    dmb     ish
    ret

arm64_lock_xchg_dd:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
arm64_lock_xchg_dd_0:
    ldaxr   x2, [x0]
    stlxr   w3, x1, [x0]
    cbnz    w3, arm64_lock_xchg_dd_0
    mov     x0, x2
    ret

arm64_lock_xchg_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
arm64_lock_xchg_d_0:
    ldaxr   w2, [x0]
    stlxr   w3, w1, [x0]
    cbnz    w3, arm64_lock_xchg_d_0
    mov     w0, w2
    ret

arm64_lock_xchg_h:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
arm64_lock_xchg_h_0:
    ldaxrh  w2, [x0]
    stlxrh  w3, w1, [x0]
    cbnz    w3, arm64_lock_xchg_h_0
    mov     w0, w2
    ret

arm64_lock_xchg_b:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
arm64_lock_xchg_b_0:
    ldaxrb  w2, [x0]
    stlxrb  w3, w1, [x0]
    cbnz    w3, arm64_lock_xchg_b_0
    mov     w0, w2
    ret

arm64_lock_storeifnull:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   x2, [x0]
    cbnz    x2, 2f
    stlxr   w3, x1, [x0]
    cbnz    w3, 1b
2:
    mov     x0, x2
    ret

arm64_lock_storeifnull_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   w2, [x0]
    cbnz    w2, 2f
    stlxr   w3, w1, [x0]
    cbnz    w3, 1b
2:
    dmb     ish
    mov     w0, w2
    ret

arm64_lock_storeifref:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   x3, [x0]
    cmp     x2, x3
    bne     2f
    stlxr   w4, x1, [x0]
    cbnz    w4, 1b
    mov     x0, x1
    ret
2:
    mov     x0, x3
    ret

arm64_lock_storeifref_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   w3, [x0]
    cmp     w2, w3
    bne     2f
    stlxr   w4, w1, [x0]
    cbnz    w4, 1b
    mov     w0, w1
    ret
2:
    mov     w0, w3
    ret

arm64_lock_storeifref2_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   w3, [x0]
    cmp     w2, w3
    bne     2f
    stlxr   w4, w1, [x0]
    cbnz    w4, 1b
2:
    mov     w0, w3
    ret

arm64_lock_decifnot0b:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxrb  w1, [x0]
    cmp     w1, #0
    beq     2f
    sub     w1, w1, #1
    stlxrb  w2, w1, [x0]
    cbnz    w2, 1b
2:
    ret

arm64_lock_storeb:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    strb    w1, [x0]
    dmb     ish
    ret

arm64_lock_decifnot0:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   w1, [x0]
    cmp     w1, #0
    beq     2f
    sub     w3, w1, #1
    stlxr   w2, w3, [x0]
    cbnz    w2, 1b
2:
    mov     w0, w1
    ret

arm64_lock_incif0:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
1:
    ldaxr   w1, [x0]
    cmp     w1, #0
    bne     2f
    mov     w1, #1
    stlxr   w2, w1, [x0]
    cbnz    w2, 1b
2:
    ret

arm64_lock_store:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    str     w1, [x0]
    dmb     ish
    ret

arm64_lock_store_dd:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    str     x1, [x0]
    dmb     ish
    ret

arm64_lock_get_b:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldrb    w0, [x0]
    ret

arm64_lock_get_d:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldr     w0, [x0]
    ret

arm64_lock_get_dd:
    cbz     x0, error_handler   // Verificação de ponteiro nulo
    dmb     ish
    ldr     x0, [x0]
    ret
