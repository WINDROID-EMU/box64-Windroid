#define _GNU_SOURCE 
 #include <stdio.h> 
 #include <stdlib.h> 
 #include <errno.h> 
 #include <string.h> 
 #include <math.h> 
 #include <signal.h> 
 #include <sys/types.h> 
 #include <unistd.h> 
  
 #include "debug.h" 
 #include "box64context.h" 
 #include "dynarec.h" 
 #include "emu/x64emu_private.h" 
 #include "x64run.h" 
 #include "x64emu.h" 
 #include "box64stack.h" 
 #include "callback.h" 
 #include "emu/x64run_private.h" 
 #include "emu/x87emu_private.h" 
 #include "x64trace.h" 
 #include "signals.h" 
 #include "dynarec_native.h" 
 #include "dynarec_arm64_private.h" 
 #include "dynarec_arm64_functions.h" 
 #include "custommem.h" 
 #include "bridge.h" 
  
 // Get a FPU scratch reg 
 int fpu_get_scratch(dynarec_arm_t* dyn, int ninst) 
 { 
     int ret = SCRATCH0 + dyn->n.fpu_scratch++; 
     if(dyn->n.ymm_used) printf_log(LOG_INFO, "Warning, getting a scratch register after getting some YMM at inst=%d\n", ninst); 
     if(dyn->n.neoncache[ret].t==NEON_CACHE_YMMR || dyn->n.neoncache[ret].t==NEON_CACHE_YMMW) { 
         // should only happens in step 0... 
         dyn->insts[ninst].purge_ymm |= (1<<dyn->n.neoncache[ret].n); // mark as purged 
         dyn->n.neoncache[ret].v = 0; // reset it 
     } 
     return ret; 
 } 
 // Get 2 consicutive FPU scratch reg 
 int fpu_get_double_scratch(dynarec_arm_t* dyn, int ninst) 
 { 
     int ret = SCRATCH0 + dyn->n.fpu_scratch; 
     if(dyn->n.ymm_used) printf_log(LOG_INFO, "Warning, getting a double scratch register after getting some YMM at inst=%d\n", ninst); 
     if(dyn->n.neoncache[ret].t==NEON_CACHE_YMMR || dyn->n.neoncache[ret].t==NEON_CACHE_YMMW) { 
         // should only happens in step 0... 
         dyn->insts[ninst].purge_ymm |= (1<<dyn->n.neoncache[ret].n); // mark as purged 
         dyn->n.neoncache[ret].v = 0; // reset it 
     } 
     if(dyn->n.neoncache[ret+1].t==NEON_CACHE_YMMR || dyn->n.neoncache[ret+1].t==NEON_CACHE_YMMW) { 
         // should only happens in step 0... 
         dyn->insts[ninst].purge_ymm |= (1<<dyn->n.neoncache[ret+1].n); // mark as purged 
         dyn->n.neoncache[ret+1].v = 0; // reset it 
     } 
     dyn->n.fpu_scratch+=2; 
     return ret; 
 } 
 // Reset scratch regs counter 
 void fpu_reset_scratch(dynarec_arm_t* dyn) 
 { 
     dyn->n.fpu_scratch = 0; 
     dyn->n.ymm_used = 0; 
     dyn->n.ymm_regs = 0; 
     dyn->n.ymm_write = 0; 
     dyn->n.ymm_removed = 0; 
 } 
 // Get a x87 double reg 
 int fpu_get_reg_x87(dynarec_arm_t* dyn, int ninst, int t, int n) 
 { 
     int i=X870; 
     while (dyn->n.fpuused[i]) ++i; 
     if(dyn->n.neoncache[i].t==NEON_CACHE_YMMR || dyn->n.neoncache[i].t==NEON_CACHE_YMMW) { 
         // should only happens in step 0... 
         dyn->insts[ninst].purge_ymm |= (1<<dyn->n.neoncache[i].n); // mark as purged 
         dyn->n.neoncache[i].v = 0; // reset it 
     } 
     dyn->n.fpuused[i] = 1; 
     dyn->n.neoncache[i].n = n; 
     dyn->n.neoncache[i].t = t; 
     dyn->n.news |= (1<<i); 
     return i; // return a Dx 
 } 
 // Free a FPU double reg 
 void fpu_free_reg(dynarec_arm_t* dyn, int reg) 
 { 
     // TODO: check upper limit? 
     dyn->n.fpuused[reg] = 0; 
     if(dyn->n.neoncache[reg].t==NEON_CACHE_YMMR || dyn->n.neoncache[reg].t==NEON_CACHE_YMMW) { 
         dyn->n.ymm_removed |= 1<<dyn->n.neoncache[reg].n; 
         if(dyn->n.neoncache[reg].t==NEON_CACHE_YMMW) 
             dyn->n.ymm_write |= 1<<dyn->n.neoncache[reg].n; 
         if(reg>SCRATCH0) 
             dyn->n.ymm_regs |= (8LL+reg-SCRATCH0)<<(dyn->n.neoncache[reg].n*4); 
         else 
             dyn->n.ymm_regs |= ((uint64_t)(reg-EMM0))<<(dyn->n.neoncache[reg].n*4); 
     } 
     if(dyn->n.neoncache[reg].t!=NEON_CACHE_ST_F && dyn->n.neoncache[reg].t!=NEON_CACHE_ST_D && dyn->n.neoncache[reg].t!=NEON_CACHE_ST_I64) 
         dyn->n.neoncache[reg].v = 0; 
     if(dyn->n.fpu_scratch && reg==SCRATCH0+dyn->n.fpu_scratch-1) 
         --dyn->n.fpu_scratch; 
 } 
 // Get an MMX double reg 
 int fpu_get_reg_emm(dynarec_arm_t* dyn, int ninst, int emm) 
 { 
     int ret = EMM0 + emm; 
     if(dyn->n.neoncache[ret].t==NEON_CACHE_YMMR || dyn->n.neoncache[ret].t==NEON_CACHE_YMMW) { 
         // should only happens in step 0... 
         dyn->insts[ninst].purge_ymm |= (1<<dyn->n.neoncache[ret].n); // mark as purged 
         dyn->n.neoncache[ret].v = 0; // reset it 
     } 
     dyn->n.fpuused[ret] = 1; 
     dyn->n.neoncache[ret].t = NEON_CACHE_MM; 
     dyn->n.neoncache[ret].n = emm; 
     dyn->n.news |= (1<<(ret)); 
     return ret; 
 } 
 // Get an XMM quad reg 
 int fpu_get_reg_xmm(dynarec_arm_t* dyn, int t, int xmm) 
 { 
     int i; 
     if(xmm>7) { 
         i = XMM8 + xmm - 8; 
     } else { 
         i = XMM0 + xmm; 
     } 
     dyn->n.fpuused[i] = 1; 
     dyn->n.neoncache[i].t = t; 
     dyn->n.neoncache[i].n = xmm; 
     dyn->n.news |= (1<<i); 
     return i; 
 } 
 int internal_mark_ymm(dynarec_arm_t* dyn, int t, int ymm, int reg) 
 { 
     if((dyn->n.neoncache[reg].t==NEON_CACHE_YMMR) || (dyn->n.neoncache[reg].t==NEON_CACHE_YMMW)) { 
         if(dyn->n.neoncache[reg].n == ymm) { 
             // already there! 
             if(t==NEON_CACHE_YMMW) 
                 dyn->n.neoncache[reg].t=t; 
             return reg; 
         } 
     } else if(!dyn->n.neoncache[reg].v) { 
         // found a slot! 
         dyn->n.neoncache[reg].t=t; 
         dyn->n.neoncache[reg].n=ymm; 
         dyn->n.news |= (1<<reg); 
         return reg; 
     } 
     return -1; 
 } 
 int is_ymm_to_keep(dynarec_arm_t* dyn, int reg, int k1, int k2, int k3) 
 { 
     if((k1!=-1) && (dyn->n.neoncache[reg].n==k1)) 
         return 1; 
     if((k2!=-1) && (dyn->n.neoncache[reg].n==k2)) 
         return 1; 
     if((k3!=-1) && (dyn->n.neoncache[reg].n==k3)) 
         return 1; 
     if((dyn->n.neoncache[reg].t==NEON_CACHE_YMMR || dyn->n.neoncache[reg].t==NEON_CACHE_YMMW) && (dyn->n.ymm_used&(1<<dyn->n.neoncache[reg].n))) 
         return 1; 
     return 0; 
 } 
  
 // Reset fpu regs counter 
 static void fpu_reset_reg_neoncache(neoncache_t* n) 
 { 
     n->fpu_reg = 0; 
     for (int i=0; i<32; ++i) { 
         n->fpuused[i]=0; 
         n->neoncache[i].v = 0; 
     } 
     n->ymm_regs = 0; 
     n->ymm_removed = 0; 
     n->ymm_used = 0; 
     n->ymm_write = 0; 
  
 } 
 void fpu_reset_reg(dynarec_arm_t* dyn) 
 { 
     fpu_reset_reg_neoncache(&dyn->n); 
 } 
  
 int neoncache_no_i64(dynarec_arm_t* dyn, int ninst, int st, int a) 
 { 
     if(a==NEON_CACHE_ST_I64) { 
         neoncache_promote_double(dyn, ninst, st); 
         return NEON_CACHE_ST_D; 
     } 
     return a; 
 } 
  
 int neoncache_get_st(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     if (dyn->insts[ninst].n.swapped) { 
         if(dyn->insts[ninst].n.combined1 == a) 
             a = dyn->insts[ninst].n.combined2; 
         else if(dyn->insts[ninst].n.combined2 == a) 
             a = dyn->insts[ninst].n.combined1; 
     } 
     for(int i=0; i<24; ++i) 
         if((dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F 
          || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_D 
          || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_I64) 
          && dyn->insts[ninst].n.neoncache[i].n==a) 
             return dyn->insts[ninst].n.neoncache[i].t; 
     // not in the cache yet, so will be fetched... 
     return NEON_CACHE_ST_D; 
 } 
  
 int neoncache_get_current_st(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     (void)ninst; 
     if(!dyn->insts) 
         return NEON_CACHE_ST_D; 
     for(int i=0; i<24; ++i) 
         if((dyn->n.neoncache[i].t==NEON_CACHE_ST_F 
          || dyn->n.neoncache[i].t==NEON_CACHE_ST_D 
          || dyn->n.neoncache[i].t==NEON_CACHE_ST_I64) 
          && dyn->n.neoncache[i].n==a) 
             return dyn->n.neoncache[i].t; 
     // not in the cache yet, so will be fetched... 
     return NEON_CACHE_ST_D; 
 } 
  
 int neoncache_get_st_f(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     /*if(a+dyn->insts[ninst].n.stack_next-st<0) 
         // The STx has been pushed at the end of instructon, so stop going back 
         return -1;*/ 
     for(int i=0; i<24; ++i) 
         if(dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F 
          && dyn->insts[ninst].n.neoncache[i].n==a) 
             return i; 
     return -1; 
 } 
 int neoncache_get_st_f_i64(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     /*if(a+dyn->insts[ninst].n.stack_next-st<0) 
         // The STx has been pushed at the end of instructon, so stop going back 
         return -1;*/ 
     for(int i=0; i<24; ++i) 
         if((dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_I64 || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F) 
          && dyn->insts[ninst].n.neoncache[i].n==a) 
             return i; 
     return -1; 
 } 
 int neoncache_get_st_f_noback(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     for(int i=0; i<24; ++i) 
         if(dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F 
          && dyn->insts[ninst].n.neoncache[i].n==a) 
             return i; 
     return -1; 
 } 
 int neoncache_get_st_f_i64_noback(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     for(int i=0; i<24; ++i) 
         if((dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_I64 || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F) 
          && dyn->insts[ninst].n.neoncache[i].n==a) 
             return i; 
     return -1; 
 } 
 int neoncache_get_current_st_f(dynarec_arm_t* dyn, int a) 
 { 
     for(int i=0; i<24; ++i) 
         if(dyn->n.neoncache[i].t==NEON_CACHE_ST_F 
          && dyn->n.neoncache[i].n==a) 
             return i; 
     return -1; 
 } 
 int neoncache_get_current_st_f_i64(dynarec_arm_t* dyn, int a) 
 { 
     for(int i=0; i<24; ++i) 
         if((dyn->n.neoncache[i].t==NEON_CACHE_ST_I64 || dyn->n.neoncache[i].t==NEON_CACHE_ST_F) 
          && dyn->n.neoncache[i].n==a) 
             return i; 
     return -1; 
 } 
 static void neoncache_promote_double_forward(dynarec_arm_t* dyn, int ninst, int maxinst, int a); 
 static void neoncache_promote_double_internal(dynarec_arm_t* dyn, int ninst, int maxinst, int a); 
 static void neoncache_promote_double_combined(dynarec_arm_t* dyn, int ninst, int maxinst, int a) 
 { 
     if(a == dyn->insts[ninst].n.combined1 || a == dyn->insts[ninst].n.combined2) { 
         if(a == dyn->insts[ninst].n.combined1) { 
             a = dyn->insts[ninst].n.combined2; 
         } else 
             a = dyn->insts[ninst].n.combined1; 
         int i = neoncache_get_st_f_i64_noback(dyn, ninst, a); 
         //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_combined, ninst=%d combined%c %d i=%d (stack:%d/%d)\n", ninst, (a == dyn->insts[ninst].n.combined2)?'2':'1', a ,i, dyn->insts[ninst].n.stack_push, -dyn->insts[ninst].n.stack_pop); 
         if(i>=0) { 
             dyn->insts[ninst].n.neoncache[i].t = NEON_CACHE_ST_D; 
             if(!dyn->insts[ninst].n.barrier) 
                 neoncache_promote_double_internal(dyn, ninst-1, maxinst, a-dyn->insts[ninst].n.stack_push); 
             // go forward is combined is not pop'd 
             if(a-dyn->insts[ninst].n.stack_pop>=0) 
                 if(!((ninst+1<dyn->size) && dyn->insts[ninst+1].n.barrier)) 
                     neoncache_promote_double_forward(dyn, ninst+1, maxinst, a-dyn->insts[ninst].n.stack_pop); 
         } 
     } 
 } 
 static void neoncache_promote_double_internal(dynarec_arm_t* dyn, int ninst, int maxinst, int a) 
 { 
     while(ninst>=0) { 
         a+=dyn->insts[ninst].n.stack_pop;    // adjust Stack depth: add pop'd ST (going backward) 
         int i = neoncache_get_st_f_i64(dyn, ninst, a); 
         //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_internal, ninst=%d, a=%d st=%d:%d, i=%d\n", ninst, a, dyn->insts[ninst].n.stack, dyn->insts[ninst].n.stack_next, i); 
         if(i<0) return; 
         dyn->insts[ninst].n.neoncache[i].t = NEON_CACHE_ST_D; 
         // check combined propagation too 
         if(dyn->insts[ninst].n.combined1 || dyn->insts[ninst].n.combined2) { 
             if(dyn->insts[ninst].n.swapped) { 
                 //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_internal, ninst=%d swapped %d/%d vs %d with st %d\n", ninst, dyn->insts[ninst].n.combined1 ,dyn->insts[ninst].n.combined2, a, dyn->insts[ninst].n.stack); 
                 if (a==dyn->insts[ninst].n.combined1) 
                     a = dyn->insts[ninst].n.combined2; 
                 else if (a==dyn->insts[ninst].n.combined2) 
                     a = dyn->insts[ninst].n.combined1; 
             } else { 
                 //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_internal, ninst=%d combined %d/%d vs %d with st %d\n", ninst, dyn->insts[ninst].n.combined1 ,dyn->insts[ninst].n.combined2, a, dyn->insts[ninst].n.stack); 
                 neoncache_promote_double_combined(dyn, ninst, maxinst, a); 
             } 
         } 
         a-=dyn->insts[ninst].n.stack_push;  // // adjust Stack depth: remove push'd ST (going backward) 
         --ninst; 
         if(ninst<0 || a<0 || dyn->insts[ninst].n.barrier) 
             return; 
     } 
 } 
  
 static void neoncache_promote_double_forward(dynarec_arm_t* dyn, int ninst, int maxinst, int a) 
 { 
     while((ninst!=-1) && (ninst<maxinst) && (a>=0)) { 
         a+=dyn->insts[ninst].n.stack_push;  // // adjust Stack depth: add push'd ST (going forward) 
         if((dyn->insts[ninst].n.combined1 || dyn->insts[ninst].n.combined2) && dyn->insts[ninst].n.swapped) { 
             //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_forward, ninst=%d swapped %d/%d vs %d with st %d\n", ninst, dyn->insts[ninst].n.combined1 ,dyn->insts[ninst].n.combined2, a, dyn->insts[ninst].n.stack); 
             if (a==dyn->insts[ninst].n.combined1) 
                 a = dyn->insts[ninst].n.combined2; 
             else if (a==dyn->insts[ninst].n.combined2) 
                 a = dyn->insts[ninst].n.combined1; 
         } 
         int i = neoncache_get_st_f_i64_noback(dyn, ninst, a); 
         //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_forward, ninst=%d, a=%d st=%d:%d(%d/%d), i=%d\n", ninst, a, dyn->insts[ninst].n.stack, dyn->insts[ninst].n.stack_next, dyn->insts[ninst].n.stack_push, -dyn->insts[ninst].n.stack_pop, i); 
         if(i<0) return; 
         dyn->insts[ninst].n.neoncache[i].t = NEON_CACHE_ST_D; 
         // check combined propagation too 
         if((dyn->insts[ninst].n.combined1 || dyn->insts[ninst].n.combined2) && !dyn->insts[ninst].n.swapped) { 
             //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double_forward, ninst=%d combined %d/%d vs %d with st %d\n", ninst, dyn->insts[ninst].n.combined1 ,dyn->insts[ninst].n.combined2, a, dyn->insts[ninst].n.stack); 
             neoncache_promote_double_combined(dyn, ninst, maxinst, a); 
         } 
         a-=dyn->insts[ninst].n.stack_pop;    // adjust Stack depth: remove pop'd ST (going forward) 
         if(dyn->insts[ninst].x64.has_next && !dyn->insts[ninst].n.barrier) 
             ++ninst; 
         else 
             ninst=-1; 
     } 
     if(ninst==maxinst) 
         neoncache_promote_double(dyn, ninst, a); 
 } 
  
 void neoncache_promote_double(dynarec_arm_t* dyn, int ninst, int a) 
 { 
     int i = neoncache_get_current_st_f_i64(dyn, a); 
     //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double, ninst=%d a=%d st=%d i=%d\n", ninst, a, dyn->n.stack, i); 
     if(i<0) return; 
     dyn->n.neoncache[i].t = NEON_CACHE_ST_D; 
     dyn->insts[ninst].n.neoncache[i].t = NEON_CACHE_ST_D; 
     // check combined propagation too 
     if(dyn->n.combined1 || dyn->n.combined2) { 
         if(dyn->n.swapped) { 
             //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double, ninst=%d swapped! %d/%d vs %d\n", ninst, dyn->n.combined1 ,dyn->n.combined2, a); 
             if(dyn->n.combined1 == a) 
                 a = dyn->n.combined2; 
             else if(dyn->n.combined2 == a) 
                 a = dyn->n.combined1; 
         } else { 
             //if(box64_dynarec_dump) dynarec_log(LOG_NONE, "neoncache_promote_double, ninst=%d combined! %d/%d vs %d\n", ninst, dyn->n.combined1 ,dyn->n.combined2, a); 
             if(dyn->n.combined1 == a) 
                 neoncache_promote_double(dyn, ninst, dyn->n.combined2); 
             else if(dyn->n.combined2 == a) 
                 neoncache_promote_double(dyn, ninst, dyn->n.combined1); 
         } 
     } 
     a-=dyn->insts[ninst].n.stack_push;  // // adjust Stack depth: remove push'd ST (going backward) 
     if(!ninst || a<0) return; 
     neoncache_promote_double_internal(dyn, ninst-1, ninst, a); 
 } 
  
 int neoncache_combine_st(dynarec_arm_t* dyn, int ninst, int a, int b) 
 { 
     dyn->n.combined1=a; 
     dyn->n.combined2=b; 
     if( neoncache_get_current_st(dyn, ninst, a)==NEON_CACHE_ST_F 
      && neoncache_get_current_st(dyn, ninst, b)==NEON_CACHE_ST_F ) 
         return NEON_CACHE_ST_F; 
     // don't combine i64, it's only for load/store 
     /*if( neoncache_get_current_st(dyn, ninst, a)==NEON_CACHE_ST_I64 
      && neoncache_get_current_st(dyn, ninst, b)==NEON_CACHE_ST_I64 ) 
         return NEON_CACHE_ST_I64;*/ 
     return NEON_CACHE_ST_D; 
 } 
  
 static int isCacheEmpty(dynarec_native_t* dyn, int ninst) { 
     if(dyn->insts[ninst].n.stack_next) { 
         return 0; 
     } 
     for(int i=0; i<24; ++i) 
         if(dyn->insts[ninst].n.neoncache[i].v) {       // there is something at ninst for i 
             if(!( 
             (dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F 
              || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_D 
              || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_I64) 
             && dyn->insts[ninst].n.neoncache[i].n<dyn->insts[ninst].n.stack_pop)) 
                 return 0; 
         } 
     return 1; 
  
 } 
  
 int fpuCacheNeedsTransform(dynarec_arm_t* dyn, int ninst) { 
     int i2 = dyn->insts[ninst].x64.jmp_insts; 
     if(i2<0) 
         return 1; 
     if((dyn->insts[i2].x64.barrier&BARRIER_FLOAT)) 
         // if the barrier as already been apply, no transform needed 
         return ((dyn->insts[ninst].x64.barrier&BARRIER_FLOAT))?0:(isCacheEmpty(dyn, ninst)?0:1); 
     int ret = 0; 
     if(!i2) { // just purge 
         if(dyn->insts[ninst].n.stack_next) 
             return 1; 
         if(dyn->insts[ninst].ymm0_out) 
             return 1; 
         for(int i=0; i<32 && !ret; ++i) 
             if(dyn->insts[ninst].n.neoncache[i].v) {       // there is something at ninst for i 
                 if(!( 
                 (dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_F 
                 || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_D 
                 || dyn->insts[ninst].n.neoncache[i].t==NEON_CACHE_ST_I64) 
                 && dyn->insts[ninst].n.neoncache[i].n<dyn->insts[ninst].n.stack_pop)) 
                     ret = 1; 
             } 
         return ret; 
     } 
     // Check if ninst can be compatible to i2 
     if(dyn->insts[ninst].n.stack_next != dyn->insts[i2].n.stack-dyn->insts[i2].n.stack_push) { 
         return 1; 
     } 
     if(dyn->insts[ninst].ymm0_out && (dyn->insts[ninst].ymm0_out&~dyn->insts[i2].ymm0_in)) 
         return 1; 
     neoncache_t cache_i2 = dyn->insts[i2].n; 
     neoncacheUnwind(&cache_i2); 
  
     for(int i=0; i<32; ++i) { 
         if(dyn->insts[ninst].n.neoncache[i].v) {       // there is something at ninst for i 
             if(!cache_i2.neoncache[i].v) {    // but there is nothing at i2 for i 
                 ret = 1; 
             } else if(dyn->insts[ninst].n.neoncache[i].v!=cache_i2.neoncache[i].v) {  // there is something different 
                 if(dyn->insts[ninst].n.neoncache[i].n!=cache_i2.neoncache[i].n) {   // not the same x64 reg 
                     ret = 1; 
                 } 
                 else if(dyn->insts[ninst].n.neoncache[i].t == NEON_CACHE_XMMR && cache_i2.neoncache[i].t == NEON_CACHE_XMMW) 
                     {/*