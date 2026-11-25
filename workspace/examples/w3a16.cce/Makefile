# Makefile for W3A16 Dequant + MatMul CCE Kernel (dav-l310 / Kirin 9020)
#
# Usage:
#   ./build.sh          - Build dynamic-shape kernel binary (output/w3a16_dequant_matmul.o)
#   ./build.sh asm      - Emit post-RA device assembly (output/w3a16_dequant_matmul.s)
#   ./build.sh symbols  - Show ELF sections and symbol table
#   ./build.sh clean    - Remove build artifacts

ASCEND_HOME ?= /usr/local/Ascend/cann-9.2.0
TOOL_BIN    ?= $(ASCEND_HOME)/tools/ccec_compiler/bin
BISHENG_CC1 ?= $(ASCEND_HOME)/tools/bisheng_compiler/bin/bisheng

CCEC        := $(TOOL_BIN)/bisheng
LLVM_LINK   := $(TOOL_BIN)/llvm-link
LLD         := $(TOOL_BIN)/ld.lld
OBJDUMP     := $(TOOL_BIN)/llvm-objdump

INC_DIR     := include
SRC_DIR     := src
OUT_DIR     ?= output
BUILD_DIR   := $(OUT_DIR)/objs

ENTRY_SRC   := $(SRC_DIR)/w3a16_dequant_matmul.cce
MOD_SRCS    := $(SRC_DIR)/gemv_m1_k1024.cce \
               $(SRC_DIR)/gemv_m1_stream.cce \
               $(SRC_DIR)/gemm_general.cce \
               $(SRC_DIR)/gemm_multi_m.cce

MOD_BCS     := $(patsubst $(SRC_DIR)/%.cce,$(BUILD_DIR)/%.bc,$(MOD_SRCS))
MODULES_BC  := $(BUILD_DIR)/modules.bc
ENTRY_DEP   := $(BUILD_DIR)/w3a16_dequant_matmul.d
DEPS        := $(MOD_BCS:.bc=.d) $(ENTRY_DEP)

TARGET      := $(OUT_DIR)/w3a16_dequant_matmul.o
ASM_OUT     := $(OUT_DIR)/w3a16_dequant_matmul.s
ARCH        := dav-l310

CCE_FLAGS   := -x cce -std=c++17 -O2 -g0 -Wall -Wextra \
               -I$(INC_DIR) \
               --cce-aicore-arch=$(ARCH) \
               --cce-aicore-only \
               --cce-auto-sync=off \
               --cce-generic-addrspace=off \
               --cce-long-scbz=true \
               -mllvm -cce-aicore-dcci-insert-for-scalar=false \
               -mllvm -cce-vf-aa-between-iters=true \
               -mllvm -cce-vf-enable-loop-fusion=false

LLD_FLAGS   := -m aicorelinux -Ttext 0 -q -static

CC1_ASM_FLAGS := -cc1 -triple hiipu64-hisilicon-cce -S -O2 -target-cpu $(ARCH) \
                 -fno-zero-initialized-in-bss -fno-jump-tables \
                 -mllvm -cce-aicore-enable-generic-addrspace=false \
                 -mllvm -cce-aicore-basicaa-disjoint-addrspace=true \
                 -mllvm -cce-aicore-hwloops=false \
                 -mllvm -cce-aicore-hwloops-threshold=0 \
                 -mllvm -cce-aicore-analysis-peel=false \
                 -mllvm -cce-aicore-dcache-safe-mode=false \
                 -mllvm -cce-aicore-loop-unroll-tti=false \
                 -mllvm -licm-safe-hoist=false \
                 -mllvm -handle-output-dep=true \
                 -mllvm -sroa-allow-skip-function=false \
                 -mllvm -misched-allow-skip-func=false \
                 -mllvm -enable-post-ra-sched=true \
                 -mllvm -misched-postra=true \
                 -mllvm -early-cse-allow-skip-function=false \
                 -mllvm -loop-unswitch-allow-skip-loop=false \
                 -mllvm -enable-vloop-lowering=true \
                 -mllvm -cce-aicore-backend-loop-lower=true \
                 -mllvm -cce-aicore-loop-unrotate=true \
                 -mllvm -cce-aicore-if-predicate=true \
                 -mllvm -cce-aicore-fix-errata-v920-001-enable=true \
                 -mllvm -cce-aicore-branch-relax=true \
                 -mllvm -cce-vf-arg-hoist-opt=false \
                 -mllvm -cce-vf-enable-vf-fusion=false \
                 -mllvm -cce-vf-enable-vf-loop-extender=false \
                 -mllvm -cce-vf-enable-loop-fusion=false \
                 -mllvm -cce-vf-enable-vf-ldst-elimination=false \
                 -mllvm -cce-vf-enable-ub-dead-st-elimination=false \
                 -mllvm -cce-vf-auto-sync=off \
                 -mllvm -cce-vf-enable-vf-ifelse-extender=false \
                 -mllvm -hoist-common-insts=true \
                 -mllvm -cce-vf-stack-size=0x2000 \
                 -mllvm -cce-simt-blockdim-specialization=false \
                 -mllvm -cce-aicore-movx8-opt=false \
                 -mllvm -vectorize-slp=false \
                 -mllvm -cce-aicore-number-of-cluster-loads=5 \
                 -mllvm -cce-aicore-long-scbz=true \
                 -mllvm -cce-aicore-jump-expand=false \
                 -mllvm -cce-sync-inst=off \
                 -mllvm -cce-aicore-disable-argmem-alias-check=false \
                 -mllvm -enable-deps-filter=off \
                 -mllvm -enable-sync-event-coloring=false \
                 -mllvm -cce-aicore-enable-sync=false \
                 -mllvm -api-deps-filter=false \
                 -mllvm -cce-aicore-cleanup-metadata=true \
                 -mllvm -cce-aicore-dcci-insert-for-scalar=false \
                 -mllvm -cce-vf-aa-between-iters=true

.PHONY: all asm symbols clean

all: $(TARGET)

$(OUT_DIR) $(BUILD_DIR):
	@mkdir -p $@

$(BUILD_DIR)/%.bc: $(SRC_DIR)/%.cce Makefile | $(BUILD_DIR)
	$(CCEC) $(CCE_FLAGS) -MMD -MP -MF $(@:.bc=.d) -MT $@ \
		-Xclang -disable-llvm-optzns -emit-llvm -c $< -o $@

$(MODULES_BC): $(MOD_BCS) | $(BUILD_DIR)
	$(LLVM_LINK) $(MOD_BCS) -o $@

$(TARGET): $(ENTRY_SRC) $(MODULES_BC) Makefile | $(OUT_DIR) $(BUILD_DIR)
	$(CCEC) $(CCE_FLAGS) -MMD -MP -MF $(ENTRY_DEP) -MT $@ \
		-cce-link-aicore-ll-module $(MODULES_BC) -c $(ENTRY_SRC) -o $(BUILD_DIR)/kernel.raw
	$(LLD) $(LLD_FLAGS) $(BUILD_DIR)/kernel.raw -o $@
	@rm -f $(BUILD_DIR)/kernel.raw

asm: $(ASM_OUT)

$(ASM_OUT): $(ENTRY_SRC) $(MODULES_BC) Makefile | $(OUT_DIR) $(BUILD_DIR)
	$(CCEC) $(CCE_FLAGS) -cce-link-aicore-ll-module $(MODULES_BC) \
		-emit-llvm -c $(ENTRY_SRC) -o $(BUILD_DIR)/combined.bc
	$(BISHENG_CC1) $(CC1_ASM_FLAGS) $(BUILD_DIR)/combined.bc -o $@
	@rm -f $(BUILD_DIR)/combined.bc

symbols: $(TARGET)
	$(OBJDUMP) -h -t $(TARGET)

clean:
	rm -rf $(OUT_DIR)

-include $(DEPS)
