/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "llvm/Support/Debug.h"

#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

#include "DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "StandardizeOp";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace mlir::triton {

void addStandardizeOpPipeline(OpPassManager &pm)
{
    pm.addPass(createPatternMatchRewritePass());
}

void registerStandardizeOpPasses()
{
    PassPipelineRegistration<>(
        "ssbuf-standardize-op", "Pipeline to standardize op for dynamic cv pipeline", addStandardizeOpPipeline);
}

} // namespace mlir::triton
