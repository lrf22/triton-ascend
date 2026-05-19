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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

static constexpr const char *DEBUG_TYPE = "analyze-scope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::dbgs() << __VA_ARGS__; \
  llvm::dbgs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;

namespace {

// Check if the module should be skipped for control flow condition processing
static LogicalResult verifyScopeWithCV(ModuleOp module)
{
  int cubeScopeCount = 0;
  int vectorScopeCount = 0;

  module.walk([&](Operation *op) {
    auto scopeOp = dyn_cast<scope::ScopeOp>(op);
    if (!scopeOp) {
      return;
    }
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr) {
      return;
    }
    if (attr.getTcoretype() == hivm::TCoreType::CUBE) {
      ++cubeScopeCount;
    } else if (attr.getTcoretype() == hivm::TCoreType::VECTOR) {
      ++vectorScopeCount;
    }
  });

  // If either CUBE or VECTOR scope is missing, skip processing
  if (cubeScopeCount == 0 || vectorScopeCount == 0) {
    LDBG("CUBE or VECTOR scope missing, skip processing.");
    return failure();
  }

  // Only skip if ALL forOps lack main_loop attr
  bool hasMainLoopForOp = false;
  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      hasMainLoopForOp = true;
    }
  });
  if (!hasMainLoopForOp) {
    LDBG("All forOps lack ssbuffer.main_loop, skip processing.");
    return failure();
  }

  return success();
}

} // namespace

void AnalyzeScopePass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeScope:\n" << module << "\n");

  if (failed(verifyScopeWithCV(module))) {
    signalPassFailure();
    return;
  }

  LDBG("After AnalyzeScope:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeScopePass()
{
  return std::make_unique<AnalyzeScopePass>();
}

} // namespace triton
} // namespace mlir