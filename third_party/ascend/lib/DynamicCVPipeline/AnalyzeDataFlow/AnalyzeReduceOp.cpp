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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

static constexpr const char *DEBUG_TYPE = "analyze-reduce-op";
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

static bool checkReduceOpsInCubeScope(ModuleOp module)
{
  bool shouldReturn = false;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr || attr.getTcoretype() != hivm::TCoreType::CUBE) {
      return WalkResult::advance();
    }

    // Walk inside CUBE scope to find reduce ops
    scopeOp.walk([&](linalg::ReduceOp reduceOp) -> WalkResult {
      LDBG("[ERROR]: Found reduce op inside CUBE scope!\n");
      shouldReturn = true;
      return WalkResult::interrupt();
    });

    if (shouldReturn) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return shouldReturn;
}

} // namespace

void AnalyzeReduceOpPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeReduceOp:\n" << module << "\n");

  if (checkReduceOpsInCubeScope(module)) {
    signalPassFailure();
    return;
  }

  LDBG("After AnalyzeReduceOp:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeReduceOpPass()
{
  return std::make_unique<AnalyzeReduceOpPass>();
}

} // namespace triton
} // namespace mlir
