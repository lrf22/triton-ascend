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
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

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
using namespace CVPipeline;

namespace {
static bool isVectorScope(scope::ScopeOp scopeOp)
{
    auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr) {
        return false;
    }
    return coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

static bool checkVecScopeMainLoop(ModuleOp module) {
  bool isMainLoop = false;

  module.walk([&](scope::ScopeOp scopeOp) {
    if (!isVectorScope(scopeOp)) {
      return WalkResult::advance();
    }

    scopeOp.walk([&](hivm::CopyOp copyOp) -> WalkResult {
      isMainLoop = true;
      return WalkResult::interrupt();
    });

    scopeOp.walk([&](scf::ForOp forOp) -> WalkResult {
      if (!forOp->hasAttr("ssbuffer.main_loop")) {
        return WalkResult::advance();
      }

      forOp.walk([&](bufferization::ToTensorOp toTensorOp) -> WalkResult {
        if (!toTensorOp->hasAttr("ssbuffer.transfer_id")) {
          return WalkResult::advance();
        }

        for (Operation *user : toTensorOp->getUsers()) {
          if (!user->hasAttr("ssbuffer.add_from_matmul")) {
            isMainLoop = true;
            break;
          }
          for (Operation *userUser : user->getUsers()) {
            if (!isa<scf::YieldOp>(userUser)) {
              isMainLoop = true;
              break;
            }
          }
        }
        return isMainLoop ? WalkResult::interrupt() : WalkResult::advance();
      });
      return isMainLoop ? WalkResult::interrupt() : WalkResult::advance();
    });
    return isMainLoop ? WalkResult::interrupt() : WalkResult::advance();
  });

  return isMainLoop;
}

static LogicalResult verifyMainLoop(ModuleOp module)
{
  // Only skip if ALL forOps lack main_loop attr
  bool hasMainLoopForOp = false;
  module.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("ssbuffer.main_loop")) {
      hasMainLoopForOp = true;
    }
  });

  if (!hasMainLoopForOp) {
    LDBG("[INFO]: No cycle of multiple iterations, the DynamicCVPipeline pass will be interrupted, and resumed to the original workflow.");
    CVPipeline::setFallbackAttr(module);
    return failure();
  }

  if (!checkVecScopeMainLoop(module)) {
    LDBG("[INFO]: No op beside matmul add in vector main loop.");
    CVPipeline::setFallbackAttr(module);
    return failure();
  };

  return success();
}

} // namespace

void AnalyzeScopePass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeScope:\n" << module << "\n");

  if (failed(verifyMainLoop(module))) {
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