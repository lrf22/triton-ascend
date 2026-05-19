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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

static constexpr const char *DEBUG_TYPE = "analyze-reduce-op";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace llvm;
using namespace mlir;
using namespace triton;

namespace {

// Traverse the defining chain of a value to find if it originates from a reduce op
static linalg::ReduceOp findReduceOpInDefiningChain(Value v,
                                                    const llvm::DenseSet<linalg::ReduceOp *> &reduceOps)
{
  llvm::DenseSet<Value> visited;
  llvm::SmallVector<Value> worklist;
  worklist.push_back(v);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (visited.count(current)) {
      continue;
    }
    visited.insert(current);

    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      // Block argument, skip
      continue;
    }

    if (auto reduceOp = dyn_cast<linalg::ReduceOp>(defOp)) {
      if (reduceOps.count(&reduceOp)) {
        return reduceOp;
      }
    }

    // Add operands to worklist for further traversal
    for (Value operand : defOp->getOperands()) {
      if (!visited.count(operand)) {
        worklist.push_back(operand);
      }
    }
  }

  return nullptr;
}

// Check scf control flow ops' operands for reduce op usage
static void checkScfControlFlowOps(scope::ScopeOp scopeOp,
                                   llvm::DenseSet<linalg::ReduceOp *> &reduceOps)
{
  auto checkScfOp = [&](Operation *scfOp) {
    for (Value operand : scfOp->getOperands()) {
      linalg::ReduceOp foundReduce = findReduceOpInDefiningChain(operand, reduceOps);
      if (foundReduce) {
        LOG_DEBUG(“remain reduceOp in cube scope for “ << scfOp->getName() << “\n”);
        reduceOps.erase(&foundReduce);
      }
    }
  };

  scopeOp.walk([&](scf::IfOp ifOp) { checkScfOp(ifOp); });
  scopeOp.walk([&](scf::ForOp forOp) { checkScfOp(forOp); });
  scopeOp.walk([&](scf::WhileOp whileOp) { checkScfOp(whileOp); });
}

// Collect all reduce ops inside a CUBE scope
static void collectReduceOpsInScope(scope::ScopeOp scopeOp,
                                    llvm::DenseSet<linalg::ReduceOp *> &reduceOps)
{
  scopeOp.walk([&](linalg::ReduceOp reduceOp) {
    reduceOps.insert(&reduceOp);
  });
}

static bool checkReduceOpsInCubeScope(ModuleOp module)
{
  bool shouldReturn = false;

  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(“hivm.tcore_type”);
    if (!attr || attr.getTcoretype() != hivm::TCoreType::CUBE) {
      return WalkResult::advance();
    }

    // Collect all reduce ops in CUBE scope
    llvm::DenseSet<linalg::ReduceOp *> reduceOps;
    collectReduceOpsInScope(scopeOp, reduceOps);

    if (reduceOps.empty()) {
      return WalkResult::advance();
    }

    // Check scf control flow ops' operands to find reduce op usage
    checkScfControlFlowOps(scopeOp, reduceOps);

    // Report remaining reduce ops with unknown reasons
    for (auto *reduceOp : reduceOps) {
      LOG_DEBUG(“[ERROR]: Unknown reduce op remaining in CUBE scope: “ << *reduceOp << “\n”);
      shouldReturn = true;
    }

    return WalkResult::advance();
  });

  return shouldReturn;
}

} // namespace

void AnalyzeReduceOpPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LOG_DEBUG("Before AnalyzeReduceOp:\n" << module << "\n");

  if (checkReduceOpsInCubeScope(module)) {
    signalPassFailure();
    return;
  }

  LOG_DEBUG("After AnalyzeReduceOp:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeReduceOpPass()
{
  return std::make_unique<AnalyzeReduceOpPass>();
}

} // namespace triton
} // namespace mlir
