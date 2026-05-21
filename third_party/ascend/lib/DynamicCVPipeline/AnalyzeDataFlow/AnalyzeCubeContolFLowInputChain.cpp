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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

static constexpr const char *DEBUG_TYPE = "analyze-cube-control-flow-input-chain";
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

static bool isCubeScope(scope::ScopeOp scopeOp)
{
    auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr) {
        return false;
    }
    return coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE;
}

static bool isControlFlowOp(Operation *op)
{
    return isa<scf::IfOp>(op) || isa<scf::ForOp>(op) || isa<scf::WhileOp>(op);
}

// 沿着 Value 的 definingOp 链向上递归搜索，检测是否存在 linalg.reduce 或
// 结果为 tensor 类型的 arith.select 操作。
// visited 用于记录已访问的 Value，避免循环引用导致无限递归。
static bool hasDefiningChainWithReduceOrTensorSelect(Value val,
                                                     DenseSet<Value> &visited)
{
    // 防止循环引用：若该 Value 已访问过，直接返回 false
    if (visited.contains(val)) {
        return false;
    }
    visited.insert(val);

    // 获取当前 Value 的定义操作，若为 BlockArgument 则 defOp 为 nullptr
    Operation *defOp = val.getDefiningOp();
    while (defOp) {
        // 检查是否为 linalg.reduce 操作
        if (isa<linalg::ReduceOp>(defOp)) {
            LDBG("Found linalg.reduce in defining chain");
            return true;
        }

        // 检查是否为 arith.select 且结果为 tensor 类型
        if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
            if (isa<RankedTensorType>(selectOp.getType())) {
                LDBG("Found arith.select with tensor result in defining chain");
                return true;
            }
        }

        // 若当前操作无操作数，defining 链无法继续向上追溯，终止搜索
        if (defOp->getNumOperands() == 0) {
            break;
        }

        // 对当前操作的所有操作数递归搜索其 defining 链
        for (Value operand : defOp->getOperands()) {
            if (hasDefiningChainWithReduceOrTensorSelect(operand, visited)) {
                return true;
            }
        }
        // 当前 defOp 的操作数已全部递归检查完毕，无需继续 while 循环
        break;
    }

    return false;
}

static bool checkControlFlowOpInputs(Operation *cfOp)
{
    DenseSet<Value> visited;

    auto checkOperands = [&](ValueRange operands) {
        for (Value operand : operands) {
            if (hasDefiningChainWithReduceOrTensorSelect(operand, visited)) {
                return true;
            }
        }
        return false;
    };

    if (auto forOp = dyn_cast<scf::ForOp>(cfOp)) {
        if (checkOperands(forOp.getInitArgs())) {
            return true;
        }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(cfOp)) {
        if (checkOperands(ifOp.getOperands())) {
            return true;
        }
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(cfOp)) {
        if (checkOperands(whileOp.getOperands())) {
            return true;
        }
    }

    return false;
}

bool checkCubeControlFlowInputChain(ModuleOp module)
{
    bool shouldReturn = false;

    module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
        if (!isCubeScope(scopeOp)) {
            return WalkResult::advance();
        }

        LDBG("Found CUBE scope");

        scopeOp.walk([&](Operation *op) -> WalkResult {
            if (!isControlFlowOp(op)) {
                return WalkResult::advance();
            }

            LDBG("Found control flow op in CUBE scope");

            if (checkControlFlowOpInputs(op)) {
                shouldReturn = true;
                return WalkResult::interrupt();
            }

            return WalkResult::advance();
        });

        if (shouldReturn) {
            return WalkResult::interrupt();
        }

        return WalkResult::advance();
    });

    return shouldReturn;
}

} // namespace

void AnalyzeCubeControlFlowInputChainPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Enter AnalyzeCubeControlFlowInputChainPass.");

  if (checkCubeControlFlowInputChain(module)) {
    setFallbackAttr(module);
    signalPassFailure();
    return;
  }

  LDBG("Exit AnalyzeCubeControlFlowInputChainPass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeCubeContolFLowInputChainPass()
{
  return std::make_unique<AnalyzeCubeControlFlowInputChainPass>();
}

} // namespace triton
} // namespace mlir