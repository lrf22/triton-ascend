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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

#include "DynamicCVPipeline/Common/Utils.h"

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "SplitMatmul";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

/**
 * Assumptions:
 * 1. cDefOp is not nullptr
 *
 * Safety: caller is responsible to ensure the assumptions
 */
static bool cIsZero(Operation *cDefOp)
{
    auto fillOp = dyn_cast<linalg::FillOp>(cDefOp);
    if (!fillOp) {
        return false;
    }
    auto filledVal = fillOp.getInputs()[0];
    auto constOp = filledVal.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
        return false;
    }
    return mlir::TypeSwitch<TypedAttr, bool>(constOp.getValueAttr())
        .Case<FloatAttr, IntegerAttr>([](auto intOrFloatAttr) { return intOrFloatAttr.getValue().isZero(); })
        .Default([](auto) { return false; });
}

/**
 * Assumptions:
 * 1. cDefOp is not nullptr
 *
 * Safety: caller is responsible to ensure the assumptions
 */
static bool cIsBroadcast1DTo2D(Operation *cDefOp, Value c)
{
    auto broadcastOp = dyn_cast<linalg::BroadcastOp>(cDefOp);
    if (!broadcastOp) {
        return false;
    }

    auto inputValue = broadcastOp.getInput();
    auto inputType = inputValue.getType();
    auto outputType = c.getType();

    auto inputTensor = dyn_cast<RankedTensorType>(inputType);
    auto outputTensor = dyn_cast<RankedTensorType>(outputType);
    if (!inputTensor || !outputTensor) {
        return false;
    }

    return inputTensor.getRank() == 1 && outputTensor.getRank() == 2;
}

static bool resultIsUsedByMatmul(Value res)
{
    return llvm::any_of(res.getUsers(), [](Operation *op) { return isa<linalg::MatmulOp>(op); });
}

// this generally should always be true, but just for safety...
static bool isFloatOrInt(RankedTensorType tensorType)
{
    auto elmType = tensorType.getElementType();
    return isa<FloatType, IntegerType>(elmType);
}

static bool shouldSplit(linalg::MatmulOp matmulOp)
{
    auto c = matmulOp.getDpsInits()[0];
    auto outputType = dyn_cast<RankedTensorType>(c.getType());
    if (!outputType) {
        LOG_DEBUG("Not split because not tensor mode matmul: " << matmulOp);
        return false;
    }
    if (!isFloatOrInt(outputType)) {
        LOG_DEBUG("Not split because not integer or float: " << matmulOp);
        return false;
    }

    // Rule 1: result is used by another matmul -> split
    if (resultIsUsedByMatmul(matmulOp.getResult(0))) {
        LOG_DEBUG("Split because result is used by another matmul: " << matmulOp);
        return true;
    }

    // Rule 2: c is block arg -> split
    auto *cDefOp = c.getDefiningOp();
    if (!cDefOp) {
        LOG_DEBUG("Split because c is block arg: " << matmulOp);
        // no defining op, split
        return true;
    }

    // Rule 3: matmul a b 0 -> do not split
    if (cIsZero(cDefOp)) {
        LOG_DEBUG("Not split because c is zero: " << matmulOp);
        return false;
    }

    // Rule 4: c is result of a broadcast operation from 1D to 2D -> do not split
    if (cIsBroadcast1DTo2D(cDefOp, c)) {
        LOG_DEBUG("Not split because c is broadcast from 1d to 2d: " << matmulOp);
        return false;
    }

    // Otherwise split:
    LOG_DEBUG("Should split: " << matmulOp);
    return true;
}

/**
 * Assumptions:
 * 1. matmul is tensor-based
 * 2. the accumulator is not zero-initialized
 * 3. the element type is integer or float (not e.g. complex type)
 *
 * Safety: The caller is responsible for ensuring the assumptions above
 */
static void splitMatmul(linalg::MatmulOp matmulOp, PatternRewriter &rewriter)
{
    auto inputs = matmulOp.getDpsInputs();
    auto a = inputs[0];
    auto b = inputs[1];
    // this is the accumulator/out operand, not result in tensor mode
    auto c = matmulOp.getDpsInits()[0];

    auto outputType = dyn_cast<RankedTensorType>(c.getType());
    if (!outputType) {
        LOG_DEBUG("Not tensor mode: " << matmulOp
                                      << "; the caller does not ensure the assumption. Cowardly doing nothing");
        return;
    }
    auto elmType = outputType.getElementType();

    Location loc = matmulOp.getLoc();

    // [Step 1] Create tensor.empty for the new accumulator tensor
    // Same shape and type as original matmul output
    SmallVector<Value> dynamicSizes;
    for (int64_t i = 0; i < outputType.getRank(); ++i) {
        if (outputType.isDynamicDim(i)) {
            dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, c, i));
        }
    }
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, outputType, dynamicSizes);

    // [Step 2] Create zero constant based on element type
    // Supports both floating-point (arith.constant float) and integer types
    Value zeroValue;
    if (auto floatType = dyn_cast<FloatType>(elmType)) {
        APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
        zeroValue = rewriter.create<arith::ConstantFloatOp>(loc, zeroAPFloat, floatType).getResult();
    } else if (auto intType = dyn_cast<IntegerType>(elmType)) {
        zeroValue = rewriter.create<arith::ConstantIntOp>(loc, 0, intType).getResult();
    } else {
        // User does not ensure assumption 3.
        return;
    }

    // [Step 3] Use linalg.fill to populate empty tensor with zero -> zero accumulator
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange {zeroValue}, ValueRange {emptyOp.getResult()});

    // [Step 5] Create new matmul using zero-filled tensor as accumulator
    // New matmul runs entirely on CUBE with no VECTOR dependency
    auto newMatmul = rewriter.create<linalg::MatmulOp>(loc, ValueRange {a, b}, ValueRange {fillOp.getResult(0)});
    NamedAttrList attrs(matmulOp->getAttrDictionary());
    constexpr StringLiteral kShouldRemoveAttrs[] = {"operandSegmentSizes", "res_attrs", "arg_attrs"};
    for (auto attr : kShouldRemoveAttrs) {
        attrs.erase(attr);
    }
    newMatmul->setAttrs(attrs);
    auto newMatmulRes = newMatmul.getResult(0);

    // [Step 6] Create add: add(new_matmul_result, outs_value)
    // This is the "c" in a*b+c, added after the matmul result
    Operation *addOp;
    if (isa<FloatType>(elmType)) {
        addOp = rewriter.create<arith::AddFOp>(loc, newMatmulRes, c).getOperation();
    } else {
        addOp = rewriter.create<arith::AddIOp>(loc, newMatmulRes, c).getOperation();
    }

    addOp->setAttr(CVPipeline::kAddFromMatmul, rewriter.getUnitAttr());
    rewriter.replaceOp(matmulOp, addOp);
}

LogicalResult SplitMatmulPattern::matchAndRewrite(linalg::MatmulOp matmulOp, PatternRewriter &rewriter) const
{
    if (!shouldSplit(matmulOp)) {
        return failure();
    }

    splitMatmul(matmulOp, rewriter);
    return success();
}
