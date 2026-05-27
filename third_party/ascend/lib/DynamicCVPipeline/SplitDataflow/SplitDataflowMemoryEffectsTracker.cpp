#include "ascend/include/DynamicCVPipeline/SplitDataflow/SplitDataflowMemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

using namespace mlir;
using namespace mlir::CVPipeline;

static constexpr const char *DEBUG_TYPE = "splitdataflow-memory-effects-tracker";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

void SplitDataflowMemoryDependenceGraph::applyEffects(Operation *op,
    ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown)
{
    LOG_DEBUG("SplitDataflowMemoryDependenceGraph::applyEffects");
    // Conservative strategy. Unknown op is to treat as barrier.
    if (unknown) {
        for (auto &slot : slots) {
            slot->lastWriter = op;
            slot->pendingReads.clear();
        }
        return;
    }

    for (auto result: op->getOpResults()) {
        if (isa<BaseMemRefType>(result.getType())) {
            if (MemSlot *s = getOrCreateSlot(result)) {
                s->dataSource = op;
                s->lastWriter = op;
                s->pendingReads.clear();
            }
        }
    }

    DenseMap<Value, SmallVector<MemSlot *>> cache;

    for (const auto &e : effects) {
        auto *eDefiningOp = e.getValue().getDefiningOp();
        if (!eDefiningOp) {
            continue;
        }
        // Reads first: prevents a self-aliasing op from clearing its own pending read.
        if (isa<MemoryEffects::Read>(e.getEffect())) {
            for (MemSlot *s : resolveAliasSlots(e.getValue(), cache)) {
                s->pendingReads.insert(eDefiningOp);
            }
        }
    }

    for (const auto &e : effects) {
        Value v = e.getValue();
        auto *eDefiningOp = e.getValue().getDefiningOp();
        if (!eDefiningOp) {
            continue;
        }
        if (isa<MemoryEffects::Allocate>(e.getEffect())) {
            if (MemSlot *s = getOrCreateSlot(v)) {
                s->dataSource = eDefiningOp;
                s->lastWriter = eDefiningOp;
                s->pendingReads.clear();
            }
        } else if (isa<MemoryEffects::Write>(e.getEffect())) {
            for (MemSlot *s : resolveAliasSlots(v, cache)) {
                // annotation::MarkOp, no data produced, is marked with Mem::Write
                if (!isa<annotation::MarkOp>(op)) {
                    s->dataSource = eDefiningOp;
                }
                s->lastWriter = eDefiningOp;
                s->pendingReads.clear();
            }
        } else if (isa<MemoryEffects::Free>(e.getEffect())) {
            SmallPtrSet<MemSlot *, INIT_SIZE> toRemove;
            if (!v) {
                LOG_DEBUG("Free of unknown value: conservatively drop all slots.");
                for (auto &slot : slots) {
                    toRemove.insert(slot.get());
                }
            } else {
                auto aliasSlots = findAliasSlots(v);
                for (auto *slot : aliasSlots) {
                    toRemove.insert(slot);
                }
            }
            llvm::erase_if(slots, [&](std::unique_ptr<MemSlot> &sp) {
                if (toRemove.contains(sp.get())) {
                    if (sp->memref) {
                        valueToSlot.erase(sp->memref);
                    }
                    return true;
                }
                return false;
            });
            cache.clear();
        }
    }
}
