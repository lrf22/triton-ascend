 #ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLITDATAFLOW_MEMORY_EFFECTS_TRACKER_H
 #define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLITDATAFLOW_MEMORY_EFFECTS_TRACKER_H

 #include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"

 namespace mlir {
 namespace CVPipeline {

 class SplitDataflowMemoryDependenceGraph : public MemoryDependenceGraph {
 public:
     SplitDataflowMemoryDependenceGraph(Operation *root, AliasAnalysis &aa)
         : MemoryDependenceGraph(root, aa) {}

 protected:
     void applyEffects(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown) override;
 };

 } // namespace CVPipeline
 } // namespace mlir

 #endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLITDATAFLOW_MEMORY_EFFECTS_TRACKER_H
