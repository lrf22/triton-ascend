# DynamicCVPipeline 代码风险总表（按 Pass 执行顺序）

**生成日期**: 2026-07-16
**数据来源**: [CodeRiskReport.md](file:///home/liurufeng/ssbuf322/triton-ascend-lrf/third_party/ascend/lib/DynamicCVPipeline/CodeRiskReport.md)

---

## Pipeline 总览

```
AddDynamicCVPipelinePass
 ├─ 1. PreCheckAvailablePass
 ├─ 2. StandardizeOpPass
 ├─ 3. PlanComputeBlockPass
 ├─ 4. ComputeBlockOptPass
 ├─ 5. SplitDataflowPass
 ├─ 6. AnalyzeDataFlowPass
 ├─ 7. SeparateMemoryFromComputePass
 ├─ 8. AllocMultiCachePass
 ├─ 9. AddControlFlowConditionPass
 └─ 10. RemoveSsbufAttrPass
```

---

## 一、公共模块风险（跨 Pass 共享）

以下风险来自公共工具类，被多个 Pass 共同使用，影响范围最广。

| 风险编号 | 严重程度 | 风险类别 | 文件 | 描述 | 影响的 Pass |
|---------|---------|---------|------|------|-----------|
| 1.1 | 高 | 空指针 | ComputeBlockIdManager.cpp:L32 | 构造函数未检查空指针 `root` | PlanComputeBlock, ComputeBlockOpt, SplitDataflow |
| 2.1 | 高 | 整数溢出 | SSBufferManager.cpp:L62 | 地址计算乘法溢出 | SplitDataflow, AddControlFlowCondition |
| 2.2 | 高 | 整数溢出 | FlagIdManager.cpp:L58 | Flag ID 溢出和 int 截断 | SplitDataflow, AllocMultiCache, AddControlFlowCondition |
| 2.4 | 中 | 整数溢出 | ComputeBlockIdManager.cpp:L69 | Block ID 溢出无检查 | PlanComputeBlock, ComputeBlockOpt |
| 3.1 | 高 | 资源泄露 | SSBufferManager.h/cpp:L45-62 | 无地址释放机制 | SplitDataflow, AddControlFlowCondition |
| 3.2 | 高 | 资源泄露 | FlagIdManager.cpp:L55-58 | Flag ID 无回收 | SplitDataflow, AllocMultiCache |
| 3.3 | 中 | 资源泄露 | ComputeBlockIdManager.cpp:L69 | Block ID 无回收 | PlanComputeBlock, ComputeBlockOpt |
| 3.4 | 中 | 资源泄露 | MemoryEffectsTracker.cpp:L88-93 | 悬空指针风险 | PlanComputeBlock, ComputeBlockOpt |
| 1.5 | 中 | 空指针 | MemoryEffectsTracker.cpp:L88-93 | 深层递归空指针传播 | PlanComputeBlock, ComputeBlockOpt |
| 1.6 | 低 | 空指针 | BufferCountManager.cpp:L55-65 | 空 ModuleOp 解引用 | SeparateMemoryFromCompute |
| 10.2 | 中 | 代码质量 | Common/Utils.cpp:L12 | 全局可变状态 g_enableCubeBlockMerge | 全局 |
| 10.5 | 低 | 代码质量 | 多个文件 | int/int64_t 类型不一致 | 全局 |
| 12.1 | 高 | 断言使用 | 全代码库 | 缺少关键不变量断言 | 全局 |
| NP-01 | 高 | 空指针 | ComputeBlockIdManager.cpp:L32 | 构造函数 `root` 未检查 | PlanComputeBlock, ComputeBlockOpt, SplitDataflow |
| NP-39 | 中 | 空指针 | BufferCountManager.cpp:L55-65 | `module_` 为空时解引用 | SeparateMemoryFromCompute |
| NP-62 | 低 | 空指针 | MemoryEffectsTracker.cpp:L88-93 | `analyzeOp` 中 `root` 未检查 | PlanComputeBlock, ComputeBlockOpt |
| NP-63 | 低 | 整数溢出 | SSBufferManager.cpp:L62 | 乘法溢出（同 2.1） | SplitDataflow, AddControlFlowCondition |
| NP-65 | 低 | — | FlagIdManager.cpp:L55-58 | `acquireId` 参数 `insertionPoint` 未使用 | SplitDataflow, AllocMultiCache |

---

## 二、顶层 Pipeline 风险

| 风险编号 | 严重程度 | 风险类别 | 文件 | 描述 |
|---------|---------|---------|------|------|
| 10.1 | 中 | 代码质量 | AddDynamicCVPipeline.cpp:L82-110 | moduleBackup 异常不安全，非 RAII 管理生命周期 |
| 11.1 | 高 | 异常处理 | AddDynamicCVPipeline.cpp:L93-110 | pass 失败后未完全恢复 IR，恢复操作可能部分失败 |

---

## 三、按 Pass 逐一梳理

### Pass 1: PreCheckAvailablePass

**子 Pass 执行顺序**: PreCheckBlacklistPass → PreCheckMatmulPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 描述 |
|---------|---------|---------|------|------|
| — | — | — | — | 本 Pass 无独立风险项 |

---

### Pass 2: StandardizeOpPass

**子 Pass 执行顺序**: PatternMatchRewritePass (SplitMatmulPattern)

| 风险编号 | 严重程度 | 风险类别 | 文件 | 描述 |
|---------|---------|---------|------|------|
| 1.3 | 中 | 空指针 | SplitMatmulPattern.cpp:L141-142 | `getDefiningOp()` 返回值未检查即解引用 |
| 4.4 | 中 | 数组越界 | SplitMatmulPattern.cpp:L88-95 | `parseMatmulInputs` 直接索引 `inputs[0]/[1]`、`inits[0]` 未检查大小 |
| NP-03 | 高 | 空指针 | SplitMatmulPattern.cpp:142-144 | `getDefiningOp()` 对 BlockArgument 返回 nullptr 后直接 `op->getParentOp()` |
| NP-04 | 高 | 数组越界 | SplitMatmulPattern.cpp:60-65 | `inputs[0]`、`inputs[1]`、`inits[0]` 未检查容器大小 |

---

### Pass 3: PlanComputeBlockPass

**子 Pass 执行顺序**: OpClassifierPass → PlanCubeBlockPass → PlanVectorBlockPass → ReorderOpsByBlockIdPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 1.2 | 高 | 空指针 | OpClassifier.cpp:L227-231 | OpClassifierPass | `matchTransposePattern` 返回 `bool` 但 `return;` 无值（UB） |
| 8.3 | 中 | 逻辑错误 | ReorderOpsByBlockId.cpp:L197-209 | ReorderOpsByBlockIdPass | `collectBlockIds` 中 `nestedOp` 在 walk 外不可用，变量覆盖 |
| 5.2 | 中 | 其他 | ComputeBlockIdManager.h:L43 | 公共 | mutex 声明但未使用 |
| NP-02 | 高 | 空指针 | OpClassifier.cpp:L227-231 | OpClassifierPass | `def` 未做空指针检查 + `return;` 在 bool 函数中是 UB |
| NP-22 | 中 | 空指针 | PlanCubeBlock.cpp:117 | PlanCubeBlockPass | `willCreateCycle` 中 `op->getBlock()` 可能返回空 |
| NP-23 | 中 | 空指针 | PlanCubeBlock.cpp:230 | PlanCubeBlockPass | `removeNonCubeOpsRecursively` 中 `op->getBlock()` 可能返回空 |
| NP-24 | 中 | 空指针 | PlanCubeBlock.cpp:~300 | PlanCubeBlockPass | `group[head]` 索引和 `getDefiningOp()` 检查 |
| NP-25 | 中 | 空指针 | PlanVectorBlockPass.cpp:70-75 | PlanVectorBlockPass | `nowOp->getBlock()` 可能返回空 |
| NP-26 | 中 | 空指针 | PlanVectorBlockPass.cpp:230 | PlanVectorBlockPass | `queue.front()` 空指针检查顺序 |
| NP-55 | 中 | 空指针 | ReorderOpsByBlockId.cpp:100-110 | ReorderOpsByBlockIdPass | `resolveToBlockOp` 可能返回 nullptr |
| 12.4 | 低 | 断言使用 | PlanCubeBlock.cpp:L110-134 | PlanCubeBlockPass | 环检测策略与 Common.cpp 不一致 |
| 9.1 | 中 | 冗余代码 | PlanCubeBlock.cpp:L110-134 | PlanCubeBlockPass | 与 Common.cpp、FixpipeOptPass.cpp 重复环检测实现 |

---

### Pass 4: ComputeBlockOptPass

**子 Pass 执行顺序**: UnifyAllocBlockPass → ReorderOpsByBlockIdPass → MergeVectorIfBlockPass → ReorderOpsByBlockIdPass → MergeCubeForBlockPass → ReorderOpsByBlockIdPass → UBUsageOptPass → ReorderOpsByBlockIdPass → FixpipeOptPass → ReorderOpsByBlockIdPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 2.3 | 高 | 整数溢出 | UBUsageOptPass.cpp:L98-124 | UBUsageOptPass | `getValueSizeInBytes` 张量大小计算溢出 + int 截断 |
| 4.3 | 中 | 数组越界 | UBUsageOptPass.cpp:L125-130 | UBUsageOptPass | 终止器操作数索引，`std::min` 限制但逻辑依赖 IR 正确性 |
| 8.4 | 中 | 逻辑错误 | FixpipeOptPass.cpp:L335-387 | FixpipeOptPass | `isSubviewFromGlobalMemory` 不可达代码 |
| 11.4 | 中 | 异常处理 | UnifyAllocBlockPass.cpp:L335-340 | UnifyAllocBlockPass | 检测到环后返回 `success()` 而非 `failure()` |
| 12.2 | 中 | 断言使用 | Common.cpp:L105-130 | 公共 | `willCreateCycle` 临时修改 block ID 缺少 RAII 回滚保证 |
| 12.3 | 中 | 断言使用 | FixpipeOptPass.cpp:L117-140 | FixpipeOptPass | 同 12.2，`willCreateCycle` 回滚无保证 + 接口不一致 |
| 5.1 | 中 | 其他 | Common.cpp:L50-80 | 公共 | `CycleDfs` 深度递归可能栈溢出 |
| 10.3 | 中 | 代码质量 | Common.cpp:L50-130 | 公共 | DFS 环检测无深度限制，时间复杂度可能指数级 |
| NP-27 | 中 | 空指针 | UBUsageOptPass.cpp:L125-130 | UBUsageOptPass | `terminator` 空指针检查已有，`yielded.getDefiningOp()` 需检查 |
| NP-28 | 中 | 空指针 | UBUsageOptPass.cpp:L155-165 | UBUsageOptPass | `getAncestorInBlock` 可能返回空（已有检查，安全） |
| NP-42 | 中 | 空指针 | Common.cpp:L105-130 | 公共 | `willCreateCycle` 中 `getBlock()` 可能返回空 |
| NP-43 | 中 | 空指针 | Common.cpp:L115-120 | 公共 | `getExecAfter` 返回的指针可能包含空 |
| NP-44 | 中 | 空指针 | FixpipeOptPass.cpp:L117-140 | FixpipeOptPass | `block` 参数未做空指针检查 |
| NP-45 | 中 | 空指针 | FixpipeOptPass.cpp:L155-165 | FixpipeOptPass | `getAncestorInBlock` 可能返回 nullptr |
| NP-53 | 中 | 空指针 | UnifyAllocBlockPass.cpp:L130-140 | UnifyAllocBlockPass | `fillOp->getBlock()` 安全（来自 dyn_cast 成功结果） |
| NP-54 | 中 | 空指针 | MergeCubeForBlockPass.cpp:L80-90 | MergeCubeForBlockPass | `forOp.getBody()` 可能返回空 |
| 9.1 | 中 | 冗余代码 | Common.cpp:L50-130 / FixpipeOptPass.cpp:L117-140 | 公共 / FixpipeOptPass | 与 PlanCubeBlock.cpp 重复环检测实现 |

---

### Pass 5: SplitDataflowPass

**子 Pass 执行顺序**: AddBlockIdForControlOpsPass → DataDependencyAnalysisPass → InterCoreTransferAndSyncPass → MarkMainLoopPass → SeparateCVScopePass → PreserveControlAttrsCanonicalizePass → RefineArgsBlockIdPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 2.5 | 中 | 整数溢出 | AddBlockIdForControlOps.cpp:L55-56 | AddBlockIdForControlOpsPass | `maxBlockId` 持续自增无溢出检查 |
| 1.4 | 中 | 空指针 | InterCoreTransferAndSync.cpp:L131-134 | InterCoreTransferAndSyncPass | `getBlockStartEnd` 返回空指针后未检查 |
| 2.7 | 低 | 整数溢出 | InterCoreTransferAndSync.cpp:L85-95 | InterCoreTransferAndSyncPass | `uint64_t` 与 0 比较永远为 false |
| 8.1 | 高 | 逻辑错误 | DataDependencyAnalysis.cpp:L148-155 | DataDependencyAnalysisPass | `isOuterOpArg` 始终返回 true，遗漏跨核心依赖 |
| 8.5 | 中 | 逻辑错误 | MarkMainLoop.cpp:L55-73 | MarkMainLoopPass | 嵌套 main_loop 检测不完整，多层嵌套处理顺序不确定 |
| 8.7 | 低 | 逻辑错误 | DataDependencyAnalysis.cpp:L544-550 | DataDependencyAnalysisPass | transpose 替换可能破坏 IR 语义 + 迭代器失效 |
| 9.5 | 低 | 冗余代码 | DataDependencyAnalysis.cpp:L94-98 | DataDependencyAnalysisPass | `isCubeOrVectorOp` 命名与逻辑不匹配 |
| 10.6 | 低 | 代码质量 | PreserveControlAttrsCanonicalize.cpp:L38-42 | PreserveControlAttrsCanonicalizePass | `debugDumpIr` 中 LOG_DEBUG 宏语法问题 |
| 11.2 | 高 | 异常处理 | DataDependencyAnalysis.cpp:L236-242 | DataDependencyAnalysisPass | `signalPassFailure()` 后继续执行，插入无效依赖 |
| 11.3 | 中 | 异常处理 | DataDependencyAnalysis.cpp:L405-420 | DataDependencyAnalysisPass | `signalPassFailure()` 后继续循环处理 iterArg |
| 4.6 | 低 | 数组越界 | RefineArgsBlockId.cpp:L42-48 | RefineArgsBlockIdPass | Block 参数索引边界情况 |
| 5.3 | 低 | 其他 | FlagIdReuse.cpp:L90-114 | InterCoreTransferAndSyncPass | 跨 Block 顺序判断不完整 |
| NP-10 | 高 | 空指针 | InterCoreTransferAndSync.cpp:117-130 | InterCoreTransferAndSyncPass | `getBlockStartEnd` 返回 `{nullptr, nullptr}` 后未检查 |
| NP-11 | 高 | 异常处理 | DataDependencyAnalysis.cpp:236-242 | DataDependencyAnalysisPass | `signalPassFailure()` 后使用 -1 构建 invalid 依赖 |
| NP-12 | 中 | 空指针 | DataDependencyAnalysis.cpp:L527 | DataDependencyAnalysisPass | `dyn_cast<OpResult>(value)` 结果未检查 |
| NP-46 | 中 | 空指针 | InterCoreTransferAndSync.cpp:148 | InterCoreTransferAndSyncPass | `getBlock()` 有空指针检查（安全） |
| NP-47 | 中 | 空指针 | InterCoreTransferAndSync.cpp:160-170 | InterCoreTransferAndSyncPass | `block` 已检查非空（安全） |
| NP-48 | 中 | 空指针 | AddMultiBufferInnerScope.cpp:50-55 | —（属 AllocMultiCache） | `forOp.getBody()` 可能返回空 |
| NP-58 | 中 | 空指针 | DataDependencyAnalysis.cpp:L399 | DataDependencyAnalysisPass | `forOp.getBody()->getTerminator()` 可能返回空 |
| NP-59 | 中 | 数组越界 | DataDependencyAnalysis.cpp:L404-407 | DataDependencyAnalysisPass | `iterArgIndex` 可能超过 iter_args 数量 |
| NP-67 | 低 | 空指针 | PlanVectorBlockPass.cpp:340-350 | —（属 PlanComputeBlock） | `block` 来自 walk 理论非空 |

---

### Pass 6: AnalyzeDataFlowPass

**子 Pass 执行顺序**: AnalyzeNamePass → AnalyzeScopePass → AnalyzeArgsPass → AnalyzeFlagPass → AnalyzeCubeContolFLowInputChainPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 2.6 | 中 | 整数溢出 | AnalyzeFlag.cpp:L62 | AnalyzeFlagPass | 硬编码上限 14 与 `MAX_FLAG_ID` 重复定义 + int 截断 |
| 4.5 | 中 | 数组越界 | AnalyzeArgs.cpp:L73-80 | AnalyzeArgsPass | yield 操作数数量可能少于 iter_args 数量 |
| NP-60 | 中 | 空指针 | AnalyzeArgs.cpp:80-85 | AnalyzeArgsPass | `body->getTerminator()` 可能返回空 + yield 索引越界 |

---

### Pass 7: SeparateMemoryFromComputePass

**子 Pass 执行顺序**: AsyncLoadHoistingPass → AddMultiBufferToGMLoadPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 4.2 | 高 | 数组越界 | LoopTransform.cpp:L42 | AddMultiBufferToGMLoadPass | `iterArgIdx + kForBodyIterArgOffset` 可能越界 |
| 3.6 | 低 | 资源泄露 | AddMultiBufferToGMLoad.cpp:L95-135 | AddMultiBufferToGMLoadPass | 嵌套 ForOp 清理不完整 |
| 9.2 | 中 | 冗余代码 | DecoupleComputeAndMemory/ & SeparateMemoryFromCompute/ | 公共 | 两个目录下 AddMultiBufferToGMLoad.cpp 重复 |
| NP-29 | 中 | 空指针 | AsyncLoadHoisting.cpp:88-90 | AsyncLoadHoistingPass | `getDefiningOp()` 空指针检查已有（安全） |
| NP-30 | 中 | 空指针 | AsyncLoadHoisting.cpp:130-135 | AsyncLoadHoistingPass | `loadOp->getBlock()` 可能返回空 |
| NP-31 | 中 | 空指针 | AsyncLoadHoisting.cpp:155-160 | AsyncLoadHoistingPass | `getDefiningOp()` 返回空表示 BlockArgument（逻辑正确） |
| NP-32 | 中 | 空指针 | DependencyAnalysis.cpp:27-30 | 公共 | `scopeBlock` 可能为 nullptr |
| NP-33 | 中 | 空指针 | DependencyAnalysis.cpp:38-43 | 公共 | `sourceOp` 和 `scopeBlock` 未做空指针检查 |
| NP-34 | 中 | 空指针 | MultiBufferLoopBodyEmission.cpp:30-35 | AddMultiBufferToGMLoadPass | `forOp.getBody()` 和 `getTerminator()` 可能返回空 |
| NP-35 | 中 | 空指针 | MultiBufferLoopBodyEmission.cpp:90-95 | AddMultiBufferToGMLoadPass | `builder.create` 极端情况可能失败 |
| NP-36 | 中 | 空指针 | MultiBufferLoopBodyEmission.cpp:100-110 | AddMultiBufferToGMLoadPass | `oldBody`/`newBody` 未检查 + `getArgument` 索引越界 |
| NP-37 | 中 | 空指针 | MultiBufferLoopBodyEmission.cpp:120-125 | AddMultiBufferToGMLoadPass | `cast<scf::YieldOp>` 在空指针上不安全 |
| NP-38 | 中 | 空指针 | AddMultiBufferToGMLoad.cpp:55-60 | AddMultiBufferToGMLoadPass | `BufferCountManager` 构造时 `module` 可能为空 |
| NP-40 | 中 | 空指针 | LoopTransform.cpp:230-240 | AddMultiBufferToGMLoadPass | `getDefiningOp()` 对 BlockArgument 返回空后直接解引用 |
| NP-41 | 中 | 空指针 | LoopTransform.cpp:250-260 | AddMultiBufferToGMLoadPass | `getDefiningOp<arith::AddIOp>()` 已检查（安全） |
| NP-68 | 低 | 空指针 | AddMultiBufferToGMLoad.cpp:100-110 | AddMultiBufferToGMLoadPass | `context.forOp` erase 后安全（reverse 迭代） |

---

### Pass 8: AllocMultiCachePass

**子 Pass 执行顺序**: AddMultiBufferInnerScopePass → AddMultiBufferOuterScopePass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 8.2 | 高 | 逻辑错误 | AddMultiBufferInnerScope.cpp:L170-195 | AddMultiBufferInnerScopePass | `getForOpPriority` 优先级分支错误：有 block_id 时应返回 `priorityBlockId(2)` 但返回了 `priorityIterArgs(3)` |
| 8.6 | 中 | 逻辑错误 | AddMultiBufferOuterScope.cpp:L275-285 | AddMultiBufferOuterScopePass | Fallback 逻辑匹配错误 flag，不同 flag 的 set/wait 可能被错误配对 |
| 11.5 | 中 | 异常处理 | AddMultiBufferOuterScope.cpp:L388 | AddMultiBufferOuterScopePass | `acquireId` 失败后无错误处理，使用无效 flag ID |
| 9.3 | 低 | 冗余代码 | AddMultiBufferOuterScope.cpp:L60-65 | AddMultiBufferOuterScopePass | 局部 `getBlockId` 与 `CVPipeline::getOpBlockId` 重复 |
| 9.4 | 低 | 冗余代码 | AddMultiBufferInnerScope.cpp:L154-160 / OuterScope.cpp | 两个子 Pass | 重复的 main_loop 检查函数 |
| 10.4 | 低 | 代码质量 | AddMultiBufferInnerScope.cpp:L154-160 | AddMultiBufferInnerScopePass | `collectNestedOps` 递归无终止保护 |
| NP-48 | 中 | 空指针 | AddMultiBufferInnerScope.cpp:50-55 | AddMultiBufferInnerScopePass | `forOp.getBody()` 可能返回空 |
| NP-49 | 中 | 空指针 | AddMultiBufferInnerScope.cpp:80-90 | AddMultiBufferInnerScopePass | `f.getBody()` 可能返回空 |
| NP-50 | 中 | 空指针 | AddMultiBufferOuterScope.cpp:60-65 | AddMultiBufferOuterScopePass | `forOp.getBody()` 可能返回空 |
| NP-51 | 中 | 空指针 | AddMultiBufferOuterScope.cpp:85-90 | AddMultiBufferOuterScopePass | `start` 参数未做空指针检查 |
| NP-52 | 中 | 空指针 | AddMultiBufferOuterScope.cpp:105-110 | AddMultiBufferOuterScopePass | 同 NP-51，`start` 未检查 |

---

### Pass 9: AddControlFlowConditionPass

**子 Pass 执行顺序**: CloneOpsPass → ProcessArgsPass → CreateIfOpsPass → InitDependentMapPass → UpdateForOpsPass → UpdateConditionInfoPass → UpdateLoopIterTimesPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 所属子 Pass | 描述 |
|---------|---------|---------|------|-----------|------|
| 4.1 | 高 | 数组越界 | CreateIfOps.cpp:L68-85 | CreateIfOpsPass | yield 值索引越界，检查在访问之后 |
| 3.5 | 中 | 资源泄露 | CloneOps.cpp:L95-119 | CloneOpsPass | 克隆操作产生的 IR 在失败路径中未清理 |
| NP-05 | 高 | 空指针 | InitDependentMap.cpp:56 | InitDependentMapPass | `consumer` 参数未做空指针检查 |
| NP-06 | 高 | 空指针 | UpdateForOps.cpp:91-92 | UpdateForOpsPass | `getBody()` 可能返回空指针 |
| NP-07 | 高 | 空指针 | UpdateForOps.cpp:99 | UpdateForOpsPass | `getTerminator()` 可能返回空，`cast<>` 不安全 |
| NP-08 | 高 | 数组越界 | CreateIfOps.cpp:68-85 | CreateIfOpsPass | 先通过 `oldYieldValues[i]` 访问再检查越界 |
| NP-09 | 高 | 数组越界 | CreateIfOps.cpp:42-55 | CreateIfOpsPass | yield 操作数索引可能超过 iter_args 数量 |
| NP-13 | 中 | 空指针 | UpdateConditionInfo.cpp:85-120 | UpdateConditionInfoPass | walk 后 `ssbufferVec0Ptrs` 可能为空 |
| NP-14 | 中 | 空指针 | UpdateConditionInfo.cpp:85 | UpdateConditionInfoPass | `info` 指针未做空指针检查 |
| NP-15 | 中 | 空指针 | UpdateForOps.cpp:99 | UpdateForOpsPass | 同 NP-07，`cast<>` 在空指针上不安全 |
| NP-16 | 中 | 空指针 | UpdateForOps.cpp:104-106 | UpdateForOpsPass | `newBlock` 可能为空 |
| NP-17 | 中 | 空指针 | CloneOps.cpp:72-80 | CloneOpsPass | `builder.clone()` 可能返回空 |
| NP-18 | 中 | 空指针 | CloneOps.cpp:55 | CloneOpsPass | 入口有空指针检查（安全） |
| NP-19 | 中 | 空指针 | ProcessArgs.cpp:32-34 | ProcessArgsPass | `getTerminator()` 返回值未再次检查 |
| NP-20 | 中 | 空指针 | ProcessArgs.cpp:119 | ProcessArgsPass | `cast<scf::YieldOp>` 在空指针上不安全 |
| NP-21 | 中 | 数组越界 | ProcessArgs.cpp:121 | ProcessArgsPass | `info.argIndex` 可能超过 `yieldOp.getNumOperands()` |
| NP-56 | 中 | 空指针 | UpdateLoopIterTimes.cpp:80-85 | UpdateLoopIterTimesPass | `dyn_cast` 和 `getAttrOfType` 已检查（安全） |
| NP-57 | 中 | 空指针 | UpdateLoopIterTimes.cpp:130-135 | UpdateLoopIterTimesPass | `buffer.getDefiningOp()` 可能返回空，`isa<>(nullptr)` 未定义 |
| NP-69 | 低 | 空指针 | AddControlFlowCondition.cpp | 公共 | `info` 来自 `getAnalysis<>` 理论非空 |

---

### Pass 10: RemoveSsbufAttrPass

| 风险编号 | 严重程度 | 风险类别 | 文件 | 描述 |
|---------|---------|---------|------|------|
| — | — | — | — | 本 Pass 无独立风险项 |

---

## 四、风险统计总表

### 按 Pass 统计风险数量

| Pass | 高风险 | 中风险 | 低风险 | 合计 |
|------|--------|--------|--------|------|
| 公共模块 | 5 | 5 | 4 | 14 |
| 顶层 Pipeline | 1 | 1 | 0 | 2 |
| 1. PreCheckAvailable | 0 | 0 | 0 | 0 |
| 2. StandardizeOp | 2 | 2 | 0 | 4 |
| 3. PlanComputeBlock | 2 | 5 | 2 | 9 |
| 4. ComputeBlockOpt | 1 | 10 | 0 | 11 |
| 5. SplitDataflow | 3 | 7 | 3 | 13 |
| 6. AnalyzeDataFlow | 0 | 3 | 0 | 3 |
| 7. SeparateMemoryFromCompute | 1 | 10 | 2 | 13 |
| 8. AllocMultiCache | 1 | 5 | 3 | 9 |
| 9. AddControlFlowCondition | 4 | 11 | 1 | 16 |
| 10. RemoveSsbufAttr | 0 | 0 | 0 | 0 |
| **合计** | **20** | **59** | **15** | **94** |

### 按风险类别统计

| 风险类别 | 高风险 | 中风险 | 低风险 | 合计 |
|---------|--------|--------|--------|------|
| 空指针解引用 | 11 | 36 | 5 | 52 |
| 整数运算溢出/反转 | 3 | 4 | 1 | 8 |
| 数组越界 | 3 | 4 | 2 | 9 |
| 资源泄露 | 2 | 4 | 2 | 8 |
| 逻辑错误 | 2 | 5 | 1 | 8 |
| 异常处理 | 2 | 4 | 1 | 7 |
| 代码质量 | 0 | 3 | 3 | 6 |
| 冗余代码 | 0 | 3 | 3 | 6 |
| 断言使用 | 1 | 2 | 1 | 4 |
| 其他 | 0 | 1 | 1 | 2 |

### 按修复优先级统计

| 优先级 | 数量 | 关键风险项 |
|--------|------|-----------|
| P0 — 立即修复 | 8 | 1.2(UB), 2.1(溢出), 4.1(越界), 4.2(越界), 8.1(逻辑), 8.2(逻辑), 11.2(异常), 12.1(断言) |
| P1 — 尽快修复 | 20 | 1.1, 1.3, 2.2, 2.3, 3.1, 3.2, 8.3, 8.5, 8.6, 9.1, 10.1, 10.2, 11.1, 11.4, 11.5, 12.2, 12.3 及 NP 高风险项 |
| P2 — 计划修复 | 35 | 1.4, 1.5, 2.4-2.6, 3.3-3.5, 4.3-4.5, 8.4, 8.7, 9.2, 10.3, 10.5, 11.3, 11.6 及 NP 中风险项 |
| P3 — 低优先级 | 31 | 1.6, 2.7, 3.6, 4.6, 5.1-5.3, 8.7, 9.3-9.5, 10.4, 10.6, 12.4 及 NP 低风险项 |

---

## 五、高风险项快速索引（按 Pass 执行顺序）

| 执行序号 | Pass | 风险编号 | 文件 | 一句话描述 |
|---------|------|---------|------|-----------|
| — | 公共 | 2.1 | SSBufferManager.cpp:L62 | 地址计算乘法溢出 |
| — | 公共 | 2.2 | FlagIdManager.cpp:L58 | Flag ID 溢出和 int 截断 |
| — | 公共 | 3.1 | SSBufferManager.h/cpp:L45-62 | 无地址释放机制 |
| — | 公共 | 3.2 | FlagIdManager.cpp:L55-58 | Flag ID 无回收 |
| — | 公共 | 12.1 | 全代码库 | 缺少关键不变量断言 |
| 0 | Pipeline | 11.1 | AddDynamicCVPipeline.cpp:L93-110 | pass 失败后未完全恢复 |
| 2 | StandardizeOp | NP-03 | SplitMatmulPattern.cpp:L142-144 | getDefiningOp() 返回空后直接解引用 |
| 2 | StandardizeOp | NP-04 | SplitMatmulPattern.cpp:L60-65 | 数组索引未检查容器大小 |
| 3 | PlanComputeBlock | 1.2 | OpClassifier.cpp:L227-231 | bool 函数中 return; 无值（UB） |
| 3 | PlanComputeBlock | NP-02 | OpClassifier.cpp:L227-231 | def 未做空指针检查 + UB |
| 4 | ComputeBlockOpt | 2.3 | UBUsageOptPass.cpp:L98-124 | 张量大小计算溢出 + int 截断 |
| 5 | SplitDataflow | 8.1 | DataDependencyAnalysis.cpp:L148-155 | isOuterOpArg 始终返回 true |
| 5 | SplitDataflow | 11.2 | DataDependencyAnalysis.cpp:L236-242 | signalPassFailure 后继续执行 |
| 5 | SplitDataflow | NP-10 | InterCoreTransferAndSync.cpp:L117-130 | getBlockStartEnd 返回空后未检查 |
| 7 | SepMemFromCompute | 4.2 | LoopTransform.cpp:L42 | iter_arg 索引越界 |
| 8 | AllocMultiCache | 8.2 | AddMultiBufferInnerScope.cpp:L170-195 | getForOpPriority 优先级分支错误 |
| 9 | AddCtrlFlowCond | 4.1 | CreateIfOps.cpp:L68-85 | yield 值索引越界 |
| 9 | AddCtrlFlowCond | NP-05 | InitDependentMap.cpp:L56 | consumer 参数未检查空指针 |
| 9 | AddCtrlFlowCond | NP-06 | UpdateForOps.cpp:L91-92 | getBody() 可能返回空 |
| 9 | AddCtrlFlowCond | NP-07 | UpdateForOps.cpp:L99 | getTerminator() + cast<> 不安全 |
| 9 | AddCtrlFlowCond | NP-08 | CreateIfOps.cpp:L68-85 | 越界检查在访问之后 |
| 9 | AddCtrlFlowCond | NP-09 | CreateIfOps.cpp:L42-55 | yield 索引可能超过 iter_args |