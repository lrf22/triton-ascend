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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/SyncAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "sync-analysis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace mlir::triton;

void mlir::triton::analyzeDepLoopInfo(DataDependencyInfo &info,
                                      DependencyInfo &dep) {
  auto blockInfoIt = info.getBlockInfoMap().find(dep.producerBlockId);
  if (blockInfoIt == info.getBlockInfoMap().end() ||
      blockInfoIt->second.Operations.empty()) {
    LOG_DEBUG("Producer block info is missing for dependency block "
              << dep.producerBlockId << "\n");
    return;
  }

  auto producerBlockOp = blockInfoIt->second.Operations.front();
  for (Operation *parent = producerBlockOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      dep.loopRegion = parent;
      dep.loopDepth = getOpLoopDepth(parent);
      break;
    }
  }
}

namespace {

// Assign flag IDs by reusing those from the dominant loop region:
// dependencies are grouped by loop depth and loop region, then regions other
// than the one with the most dependencies reuse its unique IDs round-robin.
// Dependencies in the dominant region retain their IDs and do not need
// start/end synchronization; reused dependencies do.
void assignDepFlagIds(llvm::SmallVector<DependencyInfo> &deps) {
  llvm::DenseMap<int, llvm::SmallVector<DependencyInfo *>> loopDepthGroups;
  for (auto &dep : deps) {
    if (!dep.loopRegion) {
      continue;
    }
    loopDepthGroups[dep.loopDepth].push_back(&dep);
  }

  for (auto &[loopDepth, depList] : loopDepthGroups) {
    if (depList.empty()) {
      continue;
    }

    llvm::DenseMap<int, llvm::SmallVector<DependencyInfo *>> regionGroups;
    for (auto *dep : depList) {
      int regionID = CVPipeline::getOpBlockId(dep->loopRegion).value_or(-1);
      regionGroups[regionID].push_back(dep);
    }

    auto it = regionGroups.begin();
    int maxRegion = it->first;
    size_t maxSize = it->second.size();
    ++it;

    for (; it != regionGroups.end(); ++it) {
      size_t curSize = it->second.size();
      if (curSize > maxSize) {
        maxRegion = it->first;
        maxSize = curSize;
      }
    }

    llvm::SmallVector<int> masterFlags;
    llvm::DenseSet<int> seen;
    for (auto *dep : regionGroups[maxRegion]) {
      if (seen.insert(dep->flagId).second) {
        masterFlags.push_back(dep->flagId);
      }
    }

    if (masterFlags.empty()) {
      continue;
    }

    for (auto &[region, group] : regionGroups) {
      if (region == maxRegion) {
        for (auto *dep : group) {
          dep->reuseStartEnd = false;
        }
        continue;
      }

      size_t idx = 0;
      for (auto *dep : group) {
        dep->flagId = masterFlags[idx % masterFlags.size()];
        dep->reuseStartEnd = true;
        ++idx;
      }
    }
  }
}

// Remap all distinct IDs to a contiguous range starting at zero, preserving
// the relationship between dependencies that share an ID.
void remapFlagIdsToContinuous(
    llvm::SmallVector<llvm::SmallVector<DependencyInfo> *> &depLists) {
  llvm::DenseMap<int, int> idMap;
  int nextId = 0;

  for (auto *list : depLists) {
    for (auto &dep : *list) {
      auto it = idMap.find(dep.flagId);
      if (it == idMap.end()) {
        idMap[dep.flagId] = nextId;
        dep.flagId = nextId;
        ++nextId;
      } else {
        dep.flagId = it->second;
      }
    }
  }
}

} // namespace

void mlir::triton::analyzeSyncReuse(DataDependencyInfo &info) {
  int flag_id = 0;
  for (auto &dep : info.getV2CDependencies()) {
    dep.flagId = flag_id++;
  }
  for (auto &dep : info.getC2VDependencies()) {
    dep.flagId = flag_id++;
  }
  for (auto &dep : info.getMemoryDependencies()) {
    dep.flagId = flag_id++;
  }

  assignDepFlagIds(info.getV2CDependencies());
  assignDepFlagIds(info.getC2VDependencies());
  assignDepFlagIds(info.getMemoryDependencies());

  llvm::SmallVector<llvm::SmallVector<DependencyInfo> *> depLists;
  depLists.push_back(&info.getV2CDependencies());
  depLists.push_back(&info.getC2VDependencies());
  depLists.push_back(&info.getMemoryDependencies());
  remapFlagIdsToContinuous(depLists);
}



/// 冗余边分析模块。
///
/// 使用 DenseMap + SmallVector 存储有向图，并提供统一的 addEdge 接口，
/// 方便后续扩展其他类型的边。
class RedundantEdgeAnalyzer {
public:
  using Graph = llvm::DenseMap<int, llvm::SmallVector<int>>;

  /// 添加一条有向边 from -> to。
  void addEdge(int from, int to) {
    graph_[from].push_back(to);
  }

  /// 根据 IR 中计算块的物理顺序构建隐式边。
  ///
  /// 相邻且类型相同的计算块之间存在隐式执行顺序保证。
  void buildImplicitEdges(DataDependencyInfo &info, ModuleOp module) {
    llvm::SmallVector<int> cubeBlockIds;
    llvm::SmallVector<int> vectorBlockIds;
    llvm::DenseSet<int> seenBlockIds;
    auto &blockInfoMap = info.getBlockInfoMap();

    module.walk([&](Operation *op) {
      auto blockId = CVPipeline::getOpBlockId(op);
      if (!blockId || !seenBlockIds.insert(*blockId).second) {
        return;
      }
      auto coreType = CVPipeline::getCoreType(op);
      //有可能存在同时是cube和vector的情况
    });

    auto connectAdjacentBlocks = [&](llvm::ArrayRef<int> blockIds) {
      for (size_t i = 1; i < blockIds.size(); ++i) {
        addEdge(blockIds[i - 1], blockIds[i]);
      }
    };
    connectAdjacentBlocks(cubeBlockIds);
    connectAdjacentBlocks(vectorBlockIds);
  }

  /// 分析依赖列表，标记每条依赖是否需要插入同步点。
  ///
  /// 对每条显式依赖边 u -> v，如果存在长度大于等于 2 的替代路径，
  /// 则认为该边冗余，sync = false；否则 sync = true。
  void analyzeAndMark(llvm::SmallVectorImpl<DependencyInfo> &dependencies) {
    for (auto &dep : dependencies) {
      int u = dep.producerBlockId;
      int v = dep.consumerBlockId;

      // 临时加入当前显式边，供 BFS 使用。
      addEdge(u, v);

      if (hasAlternativePath(u, v)) {
        dep.sync = false;
      } else {
        dep.sync = true;
      }

      // 分析完立即移除，避免影响后续依赖的独立判断。
      graph_[u].pop_back();
    }
  }

  /// 重置图结构，支持多轮分析。
  void reset() {
    graph_.clear();
  }

private:
  Graph graph_;

  /// 检查是否存在从 start 到 end 的替代路径，且路径长度 >= 2。
  bool hasAlternativePath(int start, int end) {
    if (start == end)
      return false;

    llvm::SmallVector<int, 8> worklist;
    llvm::DenseMap<int, bool> visited;

    worklist.push_back(start);
    visited[start] = true;

    while (!worklist.empty()) {
      int current = worklist.pop_back_val();

      auto it = graph_.find(current);
      if (it == graph_.end())
        continue;

      for (int neighbor : it->second) {
        if (neighbor == end) {
          // 如果前驱不是起点，说明路径长度 >= 2。
          if (current != start)
            return true;
          // current == start 表示直接边 start -> end，继续搜索其他路径。
        } else {
          if (!visited.contains(neighbor)) {
            visited[neighbor] = true;
            worklist.push_back(neighbor);
          }
        }
      }
    }

    return false;
  }
};