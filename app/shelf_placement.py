"""窑内层位安排：器物与层位的一一分配，最大化最小温区边界温差。

针对已保存方案的某个窑次：窑内 k 件器物与 k 个层位一一对应。层位 j
带有整数温度偏移 o_j（可正可负），器物 i 入位后的实际温度为该窑烧成
温度 T 加偏移，即 t = T + o_j，须落入器物温区闭区间 [low_i, high_i]。
器物 i 入位层位 j 的「边界温差」为实际温度到自身温区最近边界的距离：

    m_ij = min(t - low_i, high_i - t)   （可行当且仅当 m_ij >= 0）

目标：求一一分配使「全部器物边界温差的最小值」最大（瓶颈分配）；
最优值并列时，按器物编号升序排列各器物所配层位编号得到的序列，
取字典序最小者。

算法（k ≤ 14，与装窑求解器同规模）：

  1. 边界温差为整数，最优值落在 [0, max_i (high_i - low_i) // 2]
     （任一器物的边界温差不超过其温区宽度的一半）。在该整数区间上
     二分最大可行阈值 T*：判定「是否存在一一分配使每件器物边界温差
     ≥ T」即二分图完美匹配——器物 i 可接层位 j 当且仅当 m_ij ≥ T，
     用增广路算法求最大匹配，匹配数等于 k 即可行。阈值越大可用边
     越少，可行性关于阈值单调，故二分正确。
  2. 阈值 0 已不可行（存在器物无法落入温区的一一分配）时，抛出
     NoFeasibleShelfPlacementError。
  3. 字典序重建：按器物编号升序逐件确定层位，每步在剩余层位中按
     编号升序，取首个「自身满足 m_ij ≥ T* 且剩余器物仍能与剩余层位
     完成完美匹配」的层位。该贪心逐步固定序列前缀，所得层位编号
     序列在全部最优分配中字典序最小。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class NoFeasibleShelfPlacementError(ValueError):
    """无完整可行分配：无法让全部器物入位后实际温度均落入各自温区。"""


@dataclass(frozen=True)
class ShelfPlacementItem:
    """一件器物的层位安排结果。"""

    item_id: str
    shelf_id: str
    temp_offset: int
    actual_temp: int   # 实际温度 = 窑烧成温度 + 层位偏移
    temp_margin: int   # 到自身温区最近边界的温差


@dataclass(frozen=True)
class ShelfPlacement:
    """一个窑次的层位安排结果：逐件安排按器物编号升序。"""

    min_temp_margin: int
    items: tuple[ShelfPlacementItem, ...]


def _max_match(adj: Sequence[Sequence[int]], shelf_slots: int) -> int:
    """二分图最大匹配（增广路算法）。

    :param adj: adj[i] 为器物 i 可接的层位下标列表
    :param shelf_slots: 层位槽位数（层位下标取值 0..shelf_slots-1）
    :returns: 最大匹配成功的器物数
    """

    match_to = [-1] * shelf_slots  # 层位 -> 器物

    def _augment(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_to[j] < 0 or _augment(match_to[j], seen):
                match_to[j] = i
                return True
        return False

    matched = 0
    for i in range(len(adj)):
        if _augment(i, [False] * shelf_slots):
            matched += 1
    return matched


def optimal_shelf_placement(
    items: Sequence[tuple[str, int, int]],
    firing_temp: int,
    shelves: Sequence[tuple[str, int]],
) -> ShelfPlacement:
    """求器物与层位的一一分配，使最小边界温差最大。

    最优值并列时，按器物编号升序排列所配层位编号得到的序列取字典序
    最小；返回的逐件安排亦按器物编号升序。

    :param items: 窑内器物 ``(编号, 温区下界, 温区上界)`` 序列（1～14 件）
    :param firing_temp: 该窑烧成温度（整数）
    :param shelves: 层位 ``(编号, 整数温度偏移)`` 序列，数量须与器物数
        相同，编号互不重复
    :returns: 逐件层位安排与最大化后的最小边界温差
    :raises NoFeasibleShelfPlacementError: 不存在让全部器物实际温度均
        落入各自温区闭区间的一一分配
    :raises ValueError: 器物或层位为空 / 超过 14 件 / 数量不一致 /
        层位编号重复 / 温区下界大于上界
    """

    n = len(items)
    if n == 0:
        raise ValueError("至少需要一件器物")
    if n > 14:
        raise ValueError("器物数量不得超过 14 件")
    if len(shelves) != n:
        raise ValueError(
            f"层位数量须与器物数相同: {len(shelves)} 个层位 != {n} 件器物"
        )
    if len({s[0] for s in shelves}) != n:
        raise ValueError("层位编号必须唯一")
    for item_id, low, high in items:
        if low > high:
            raise ValueError(
                f"器物 {item_id} 的温度区间非法: 下界 {low} 大于上界 {high}"
            )

    # 器物按编号升序、层位按编号升序排列；下标 i / j 对应排序后次序。
    ordered_items = sorted(items, key=lambda it: it[0])
    ordered_shelves = sorted(shelves, key=lambda s: s[0])

    # margin[i][j]：器物 i 入位层位 j 的边界温差（负数即不可行）
    margin = [
        [
            min(
                firing_temp + ordered_shelves[j][1] - ordered_items[i][1],
                ordered_items[i][2] - firing_temp - ordered_shelves[j][1],
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    def _feasible(threshold: int) -> bool:
        adj = [
            [j for j in range(n) if margin[i][j] >= threshold]
            for i in range(n)
        ]
        return _max_match(adj, n) == n

    # ---- 1. 二分最大可行阈值（边界温差为整数，可行性关于阈值单调） ----
    if not _feasible(0):
        raise NoFeasibleShelfPlacementError(
            f"无完整可行分配：{n} 件器物与 {n} 个层位无法一一对应，"
            "使每件器物入位后的实际温度（烧成温度 + 层位偏移）"
            "均落入其温度区间"
        )
    lo = 0
    hi = max(it[2] - it[1] for it in ordered_items) // 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    best_margin = lo

    # ---- 2. 字典序最小重建：按器物编号升序逐件取最小可行层位 ----
    # 不变式：剩余器物与剩余层位之间恒存在阈值 best_margin 下的完美
    # 匹配，故每步必有可取层位。
    assign = [-1] * n  # 器物 -> 层位
    used = [False] * n
    for i in range(n):
        rest_count = n - i - 1
        for j in range(n):
            if used[j] or margin[i][j] < best_margin:
                continue
            # 试配 (i, j)：剩余器物须仍能与剩余层位完成完美匹配
            rest_adj = [
                [
                    j2
                    for j2 in range(n)
                    if not used[j2] and j2 != j and margin[i2][j2] >= best_margin
                ]
                for i2 in range(i + 1, n)
            ]
            if _max_match(rest_adj, n) == rest_count:
                assign[i] = j
                used[j] = True
                break
        else:  # 不变式保证必有可取层位，理论不可达
            raise AssertionError("层位安排重建失败：当前无层位可配")

    result_items = tuple(
        ShelfPlacementItem(
            item_id=ordered_items[i][0],
            shelf_id=ordered_shelves[assign[i]][0],
            temp_offset=ordered_shelves[assign[i]][1],
            actual_temp=firing_temp + ordered_shelves[assign[i]][1],
            temp_margin=margin[i][assign[i]],
        )
        for i in range(n)
    )
    # 全部器物边界温差均 >= best_margin；若均严格更大则与 best_margin
    # 的最大性矛盾，故最小值恰为 best_margin。
    assert min(it.temp_margin for it in result_items) == best_margin

    return ShelfPlacement(min_temp_margin=best_margin, items=result_items)
