"""shelf_placement 单元测试：固定用例 + 小规模全排列暴力枚举交叉验证。

暴力参考实现枚举器物到层位的全部排列，直接按定义取「最小边界温差最大、
并列时层位编号序列字典序最小」，与被测模块的「二分阈值 + 增广路匹配 +
贪心重建」相互独立。
"""

import os
import random
import sys
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shelf_placement import (  # noqa: E402
    NoFeasibleShelfPlacementError,
    optimal_shelf_placement,
)


def _brute_force(items, firing_temp, shelves):
    """枚举全部排列，返回 (最优最小温差, 编号升序器物对应的层位编号序列)。

    无可行分配返回 None。
    """

    ordered_items = sorted(items, key=lambda it: it[0])
    n = len(ordered_items)
    best_margin = None
    best_seq = None
    for perm in permutations(range(n)):
        # perm[i]：编号升序第 i 件器物入位的层位下标
        margins = []
        feasible = True
        for i, j in enumerate(perm):
            _, low, high = ordered_items[i]
            actual = firing_temp + shelves[j][1]
            if not low <= actual <= high:
                feasible = False
                break
            margins.append(min(actual - low, high - actual))
        if not feasible:
            continue
        m = min(margins)
        seq = tuple(shelves[perm[i]][0] for i in range(n))
        if (
            best_margin is None
            or m > best_margin
            or (m == best_margin and seq < best_seq)
        ):
            best_margin = m
            best_seq = seq
    if best_margin is None:
        return None
    return best_margin, best_seq


def _check_result(result, items, firing_temp, shelves):
    """校验结果的基本性质：一一对应、温度落区、最小温差一致。"""

    n = len(items)
    assert len(result.items) == n
    # 逐件按器物编号升序，层位一一对应
    assert [it.item_id for it in result.items] == sorted(i[0] for i in items)
    assert sorted(it.shelf_id for it in result.items) == sorted(
        s[0] for s in shelves
    )
    shelf_by_id = {s[0]: s[1] for s in shelves}
    interval_by_id = {i[0]: (i[1], i[2]) for i in items}
    margins = []
    for it in result.items:
        assert it.temp_offset == shelf_by_id[it.shelf_id]
        assert it.actual_temp == firing_temp + it.temp_offset
        low, high = interval_by_id[it.item_id]
        assert low <= it.actual_temp <= high  # 实际温度落入温区闭区间
        assert it.temp_margin == min(
            it.actual_temp - low, high - it.actual_temp
        )
        margins.append(it.temp_margin)
    assert min(margins) == result.min_temp_margin


def test_basic_two_items():
    # 烧成温度 150；p1 [120,220]，p2 [150,300]
    # p1->L1(+40)=190 余 30，p2->L2(+20)=170 余 20，最小 20
    # p1->L2(+20)=170 余 50，p2->L1(+40)=190 余 40，最小 40（最优）
    items = [("p1", 120, 220), ("p2", 150, 300)]
    shelves = [("L1", 40), ("L2", 20)]
    result = optimal_shelf_placement(items, 150, shelves)
    assert result.min_temp_margin == 40
    assert [(it.item_id, it.shelf_id) for it in result.items] == [
        ("p1", "L2"),
        ("p2", "L1"),
    ]
    assert [(it.actual_temp, it.temp_margin) for it in result.items] == [
        (170, 50),
        (190, 40),
    ]
    _check_result(result, items, 150, shelves)


def test_boundary_inclusive_and_negative_offset():
    # 实际温度恰等于温区边界可行（闭区间），边界温差为 0；偏移可为负
    items = [("a", 100, 200)]
    shelves = [("S", -50)]
    result = optimal_shelf_placement(items, 150, shelves)
    assert result.min_temp_margin == 0
    assert result.items[0].actual_temp == 100
    assert result.items[0].temp_margin == 0
    _check_result(result, items, 150, shelves)


def test_tie_break_lexicographic():
    # 两个层位偏移相同：两种分配的最小温差一致（p2 入位 170，余 20；
    # p1 入位 170，余 50），按器物编号升序比较层位编号序列，取 (L1, L2)
    items = [("p1", 120, 220), ("p2", 150, 300)]
    shelves = [("L2", 20), ("L1", 20)]  # 请求次序打乱不影响结果
    result = optimal_shelf_placement(items, 150, shelves)
    assert result.min_temp_margin == 20
    assert [it.shelf_id for it in result.items] == ["L1", "L2"]
    _check_result(result, items, 150, shelves)


def test_tie_break_with_distinct_offsets():
    # 三件器物、三种偏移均相同的层位：全部排列等价，取字典序最小前缀
    items = [("b", 0, 1000), ("a", 0, 1000), ("c", 0, 1000)]
    shelves = [("s3", 5), ("s1", 5), ("s2", 5)]
    result = optimal_shelf_placement(items, 100, shelves)
    assert [it.shelf_id for it in result.items] == ["s1", "s2", "s3"]
    _check_result(result, items, 100, shelves)


def test_no_feasible_raises():
    # b 的温区 [150,300] 容不下任何偏移后的实际温度（150-10=140、150-20=130）
    items = [("a", 100, 200), ("b", 150, 300)]
    shelves = [("S1", -10), ("S2", -20)]
    try:
        optimal_shelf_placement(items, 150, shelves)
    except NoFeasibleShelfPlacementError:
        pass
    else:  # pragma: no cover
        raise AssertionError("应当抛出 NoFeasibleShelfPlacementError")


def test_hall_blocking_raises():
    # 两件器物只能共用同一可行层位：单个可行温度各自存在，但无一一分配
    items = [("a", 100, 110), ("b", 105, 115)]
    shelves = [("S1", 0), ("S2", 100)]
    try:
        optimal_shelf_placement(items, 108, shelves)
    except NoFeasibleShelfPlacementError:
        pass
    else:  # pragma: no cover
        raise AssertionError("应当抛出 NoFeasibleShelfPlacementError")


def test_input_validation():
    items = [("a", 100, 200)]
    shelves = [("S", 0)]
    for bad_items, bad_shelves in [
        ([], shelves),                              # 器物为空
        (items, []),                                # 层位为空（数量不一致）
        (items, [("S", 0), ("T", 1)]),              # 层位多于器物
        (items * 2, [("S", 0), ("S", 1)]),          # 层位编号重复
        ([("a", 300, 200)], shelves),               # 温区下界大于上界
    ]:
        try:
            optimal_shelf_placement(bad_items, 150, bad_shelves)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(
                f"应当抛出 ValueError: items={bad_items}, shelves={bad_shelves}"
            )
    too_many = [(f"i{k}", 0, 10) for k in range(15)]
    try:
        optimal_shelf_placement(too_many, 0, [(f"s{k}", 0) for k in range(15)])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("超过 14 件应当抛出 ValueError")


def test_random_against_brute_force():
    rng = random.Random(20260926)
    shelf_id_pool = ["a", "b", "c2", "c10", "L1", "L2", "upper", "lower"]
    feasible_trials = 0
    infeasible_trials = 0
    for trial in range(3000):
        n = rng.randint(1, 6)
        firing_temp = rng.randint(0, 200)
        items = []
        for i in range(n):
            low = rng.randint(0, 200)
            high = low + rng.randint(0, 120)
            items.append((f"i{i}", low, high))
        offsets = [rng.randint(-160, 160) for _ in range(n)]
        ids = rng.sample(shelf_id_pool, n)
        shelves = list(zip(ids, offsets))
        rng.shuffle(shelves)

        expected = _brute_force(items, firing_temp, shelves)
        if expected is None:
            infeasible_trials += 1
            try:
                optimal_shelf_placement(items, firing_temp, shelves)
            except NoFeasibleShelfPlacementError:
                continue
            raise AssertionError(
                f"trial {trial}: 应当抛出 NoFeasibleShelfPlacementError, "
                f"items={items}, firing_temp={firing_temp}, shelves={shelves}"
            )

        feasible_trials += 1
        exp_margin, exp_seq = expected
        result = optimal_shelf_placement(items, firing_temp, shelves)
        got_seq = tuple(it.shelf_id for it in result.items)
        assert result.min_temp_margin == exp_margin, (
            f"trial {trial}: 最小温差 {result.min_temp_margin} != {exp_margin}, "
            f"items={items}, firing_temp={firing_temp}, shelves={shelves}"
        )
        assert got_seq == exp_seq, (
            f"trial {trial}: 层位序列 {got_seq} != {exp_seq}, "
            f"items={items}, firing_temp={firing_temp}, shelves={shelves}"
        )
        _check_result(result, items, firing_temp, shelves)

    # 随机参数应同时覆盖可行与不可行两类情形
    assert feasible_trials > 0 and infeasible_trials > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n全部 {len(fns)} 个测试通过")
