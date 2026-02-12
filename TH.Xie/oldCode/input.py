"""
Cyclodehydrogenation 最小重复单元 Demo

功能：
1. 用户用字符矩阵描述最小重复单元：
   - '1' 表示一个苯环（正六边形）
   - 其它字符表示没有
2. 把每一个苯环转换成 BenzeneHex 类对象（带一堆成员变量）
3. 在水平方向 head-to-tail 重复 repeat 次（Polymerization）
4. 把扩展后的 BenzeneHex 列表：
   - 画成 PNG 图（polymer.png）
   - 存成 JSON（polymer_hexes.json），供后续“切割模块”使用
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple
import math
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ===========================
# 1. 每一个苯环定义成一个类
# ===========================

@dataclass
class BenzeneHex:
    """表示一个苯环（正六边形）的所有信息"""
    id: int            # 全局唯一编号
    row: int           # 在最小单元中的行索引
    col: int           # 在最小单元中的列索引
    repeat_index: int  # 属于第几个重复单元（0,1,2,...）
    cx: float          # 中心 x 坐标（用来画图 / 后续几何切割）
    cy: float          # 中心 y 坐标
    size: float        # 六边形边长（正六边形）

    # 预留给你切割用的扩展信息，比如属于哪条链、是否在边缘等
    tag: str = ""      # 可以随意写，后面切割时根据 tag 判断


# ===========================
# 2. 把用户输入的字符矩阵 → 网格上的 (row, col)
# ===========================

def parse_pattern(pattern_str: str, on_char: str = "1") -> Tuple[List[Tuple[int, int]], int, int]:
    """
    把最小重复单元（字符矩阵）解析成有苯环的格子坐标列表。

    pattern_str 例子：
        1
        1
        1
        1

    on_char : 哪个字符表示“这里有一个苯环”。

    返回：
        cells  : [(row, col), ...]
        height : 行数
        width  : 列数
    """
    lines = [line.rstrip("\n") for line in pattern_str.strip("\n").splitlines()]
    if not lines:
        raise ValueError("输入图案为空，请在 pattern_str 中写入最小重复单元。")

    width = max(len(line) for line in lines)
    height = len(lines)

    cells: List[Tuple[int, int]] = []
    for r, line in enumerate(lines):
        # 行右侧不足宽度的用空格补齐，保证每行长度一致
        line = line.ljust(width)
        for c, ch in enumerate(line):
            if ch == on_char:
                cells.append((r, c))

    return cells, height, width


# ===========================
# 3. 网格 (row, col) → 正六边形中心坐标
# ===========================

def grid_to_center(row: int, col: int, size: float = 1.0) -> Tuple[float, float]:
    """
    把蜂窝网格中的 (row, col) 转成平面坐标 (cx, cy)。

    使用“尖顶六边形(pointy-top) + odd-r offset”布局：
      - 横向邻居中心距: sqrt(3) * size
      - 纵向行距      : 1.5 * size
      - 奇数行整体往右偏半个 hex
    """
    w = math.sqrt(3) * size      # 水平方向相邻中心距离
    v_step = 1.5 * size          # 垂直方向相邻行距

    x = w * (col + 0.5 * (row % 2))
    y = - v_step * row
    return x, y


# ===========================
# 4. 构建 Polymer：最小单元 head-to-tail 重复 repeat 次
# ===========================

def build_polymer(
    pattern_str: str,
    repeat: int,
    size: float = 1.0
) -> List[BenzeneHex]:
    """
    根据最小重复单元 pattern_str，在水平方向重复 repeat 次，
    返回所有 BenzeneHex 对象的列表。
    """
    cells, height, width = parse_pattern(pattern_str)

    hexes: List[BenzeneHex] = []
    hid = 0  # 全局 id

    for rep in range(repeat):
        # 每重复一次，整体在 col 方向平移一个最小单元的宽度
        col_offset = rep * width
        for (r, c) in cells:
            global_col = c + col_offset
            cx, cy = grid_to_center(r, global_col, size=size)

            hex_obj = BenzeneHex(
                id=hid,
                row=r,
                col=c,
                repeat_index=rep,
                cx=cx,
                cy=cy,
                size=size,
                tag=""  # 先留空，后续切割可以按需要填
            )
            hexes.append(hex_obj)
            hid += 1

    return hexes


# ===========================
# 5. 画出所有苯环（正六边形）
# ===========================

def hex_vertices(hex_obj: BenzeneHex) -> List[Tuple[float, float]]:
    """根据中心坐标和边长，生成一个正六边形 6 个顶点坐标"""
    cx, cy, size = hex_obj.cx, hex_obj.cy, hex_obj.size
    verts: List[Tuple[float, float]] = []
    # 从正上方开始，逆时针每 60° 一个点
    for k in range(6):
        angle_deg = 90 + 60 * k
        angle_rad = math.radians(angle_deg)
        x = cx + size * math.cos(angle_rad)
        y = cy + size * math.sin(angle_rad)
        verts.append((x, y))
    return verts


def draw_hexes(hexes: List[BenzeneHex], filename: str = "polymer.png") -> None:
    """把所有 BenzeneHex 画成线框六边形，并保存成 PNG"""
    if not hexes:
        raise ValueError("没有任何苯环可画。")

    fig, ax = plt.subplots(figsize=(10, 3))

    xs, ys = [], []
    for h in hexes:
        verts = hex_vertices(h)
        poly = Polygon(
            verts,
            closed=True,
            edgecolor="black",
            facecolor="none",
            linewidth=1.0
        )
        ax.add_patch(poly)
        xs.append(h.cx)
        ys.append(h.cy)

    margin = 2 * hexes[0].size
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# ===========================
# 6. 存储：把所有 BenzeneHex 写入 JSON
# ===========================

def save_hexes_json(hexes: List[BenzeneHex], filename: str = "polymer_hexes.json") -> None:
    """把 BenzeneHex 列表序列化成 JSON，便于后续切割模块读取"""
    data = [asdict(h) for h in hexes]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===========================
# 7. 示例入口
# ===========================

def main():
    # ====== 用户定义的最小重复单元（Cyclodehydrogenation 后的图形）======
    # 这里只举例：竖直 4 个苯环（类似你之前发的图2）
    # 你可以随便改成任意小图形，只要用 '1' 标出有苯环的位置即可。
    pattern_str = """
1
11
1
11
"""

    # ====== Polymerization：head-to-tail 重复多少次 ======
    # 之后“切割模块”如果需要扩大一倍，就把 repeat 从 1 改成 2 再生成一遍
    repeat = 5       # 类似图里那种长条，可以改成 5、10、30 等
    size = 1.0         # 正六边形边长，可以调大调小

    # 构建 polymer：每一个苯环 = 一个 BenzeneHex 实例
    hexes = build_polymer(pattern_str, repeat=repeat, size=size)

    print(f"生成苯环数量: {len(hexes)}")
    print("前 5 个苯环对象示例（后续切割就靠这些成员变量）：")
    for h in hexes[:5]:
        print(h)

    # 存储到 JSON，供后续“切割模块”使用
    save_hexes_json(hexes, "polymer_hexes.json")

    # 画出结构图，方便肉眼检查
    draw_hexes(hexes, "polymer.png")
    print("已生成 polymer.png 和 polymer_hexes.json")


if __name__ == "__main__":
    main()

