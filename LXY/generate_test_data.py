import os
import pathlib

# 1. 定义数据存放的文件夹名称 (需要与 gnr_loader.py 中使用的保持一致)
DATA_FOLDER = "data_files"

# 2. 定义测试 SMILES 数据集
# 命名参考你图片中的三种结构
TEST_SMILES_DATA = {
    "precursor_1_dibromo.smi":
    # Precursor 1: 1,4-dibromo-2,5-bis(p-tolyl)benzene, 简化为二溴联苯片段
        "BrC1=CC(Br)=C(C=C1)C2=CC=CC=C2",

    "polymer_head_to_tail.smi":
    # Head-to-Tail 聚合链 (代表图中的中间聚合物)
        "C1=CC=C(C=C1)C2=C(Br)C=C(Br)C=C2C3=CC=CC=C3.C1=CC=C(C=C1)C2=C(Br)C=C(Br)C=C2C3=CC=CC=C3",
    # 注意：这里我们使用一个较长的、但仍然是小分子的结构来模拟聚合片段，
    # 实际高分子的SMILES会更长。这里用一个长的联苯片段来代表聚合链。

    "gnr_7ac_segment.smi":
    # 7-AGNR (Armchair GNR) 的一个片段（代表图中的最终产物）
    # 这是一个多环芳烃结构，模拟 GNR 最终的平面结构
        "C1=CC2=C3C=CC4=C5C=CC6=CC=CC7=C6C5=C4C=C3C=C1C2=C7",

    "precursor_2_steric.smi":
    # Precursor 2 的一个片段，用来模拟位阻效应
        "BrC1=CC=C(C=C1)C2=C(Br)C=C(Br)C=C2C3=CC=CC=C3",

    "gnr_heterojunction_segment.smi":
    # 异质结 GNR 的一个片段
        "C1=CC2=C3C=CC4=C5C=CC6=CC=CC7=C6C5=C4C=C3C=C1C2=C7.C1=CC2=C3C=CC4=C5C=CC6=CC=CC7=C6C5=C4C=C3C=C1C2=C7",

    "alkane_C100.smi":
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",

    "branched_C100_v1.smi":
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)CCCCCCCCCCCC",

    "branched_C100_v2.smi":
        "CCCCCCCCCCCCCCCCCCCCCCCCCC(C)CCCCCCCCCCCCCCCC(CC)CCCCCCCCCCCCCCCC(C)CCCCCCCCCCCCCCCC",
}



def generate_smiles_files():
    """创建数据文件夹并保存 SMILES 文件"""

    # 找到脚本目录
    current_dir = pathlib.Path(__file__).resolve().parent
    data_dir = current_dir / DATA_FOLDER

    print(f"尝试创建数据文件夹: {data_dir}")
    # 创建目录，如果目录已存在则不会报错
    data_dir.mkdir(exist_ok=True)

    print("-" * 30)
    print("开始生成 SMILES 文件...")

    for filename, smiles in TEST_SMILES_DATA.items():
        file_path = data_dir / filename

        # 将 SMILES 写入文件
        try:
            with open(file_path, 'w') as f:
                f.write(smiles + '\n')  # 确保 SMILES 占一行
            print(f"[成功] 生成文件: {filename}")
        except Exception as e:
            print(f"[失败] 写入文件 {filename} 失败: {e}")

    print("-" * 30)
    print("数据生成完毕。")


if __name__ == "__main__":
    generate_smiles_files()