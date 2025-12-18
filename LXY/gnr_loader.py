import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import os


class CarbonNode:
    """存储单个碳原子的信息"""

    def __init__(self, atom_id, element, x, y, hybridization):
        self.id = atom_id
        self.element = element
        self.pos = (x, y)  # 存储 (x, y) 坐标
        self.hybridization = hybridization  # 存储杂化类型 (SP2, SP3)

    def __repr__(self):
        return f"Node({self.id}, {self.element}, {self.hybridization})"


class GNRGraph:
    """存储整个高分子图结构"""

    def __init__(self):
        self.nodes = {}  # 字典 {id: CarbonNode}
        self.edges = []  # 列表 [(id1, id2, bond_type)]
        # 为了方便后续算法，我们同时也维护一个 NetworkX 图对象
        self.nx_graph = nx.Graph()

    def add_node(self, atom_id, element, x, y, hybridization):
        node = CarbonNode(atom_id, element, x, y, hybridization)
        self.nodes[atom_id] = node
        self.nx_graph.add_node(atom_id, pos=(x, y), label=element)

    def add_edge(self, u, v, bond_type):
        self.edges.append((u, v, bond_type))
        self.nx_graph.add_edge(u, v, type=bond_type)

    def get_summary(self):
        return f"GNR Structure: {len(self.nodes)} Atoms, {len(self.edges)} Bonds."


# --- 2. 核心功能函数 ---

def read_smiles_file(file_path):
    """读取文件中的第一行作为 SMILES"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        # 假设文件里只有一行 SMILES，或者第一行就是
        smiles = f.readline().strip()
    return smiles


def parse_smiles_to_gnr(smiles):
    """
    将 SMILES 转换为自定义的 GNRGraph 数据结构
    关键步骤：必须生成 2D 坐标，因为 SMILES 本身不包含空间信息
    """
    # 1. RDKit 读取
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    # 4. 初始化我们的自定义图
    gnr_graph = GNRGraph()

    # 5. 提取原子 (Nodes)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        hyb = str(atom.GetHybridization())  # 获取杂化方式 (SP2, SP3)

        # 获取坐标 (RDKit 的坐标通常是 Angstrom 或者是绘图单位)
        pos = conf.GetAtomPosition(idx)

        gnr_graph.add_node(idx, sym, pos.x, pos.y, hyb)

    # 6. 提取键 (Edges)
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        b_type = str(bond.GetBondType())  # SINGLE, DOUBLE, AROMATIC
        gnr_graph.add_edge(u, v, b_type)

    return mol, gnr_graph

def verify_and_visualize(mol, gnr_graph, output_prefix="output"):
    print("-" * 30)
    print("验证信息 / Verification Info")
    print("-" * 30)
    print(gnr_graph.get_summary())

    # --- 验证 1: 文本形式的“邻接表” (Adjacency List) ---
    print("\n前 5 个原子的连接关系示例:")
    for i in range(min(5, len(gnr_graph.nodes))):
        neighbors = list(gnr_graph.nx_graph.neighbors(i))
        node_info = gnr_graph.nodes[i]
        print(f"ID {i} ({node_info.element}, {node_info.hybridization}) 连着 -> {neighbors}")

    # --- 验证 2: 输出自定义数据结构的图 (NetworkX + Matplotlib) ---
    # 这张图现在是唯一的可视化验证手段，它依赖于你存储的 (x, y) 坐标。

    plt.figure(figsize=(10, 6))

    # 获取节点位置 (从自定义的 gnr_graph 中提取的坐标)
    pos = nx.get_node_attributes(gnr_graph.nx_graph, 'pos')

    # 获取节点标签 (例如 'C', 'Br')
    labels = nx.get_node_attributes(gnr_graph.nx_graph, 'label')

    # 绘制
    nx.draw(gnr_graph.nx_graph, pos,
            labels=labels,  # 使用原子符号作为标签
            with_labels=True,
            node_color='lightblue',
            node_size=600,
            font_size=10,
            edge_color='gray')

    plt.title("Visualized from Custom Data Structure (NetworkX)")
    nx_img_path = f"{output_prefix}_custom_struct.png"
    plt.savefig(nx_img_path)
    plt.close()
    print(f"[成功] 自定义数据结构可视化已保存为: {nx_img_path}")
    print("\n结论: 请检查图片结构是否符合预期。")

# --- 主程序入口 ---

if __name__ == "__main__":
    file_path = "smile/gnr_7ac_segment.smi"

    # 1. 读取
    print(f"正在读取文件: {file_path} ...")
    try:
        smiles_str = read_smiles_file(file_path)
    except FileNotFoundError as e:
        print(f"错误：文件未找到。请检查路径是否正确: {e}")
        exit()  # 遇到错误就退出程序

    # 2. 解析存储
    print("正在解析并构建图结构...")
    try:
        rdkit_mol, my_gnr = parse_smiles_to_gnr(smiles_str)
    except ValueError as e:
        print(f"错误：SMILES 字符串解析失败。请检查 SMILES 内容: {smiles_str[:50]}...")
        exit()

    # 3. 验证输出
    verify_and_visualize(rdkit_mol, my_gnr, output_prefix="gnr_check")