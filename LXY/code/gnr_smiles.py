from typing import List, Dict, Tuple
from rdkit import Chem
from gnr_types import BenzeneHex

def extract_submol_from_hexes(original_mol, hexes_subset: List[BenzeneHex]):
    """辅助函数：根据一组 Hex 生成 RDKit 分子对象"""
    atom_indices = set()
    for h in hexes_subset:
        for aid in h.atom_indices:
            atom_indices.add(aid)

    if not atom_indices: return None

    rw_mol = Chem.RWMol()
    old_to_new_map = {}
    sorted_old_indices = sorted(list(atom_indices))

    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        new_idx = rw_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        old_to_new_map[old_idx] = new_idx

    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in atom_indices and n_idx > old_idx:
                bond = original_mol.GetBondBetweenAtoms(old_idx, n_idx)
                if bond:
                    rw_mol.AddBond(old_to_new_map[old_idx], old_to_new_map[n_idx], bond.GetBondType())

    return rw_mol


def generate_monomer_smiles_periodic(original_mol, all_hexes: List[BenzeneHex],
                                     global_cutting_plan: Dict[int, List[Tuple[int, int]]],
                                     k: int, max_col: int, filename: str):
    """
    智能周期性提取法：
    1. 全局移除红线环。
    2. 在 GNR 中间位置使用滑动窗口 (Sliding Window)，窗口宽度为 K。
    3. 寻找一个相位 (Shift)，使得该窗口内提取出的分子是“最大连通”的。
    这能解决单体跨越 Col 边界导致被切断的问题。
    """

    # 1. 确定全局移除的环
    removed_hex_ids = set()
    for path in global_cutting_plan.values():
        for (u, v) in path:
            removed_hex_ids.add(u)
            removed_hex_ids.add(v)

    # 2. 筛选保留的环
    kept_hexes = [h for h in all_hexes if h.id not in removed_hex_ids]
    if not kept_hexes:
        print("    [失败] 所有环都被切除")
        return

    # 3. 设定滑动窗口的搜索区域 (避免边缘效应，取中间一段)
    # 我们选择分子中间的一个周期作为基准
    mid_col = max_col // 2
    # 将 mid_col 对齐到 K 的倍数作为基准搜索起点
    base_start_col = (mid_col // k) * k
    if base_start_col < 0: base_start_col = 0

    best_mol = None
    max_atoms = 0

    # 4. 尝试 K 种相位偏移 (Shift 0, 1, ..., K-1)
    # 只要有一种偏移能“完美框住”单体（即不切断蓝色键），我们就选它
    for shift in range(k):
        # 窗口范围: [start, end)
        window_start = base_start_col + shift
        window_end = window_start + k

        # 收集窗口内的保留环
        window_hexes = [h for h in kept_hexes if window_start <= h.col < window_end]
        if not window_hexes: continue

        # 生成分子
        mol = extract_submol_from_hexes(original_mol, window_hexes)
        if not mol: continue

        # 检查连通性
        # 我们希望提取出的是一个完整的单体，而不是断裂的碎片
        frags = Chem.GetMolFrags(mol.GetMol(), asMols=True, sanitizeFrags=False)
        if not frags: continue

        # 获取最大的碎片（假设单体是最大的那部分）
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())

        # 评分：优先选择原子数最多的完整碎片
        # 逻辑是：如果窗口切坏了单体，原子数会变少（因为一部分原子被切到窗口外了）
        # 只有当窗口完美对齐时，原子数才是最大的
        if largest_frag.GetNumAtoms() > max_atoms:
            max_atoms = largest_frag.GetNumAtoms()
            best_mol = largest_frag

    # 5. 保存结果
    if best_mol:
        try:
            Chem.SanitizeMol(best_mol)
            smi = Chem.MolToSmiles(best_mol)
            with open(filename, 'w') as f:
                f.write(smi)
            print(f"    [SMILES] 已保存: {filename} (原子数: {max_atoms})")
        except Exception as e:
            print(f"    [SMILES 错误] {filename}: {e}")
    else:
        print(f"    [SMILES 警告] 未能提取有效单体: {filename}")
