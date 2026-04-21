import os
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from gnr_types import BenzeneHex

def extract_submol_from_hexes(original_mol, hexes_subset: List[BenzeneHex]):
    """提取原始未封端分子骨架"""
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

def get_atom_to_hexes_map(all_hexes: List[BenzeneHex]) -> Dict[int, List[BenzeneHex]]:
    """建立 原子ID 到 苯环对象 的反向映射"""
    mapping = {}
    for h in all_hexes:
        for idx in h.atom_indices:
            mapping.setdefault(idx, []).append(h)
    return mapping

def extract_all_capped_monomers(original_mol, window_hexes: List[BenzeneHex], all_hexes: List[BenzeneHex], removed_hex_ids: set) -> List[Chem.Mol]:
    """
    终极化学真理版：
    - Ullmann 主链 (Br) 固定在对角的 Alpha 键（即视觉上的 Top-Right 和 Bottom-Left）。
    - Scholl 海湾 (C) 在单侧剩余的两个断键中二选一进行 $2 \times 2$ 组合。
    - 完美生成 1,5-二溴 的 4 种甲基异构体！
    """
    monomer_atom_indices = set(aid for h in window_hexes for aid in h.atom_indices)
    if not monomer_atom_indices: return []

    base_rw_mol = Chem.RWMol()
    old_to_new_map = {}
    sorted_old_indices = sorted(list(monomer_atom_indices))

    for old_idx in sorted_old_indices:
        atom = original_mol.GetAtomWithIdx(old_idx)
        new_idx = base_rw_mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        base_rw_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
        old_to_new_map[old_idx] = new_idx

    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in monomer_atom_indices and n_idx > old_idx:
                bond = original_mol.GetBondBetweenAtoms(old_idx, n_idx)
                if bond:
                    base_rw_mol.AddBond(old_to_new_map[old_idx], old_to_new_map[n_idx], bond.GetBondType())

    conf = original_mol.GetConformer()

    # 1. 找到所有向外的断键
    broken_bonds = []
    for old_idx in sorted_old_indices:
        orig_atom = original_mol.GetAtomWithIdx(old_idx)
        for neighbor in orig_atom.GetNeighbors():
            if neighbor.GetIdx() not in monomer_atom_indices:
                broken_bonds.append(old_idx)
                break

    if not broken_bonds: return []

    # 2. 将断键分为左右两侧，并严格按 Y 坐标排序 (0为Top，-1为Bottom)
    cx = sum(conf.GetAtomPosition(idx).x for idx in broken_bonds) / len(broken_bonds)
    left_bonds = sorted([u for u in broken_bonds if conf.GetAtomPosition(u).x < cx],
                        key=lambda u: conf.GetAtomPosition(u).y, reverse=True)
    right_bonds = sorted([u for u in broken_bonds if conf.GetAtomPosition(u).x >= cx],
                         key=lambda u: conf.GetAtomPosition(u).y, reverse=True)

    # 3. 严格分配逻辑：强制 1,5-二溴 (对角 Alpha 键)，甲基在剩余边缘二选一
    left_choices = []
    if len(left_bonds) >= 3:
        # 左侧的 5 位 (Bottom-Left) 固定为 Br
        br_bond_left = left_bonds[-1]
        left_choices = [
            {br_bond_left: 'Br', left_bonds[0]: 'C'}, # 甲基在 8 位 (Top-Left)
            {br_bond_left: 'Br', left_bonds[1]: 'C'}  # 甲基在 6 位 (Middle-Left)
        ]
    elif len(left_bonds) == 2:
        left_choices = [
            {left_bonds[1]: 'Br', left_bonds[0]: 'C'},
            {left_bonds[0]: 'Br', left_bonds[1]: 'C'}
        ]
    elif len(left_bonds) == 1:
        left_choices = [{left_bonds[0]: 'Br'}]
    else:
        left_choices = [{}]

    right_choices = []
    if len(right_bonds) >= 3:
        # 右侧的 1 位 (Top-Right) 固定为 Br
        br_bond_right = right_bonds[0]
        right_choices = [
            {br_bond_right: 'Br', right_bonds[-1]: 'C'}, # 甲基在 4 位 (Bottom-Right)
            {br_bond_right: 'Br', right_bonds[1]: 'C'}   # 甲基在 2 位 (Middle-Right)
        ]
    elif len(right_bonds) == 2:
        right_choices = [
            {right_bonds[0]: 'Br', right_bonds[1]: 'C'},
            {right_bonds[1]: 'Br', right_bonds[0]: 'C'}
        ]
    elif len(right_bonds) == 1:
        right_choices = [{right_bonds[0]: 'Br'}]
    else:
        right_choices = [{}]

    # 4. 组装最终的 4 种完美单体异构体
    capped_mols = []
    for lc in left_choices:
        for rc in right_choices:
            rw_mol = Chem.RWMol(base_rw_mol)
            
            # 应用左侧分配
            for u, cap_type in lc.items():
                cap = rw_mol.AddAtom(Chem.Atom(cap_type))
                rw_mol.AddBond(old_to_new_map[u], cap, Chem.BondType.SINGLE)
                
            # 应用右侧分配
            for u, cap_type in rc.items():
                cap = rw_mol.AddAtom(Chem.Atom(cap_type))
                rw_mol.AddBond(old_to_new_map[u], cap, Chem.BondType.SINGLE)
                
            try:
                Chem.SanitizeMol(rw_mol)
                capped_mols.append(rw_mol.GetMol())
            except:
                pass
            
    return capped_mols

def generate_monomer_smiles_periodic(original_mol, all_hexes: List[BenzeneHex],
                                     global_cutting_plan: Dict[int, List[Tuple[int, int]]],
                                     k: int, max_col: int,
                                     raw_smi_filename: str, capped_smi_filename: str, img_filename: str):
    
    removed_hex_ids = set()
    for path in global_cutting_plan.values():
        for (u, v) in path:
            removed_hex_ids.add(u)
            removed_hex_ids.add(v)

    kept_hexes = [h for h in all_hexes if h.id not in removed_hex_ids]
    if not kept_hexes: return

    # 只保留 max_atoms 保证骨架完整，去除阻挡不同相位的严苛过滤器
    max_atoms = 0
    valid_windows = []
    
    for start_col in range(max_col - k + 1):
        window_hexes = [h for h in kept_hexes if start_col <= h.col < start_col + k]
        if not window_hexes: continue

        mol = extract_submol_from_hexes(original_mol, window_hexes)
        if not mol: continue

        frags = Chem.GetMolFrags(mol.GetMol(), asMols=True, sanitizeFrags=False)
        if not frags: continue
        
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        num_atoms = largest_frag.GetNumAtoms()

        if num_atoms > max_atoms:
            max_atoms = num_atoms
            valid_windows = [window_hexes]
        elif num_atoms == max_atoms:
            valid_windows.append(window_hexes)

    unique_raw_smiles = set()
    raw_results = []
    unique_capped_smiles = set()
    capped_results = []

    for window_hexes in valid_windows:
        raw_mol_rw = extract_submol_from_hexes(original_mol, window_hexes)
        if raw_mol_rw:
            frags = Chem.GetMolFrags(raw_mol_rw.GetMol(), asMols=True, sanitizeFrags=False)
            if frags:
                best_raw_mol = max(frags, key=lambda m: m.GetNumAtoms())
                try:
                    Chem.SanitizeMol(best_raw_mol)
                    smi = Chem.MolToSmiles(best_raw_mol)
                    if smi not in unique_raw_smiles:
                        unique_raw_smiles.add(smi)
                        raw_results.append(best_raw_mol)
                except: pass

        capped_mols = extract_all_capped_monomers(original_mol, window_hexes, all_hexes, removed_hex_ids)
        for capped_mol in capped_mols:
            frags = Chem.GetMolFrags(capped_mol, asMols=True, sanitizeFrags=True)
            if frags:
                best_capped_mol = max(frags, key=lambda m: m.GetNumAtoms())
                try:
                    smi = Chem.MolToSmiles(best_capped_mol)
                    if smi not in unique_capped_smiles:
                        unique_capped_smiles.add(smi)
                        capped_results.append(best_capped_mol)
                except: pass

    # 保存文件
    for idx, mol in enumerate(raw_results):
        smi = Chem.MolToSmiles(mol)
        out_name = raw_smi_filename if len(raw_results) == 1 else raw_smi_filename.replace(".smi", f"_{idx+1}.smi")
        try:
            with open(out_name, 'w') as f:
                f.write(smi)
            print(f"    [成功] 原始骨架保存: {os.path.basename(out_name)}")
        except: pass

    for idx, mol in enumerate(capped_results):
        smi = Chem.MolToSmiles(mol)
        out_smi_name = capped_smi_filename if len(capped_results) == 1 else capped_smi_filename.replace(".smi", f"_{idx+1}.smi")
        out_img_name = img_filename if len(capped_results) == 1 else img_filename.replace(".png", f"_{idx+1}.png")
        
        try:
            with open(out_smi_name, 'w') as f:
                f.write(smi)
            AllChem.Compute2DCoords(mol)
            Draw.MolToFile(mol, out_img_name, size=(600, 600))
            print(f"    [成功] 智能封端保存: {os.path.basename(out_smi_name)} | 图像: {os.path.basename(out_img_name)}")
        except: pass
