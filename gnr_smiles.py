import os
from pathlib import Path
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from gnr_types import BenzeneHex
import math

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

    # 2. 将断键分为左右两侧，并建立局部相对坐标系 (抵抗全局分子倾斜)
    if len(window_hexes) == 2:
        h1, h2 = window_hexes[0], window_hexes[1]
        # 确保 h1 在左，h2 在右
        if h1.col > h2.col or (h1.col == h2.col and h1.cx > h2.cx):
            h1, h2 = h2, h1

        # 建立局部坐标系
        vec_x = (h2.cx - h1.cx, h2.cy - h1.cy)
        length_x = math.hypot(vec_x[0], vec_x[1]) if math.hypot(vec_x[0], vec_x[1]) > 0 else 1.0
        ux = (vec_x[0] / length_x, vec_x[1] / length_x)
        uy = (-ux[1], ux[0])  # 局部 Y 轴 (指向上)

        mcx = (h1.cx + h2.cx) / 2
        mcy = (h1.cy + h2.cy) / 2

        def get_local_y(u):
            pos = conf.GetAtomPosition(u)
            return (pos.x - mcx) * uy[0] + (pos.y - mcy) * uy[1]

        left_bonds = [u for u in broken_bonds if u in h1.atom_indices and u not in h2.atom_indices]
        right_bonds = [u for u in broken_bonds if u in h2.atom_indices and u not in h1.atom_indices]
        bridgeheads = set(h1.atom_indices).intersection(set(h2.atom_indices))
    else:
        # 保底逻辑 (处理非 K=2 的情况)
        bridgeheads = set(aid for aid in sorted_old_indices if sum(
            1 for n in original_mol.GetAtomWithIdx(aid).GetNeighbors() if n.GetIdx() in monomer_atom_indices) >= 3)
        cx = sum(conf.GetAtomPosition(idx).x for idx in broken_bonds) / len(broken_bonds)
        left_bonds = [u for u in broken_bonds if conf.GetAtomPosition(u).x < cx]
        right_bonds = [u for u in broken_bonds if conf.GetAtomPosition(u).x >= cx]

        def get_local_y(u):
            return conf.GetAtomPosition(u).y

    # 3. 严格拓扑分类：Alpha连着桥头碳，Beta不连
    def categorize_and_sort(bonds):
        alphas, betas = [], []
        for u in bonds:
            if any(n.GetIdx() in bridgeheads for n in original_mol.GetAtomWithIdx(u).GetNeighbors()):
                alphas.append(u)
            else:
                betas.append(u)
        # 统一按局部 Y 轴从上到下排序
        alphas.sort(key=get_local_y, reverse=True)
        betas.sort(key=get_local_y, reverse=True)
        return alphas, betas

    left_alphas, left_betas = categorize_and_sort(left_bonds)
    right_alphas, right_betas = categorize_and_sort(right_bonds)

    left_choices = []
    if len(left_alphas) >= 1:
        br_bond_left = left_alphas[-1]  # Bottom-Alpha (5)
        choices = []
        if len(left_alphas) >= 2:
            choices.append({br_bond_left: 'Br', left_alphas[0]: 'C'})  # Top-Alpha (8)
        if len(left_betas) >= 2:
            choices.append({br_bond_left: 'Br', left_betas[-1]: 'C'})  # Bottom-Beta (6)
        elif len(left_betas) == 1:
            choices.append({br_bond_left: 'Br', left_betas[0]: 'C'})
        left_choices = choices if choices else [{br_bond_left: 'Br'}]
    else:
        left_choices = [{}]

    right_choices = []
    if len(right_alphas) >= 1:
        br_bond_right = right_alphas[0]  # Top-Alpha (1)
        choices = []
        if len(right_alphas) >= 2:
            choices.append({br_bond_right: 'Br', right_alphas[-1]: 'C'})  # Bottom-Alpha (4)
        if len(right_betas) >= 1:
            choices.append({br_bond_right: 'Br', right_betas[0]: 'C'})  # Top-Beta (2)
        right_choices = choices if choices else [{br_bond_right: 'Br'}]
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

    if valid_windows:
        target_window = valid_windows[0]
        capped_mols = extract_all_capped_monomers(original_mol, target_window, all_hexes, removed_hex_ids)
        for capped_mol in capped_mols:
            frags = Chem.GetMolFrags(capped_mol, asMols=True, sanitizeFrags=True)
            if frags:
                best_capped_mol = max(frags, key=lambda m: m.GetNumAtoms())
                # 直接添加，不进行 SMILES 字符串去重
                capped_results.append(best_capped_mol)

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
        if len(capped_results) == 1:
            out_smi_name = capped_smi_filename
            out_img_name = img_filename
        else:
            smi_p = Path(capped_smi_filename)
            out_smi_name = str(smi_p.parent / f"{smi_p.stem}_{idx+1}{smi_p.suffix}")
            img_p = Path(img_filename)
            out_img_name = str(img_p.parent / f"{img_p.stem}_{idx+1}{img_p.suffix}")
        try:
            with open(out_smi_name, 'w') as f:
                f.write(smi)
            AllChem.Compute2DCoords(mol)
            drawer = Draw.MolDraw2DSVG(600, 600)
            opts = drawer.drawOptions()
            opts.addAtomIndices = False
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            with open(out_img_name, 'w') as f:
                f.write(drawer.GetDrawingText())
            print(f"    [成功] 智能封端保存: {os.path.basename(out_smi_name)} | 图像: {os.path.basename(out_img_name)}")
        except Exception as exc:
            print(f"    [错误] 智能封端图像生成失败 {os.path.basename(out_smi_name)}: {exc}")
